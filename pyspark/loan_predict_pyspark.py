#!/usr/bin/env python
# coding: utf-8

# # Loan Approval
# Building a Predictive Model with PySpark and MLlib<BR>
# Summer 2025: Foundations of Cloud Computing for Data Science
# #### Lisa Over

# In[1]:


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pyspark.sql.functions import col, sum, count, when, isnan, skewness, kurtosis, log1p, min, max, lit, expr
from pyspark.sql.types import DoubleType, IntegerType, FloatType, StringType
from pyspark.ml.stat import Correlation
from scipy.stats import chi2_contingency
from pyspark.ml.feature import Imputer, PCA
from pyspark.ml.linalg import Vectors

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


# ##### Define the input and output paths

# In[2]:


input_path = "data/loan_data.csv"
output_path = "output/output.txt"


# ##### Create the Spark Session

# In[3]:


spark = SparkSession.builder.appName("LoanApproval").getOrCreate()


# ##### Read the data

# In[4]:


df = spark.read.csv(input_path, header=True, inferSchema=True)


# ## EDA

# In[5]:


df.printSchema()


# In[6]:


df.count()


# In[7]:


len(df.columns)


# In[8]:


df.select("Dependents").distinct().show()


# In[9]:


df.groupBy("Dependents").count().show()


# In[10]:


df.select("Loan_ID").distinct().count()


# In[11]:


df.show()


# In[12]:


df.describe().show()


# #### Display missing value counts

# In[13]:


null_df = df.select([count(when(col(c).contains('None') | \
                                col(c).contains('NULL') | \
                                (col(c) == '') | \
                                col(c).isNull() | \
                                isnan(c), c
                               )).alias(c)
                     for c in df.columns])

null_df.show()


# #### Create a new variable that is the sum of ApplicantIncome and CoapplicantIncome
# This new variable will not be used in conjunction with either ApplicantIncome or CoapplicantIncome. It is created for exploration and may or may not end up in the model.

# In[14]:


df = df.withColumn("TotalIncome", expr("ApplicantIncome + CoapplicantIncome"))


# #### Visualize the data using Pandas and Seaborn

# In[15]:


vis_df = df.withColumn('Credit_History', df.Credit_History.cast(StringType())).toPandas()


# In[16]:


numerical_columns = vis_df.select_dtypes(include='number').columns.tolist()
numerical_columns


# In[17]:


categorical_columns = vis_df.select_dtypes(include='object').columns.tolist()
categorical_columns.remove('Loan_ID')
categorical_columns


# In[18]:


vis_df.head()


# In[19]:


vis_df.describe()


# In[20]:


for c in categorical_columns:
    print(vis_df[c].value_counts())
    print('Missing: ' + str(vis_df[c].isna().sum()))
    print()


# In[21]:


categorical_columns.remove('Loan_Status')


# #### Evaluate Outliers

# In[22]:


for c in numerical_columns:

    Q1 = vis_df[c].quantile(0.25)
    Q3 = vis_df[c].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    vis_df[c + '_out'] = [1 if x < lower_bound or x > upper_bound else 0 for x in vis_df[c]]
    vis_df[c + '_low'] = [1 if x < lower_bound else 0 for x in vis_df[c]]
    vis_df[c + '_high'] = [1 if x > upper_bound else 0 for x in vis_df[c]]

    outliers_iqr = vis_df[(vis_df[c] < lower_bound) | (vis_df[c] > upper_bound)]
    print("\nOutliers for '" + c + "' using IQR method:\n", outliers_iqr.shape[0])


# In[23]:


vis_df['ApplicantIncome_out'].sum()


# In[24]:


data = [("ApplicantIncome", int(vis_df['ApplicantIncome_out'].sum()), int(vis_df['ApplicantIncome_low'].sum()), int(vis_df['ApplicantIncome_high'].sum())),
        ("CoapplicantIncome", int(vis_df['CoapplicantIncome_out'].sum()), int(vis_df['CoapplicantIncome_low'].sum()), int(vis_df['CoapplicantIncome_high'].sum())),
        ("LoanAmount", int(vis_df['LoanAmount_out'].sum()), int(vis_df['LoanAmount_low'].sum()), int(vis_df['LoanAmount_high'].sum())),
        ("Loan_Amount_Term", int(vis_df['Loan_Amount_Term_out'].sum()), int(vis_df['Loan_Amount_Term_low'].sum()), int(vis_df['Loan_Amount_Term_high'].sum()))]

columns = ["Variable", "Total_Outliers", "Low_Outliers", "High_Outliers"]

outliers_df = spark.createDataFrame(data, columns)


# In[25]:


outliers_df.show()


# #### Boxplots of numerical columns and `Loan_Status`

# In[26]:


for c in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=vis_df[c], x=vis_df['Loan_Status'])
    plt.title('Boxplot to Visualize Outliers for ' + c)
    plt.show()

    for i in categorical_columns:
        print(vis_df[i].value_counts())


# #### Correlation Matrix Heatmap

# In[27]:


correlation_matrix = vis_df.loc[:, numerical_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# #### Pair Plot

# In[28]:


sns.pairplot(vis_df.loc[:,numerical_columns])
plt.show()


# ### Log transform numerical variables

# In[29]:


trx_df = df.withColumn("ApplicantIncomeLog", log1p(col("ApplicantIncome"))) \
                    .withColumn("CoapplicantIncomeLog", log1p(col("CoapplicantIncome"))) \
                    .withColumn("LoanAmountLog", log1p(col("LoanAmount"))) \
                    .withColumn("LoanAmountTermLog", log1p(col("Loan_Amount_Term")))


# #### Skewness and Kurtosis

# In[30]:


df.select(skewness('ApplicantIncome'),kurtosis('ApplicantIncome')).show()


# In[31]:


trx_df.select(skewness('ApplicantIncomeLog'),kurtosis('ApplicantIncomeLog')).show()


# In[32]:


df.select(skewness('CoapplicantIncome'),kurtosis('CoapplicantIncome')).show()


# In[33]:


trx_df.select(skewness('CoapplicantIncomeLog'),kurtosis('CoapplicantIncomeLog')).show()


# In[34]:


df.select(skewness('LoanAmount'),kurtosis('LoanAmount')).show()


# In[35]:


trx_df.select(skewness('LoanAmountLog'),kurtosis('LoanAmountLog')).show()


# In[36]:


df.select(skewness('Loan_Amount_Term'),kurtosis('Loan_Amount_Term')).show()


# In[37]:


trx_df.select(skewness('LoanAmountTermLog'),kurtosis('LoanAmountTermLog')).show()


# In[38]:


vis_df2 = trx_df.toPandas()


# In[39]:


log_columns = ['ApplicantIncomeLog', 'CoapplicantIncomeLog', 'LoanAmountLog', 'LoanAmountTermLog']


# In[40]:


sns.pairplot(vis_df2.loc[:,log_columns])
plt.show()


# ### Impute Missing Values (Numeric)
# Impute missing numeric variables before discretizing them. The mode is an appropriate measure to use to impute missing discrete values such as Loan_Amount_term. The mean is an appropriate measure to impute a normally distributed variable such as the log of LoanAmount.

# In[41]:


i_mean = Imputer(strategy='mean', inputCols=['LoanAmountLog'], outputCols=['LoanAmountLogImp'])
i_mode = Imputer(strategy='mode', inputCols=['Loan_Amount_Term'], outputCols=['LoanTermImp'])


imputer_model_mean = i_mean.fit(trx_df)
trx_df2=imputer_model_mean.transform(trx_df)

imputer_model_mode = i_mode.fit(trx_df2)
trx_df3=imputer_model_mode.transform(trx_df2)


# ### Discretize Continuous Variables

# Quartile bins will be used to discretize ApplicantIncomeLog and LoanAmountLog. This type of discretization divides the data into 4 bins with each bin containing the same number of data points. The balanced distribution of data points across bins is helpful when the response variable is imbalanced, and in this dataset, 69% of loans are approved. 
# 
# The variables CoapplicantIncome and Loan_Term_Amount will be discretized in a custom manner. The log transformation did not improve the distribution, and keeping the original values makes binning straight forward and easy to interpret. Forty-four percent of CoapplicantIncome values are zero, and will be discretized as a binary variable: the coapplicant has an income or does not have an income. Eighty-three percent of Loan_Term_Amount values are 360 months, and will be discretized as 

# #### Collect quantile values for binning

# In[42]:


cols_to_bin = ['ApplicantIncomeLog','LoanAmountLogImp','CoapplicantIncome','LoanTermImp']


# In[43]:


bin_dict = {}

for c in cols_to_bin:
    bin_dict[c] = []
    # Define 25th, 50th, and 75th percentiles for bin widths
    quantile_probabilities = [0.25, 0.50, 0.75]

    # Set the relative error to 0 for exact calculation
    relative_error = 0.0

    # Calculate the quartiles for each column
    quartiles = trx_df3.approxQuantile(c, quantile_probabilities, relative_error)

    # Extract the individual quartiles
    q1 = quartiles[0]  # First Quartile (25th percentile)
    q2 = quartiles[1]  # Second Quartile (Median, 50th percentile)
    q3 = quartiles[2]  # Third Quartile (75th percentile)
    q4 = trx_df3.select(max(c)).collect()[0][0]

    bin_dict[c].append(q1)
    bin_dict[c].append(q2)
    bin_dict[c].append(q3)
    bin_dict[c].append(q4)


# In[44]:


bin_dict


# In[45]:


trx_df3.describe("ApplicantIncomeLog", "CoapplicantIncome", "LoanAmountLogImp", "LoanTermImp").show()


# In[46]:


trx_df4 = trx_df3.withColumn(
        "ApplicantIncomeLogCat",
        when(trx_df3.ApplicantIncomeLog <= bin_dict["ApplicantIncomeLog"][0], "min-Q1")
        .when(trx_df3.ApplicantIncomeLog <= bin_dict["ApplicantIncomeLog"][1], "Q1-Q2")
        .when(trx_df3.ApplicantIncomeLog <= bin_dict["ApplicantIncomeLog"][2], "Q2-Q3")
        .when(trx_df3.ApplicantIncomeLog <= bin_dict["ApplicantIncomeLog"][3], "Q3-max")
        .otherwise(lit(None))
).withColumn(
       "CoapplicantIncomeCat",
        when(trx_df3.CoapplicantIncome == 0, "0")
        .otherwise(">0")
).withColumn(
       "LoanAmountLogCat",
        when(trx_df3.LoanAmountLogImp <= bin_dict["LoanAmountLogImp"][0], "min-Q1")
        .when(trx_df3.LoanAmountLogImp <= bin_dict["LoanAmountLogImp"][1], "Q1-Q2")
        .when(trx_df3.LoanAmountLogImp <= bin_dict["LoanAmountLogImp"][2], "Q2-Q3")
        .when(trx_df3.LoanAmountLogImp <= bin_dict["LoanAmountLogImp"][3], "Q3-max")
        .otherwise(lit(None))
).withColumn(
       "LoanTermCat",
        when(trx_df3.LoanTermImp < 360, "<360")
        .when(trx_df3.LoanTermImp == 360, "360")
        .when(trx_df3.LoanTermImp > 360, ">360")
        .otherwise(lit(None))
)


# In[47]:


trx_df4.groupBy("ApplicantIncomeLogCat").count().orderBy("count", ascending=False).show()


# In[48]:


trx_df4.groupBy("CoapplicantIncomeCat").count().orderBy("count", ascending=False).show()


# In[49]:


trx_df4.groupBy("LoanAmountLogCat").count().orderBy("count", ascending=False).show()


# In[50]:


trx_df4.groupBy("LoanTermCat").count().orderBy("count", ascending=False).show()


# #### Select variables for further processing

# In[51]:


trx_df4.printSchema()


# In[52]:


trx_df5 = trx_df4.select('Loan_ID', 'Gender','Married','Dependents','Education','Self_Employed','ApplicantIncomeLogCat','CoapplicantIncomeCat','LoanAmountLogCat','LoanTermCat','Credit_History','Property_Area','Loan_Status')


# #### Bar graphs of categorical variables and `Loan_Status` using Pandas

# In[53]:


vis_df3 = trx_df5.toPandas()


# In[54]:


categorical_columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncomeLogCat','CoapplicantIncomeCat','LoanAmountLogCat','LoanTermCat','Credit_History','Property_Area']


# In[55]:


for c in categorical_columns:
    sns.catplot(data=vis_df3, x=c, col="Loan_Status", kind="count")
    plt.suptitle('Count of applicants by ' + c + ' and Loan_Status', y=1.02)
    plt.show()


# ### Index categorical variables

# In[56]:


trx_df5.printSchema()


# In[57]:


trx_df5.count()


# In[58]:


len(trx_df5.columns)


# In[59]:


gender_indexer = StringIndexer(inputCol='Gender', outputCol='GenderIdx', handleInvalid='keep')
married_indexer = StringIndexer(inputCol='Married', outputCol='MarriedIdx', handleInvalid='keep')
dependents_indexer = StringIndexer(inputCol='Dependents', outputCol='DependentsIdx', handleInvalid='keep')
education_indexer = StringIndexer(inputCol='Education', outputCol='EducationIdx', handleInvalid='keep')
self_employed_indexer = StringIndexer(inputCol='Self_Employed', outputCol='SelfEmployedIdx', handleInvalid='keep')
property_area_indexer = StringIndexer(inputCol='Property_Area', outputCol='PropertyAreaIdx', handleInvalid='keep')
app_income_indexer = StringIndexer(inputCol='ApplicantIncomeLogCat', outputCol='AppIncomeIdx', handleInvalid='keep')
coapp_income_indexer = StringIndexer(inputCol='CoapplicantIncomeCat', outputCol='CoappIncomeIdx', handleInvalid='keep')
loan_amt_indexer = StringIndexer(inputCol='LoanAmountLogCat', outputCol='LoanAmountIdx', handleInvalid='keep')
loan_term_indexer = StringIndexer(inputCol='LoanTermCat', outputCol='LoanTermIdx', handleInvalid='keep')
loan_status_indexer = StringIndexer(inputCol='Loan_Status', outputCol='LoanStatusIdx', handleInvalid='keep')


# In[60]:


genderModel = gender_indexer.fit(trx_df5)
idx_df = genderModel.transform(trx_df5)

marriedModel = married_indexer.fit(idx_df)
idx_df2 = marriedModel.transform(idx_df)

dependentsModel = dependents_indexer.fit(idx_df2)
idx_df3 = dependentsModel.transform(idx_df2)

eduModel = education_indexer.fit(idx_df3)
idx_df4 = eduModel.transform(idx_df3)

self_empModel = self_employed_indexer.fit(idx_df4)
idx_df5 = self_empModel.transform(idx_df4)

propertyModel = property_area_indexer.fit(idx_df5)
idx_df6 = propertyModel.transform(idx_df5)

appIncModel = app_income_indexer.fit(idx_df6)
idx_df7 = appIncModel.transform(idx_df6)

coappIncModel = coapp_income_indexer.fit(idx_df7)
idx_df8 = coappIncModel.transform(idx_df7)

loanTermModel = loan_term_indexer.fit(idx_df8)
idx_df9 = loanTermModel.transform(idx_df8)

loanAmountModel = loan_amt_indexer.fit(idx_df9)
idx_df10 = loanAmountModel.transform(idx_df9)

statusModel = loan_status_indexer.fit(idx_df10)
idx_df11 = statusModel.transform(idx_df10)

idx_df11.show()


# ### Impute Missing Values (Categorical)
# ...except for LoanAmountIdx and LoanTermIdx for which missing values have already been imputed.<br>
# The mode is the appropriate measure to use to impute missing categorical values.

# In[61]:


i_mode = Imputer(strategy='mode', inputCols=['GenderIdx', 'MarriedIdx', 'DependentsIdx', 'SelfEmployedIdx', 'Credit_History'], outputCols=['GenderIdxImp', 'MarriedIdxImp', 'DependentsIdxImp', 'SelfEmployedIdxImp', 'CreditHistoryImp'])

imputer_model_mode = i_mode.fit(idx_df11)
trx_df6=imputer_model_mode.transform(idx_df11)


# In[62]:


trx_df6.printSchema()


# In[63]:


trx_df7 = trx_df6.select('GenderIdxImp','MarriedIdxImp','DependentsIdxImp','EducationIdx','SelfEmployedIdxImp','PropertyAreaIdx','CreditHistoryImp','AppIncomeIdx','CoappIncomeIdx','LoanAmountIdx','LoanTermIdx', 'LoanStatusIdx')


# ### Feature Selection

# Using Chi Square Tests of Independence, the following categorical variables are associated with the response LoanStatusIdx: GenderIdxImp, DependentsIdxImp, SelfEmployedIdxImp, CreditHistoryImp, CoappIncomeIdx, and LoanTermIdxImp. 

# In[64]:


features_for_chi2test = ['GenderIdxImp','MarriedIdxImp','DependentsIdxImp','EducationIdx','SelfEmployedIdxImp','PropertyAreaIdx','CreditHistoryImp','AppIncomeIdx','CoappIncomeIdx','LoanAmountIdx','LoanTermIdx']


# In[65]:


features_for_chi2test


# In[66]:


# Derived from https://www.machinelearningplus.com/pyspark/pyspark-chi-square-test/
for f in features_for_chi2test:
    contingency_table = trx_df7.stat.crosstab(f, "LoanStatusIdx")
    contingency_table_df = contingency_table.toPandas()
    contingency_table_df = contingency_table_df.set_index(f + '_LoanStatusIdx')
    # Chi Square Test
    chi2, p_value, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table_df)
    print('----------------------------------------')
    print(f + '_LoanStatusIdx')
    print('Chi Square Test')
    print('----------------------------------------')
    print(" ")
    print("Chi-Square Statistic:", chi2)
    print("P-Value:", p_value)
    print("Degrees of Freedom:", degrees_of_freedom)
    print(" ")
    print('Contingency Table')
    print(contingency_table_df)
    print(" ")
    print('Expected Frequencies')
    print(pd.DataFrame(expected_frequencies, index=contingency_table_df.index, columns=contingency_table_df.columns))
    print(" ")
    print(" ")


# #### Chi Sq Selector

# In[67]:


features_for_chisq = ['GenderIdxImp','MarriedIdxImp','DependentsIdxImp','EducationIdx','SelfEmployedIdxImp','PropertyAreaIdx','CreditHistoryImp','LoanAmountIdx','LoanTermIdx']


# In[68]:


chisq_assembler = VectorAssembler(inputCols=features_for_chisq, outputCol="assembled_features_for_chisq")


# In[69]:


chisq_selector = ChiSqSelector(numTopFeatures=5, featuresCol="assembled_features_for_chisq",
                         outputCol="chisqFeatures", labelCol="LoanStatusIdx")


# In[70]:


stages = [chisq_assembler, chisq_selector]


# In[71]:


pipeline = Pipeline(stages=stages)
assembleSelectModel = pipeline.fit(trx_df7)


# In[72]:


trx_df8 = assembleSelectModel.transform(trx_df7)


# In[73]:


print("ChiSqSelector output with top %d features selected" % chisq_selector.getNumTopFeatures())
trx_df8.select('chisqFeatures').show()


# ### Encode categorical variables

# In[74]:


trx_df8.printSchema()


# In[75]:


gender_encoder = OneHotEncoder(inputCol='GenderIdxImp', outputCol='GenderEnc')
married_encoder = OneHotEncoder(inputCol='MarriedIdxImp', outputCol='MarriedEnc')
dependents_encoder = OneHotEncoder(inputCol='DependentsIdxImp', outputCol='DependentsEnc')
education_encoder = OneHotEncoder(inputCol='EducationIdx', outputCol='EducationEnc')
selfEmp_encoder = OneHotEncoder(inputCol='SelfEmployedIdxImp', outputCol='SelfEmployedEnc')
property_encoder = OneHotEncoder(inputCol='PropertyAreaIdx', outputCol='PropertyAreaEnc')
credit_encoder = OneHotEncoder(inputCol='CreditHistoryImp', outputCol='CreditHistoryEnc')
appInc_encoder = OneHotEncoder(inputCol='AppIncomeIdx', outputCol='AppIncomeEnc')
coappInc_encoder = OneHotEncoder(inputCol='CoappIncomeIdx', outputCol='CoappIncomeEnc')
loanAmt_encoder = OneHotEncoder(inputCol='LoanAmountIdx', outputCol='LoanAmountEnc')
loanTerm_encoder = OneHotEncoder(inputCol='LoanTermIdx', outputCol='LoanTermEnc')


# In[76]:


stages = [gender_encoder, married_encoder, dependents_encoder, education_encoder, selfEmp_encoder, property_encoder, credit_encoder, appInc_encoder, coappInc_encoder, loanAmt_encoder, loanTerm_encoder]


# In[77]:


pipeline = Pipeline(stages=stages)
encodedModel = pipeline.fit(trx_df8)


# In[78]:


trx_df9 = encodedModel.transform(trx_df8)
trx_df9.printSchema()


# ### Create dataset for models

# In[79]:


df_for_models = trx_df9.select('GenderEnc','MarriedEnc','DependentsEnc','EducationEnc','SelfEmployedEnc','PropertyAreaEnc','CreditHistoryEnc','AppIncomeEnc','CoappIncomeEnc','LoanAmountEnc','LoanTermEnc','chisqFeatures','LoanStatusIdx')


# In[80]:


df_for_models.show()


# ## Models

# #### Split dataset into train and test

# In[81]:


train_data, test_data = df_for_models.randomSplit([0.7, .3])


# #### Define assemblers

# In[82]:


input_list = ['AppIncomeEnc','SelfEmployedEnc','CreditHistoryEnc','LoanAmountEnc','LoanTermEnc']


# In[83]:


assembler = VectorAssembler(inputCols=input_list,
                                    outputCol='features')


# In[84]:


assembler_chi2selector = VectorAssembler(inputCols=['chisqFeatures'],
                                    outputCol='features')


# #### Logistic Regression Using Features Selected Manually from Chi Square Tests

# In[85]:


log_reg = LogisticRegression(featuresCol='features',
                             labelCol='LoanStatusIdx')


# In[86]:


pipeline = Pipeline(stages=[assembler, log_reg])
fit_model = pipeline.fit(train_data)


# In[87]:


results = fit_model.transform(test_data)
results.show()


# In[88]:


res = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='LoanStatusIdx')
ROC_AUC = res.evaluate(results)


# In[89]:


ROC_AUC


# #### Logistic Regression Using ChiSqSelector Results

# In[90]:


pipeline = Pipeline(stages=[assembler_chi2selector, log_reg])
fit_model_chi2selector = pipeline.fit(train_data)


# In[91]:


results_chi2selector = fit_model_chi2selector.transform(test_data)
results_chi2selector.show()


# In[92]:


ROC_AUC_chi2selector = res.evaluate(results_chi2selector)


# In[93]:


ROC_AUC_chi2selector


# In[ ]:




