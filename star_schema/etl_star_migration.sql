/*
 The dates in the dim_time dimension are derived from the min and max visit_date in the emr.viist table
 Every month/year between the earliest and latest dates are represented. The table consists of 
 start_dt and end_dt, the first and last days of the month, respectively. All counts in the 
 fact tables are calculated by month. This table also contains the month name and year as 
 separate variables and the number of days in the month so rates and frequencies can be derived.
 Finally, the dimension also contains the quarter for ease of producing quarterly reports.
 
 The temporary table dt_range selects the min and max visit dates. The 
 temporary table temp_dates uses recursion to create all days within the date range and 
 creates the start_dt and end_dt by selecting unique month/year combinations and casting 
 to dates.
*/
CREATE TEMPORARY TABLE IF NOT EXISTS dt_range AS (
SELECT MIN(visit_date) AS start_dt, MAX(visit_date) AS stop_dt FROM emr.visit
);

#SELECT * FROM dt_range;

#DROP TABLE temp_dates
CREATE TEMPORARY TABLE IF NOT EXISTS temp_dates AS (
WITH RECURSIVE dates(dt) AS (
    SELECT MIN(visit_date) AS dt FROM emr.visit
    UNION ALL
    SELECT DATE_ADD(dt, INTERVAL 1 DAY)
    FROM dates
    WHERE dt <= (SELECT stop_dt FROM dt_range)
)
SELECT DISTINCT CAST(CONCAT(CAST(EXTRACT(YEAR FROM dt) AS CHAR), '-', CAST(EXTRACT(MONTH FROM dt) AS CHAR), '-', '01') AS DATE) AS start_dt
		,LAST_DAY(CAST(CONCAT(CAST(EXTRACT(YEAR FROM dt) AS CHAR), '-', CAST(EXTRACT(MONTH FROM dt) AS CHAR), '-', '01') AS DATE)) AS end_dt
    FROM dates
);

#SELECT * FROM temp_dates;

INSERT INTO lao39.dim_time
SELECT NULL AS date_id
	,start_dt
	,end_dt
    ,EXTRACT(DAY FROM end_dt) AS days_in_month
	,DATE_FORMAT(start_dt, '%M') AS month
	,EXTRACT(YEAR FROM start_dt) AS year
	,EXTRACT(QUARTER FROM start_dt) AS quarter
FROM temp_dates
ORDER BY start_dt;

/*
The dim_patient dimension was populated directly from the emr.patient table.
Two addiitonal variables were derived from dob: birth_month and birth_year. 
The age of the patient would not be appropriate in this table because the
patient's care could span many years. The birth_month and birth_year variables
will allow analysts to filter patients by birth_year or birth_year and birth_month
to effectively drill down by age. The gender was also migrated so analysts could
drill down by gender.
*/
INSERT INTO lao39.dim_patient
SELECT patient_id, first_name, last_name, gender, dob
	,DATE_FORMAT(dob, '%M') AS birth_month
	,EXTRACT(YEAR FROM dob) AS birth_year 
FROM emr.patient
WHERE patient_id IS NOT NULL;

/*
The provider, lab, procedure, and diagnosis dimensions were populated
directly from the respective emr tables.
*/

INSERT INTO lao39.dim_provider
SELECT * FROM emr.provider
WHERE provider_id IS NOT NULL;

INSERT INTO lao39.dim_lab
SELECT * FROM emr.lab
WHERE lab_id IS NOT NULL;

INSERT INTO lao39.dim_procedure
SELECT * FROM emr.clinical_procedures
WHERE procedure_id IS NOT NULL;

INSERT INTO lao39.dim_diagnosis
SELECT * FROM emr.diagnosis
WHERE diagnosis_id IS NOT NULL;

/*
The fact_patient fact table consists of four counts related to provider-patient-month/year
combinations: visit_count, lab_count, procedure_count, and diagnosis_count. Each count
is calculated separately in common table expression and joined on patient_id, provider_id,
and date_id.
*/

INSERT INTO lao39.fact_patient
WITH vtime AS (
	SELECT v.visit_id, v.provider_id, v.patient_id, t.date_id
		,v.visit_date, t.start_dt, t.end_dt
	FROM emr.visit v
	LEFT JOIN lao39.dim_time t
	ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt
)
,vcnt AS (
	SELECT t.patient_id, t.provider_id, t.date_id, COUNT(visit_id) AS visit_count
	FROM vtime t
    GROUP BY t.patient_id, t.provider_id, t.date_id
    ORDER BY t.patient_id, t.provider_id, t.date_id
)
#SELECT * FROM vcnt
,lcnt AS (
	SELECT t.patient_id, t.provider_id, t.date_id, COUNT(lab_id) AS lab_count
	FROM vtime t
    LEFT JOIN emr.visit_lab a
    ON t.visit_id = a.visit_id
    GROUP BY t.patient_id, t.provider_id, t.date_id
    ORDER BY t.patient_id, t.provider_id, t.date_id
)
#SELECT * FROM lcnt
,pcnt AS (
	SELECT t.patient_id, t.provider_id, t.date_id, COUNT(procedure_id) AS procedure_count
	FROM vtime t
    LEFT JOIN emr.visit_procedure a
    ON t.visit_id = a.visit_id
    GROUP BY t.patient_id, t.provider_id, t.date_id
    ORDER BY t.patient_id, t.provider_id, t.date_id
)
#SELECT * FROM pcnt
,dcnt AS (
	SELECT t.patient_id, t.provider_id, t.date_id, COUNT(diagnosis_id) AS diagnosis_count
	FROM vtime t
    LEFT JOIN emr.visit_diagnosis a
    ON t.visit_id = a.visit_id
    GROUP BY t.patient_id, t.provider_id, t.date_id
    ORDER BY t.patient_id, t.provider_id, t.date_id
)
#SELECT * FROM dcnt
SELECT DISTINCT t.provider_id, t.patient_id, t.date_id
	,visit_count, lab_count, procedure_count, diagnosis_count
FROM vtime t
LEFT JOIN vcnt v
ON t.patient_id = v.patient_id AND t.provider_id = v.provider_id AND t.date_id = v.date_id
LEFT JOIN lcnt l
ON t.patient_id = l.patient_id AND t.provider_id = l.provider_id AND t.date_id = l.date_id
LEFT JOIN pcnt p
ON t.patient_id = p.patient_id AND t.provider_id = p.provider_id AND t.date_id = p.date_id
LEFT JOIN dcnt d
ON t.patient_id = d.patient_id AND t.provider_id = d.provider_id AND t.date_id = d.date_id
ORDER BY t.patient_id, t.provider_id, t.date_id;

/*
The fact_lab fact table consists of three counts for provider-lab-month/year 
combinations: patient_count, visit_count and lab_count. Counts were calculated 
for each provider and each type of lab for each month/year in the dim_time dimension. 
This allows analysts to drill down by lab or groups of labs.
*/

INSERT INTO lao39.fact_lab
WITH ltime AS (
	SELECT v.provider_id, v.patient_id, a.lab_id, t.date_id
		,v.visit_date, t.start_dt, t.end_dt 
	FROM emr.visit v
	LEFT JOIN emr.visit_lab a
	ON a.visit_id = v.visit_id
	LEFT JOIN lao39.dim_time t
	ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt
    WHERE a.lab_id IS NOT NULL
)
SELECT provider_id, lab_id, date_id
	,COUNT(patient_id) AS patient_count
    ,COUNT(visit_date) AS visit_count
    ,COUNT(lab_id) AS lab_count
FROM ltime
GROUP BY provider_id, lab_id, date_id
ORDER BY provider_id, lab_id, date_id;

/*
The fact_procedure fact table consists of three counts for provider-procedure-month/year 
combinations: patient_count, visit_count and procedure_count. Counts were calculated 
for each provider and each type of procedure for each month/year in the dim_time dimension. 
This allows analysts to drill down by procedure or groups of procedures.
*/

INSERT INTO lao39.fact_procedure
WITH ptime AS (
	SELECT v.provider_id, v.patient_id, a.procedure_id, t.date_id
		,v.visit_date, t.start_dt, t.end_dt 
	FROM emr.visit v
	LEFT JOIN emr.visit_procedure a
	ON a.visit_id = v.visit_id
	LEFT JOIN lao39.dim_time t
	ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt
    WHERE a.procedure_id IS NOT NULL
)
SELECT provider_id, procedure_id, date_id
	,COUNT(patient_id) AS patient_count
    ,COUNT(visit_date) AS visit_count
    ,COUNT(procedure_id) AS procedure_count
FROM ptime
GROUP BY provider_id, procedure_id, date_id
ORDER BY provider_id, procedure_id, date_id;

/*
The fact_diagnosis fact table consists of three counts for provider-diagnosis-month/year 
combinations: patient_count, visit_count and diagnosis_count. Counts were calculated 
for each provider and each type of diagnosis for each month/year in the dim_time dimension. 
This allows analysts to drill down by diagnosis or groups of diagnoses.
*/
INSERT INTO lao39.fact_diagnosis
WITH dtime AS (
	SELECT v.provider_id, v.patient_id, a.diagnosis_id, t.date_id
		,v.visit_date, t.start_dt, t.end_dt 
	FROM emr.visit v
	LEFT JOIN emr.visit_diagnosis a
	ON a.visit_id = v.visit_id
	LEFT JOIN lao39.dim_time t
	ON v.visit_date >= t.start_dt AND v.visit_date <= t.end_dt
    WHERE a.diagnosis_id IS NOT NULL
)
SELECT provider_id, diagnosis_id, date_id
	,COUNT(patient_id) AS patient_count
    ,COUNT(visit_date) AS visit_count
    ,COUNT(diagnosis_id) AS diagnosis_count
FROM dtime
GROUP BY provider_id, diagnosis_id, date_id
ORDER BY provider_id, diagnosis_id, date_id;






