-- Define constants
%declare NUMBER_TO_RETURN 10;
%declare DATA_TOKEN 'search_query';

-- Load the data
weblog_hd = LOAD '/user/lao39/logs/web_log.csv' USING PigStorage(',') 
        AS (timestamp:chararray, user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray);

-- Filter out the file header
weblog = FILTER weblog_hd BY timestamp != 'timestamp';

-- Get each line
weblog_lines = FOREACH weblog GENERATE FLATTEN($DATA_TOKEN) AS token;

-- Group tokens
grouped = GROUP weblog_lines BY token;

-- Count number of tokens in group
token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_lines) AS count;

-- Sort in descending order by count
tokens_ordered = ORDER token_counts BY count DESC;

-- Take the top # tokens
top_tokens = LIMIT tokens_ordered $NUMBER_TO_RETURN;

-- Store the result to an output directory in HDFS
STORE top_tokens INTO '/user/lao39/output/freq_pig';

