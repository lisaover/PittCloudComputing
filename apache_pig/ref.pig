-- Define constants
%declare DATA_TOKEN 'referrer';

-- Load the data
weblog_hd = LOAD '/user/lao39/logs/web_log.csv' USING PigStorage(',') 
        AS (user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray);

-- Remove spaces in search_query
weblog = FOREACH weblog_hd GENERATE user_id, browser, os, referrer, REPLACE(search_query, ' ', '') AS search_query;

-- Get each line
weblog_lines = FOREACH weblog GENERATE FLATTEN(TOKENIZE($DATA_TOKEN)) AS token;

-- Group tokens
grouped = GROUP weblog_lines BY token;

-- Count number of tokens in group
token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_lines) AS count;

-- Find the maximum count
max_count_data = FOREACH (GROUP token_counts ALL) GENERATE MAX(token_counts.count) AS max_count;

-- Filter records with max_count
max_tokens = JOIN token_counts BY count, max_count_data BY max_count;

-- Store the result to an output directory in HDFS
STORE max_tokens INTO '/user/lao39/output/ref_pig';

