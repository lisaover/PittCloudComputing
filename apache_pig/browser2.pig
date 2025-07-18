-- Define constants
%declare DATA_TOKEN 'browser';

-- Load the data
weblog = LOAD '/user/lao39/logs/web_log.csv' USING PigStorage(',') 
        AS (user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray);

-- Concatenate browser and os and select token
--weblog_new = FOREACH weblog GENERATE CONCAT(browser, CONCAT(':', os)) AS browser_os;

-- Group by browser and os
weblog_group = GROUP weblog BY ($DATA_TOKEN);

-- Get each token and SUM 
group_counts = FOREACH weblog_group { GENERATE FLATTEN(group) AS ($DATA_TOKEN), SUM(weblog.os) AS os_count;

-- Group tokens
--grouped = GROUP weblog_token BY token;

-- Count number of tokens in group
--token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_token) AS count;

-- Find the maximum count
max_count_data = FOREACH (GROUP group_counts ALL) GENERATE MAX(group_counts.count) AS max_count;

-- Filter records with max_count
max_tokens = JOIN group_counts BY count, max_count_data BY max_count;

-- Store the result to an output directory in HDFS
STORE max_tokens INTO '/user/lao39/output/browser_pig2';

