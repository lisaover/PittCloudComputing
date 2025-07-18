-- Define constants
%declare DATA_TOKEN 'browser_os';

-- Load the data
weblog = LOAD '/user/lao39/logs/web_log.csv' USING PigStorage(',') 
        AS (user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray);

-- Concatenate browser and os and select token
weblog_new = FOREACH weblog GENERATE CONCAT(browser, CONCAT(':', os)) AS browser_os;

-- Get each line
weblog_token = FOREACH weblog_new GENERATE $DATA_TOKEN AS token;

-- Group tokens
grouped = GROUP weblog_token BY token;

-- Count number of tokens in group
token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_token) AS count;

-- Find the maximum count
max_count_data = FOREACH (GROUP token_counts ALL) GENERATE MAX(token_counts.count) AS max_count;

-- Filter records with max_count
max_tokens = JOIN token_counts BY count, max_count_data BY max_count;

-- Store the result to an output directory in HDFS
STORE max_tokens INTO '/user/lao39/output/browser_pig';

