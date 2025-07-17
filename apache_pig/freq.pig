-- Define constants
%declare NUMBER_TO_RETURN 10;
%declare DATA_TOKEN 'search_query';

-- Load the data
weblog = LOAD '/user/lao39/logs/web_log.csv' USING PigStorage(',') 
        AS (user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray);

-- Remove spaces in search_query
weblog_new = FOREACH weblog GENERATE REPLACE(search_query, ' ', '.') AS search_query;

-- Get each line
weblog_token = FOREACH weblog_new GENERATE FLATTEN(TOKENIZE($DATA_TOKEN)) AS token;

-- Group tokens
grouped = GROUP weblog_token BY token;

-- Count number of tokens in group
token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_token) AS count;

-- Sort in descending order by count
tokens_ordered = ORDER token_counts BY count DESC;

-- Take the top # tokens - THERE IS A KNOWN BUG WITH LIMIT
--top_tokens = LIMIT tokens_ordered $NUMBER_TO_RETURN;

tokens_ordered_clean = FOREACH tokens_ordered GENERATE REPLACE(token, '.', ' ') AS token;

-- Store the result to an output directory in HDFS
STORE tokens_ordered_clean INTO '/user/lao39/output/freq_pig';

