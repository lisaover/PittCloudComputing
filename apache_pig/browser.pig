-- Define constants
DEFINE DATA_TOKEN 'browser, os';

-- Load the data
weblog_hd = LOAD '/data/web_log.csv' USING PigStorage(',') 
        AS (timestamp:chararray, user_id:chararray, search_query:chararray, browser:chararray, os:chararray, referrer:chararray)

-- Filter out the file header
weblog = FILTER weblog_hd BY timestamp != 'timestamp';

-- Get each line
weblog_lines = FOREACH weblog GENERATE FLATTEN(DATA_TOKEN) AS token;

-- Group tokens
grouped = GROUP weblog_lines BY token;

-- Count number of tokens in group
token_counts = FOREACH grouped GENERATE group AS token, COUNT(weblog_lines) AS count;

-- Find the maximum count
max_count_data = FOREACH (GROUP token_counts ALL) GENERATE MAX(token_counts.count) AS max_count;

-- Filter records with max_count
max_tokens = JOIN token_counts BY count, max_count_data BY max_count;

