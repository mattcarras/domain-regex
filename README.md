# domain_regex.py
Performs semi-optimized regular expression matching given date range constraints, grouping by given domains. The expressions do not need to contain any actual regular expressions, instead defaulting to simple word and phrase matching. 

Requires numpy and pandas.

Last tested on Python 3.8.2 64-bit with pandas 1.1.0 and numpy 1.19.1.

## History
This was part of a project to parse data from Discord. As such it's designed to work with columns representing Date/Timestamp, Message, UserId, and Username, though theoretically only the Message column is required.

## Performance
The function combines all regular expressions before processing, significantly speeding up execution. The matched expressions are then parsed again to figure out which domains they fall in and whether any date ranges need to be filtered out per domain (the more complex this is, the longer it takes). Adding simple regular expressions like "widgets?" to match singular/plural does not seem to necessarily speed up execution. The function makes heavy use of vectorization wherever possible.

Below were tested using a Surface Book 2 (6th Gen Core i7). Actual workstations should see significant improvements.

Each test used the same domain regex file with 535 domain/expression combinations, most simple, some complex, some with date ranges.

- 85k messages
- 1k unique usernames/userids
- **Avg speed over 3 iterations: ~26s**
<br />

- 848k messages
- 10k unique usernames/userids
- **Avg speed over 3 iterations: ~305s**

## Duplicated Timestamp Error
The main `domain_regex` function will raise an error if the given Date column contains any duplicated timestamps. You can suppress this error by giving `verify_timestamps=False`, but the script will likely error afterwards with a duplicate axis error when trying to re-join the filtered matched expressions with the main dataframe.

You can check for duplicate timestamps in your dataframe beforehand:
```
>>> mask = df.duplicated(subset=['Date'], keep=False)
>>> mask.any()
>>> df[mask]
```
## Example usage
```
python -i domain_regex.py
>>> df_input = pd.read_csv("fake_discord_chat_data.csv", encoding='utf-8', parse_dates=['Date'], infer_datetime_format=True)
>>> df_matches = domain_regex(df_input, "domain_regex.csv", cols={'Date':'Date','Message':'Content','UserId':'AuthorID','Username':'Author'}, print_stats=True, file_stats_prefix='stats')
```

## Input file Notes
Supported input domain regular expression CSV columns: `Domain,Expression,Allow Partial Match,Begin Date,End Date`
Any other columns, such as the "Comments" column in the example file, will be ignored.

Supported input dataframe columns: `Date,Message,UserId,Username`
These columns are mapped by the `cols={...}` parameter. Any other columns are ignored. The only required mapping is `Message`. 

## Useful advanced regular expressions
```
WORD1(?:[\w'-]+\s+){0,6}?WORD2
```
Allows 0-6 words or spaces in-between with at least one space (word1 NEAR word2). Where WORD1 and WORD2 are whatever you define.
