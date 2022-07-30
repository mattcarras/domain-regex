"""
domain_regex.py

Performs semi-optimized regular expression matching given date range constraints, grouping by given domains.
Supports all regular expressions supported by python's regex module (re).

Third-party modules required: numpy, pandas

Last tested on Python 3.8.2 64-bit with pandas 1.1.0 and numpy 1.19.1.

Author: Matt Carras
Source Link: https://github.com/mattcarras/domain-regex
---------------------------------------------------------------------------------
Usage Notes

The function will raise an error if the given Date column contains any duplicated timestamps. You can suppress this error by giving verify_timestamps=False, but the script will likely error afterwards with a duplicate axis error when trying to re-join the filtered matched expressions with the main dataframe.
---------------------------------------------------------------------------------
Example usage

python -i domain_regex.py
>>> df_input = pd.read_csv("fake_discord_chat_data.csv", encoding='utf-8', parse_dates=['Date'], infer_datetime_format=True)
>>> df_matches = domain_regex(df_input, "domain_regex.csv", cols={'Date':'Date','Message':'Content','UserId':'AuthorID','Username':'Author'}, print_stats=True, file_stats_prefix='stats')
---------------------------------------------------------------------------------
Input file Notes

Supported input domain regular expression CSV columns: Domain,Expression,Allow Partial Match,Begin Date,End Date
Any other columns, such as "Comments", will be ignored.

Supported input dataframe columns: Date,Message,UserId,Username
These columns are mapped by the cols={...} parameter. Any other columns are ignored. The only required mapping is 'Message'. 
---------------------------------------------------------------------------------
Useful advanced regular expressions

WORD1(?:[\w'-]+\s+){0,6}?WORD2

Allows 0-6 words or spaces in-between with at least one space (word1 NEAR word2).
Where WORD1 and WORD2 are whatever you define.
"""

import warnings
import re 
from collections import OrderedDict
from distutils.util import strtobool

# Third-party modules: pip install pandas
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

# -- START GLOBAL CONSTANTS -- 
# If True then Domains are grouped case-sensitive. Otherwise all Domains are converted to lowercase.
DOMAINS_ARE_CASE_SENSITIVE=True

# A string format with a stricter word boundary for regex / domain expression matching.
# \b matches between \w and \W, but that includes some characters you may want to exclude, such as @ and # (for mentions) and possibly parts of contractions or hyphenated words.
# The default string format includes only specific punctuation, mdash, and possessive nouns. It will still likely pick up pieces of URLs, like domains.
# Note the word boundary is only added when the expression's 'Allow Partial Match' column is left blank or set to anything other than 'False', '0' or 'Off'.
# Default word boundary: r'(?:(?<=[\s;:,.?!*"()\[\]])|(?<=\-\-)|^){0}(?=[\s;:,.?!*"()\[\]]|\-\-|\'s|$)'
# Looser word boundary: r"\b(?:(?<![@&#'\-])|(?<=\-\-)){0}\b(?:(?!['\-])|(?=\-\-|'s))"
# Stricter word boundary: r'(?:(?<!\S)|(?<=\-\-)){0}(?=[\s;:,.?!*"()\[\]]|\-\-|\'s|$)'
# Loosest word boundary (will match mentions): r'\b{0}\b'
FORMAT_RE_CUSTOM_WORDBOUNDARY = r'(?:^|(?<=[\s;:,.?!*"()/\\\[\]])|(?<=\-\-)){0}(?=$|[\s;:,.?!*"()/\\\[\]]|\-\-|\'s)'

# -- END GLOBAL CONSTANTS -- 

# -- User defined errors and warnings --
class ParseWarning(UserWarning):
    pass
class ParseError(RuntimeError):
    pass
class BadParametersWarning(UserWarning):
    pass
class ParametersWarning(UserWarning):
    pass
# -- END User defined errors and warnings --

def output_to_csv(df, csvoutput, suffix=None, print_msg=None, quiet=False):
    """Output given dataframe to a CSV file, optionally printing a message.

       Args:
           df (DataFrame): Pandas DataFrame to write to a CSV file.
           csvoutput (string): If suffix is given this is the filename prefix, otherwise it's the full filename including extension.
           suffix (string): Default None. If given treats csvoutput_fp as the prefix for the resulting filename. Should not include extension.
           print_msg (string): Default None. If given prints a message suffixed by the filename.
           quiet (boolean): Default False. Ignore print_msg and don't print anything to console.
       Returns:
           None.
    """
    if suffix:
        ofn = '{0}_{1}.csv'.format(csvoutput, suffix)
    else:
        ofn = csvoutput
    if print_msg and not quiet:
        print('** {0} [{1}]...'.format(print_msg, ofn))
    df.to_csv(ofn, encoding='utf-8')
    return
# END def output_csv

def apply_listlike_value_counts(obj,col=None,lowercase=False,dropna=True,nested=False):
    """
    Return a series with the value counts of strings within list-like objects from each row of a Series or DataFrame.
    
    Args:
        obj (Series or DataFrame): Pandas Series or DataFrame (if col is given).
        col (string): Optional column for DataFrames.
        lowercase (bool): If given, call str.lower() on values in arrays.
        dropna (bool): drop NA values from the Series or DataFrame before processing.
        nested (bool or key): If given, assume each value is a nested list-like. If an int or key is given, only count those elements.
    Returns:
        Pandas Series with value counts.
    """
    if col:
        obj = obj[col]
    if dropna:
        obj.dropna(inplace=True)
    if nested: 
        if not isinstance(nested,bool):
            sr = pd.Series([x.lower() if lowercase and isinstance(x,str) else x for l in obj for x in l[nested] if x]).value_counts()
        else:
            sr = pd.Series([x.lower() if lowercase and isinstance(x,str) else x for l1 in obj for l2 in l1 for x in l2 if x]).value_counts()
    else:
        sr = pd.Series([x.lower() if lowercase and isinstance(x,str) else x for l in obj for x in l if x]).value_counts()
    return sr
# END def apply_listlike_value_counts

def save_domain_match(dict_matches, match, domain, ts=None, begin_ts=None, end_ts=None, matches_to_remove=None):
    """ Save matching domains in a dict. Also optionally filters matches and the domain by a given date range.
    
        Args:
            dict_matches (dict): Dictionary used to save matches.
            match (string): Matched expression.
            domain (string): Domain to add to matches.
            ts (timestamp): Current timestamp to evaluate.
            begin_ts (timestamp): Beginning of the valid time range. One or both must be set to limit by date/time.
            end_ts (timestamp): Ending of the valid time range. One or both must be set to limit by date/time.
            matches_to_remove (dict): Dictionary of matches to remove due to being outside the given time range.
        Returns:
            dictionary of match data in the form of {'Matches from Domain Expressions':list, 'Matching Domains':list}
    """
    # If given a timestamp and time range, only return the domain if it's within the range
    # TODO: Surely there's a better way to implement this...
    if ts and (begin_ts or end_ts):
        if (not isinstance(begin_ts, pd.Timestamp) or ts >= begin_ts) and (not isinstance(end_ts, pd.Timestamp) or ts <= end_ts):
            if matches_to_remove:
                matches_to_remove[match] = False
        else:
            # Match is within given time range and needs to be removed
            domain = None
            if matches_to_remove and matches_to_remove[match] is None:
                matches_to_remove[match] = True
    if domain:
        if not match in dict_matches:
            dict_matches[match] = [domain]
        elif not domain in dict_matches[match]:
            dict_matches[match] += [domain]
    return domain
# END def save_domain_match

def apply_domain_matches_by_date(df, df_regex, dict_matches, case=False):
    """ Generate domain matches and filter both by date ranges. This is far slower than using a pair of simpler list comprehensions.
        The return is intended to be used with DataFrame.apply and result_type='expand' or result_type='broadcast' to expand into columns.
        
        Args:
            df (dataframe): Pandas DataFrame to analyze.
            df_regex (dataframe): Pandas DataFrame of regular expressions as imported by output_domain_regex_analysis.
            dict_matches (array-like): Dictionary of matches passed to save_domain_match.
            case (bool): Default False. Treat all expressions as case-sensitive if True (do not call str.lower()).
        Returns:
            dictionary of match data in the form of {'Matches from Domain Expressions':list, 'Matching Domains':list}
    """
    if pd.api.types.is_list_like(df['Matches from Domain Expressions']):
        if not case:
            re_flags=re.IGNORECASE
        else:
            re_flags=None
        # Initialize a dict of existing matches that should be removed due to being outside of the given time range
        matches_to_remove = dict.fromkeys([s.lower() if not case else s for s in df['Matches from Domain Expressions']])
        matches = list(matches_to_remove)
        matching_domains = [save_domain_match(dict_matches, s, domain, df.name, begin_date, end_date, matches_to_remove) for s in matches for domain,exp,begin_date,end_date,allow_partial in zip(df_regex['Domain'],df_regex['Expression'],df_regex['Begin Date'],df_regex['End Date'],df_regex['Allow Partial Match']) if re.match(FORMAT_RE_CUSTOM_WORDBOUNDARY.format(exp) if not allow_partial else exp, s, re_flags)]
        matches = [s for s in matches if not matches_to_remove[s]]
        matching_domains = list(dict.fromkeys(matching_domains))
    else:
        matches = df['Matches from Domain Expressions']
        matching_domains = []
    
    return {'Matches from Domain Expressions':matches,
            'Matching Domains':matching_domains}
# END def apply_domain_matches_by_date

def join_formatted_regex(sr, mask_allow_partial, case=False, compile=False):
    """ Combine strings containing regex in the given pandas Series into one expression joined by |.
        Expressions that do not allow partial matching get are enclosed in a word boundary format defined by a global.
        Note that expressions are considered case-insensitive by default.
        
        Args:
            sr (Series): pandas Series of regular expressions.
            mask_allow_partial (array-like): Either a numpy array, pandas Series, or list-like with boolean values determining which expression (if any) is allowed partial matching.
            case (bool): Default False. Treat all expressions as case-sensitive if True (do not call str.lower()).
        Returns:
            string of joined regular expressions.
    """
    # Start by joining all the expressions that allow partial matching.
    s = '|'.join(sr[mask_allow_partial].str.lower().unique() if not case else sr[mask_allow_partial].unique())
    # Enclose the expressions that don't require partial matching in (?:exp1|exp2|exp3|...) before adding word boundary formatting.
    if not mask_allow_partial.all():
        s += FORMAT_RE_CUSTOM_WORDBOUNDARY.format(r'(?:{0})'.format('|'.join((sr[~mask_allow_partial].str.lower().unique() if not case else sr[~mask_allow_partial].unique()))))
    # Separating out and joining the expressions in this manner, rather than adding word boundaries to each expression, significantly speeds up processing.
    return s
# END def join_formatted_regex

def domain_regex(df_input, file_domainregex, cols={'Date':'Date','Message':'Message'}, print_stats=True, file_stats_prefix=None, print_messages=False, file_messages_prefix=None, match_related_minutes_before=60, match_related_minutes_after=60, case=False, verify_timestamps=True, verbose=True, **kwargs):
    """Output Domain Expression/Regex analysis with matching rows and occurrence counts. 
       
       Any additional arguments passed into kwargs are ignored.

    Args:
       df_input (DataFrame): Pandas DataFrame to analyze and output.
       cols (dict): Dictionary of column mappings. Valid definitions: 'Date', 'Message', 'Username', 'UserId'. Must contain at least 'Message' definition. Default maps both 'Date' and 'Message' 1:1.
       file_domainregex (string): Full or relative filepath to the Domain Expression/Regex CSV file. An example file is included on the github.
       print_stats (boolean): Default True. Print all stats to console.
       file_stats_prefix (string): Save stats using given filepath prefix. Ex: "stats"
       print_messages (boolean): Default False. Print messages to console (this may be a lot).
       file_messages_prefix (string): Save messages using given filepath prefix. Ex: "messages"
       match_related_minutes_before (int): Default 60. When ouputting messages, include given number of minutes before matching messages.
       match_related_minutes_before (int): Default 60. When ouputting messages, include given number of minutes after matching messages.
       case (boolean): Default False. If set to True all expressions are treated as case-sensitive.
       verify_timestamps (boolean): Default True. Verify all dates and timestamps are not duplicated or out-of-order. If duplicated, raise an error.
       verbose (boolean): Default True. Print additional details.
    Returns:
       pandas DataFrame with matching messages.
    """
    # Default Columns = Domain,Expression,Begin Date,End Date
    IR_DATE_COLS=['Begin Date','End Date']
    
    # Parameter validation.
    if not cols:
        raise ValueError('Column definitions are required')
    if not 'Message' in cols.keys():
        raise ValueError('Column definition for "Message" is required')
    if not match_related_minutes_before is None:
        if match_related_minutes_before < 0:
            raise ValueError('match_related_minutes_before cannot be less than 0')
        elif not 'Date' in cols:
            raise ValueError('match_related_minutes_before requires "Date" column definition')
    if not match_related_minutes_after is None:
        if match_related_minutes_after < 0:
            raise ValueError('match_related_minutes_after cannot be less than 0')
        elif not 'Date' in cols:
            raise ValueError('match_related_minutes_after requires "Date" column definition')
            
    if df_input.empty:
        warnings.warn('Given empty dataframe', stacklevel=2, category=BadParametersWarning)
            
    if not case:
        re_flags=re.IGNORECASE
    else:
        re_flags=None
    
    use_dt_index = 'Date' in cols.keys()
    if use_dt_index:
        # Check to see if we have any duplicate or out-of-order timestamps
        if verify_timestamps:
            mask = df_input.duplicated(subset=[cols['Date']], keep=False)
            if mask.any():
                # df_input.sort_values('Date',ascending=False).drop_duplicates('Date', keep='last')
                raise ValueError('input dataframe contains {0} seemingly duplicated timestamps in [{1}] column. Give verify_timestamps=False if you want to try processing this dataframe anyway.'.format(len(df_input[mask]), cols['Date']))
            date_min = df_input.index.min()
            date_max = df_input.index.max()
            if ((df_input.at[0, cols['Date']] != date_min and df_input.at[0, cols['Date']] != date_max) or (df_input.at[0, cols['Date']] == date_min and (df_input[cols['Date']].values[:-1] > df_input[cols['Date']].values[1:]).any()) or (df_input.at[0, cols['Date']] == date_max and (df_input[cols['Date']].values[:-1] < df_input[cols['Date']].values[1:]).any())):
                warnings.warn('[{0}] column appears to either have one or more rows out-of-order and/or incorrectly parsed dates. This may cause problems when parsing time ranges.'.format(cols['Date']), stacklevel=2, category=ParseWarning)
        # Set DatetimeIndex if given column
        if not isinstance(df_input.index, pd.DatetimeIndex):
            df_input = df_input.set_index(cols['Date'])
           
    if verbose:
        print('** Reading domains and regular expressions/strings from [{0}]...'.format(file_domainregex))
    df_regex = pd.read_csv(file_domainregex, encoding='utf-8', parse_dates=IR_DATE_COLS, infer_datetime_format=True, dtype={'Allow Partial Match':str})
    
    # Transform the Allow Partial Match column into bools.
    df_regex['Allow Partial Match'] = df_regex['Allow Partial Match'].transform(lambda s: (not (not isinstance(s, str) or not s or not strtobool(s))))
    
    # Get a list of rows that have a Begin and/or End Date set. We will filter this separately.
    # TODO: Figure out a better way to do this than later dropping matches.
    df_regex_by_date = df_regex.dropna(how='all', subset=IR_DATE_COLS)
    
    # First join all the expressions per domain into | separated strings (unique expressions only). We'll be using this later.
    # The FORMAT_RE_CUSTOM_WORDBOUNDARY string format global is used to separate words and phrases.
    sr_regex = df_regex.groupby(df_regex['Domain'].str.lower() if not DOMAINS_ARE_CASE_SENSITIVE else 'Domain').apply(lambda df: join_formatted_regex(df['Expression'], df['Allow Partial Match'].values, case))
    
    try:
        # Next combine into one big regular expression for findall(). Matching will be case-insensitive due to re.IGNORECASE flag.
        # TODO: Check to see if we can improve performance by parsing and removing duplicate patterns before combining.
        regex = re.compile('|'.join(sr_regex.values), re.IGNORECASE)
    except re.error:
        # Try to find which expression may have given the error.
        for exp in df['Expression']:
            try:
                regex = re.compile(exp)
            except:
                raise re.error('Invalid regular expression in [{0}]: {1}'.format(file_domainregex, exp))
        raise re.error('Caught regex parse error. Likely invalid expression in [{0}]'.format(file_domainregex))
    
    if verbose:
        print('** Searching for matching expressions and domains, this may take a minute...')
    # Get series of matches for each row using str.findall(). This will return array values.
    # It should be way faster to do findall and then apply over matches.
    # TODO: Could we somehow combine this with getting matching domains without needing to use regular expressions twice?
    sr = df_input[cols['Message']].str.findall(regex)
    
    # Add it back into the dataframe and then use it to filter the dataframe.
    df_input['Matches from Domain Expressions'] = sr
    
    # Filtered subset will contain non-matching rows with empty lists and rows with blank (NaN) in Content column, so drop those rows.
    df = df_input[sr.astype(bool)].dropna(subset=[cols['Message']])
    
    if df.empty:
        warnings.warn('No domain expression matches found in given input dataframe. Nothing to output.', stacklevel=2, category=ParseWarning)
        df_matches = None
    else:
        # Attempt to do a much more resource-intensive process that filters out given date ranges, if we're given any
        # TODO: Figure out a better way to do this
        if use_dt_index and not df_regex_by_date is None and not df_regex_by_date.empty:
            if verbose:
                print('** Excluding matches outside of date ranges specified in domain expression file, this may take a minute...')
            dict_map_matching_domains={}
            df2 = df.apply(apply_domain_matches_by_date, axis=1, result_type='expand', df_regex=df_regex, dict_matches=dict_map_matching_domains, case=case)       
            sr = df2['Matches from Domain Expressions']
            df['Matches from Domain Expressions'] = sr[sr.astype(bool)]
            sr = df2['Matching Domains']
            df['Matching Domains'] = sr[sr.astype(bool)]
        else:
            # Remove any duplicates from the matches by converting the lists into dict keys and then back into lists.
            df['Matches from Domain Expressions'] = df['Matches from Domain Expressions'].transform(lambda li: list(dict.fromkeys(li)))

            # Loop over matches and make a series with domain per match, again using list comprehension.
            # We can't map them directly since each value is a list in the DataFrame column and each expression is a regular expression.
            # It should still be far faster to do it over previous matches from Series.str.findall() even through we're using nested loops here.
            # We'll also save a dict mapping of which domains matched the word/phrase.
            dict_map_matching_domains = {}
            df['Matching Domains'] = df['Matches from Domain Expressions'].apply(lambda li: list(dict.fromkeys([save_domain_match(dict_map_matching_domains,s.lower() if not case else s, domain) for s in li for domain,exp in zip(sr_regex.index,sr_regex) if re.match(exp,s,re_flags)])))
        # We'll be returning this.
        df_matches = df
        
        # TODO: Figure out a better way to do this, without iterating and also getting back possibly duplicate timestamps
        if use_dt_index and (match_related_minutes_before or match_related_minutes_after) and (print_messages or file_messages_prefix):
            if verbose:
                print('** Collecting all rows {0} minutes before and {1} minutes after for domain expression matching rows, this may take a minute...'.format(match_related_minutes_before,match_related_minutes_after))
            # Use an OrderedDict to remove dupes while preserving order of timestamps.
            idxs = list(OrderedDict.fromkeys([ts2 for ts in df.index for ts2 in df_input.index[((df_input.index >= ts) & (df_input.index <= (ts + DateOffset(minutes=match_related_minutes_after)))) | ((df_input.index <= ts) & (df_input.index >= ts - DateOffset(minutes=match_related_minutes_before)))]]))
            # If verify_timestamps is enabled, verify the timestamps are returned in order and not duplicated.
            # Duplicated or out-of-order timestamps are more likely the result of problems with the index.
            if verify_timestamps:
                ar = np.array(idxs)
                if (ar[:-1] <= ar[1:]).any():
                    warnings.warn('Filtered timestamps may be either out-of-order or duplicated', stacklevel=2)
            df = df_input.loc[idxs]
            # Add matching domains into the DataFrame. This result should always be larger than the dataframe that only has matching rows.
            df['Matches from Domain Expressions'] = df_matches['Matches from Domain Expressions']
            df['Matching Domains'] = df_matches['Matching Domains']
        
        if print_messages:
            print(df)
        if not file_messages_prefix is None:
            output_to_csv(df, file_messages_prefix, 'messages_matching_domain_exp', \
                          'Outputting rows with domain expression matches to', not verbose)
        # Set the dataframe we're currently working on back to just matches.
        df = df_matches
        
        # Get count of occurrences per UserId using a custom function to unpack the arrays inside of the series.
        if 'UserId' in cols.keys():
            if verbose:
                print('** Counting number of occurrences for each matching word or phrase per UserId and total number overall per match...')
            # Save this grouping for later.
            grouped = df.groupby(cols['UserId'])
            # Make a grouping of Usernames and UserIds, then do the count.
            if 'Username' in cols.keys():
                df_usernames = grouped.agg(Usernames=(cols['Username'], 'unique'))
            df = grouped.apply(apply_listlike_value_counts,col='Matches from Domain Expressions',lowercase=True,dropna=True)
            # We may get back a DataFrame if we have only one matching expression.
            if isinstance(df, pd.DataFrame):
                df = df.stack()
            df = df.to_frame('Occurrences')
            df.index.names = [cols['UserId'],'Match']
            # Add column for matching domains.
            df['Domains'] = df.index.get_level_values('Match').map(dict_map_matching_domains)
            # Output columns
            ocols = ['Match', 'Occurrences', 'Domains']
            # Add Usernames column mapped to UserID.
            if 'Username' in cols.keys():
                df2 = df.join(df_usernames)
                ocols = ['Usernames'] + ocols
            if print_stats:
                print( df2 )
            if not file_stats_prefix is None:
                df2.reset_index('Match', inplace=True)
                output_to_csv(df2[ocols], file_stats_prefix, 'domain_exp_count_per_userid', \
                             'Outputting count of domain expression matches per UserID to', not verbose)

        # Get total count of occurrences per match.
        df = df.groupby('Match').agg(**{'Total Occurrences':('Occurrences','sum')})
        df['Domains'] = df.index.map(dict_map_matching_domains)
        if print_stats:
            print( "\n" + str(df) )
        if not file_stats_prefix is None:
            output_to_csv(df, file_stats_prefix, 'domain_exp_total_count', \
                         'Outputting total count per matching expression to', not verbose)

        if 'UserId' in cols.keys():
            if verbose:
                print('** Counting number of occurrences for each domain per UserId...')
            df = grouped.apply(apply_listlike_value_counts,col='Matching Domains',lowercase=True,dropna=True)
            # We may get back a DataFrame if we have only one matching domain.
            if isinstance(df, pd.DataFrame):
                df = df.stack()
            df = df.to_frame('Occurrences')
            df.index.names = [cols['UserId'],'Domain']
            # Output columns
            ocols = ['Domain', 'Occurrences']
            # Add Usernames column mapped to UserID.
            if 'Username' in cols.keys():
                df2 = df.join(df_usernames)
                ocols = ['Usernames'] + ocols
            if print_stats:
                print( df2 )
            if not file_stats_prefix is None:
                df2.reset_index('Domain', inplace=True)
                output_to_csv(df2[ocols], file_stats_prefix, 'domain_count_per_userid', \
                             'Outputting count of domain matches per UserId to', not verbose)

        # Get total count of occurrences per domain.
        df = df.groupby('Domain').agg(**{'Total Occurrences':('Occurrences','sum')})
        if print_stats:
            print( "\n" + str(df) )
        if not file_stats_prefix is None:
            output_to_csv(df, file_stats_prefix, 'domain_total_count', \
                         'Outputting total count of domain matches to', not verbose)
        
        # If we still have the column we added to df_input, drop it before returning.
        if 'Matches from Domain Expressions' in df_input.columns:
            df_input.drop(['Matches from Domain Expressions'], axis=1, inplace=True)
    return df_matches
    
# END def output_domain_regex_analysis