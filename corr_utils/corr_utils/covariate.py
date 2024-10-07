# author: Noel Kronenberg

import corr_utils

import pandas as pd
from enum import Enum
import numpy as np
import operator
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)

# default key for unqiue rows and merging of DataFrames
global default_key
# default_key:str = corr_utils.default_key

def extract_csv_data(d_path:str, f_name:str, delimiter:str='|', col_dict:dict=None, filter_dict:'dict[str, list:str]'=None, exact_match:bool=False, remove_prefix:bool=False) -> pd.DataFrame:
    """
    Extracts data from a CSV file.

    Parameters:
        d_path (str): The path to the directory containing the CSV file.
        f_name (str): The name of the CSV file.
        delimiter (str, optional): The delimiter used in the CSV file. Defaults to '|'.
        col_dict (dict, optional): A dictionary mapping column names to new names. Defaults to None.
        filter_dict (dict[str, list:str], optional): A dictionary where keys are column names and values are lists of values to filter rows by (may include regex pattern for exact_match=False). Defaults to None.
        exact_match (bool, optional): If True, performs exact matching when filtering. Defaults to False.
        remove_prefix (bool, optional): If True, removes prefix from default_key. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the CSV file.
    """

    # read CSV and rename CSV if applicable
    if col_dict:
        df = pd.read_csv(d_path + f_name, delimiter=delimiter, usecols=list(col_dict.keys()), dtype='object').rename(columns=col_dict)
    else:
        df = pd.read_csv(d_path + f_name, delimiter=delimiter, dtype='object')

    # filter rows for given values
    if filter_dict:
            for column, values in filter_dict.items():
                if exact_match:
                    df = df[df[column].isin(values)]
                else:
                    regex = '|'.join(values)
                    df = df[df[column].str.contains(regex, na=False, regex=True)]

    if remove_prefix:
        df[default_key] = df[default_key].str.split('_').str[-1]

    return df

def extract_df_data(df:pd.DataFrame, col_dict:dict=None, filter_dict:'dict[str, list:str]'=None, exact_match:bool=False, remove_prefix:bool=False, drop:bool=False) -> pd.DataFrame:
    """
    Extracts data from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to operate on.
        col_dict (dict, optional): A dictionary mapping column names to new names. Defaults to None.
        filter_dict (dict[str, list:str], optional): A dictionary where keys are column names and values are lists of values to filter rows by (may include regex pattern for exact_match=False). Defaults to None.
        exact_match (bool, optional): If True, performs exact matching when filtering. Defaults to False.
        remove_prefix (bool, optional): If True, removes prefix from default_key. Defaults to False.
        drop (bool, optional): If True, drop all columns not specified in col_dict.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the original DataFrame.
    """

    # rename columns if applicable
    if col_dict:
        df = df.rename(columns=col_dict)

        if drop:
            df = df.filter(items=col_dict.values())

    # filter rows for given values
    if filter_dict:
            for column, values in filter_dict.items():
                if exact_match:
                    df = df[df[column].isin(values)]
                else:
                    if df[column].dtype == 'object':
                        regex = '|'.join(values)
                        df = df[df[column].str.contains(regex, na=False, regex=True)]
                    else:
                        df = df[df[column].isin(values)]

    if remove_prefix:
        df[default_key] = df[default_key].str.split('_').str[-1]

    return df

def extract_sql_data(table:str, db, col_dict:dict=None, filter_dict:'dict[str, list:str]'=None, exact_match:bool=False) -> pd.DataFrame:
    """
    Extracts data from an SQL Table.

    Parameters:
        table (str): Name of the SQL table.
        db: Connection to the database that contains the table.
        col_dict (dict, optional): A dictionary mapping column names to new names. Defaults to None.
        filter_dict (dict[str, list:str], optional): A dictionary where keys are column names and values are lists of values to filter rows by. Defaults to None.
        exact_match (bool, optional): If True, performs exact matching when filtering. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the SQL table.
    """

    query = f'SELECT * FROM {table}'

    # rename columns if applicable
    if col_dict:
        col_mappings = ', '.join([f'{old_name} AS {new_name}' for old_name, new_name in col_dict.items()])
        query = query.replace('*', col_mappings)

    # filter rows for given values
    if filter_dict:
            conditions = []
            for column, values in filter_dict.items():
                if exact_match:
                    condition = f"{column} IN ({', '.join(map(repr, values))})"
                else:
                    condition = ' OR '.join([f"{column} LIKE {repr('%%' + value + '%%')}" for value in values])
                conditions.append(condition)
            query += ' WHERE ' + ' AND '.join(conditions)

    return pd.read_sql_query(sql=query, con=db)

def extract_by_priority(df:pd.DataFrame, column:str, priority_order:list) -> pd.DataFrame:
    """
    Extracts data from a DataFrame by priority of values, keeping only the first instance.

    Parameters:
        df (pandas.DataFrame): The DataFrame to operate on.
        column (str): Name of the column of the values.
        priority_order (list): List of values ordered by priority.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the original DataFrame.
    """

    df_temp = df.copy()

    priority_dict = {value: i for i, value in enumerate(priority_order)}
    df_temp['priority'] = df_temp[column].map(priority_dict)
    df_temp = df_temp.dropna(subset=['priority'])
    df_temp['priority'] = df_temp['priority'].astype(int)

    df_sorted = df_temp.sort_values(by=[default_key, 'priority'])
    df_filtered = df_sorted.drop_duplicates(subset=default_key, keep='first')

    df_filtered = df_filtered.drop(columns='priority')

    return df_filtered

def filter_time_around(df:pd.DataFrame, time_column:str, time_reference_db:pd.DataFrame, time_reference_db_column:str, time_lower_bound:int=0, time_upper_bound:int=0, drop:bool=False) -> pd.DataFrame:
    """
    Filters a DataFrame for time difference to a reference value. Also prints the number of removed rows.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data to filter.
        time_column (str): The name of the column containing time information in df.

        time_reference_db (pandas.DataFrame): DataFrame containing reference times for each default_key.
        time_reference_db_column (str): The name of the column containing reference times in time_reference_db.

        time_lower_bound (int, optional): Lower bound for time-based filtering (in minutes). Defaults to 0.
        time_upper_bound (int, optional): Upper bound for time-based filtering (in minutes). Defaults to 0.

        drop (bool, optional): If True, drop the reference column(s) after filtering. Default is False.

    Returns:
        pandas.DataFrame: A DataFrame containing the filtered data in addition to a time_difference (in minutes) column.
    """

    # add reference times to df (matched on default_key)
    merged_df = pd.merge(df, time_reference_db, on=default_key, how='left')
    
    # calculate the time differences (in minutes)
    merged_df['time_difference'] = (merged_df[time_column] - merged_df[time_reference_db_column]) / pd.Timedelta(minutes=1)
    
    # filter df based on time bounds
    filtered_df = merged_df[
        (merged_df['time_difference'] >= -time_lower_bound) &
        (merged_df['time_difference'] <= time_upper_bound)
    ]

    if drop:
        filtered_df.drop(columns=[time_reference_db_column], inplace=True)

    get_amount_removed_rows(df, filtered_df)
    
    return filtered_df

def filter_time_between(df:pd.DataFrame, time_column:str, df_time_reference:pd.DataFrame, df_time_reference_column_upper_bound:str, merge_on:str, between:bool=False, df_time_reference_column_lower_bound:str=None, drop:bool=False) -> pd.DataFrame:
    """
    Filters a DataFrame based on a time comparison between two DataFrames. Also prints the number of removed rows.

    Parameters:
        df (pd.DataFrame): The DataFrame to be filtered.
        time_column (str): The column name in df representing the time to be filtered.

        df_time_reference (pd.DataFrame): The DataFrame containing the reference time data.
        df_time_reference_column_upper_bound (str): The column name in df_time_reference representing the reference time as an upper bound.
        merge_on (str): The column name used to merge df with df_time_reference.

        between (bool, optional): If True, also check for lower bound. Default is False.
        df_time_reference_column_lower_bound (str, optional): The column name in df_time_reference representing the reference time as a lower bound. Default is None.
        
        drop (bool, optional): If True, drop the reference column(s) after filtering. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows where the time is before (and optionally also after) the corresponding time(s) in df_time_reference.
    """

    if between:
        merged_df = pd.merge(df, df_time_reference[[merge_on, df_time_reference_column_upper_bound, df_time_reference_column_lower_bound]], on=merge_on, how='left', suffixes=('', ''))
        lower_bound = merged_df[time_column] > merged_df[df_time_reference_column_lower_bound]
        upper_bound = merged_df[time_column] < merged_df[df_time_reference_column_upper_bound]
        filtered_df = merged_df[lower_bound & upper_bound].copy()
        if drop:
            filtered_df.drop(columns=[df_time_reference_column_upper_bound, df_time_reference_column_lower_bound], inplace=True)
    else:
        merged_df = pd.merge(df, df_time_reference[[merge_on, df_time_reference_column_upper_bound]], on=merge_on, how='left', suffixes=('', ''))
        upper_bound = merged_df[time_column] < merged_df[df_time_reference_column_upper_bound]
        filtered_df = merged_df[upper_bound].copy()
        if drop:
            filtered_df.drop(columns=[df_time_reference_column_upper_bound], inplace=True)

    get_amount_removed_rows(df, filtered_df)

    return filtered_df

class AggregationMethod(Enum):
    MINIMUM = 1
    MEDIAN = 2
    MAXIMUM = 3
    SUM = 4

def aggregate_data(df:pd.DataFrame, column:str, method:AggregationMethod, rename:bool=False) -> pd.DataFrame:
    """
    Calculates the desires statistic of a DataFrame column for each default_key.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        column (str): The name of the column containing the data to aggregate.
        method (AggregationMethod): The method to aggregate with.
        rename (bool, optional): Rename aggregated column. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the aggregated data.
    """

    if method == AggregationMethod.MINIMUM:
        aggregated_df = df.groupby(default_key)[column].min().reset_index()
        if rename:
            aggregated_df = aggregated_df.rename(columns={column: f'{column}_minimum'})
    elif method == AggregationMethod.MEDIAN:
        aggregated_df = df.groupby(default_key)[column].median().reset_index()
        if rename:
            aggregated_df = aggregated_df.rename(columns={column: f'{column}_median'})
    elif method == AggregationMethod.MAXIMUM:
        aggregated_df = df.groupby(default_key)[column].max().reset_index()
        if rename:
            aggregated_df = aggregated_df.rename(columns={column: f'{column}_maximum'})
    elif method == AggregationMethod.SUM:
        aggregated_df = df.groupby(default_key)[column].sum().reset_index()
        if rename:
            aggregated_df = aggregated_df.rename(columns={column: f'{column}_sum'})
 
    return aggregated_df

def exclude_rows(df:pd.DataFrame, column:str, items:list, filter_operator=None, drop:bool=False) -> pd.DataFrame:
    """
    Removes specified rows with items from a DataFrame based on the given column. Also prints the number of rows removed. Note that NaN values might also be excluded.

    Parameters:
        df (pd.DataFrame): The DataFrame to operate on.
        column (str): The name of the column to filter items from.
        items (list): A list of items to exclude from the DataFrame. Must only contain one value if a filter_operator is provided.
        drop (bool, optional): If True, drop the specified column after excluding items. Default is False.
        filter_operator (optional): If specfied, this operator (e.g. operator.ge, operator.le, operator.gt, operator.lt) will be used to exclude rows. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with specified rows containing the items excluded.
    """

    df_res = df.copy()

    if filter_operator == None:
        df_res = df_res[~df_res[column].isin(items)]
        if drop: 
            df_res = df_res.drop(columns=[column])
    else:
        if len(items) > 1:
            raise ValueError(f'If a filter_operator is provided, items must contain just one value.')
        else:
            df_res = df_res[filter_operator(items[0], df_res[column])]

    get_amount_removed_rows(df, df_res)

    return df_res

import pandas as pd

def clean_values(df:pd.DataFrame, reference_values:str, drop_rows:bool=False) -> pd.DataFrame:
    """
    Cleans clinically implausible values in a DataFrame based on given definition. Sets implausible values to NaN (if not specified otherwise).

    Parameters:
        df (pandas.DataFrame): The DataFrame to clean.
        reference_values (str): Path to the CSV file containing min and max values for each variable ('variable', 'min_value', 'max_value').
        drop_rows (bool, optional): Drop rows with implausible values. Defaults to False.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    df_res = df.copy()
    df_reference = pd.read_csv(reference_values)
    total_rows = len(df_res)
    
    # ensure CSV is correct
    required_columns = {'variable', 'min_value', 'max_value'}
    if not required_columns.issubset(df_reference.columns):
        raise ValueError(f'The CSV file must contain the columns: {required_columns}')
    
    mask = pd.Series([True] * len(df_res))

    # NOTE: optimise to only calculate this section when not drop_rows (currently needed for count)

    # check values
    for _, row in df_reference.iterrows():
        variable = row['variable']
        min_value = row['min_value']
        max_value = row['max_value']
        
        if variable in df_res.columns:
            # mark rows with implausible values
            mask &= df_res[variable].apply(lambda x: pd.isna(x) or (min_value <= x <= max_value))

    if drop_rows:
        df_res = df_res[mask]
    else:
        # check values
        for _, row in df_reference.iterrows():
            variable = row['variable']
            min_value = row['min_value']
            max_value = row['max_value']
            
            if variable in df_res.columns:
                # set implausible values to NaN
                df_res[variable] = df_res[variable].apply(lambda x: x if pd.isna(x) or (min_value <= x <= max_value) else None)

    implausible_rows = total_rows - mask.sum()
    print(f'Number of rows with implausible values: {implausible_rows} ({((implausible_rows / total_rows) * 100):.2f}%)')

    return df_res

def create_subgroups(df:pd.DataFrame, conditions:list) -> pd.DataFrame:
    """
    Create subgroups in a DataFrame based on specified conditions for a given row.

    Parameters:
    	df (pd.DataFrame): Input DataFrame.
    	conditions (list of tuples): A list of conditions to apply and the name in the format: (lambda function, condition name).

    Returns:
	    pd.DataFrame: DataFrame with new one-hot encoded columns for each subgroup.
    """

    df_res = df.copy()

    for cond in conditions:
        condition_lambda = cond[0]
        condition_label = cond[1]
        df_res[condition_label] = df_res.apply(lambda row: 1 if condition_lambda(row) else 0, axis=1)

    return df_res

def collect_subgroups(df:pd.DataFrame, conditions:list) -> dict:
    """
    Collect subgroups in a dict of DataFrames.

    Parameters:
    	df (pd.DataFrame): DataFrame containing all data with indicators for the subgroups. Works with create_subgroups().
    	conditions (list of tuples): A list of conditions to apply and the name in the format: (lambda function, condition name).

    Returns:
	    dict: A dict with the subgroup name as keys and the DataFrame as values.
    """
     
    subgroups = {}
    for _, name in conditions:
        subgroups[name] = df[df[name] == 1]

    return subgroups

def combine_date_time(df:pd.DataFrame, date_column:str, time_column:str, new_name:str) -> pd.DataFrame:
    """
    Combines date and time columns into a singular date time column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the date and time columns.
        date_column (str): The name of the column containing the date information.
        time_column (str): The name of the column containing the time information.
        new_name (str): The name to assign to the new date time column.

    Returns:
        pandas.DataFrame: The DataFrame with the new date time column.
    """

    df_res = df.copy() 

    df_res[new_name] = df_res[date_column].astype(str) + df_res[time_column].astype(str).str.zfill(6)
    df_res[new_name] = pd.to_datetime(df_res[new_name], format='%Y%m%d%H%M%S', errors='coerce')

    df_res.drop([date_column, time_column], axis=1, inplace=True)

    return df_res

def add_missing_columns(df:pd.DataFrame, df_to_add:pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing columns to a DataFrame from a given other DataFrame, merged on default_key.

    Parameters:
        df (pd.DataFrame): The DataFrame to add missing columns to.
        df_to_add (pd.DataFrame): The DataFrame to add missing columns from.

    Returns:
        pd.DataFrame: A DataFrame containing all columns from df_to_add that had been missing.
    """

    missing_columns = list(set(df_to_add.columns) - set(df.columns))
    missing_columns.append(default_key)

    return pd.merge(df, df_to_add[missing_columns], on=default_key, how='left', suffixes=('', ''))

import pandas as pd

def handle_duplicates(df:pd.DataFrame, column:str, drop_duplicates:bool=True, testing:bool=False) -> pd.DataFrame:
    """
    Handle duplicate occurrences in a DataFrame. Prints the number of duplicates.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column to check for duplicates.
        drop_duplicates (bool): Whether to drop the duplicates. Default is True.
        testing (bool): Whether to return parameters for testing purposes. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned input DataFrame (if drop_duplicates == True).
    """

    initial_row_count = len(df)
    df_res = df.copy()

    duplicates = df_res.duplicated(subset=column, keep=False)
    duplicate_count = duplicates.sum()
    unique_duplicate_count = df_res.loc[duplicates, column].nunique()

    print(f"Number of duplicate {column}'s: {duplicate_count} ({((duplicate_count / initial_row_count) * 100):.2f}%)")
    print(f"Number of unique duplicate {column}'s: {unique_duplicate_count} ({((unique_duplicate_count / initial_row_count) * 100):.2f}%)")

    if testing:
            return duplicate_count, unique_duplicate_count

    if drop_duplicates:
        df_res = df_res.drop_duplicates(subset=column)
        return df_res

def get_amount_removed_rows(initial:pd.DataFrame, new:pd.DataFrame) -> None:
    """
    Print number and percentage of rows removed (or added).

    Parameters:
        initial (pandas.DataFrame): The initial DataFrame.
        new (pandas.DataFrame): The new DataFrame.

    Returns:
        None
    """
     
    removed_rows = len(initial) - len(new)

    if (removed_rows >= 0):
        print(f'Number of removed rows: {removed_rows} ({((removed_rows / len(initial))*100):.4f}%)')
    else:
        removed_rows *= (-1)
        print(f'Number of added rows: {removed_rows} ({((removed_rows / len(initial))*100):.4f}%)')

def get_eda_metrics(df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with exploratory data analysis (EDA) metrics for each column in the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame for which to compute the EDA metrics.
    
    Returns:
        pd.DataFrame: A DataFrame containing the EDA metrics for each column in the input DataFrame.
    """

    metrics = {
        'total_entries': df.count(),
        'missing_values': df.isna().sum(),
        'unique_values': df.nunique(),
        'duplicate_entries': [], 
        'mean': [],
        'median': [],
        'std_dev': [],
        'min': [],
        'max': []
    }

    # check if numerical
    for col in df.columns:
        metrics['duplicate_entries'].append(df[col].duplicated().sum())

        if pd.api.types.is_numeric_dtype(df[col]):
            metrics['mean'].append(round(df[col].mean(), 4))
            metrics['median'].append(round(df[col].median(), 4))
            metrics['std_dev'].append(round(df[col].std(), 4))
            metrics['min'].append(df[col].min())
            metrics['max'].append(df[col].max())

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            mean = df[col].mean()
            median = df[col].median()

            # check for missings and format
            if pd.isna(mean):
                metrics['mean'].append('NaT')  
            else:
                metrics['mean'].append(mean.strftime('%Y-%m-%d %H:%M:%S'))

            if pd.isna(median):
                metrics['median'].append('NaT')  
            else:
                metrics['median'].append(median.strftime('%Y-%m-%d %H:%M:%S'))

            # metrics['mean'].append(df[col].mean().strftime('%Y-%m-%d %H:%M:%S'))
            # metrics['median'].append(df[col].median().strftime('%Y-%m-%d %H:%M:%S'))

            # timedelta (reference: https://stackoverflow.com/a/8907269)
            raw_std = df[col].std()  
            days = raw_std.days
            hours, remainder = divmod(raw_std.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            hours = '{:02}'.format(hours)
            minutes = '{:02}'.format(minutes)
            seconds = '{:02}'.format(seconds)
            formatted_std = '%s days %s:%s:%s' % (days, hours, minutes, seconds)
            metrics['std_dev'].append(formatted_std)

            metrics['min'].append(df[col].min())
            metrics['max'].append(df[col].max())

        elif pd.api.types.is_timedelta64_dtype(df[col]):
            metrics['mean'].append(df[col].mean())
            metrics['median'].append(df[col].median())
            
            raw_std = df[col].std()
            days = raw_std.days
            hours, remainder = divmod(raw_std.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            hours = '{:02}'.format(hours)
            minutes = '{:02}'.format(minutes)
            seconds = '{:02}'.format(seconds)
            formatted_std = '%s days %s:%s:%s' % (days, hours, minutes, seconds)
            metrics['std_dev'].append(formatted_std)

            metrics['min'].append(df[col].min())
            metrics['max'].append(df[col].max())

        else:
            metrics['mean'].append(np.nan)
            metrics['median'].append(np.nan)
            metrics['std_dev'].append(np.nan)
            metrics['min'].append(np.nan)
            metrics['max'].append(np.nan)
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'variable'}, inplace=True)
    
    return df_metrics