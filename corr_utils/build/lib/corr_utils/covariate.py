# author: Noel Kronenberg
# version: 01.09.2024 16.00

import pandas as pd
from enum import Enum
import numpy as np
import operator

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from dcurves import dca, plot_graphs

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

plt.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)

# default key for unqiue rows and merging of DataFrames
global default_key
default_key:str = 'case_id' 

def set_default_key(key:str='case_id') -> None:
    """
    Sets the default key for merging operations.

    Parameters:
        key (str): The key to set. Defaults to 'case_id'.

    Returns:
        None
    """

    global default_key 
    default_key = key

# START: Covariate Function

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
    Removes specified rows with items from a DataFrame based on the given column. Also prints the number of rows removed.

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
    Cleans clinically implausible values in a DataFrame based on given definition. Sets implausible values to NaN.

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

def evaluate_subgroups(subgroups:dict, score_columns:list, outcome_column:str, test_size=0.2, dca_y_limits:list=[-0.01, 0.02], dca_thresholds:np.ndarray=np.arange(0, 0.10, 0.01)) -> None:
    """
    Validates a score on all important metrics.

    Parameters:
        subgroups (dict): A dict of subgroup DataFrames ('name': DataFrame). Works with create_subgroups().
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        dca_y_limits (list, optional): Y-axis limits for the decision curve analysis (DCA) plot. Defaults to [-0.01, 0.02].
        dca_thresholds (np.ndarray, optional): Array of thresholds for DCA. Defaults to np.arange(0, 0.10, 0.01).

    Returns:
        None: Prints the results
    """
     
    for subgroup in subgroups.keys():
        print('########################################')
        print(subgroup, '\n')
        validate_score(subgroups[subgroup], score_columns, outcome_column, test_size, dca_y_limits, dca_thresholds)

# END: Covariate Function

# START: Utils

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
        print(f'Number of removed rows: {removed_rows} ({((removed_rows / len(initial))*100):.2f}%)')
    else:
        removed_rows *= (-1)
        print(f'Number of added rows: {removed_rows} ({((removed_rows / len(initial))*100):.2f}%)')

# END: Utils

# START: Analysis

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

def get_probabilities(df:pd.DataFrame, score_column:str, outcome_column:str, plot:bool=False) -> pd.DataFrame:
    """
    Calculate the probabilities for an ouctome given a score value.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        plot (bool): Plot the resulting probabilities. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the probabilities.
    """
       
    df_probabilities = df.copy().groupby(score_column)[outcome_column].mean().reset_index()
    df_probabilities.columns = [score_column, f'{outcome_column}_probability']

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_probabilities[score_column], df_probabilities[f'{outcome_column}_probability'], marker='o', linestyle='-', color='black')
        plt.title(f'{outcome_column}_probability vs. {score_column}')
        plt.xlabel(score_column)
        plt.ylabel(f'{outcome_column}_probability')
        plt.grid(True)
        plt.show()

    return df_probabilities

def get_probabilities_for_test(df:pd.DataFrame, score_column:str, outcome_column:str, test_size=0.2) -> pd.DataFrame:
    """
    Calculate the probabilities for an ouctome given a score value and return a test set that was excluded in that calculation. 

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (str, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
    
    Returns:
        pd.DataFrame: A DataFrame containing a test set with the probabilities.
    """

    df_temp = df.copy()

    if test_size != 0:
        df_train, df_test = train_test_split(df_temp, test_size=test_size, random_state=42)
    else:
        df_train = df_test = df_temp

    df_probabilities = get_probabilities(df_train, score_column, outcome_column)
    df_test = pd.merge(df_test[[score_column, outcome_column]], df_probabilities, on=score_column, how='left')

    return df_test

# END: Analysis

# START: Validation

def get_auroc(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2) -> None:
    """
    Plots the ROC curve with AUC for a score and outcome.

    Parameters:
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    for score_column in score_columns:
        df_test = get_probabilities_for_test(df, score_column, outcome_column, test_size) # NOTE: test_size == 0 is handled in get_probabilities_for_test

        y_true = df_test[outcome_column]
        y_score = df_test[score_column]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f'{score_column} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--') # baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# reference: https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
def get_auprc(df: pd.DataFrame, score_columns: list, outcome_column: str, test_size: float = 0.2) -> None:
    """
    Plots the Precision-Recall curve (PRC) with AUC for a score and outcome.

    Parameters:
        score_columsn (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        label_name (str): Name of the model or label for legend.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    for score_column in score_columns:
        df_test = get_probabilities_for_test(df, score_column, outcome_column, test_size)  # NOTE: test_size == 0 is handled in get_probabilities_for_test

        y_true = df_test[outcome_column]
        y_score = df_test[score_column]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        # baseline (ratio of positive instances)
        baseline = sum(y_true) / len(y_true)

        plt.plot(recall, precision, label=f'{score_column} (AUPRC = {auc_pr:.2f})')    

    plt.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline = {baseline:.2f}') # baseline
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def prepare_regression_data(df_input_data:pd.DataFrame, scale:bool=True, intercept:bool=True, categorical_columns:list=[]) -> pd.DataFrame:
    """
    Prepares data for regression. Adds an intercept and optionally scales the data and also one-hot encodes categorical variables.
    
    Parameters:
        df_input_data (pandas.DataFrame): DataFrame containing the data.
        scale (bool, optional): If True, data will be scaled (range(0,1)). Defaults to True.
        intercept (bool, optional): If True, an intercept will be added. Defaults to True.
        categorical_columns (list, optional): Columns containing categorical data (will be one-hot encoded). Defaults to [].

    Returns:
        pandas.DataFrame: DataFrame with the prepared data.
    """
      
    # one-hot encoding for categorical columns
    df_data = pd.get_dummies(df_input_data.copy(), columns=categorical_columns, drop_first=True)

    # scale data
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        ndarray_data = scaler.fit_transform(df_data)
        df_data = pd.DataFrame(ndarray_data, columns=df_data.columns, index=df_data.index) # convert back to df

    # add intercept
    if intercept:
        df_data = sm.add_constant(df_data)

    return df_data

class Model(Enum):
    SKLEARN = 1
    STATS = 2

def get_regression(X:pd.DataFrame, y:pd.Series, test_size:float=0.2, label_name:str='', categorical_columns=[], use_lasso:bool=True, intercept:bool=True, scale_data:bool=True, selected_model:Model=Model.STATS) -> pd.DataFrame:
    """
    Trains a logistic regression model on the given data. Plots the results.
    
    Parameters:
        X (pandas.DataFrame): Features for training the model.
        y (pandas.Series): Target variable.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        label_name (str, optional): Label for title and legend.
        categorical_columns (list): A list of columns to consider as categorical (will be one-hot encoded). Defaults to [].
        use_lasso (bool, optional): Regularize the regression with lasso. Defaults to True.
        intercept (bool, optional): If True, an intercept will be added. Defaults to True.
        scale_data (bool, optional): If True, data will be scaled (range(0,1)). Defaults to True.
        selected_model (Model, optional): Selection of which regression model to use (Model.SKLEARN = sklearn for prediction tasks, Model.STATS = statsmodels for a more statistical approach). Defaults to STATS.

    Returns:
        pandas.DataFrame: A DataFrames for the weights.
    """

    # data preparation

    X = prepare_regression_data(X, scale=scale_data, intercept=intercept, categorical_columns=categorical_columns)

    if test_size != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else: # NOTE: check sensibility
        X_train = X_test = X
        y_train = y_test = y

    # training

    if selected_model == Model.SKLEARN:
        if use_lasso:
                model = LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=10000, C=1.0)
        else:
            model = LogisticRegression(random_state=42)

        pipe = make_pipeline(model) # apply scaling on training data (reference: https://scikit-learn.org/stable/modules/preprocessing.html)
        pipe.fit(X_train, y_train) 
        y_pred_proba = pipe.predict_proba(X_test)[:, 1] # positive class probabilities
    
    if selected_model == Model.STATS:
        model = sm.Logit(y_train, X_train)
        if use_lasso:
            result = model.fit_regularized(method='l1', alpha=1.0)
        else:
            result = model.fit()

        y_pred_proba = result.predict(X_test)

    # results

    if selected_model == Model.STATS:

        print(result.summary())

        # odds ratios (reference: https://stackoverflow.com/a/47740828)

        params = result.params
        odds_ratios = result.conf_int()
        odds_ratios['Odds Ratio'] = params
        odds_ratios.columns = ['5%', '95%', 'Odds Ratio']
        print(np.exp(odds_ratios))

        # AU-ROC

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{label_name} Logistic Regression (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # AU-PRC

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)

        # baseline (ratio of positive instances)
        baseline = sum(y_test) / len(y_test)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'{label_name} (AUPRC = {auc_pr:.2f})')    
        plt.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline = {baseline:.2f}') # baseline
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # weights

        feature_names = X.columns
        weights = result.params.values.flatten()
        df_weights = pd.DataFrame({'feature': feature_names, 'weight': weights})
        df_weights = df_weights.sort_values(by='weight', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(df_weights['feature'], df_weights['weight'])
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(f'{label_name} Feature Weights')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.show()

        # p-values

        p_values = result.pvalues
        df_p_values = pd.DataFrame({'feature': feature_names, 'p-value': p_values})
        df_p_values = df_p_values.sort_values(by='p-value')
        df_p_values['significant (< 0.05)'] = df_p_values['p-value'] < 0.05

        print(df_p_values)

    if selected_model == Model.SKLEARN:
        # AU-ROC

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{label_name} Logistic Regression (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # weights

        feature_names = X.columns
        weights = pipe.named_steps['logisticregression'].coef_.flatten()
        df_weights = pd.DataFrame({'feature': feature_names, 'weight': weights})
        df_weights = df_weights.sort_values(by='weight', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(df_weights['feature'], df_weights['weight'])
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(f'{label_name} Feature Weights')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.show()

    return df_weights

def normalize_weights(df:pd.DataFrame, weight_column:str='weight', threshold:float=0.05, save_to:str=None) -> pd.DataFrame:
    """
    Normalize the weights in a DataFrame relative to the smallest non-zero absolute weight and set near-zero values to 0.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the weights and variables.
        weight_column (str, optional): Column inside the DataFrame containing the weights. Defaults to 'weight'.
        threshold (float, optional): Threshold for when to consider a value to be 0. Defaults to 0.05.
        save_to (str: optional): Path to where a csv of the weights should be saved. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame with the normalized weights.
    """

    df_temp = df.copy()

    df_temp.loc[df_temp[weight_column].abs() < threshold, weight_column] = 0

    min_weight = df_temp[df_temp[weight_column] != 0][weight_column].abs().min()
    df_temp['normalized_weight'] = df_temp[weight_column] / min_weight

    if save_to != None:
        df_temp.to_csv(save_to, index=False)

    df_temp.drop(columns=[weight_column], inplace=True)

    return df_temp

def load_weights(df:pd.DataFrame, features:list, feature_column:str='feature', weight_column:str='normalized_weight') -> dict:
    """
    Loads weights for specified features from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'feature' and 'normalized_weight' columns.
        features (list): Features to load weights for.
        feature_column (str): Column name with the feature names. Defaults to 'feature'.
        weight_column (str): Column name with the weights. Defaults to 'normalized_weight'.

    Returns:
        dict: Feature names mapped to their weights. Unfound features get `None`.
    """

    weights = {}
    for feature in features:
        try:
            weight = float(df[df[feature_column] == feature][weight_column])
            weights[feature] = weight
        except KeyError:
            print(f"Warning: {feature} not found in DataFrame.")
            weights[feature] = None

    return weights

def get_brier(df:pd.DataFrame, score_column:str, outcome_column:str, test_size=0.2) -> None:
    """
    Calculate and print the Brier score for a score and outcome.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (str): Proportion of the dataset to include in the test split. Defaults to 0.2.
    
    Returns:
        None
    """

    df_test = get_probabilities_for_test(df, score_column, outcome_column, test_size) # NOTE: test_size == 0 is handled in get_probabilities_for_test

    if df_test[f'{outcome_column}_probability'].isnull().sum() > 0:
        print(f'Probabilities for {score_column} contain missings.')
        df_test = exclude_rows(df_test, f'{outcome_column}_probability', [np.nan]) # NOTE: investigate missings

    # reference: https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss
    y_true = df_test[outcome_column].astype(int)
    y_prob = df_test[f'{outcome_column}_probability']
    brier_score = brier_score_loss(y_true, y_prob)

    print(f'Brier score for {score_column} and {outcome_column}: {brier_score}')

def get_calibration(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, groups:int=10) -> None:
    """ 
    Plots the calibration plot for a given score and outcome. 
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        groups (int, optional): Number of groups to split the data into. Defaults to 10.
    
    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    for score_column in score_columns:
        df_test = get_probabilities_for_test(df, score_column, outcome_column, test_size) # NOTE: test_size == 0 is handled in get_probabilities_for_test
        
        observed = df_test[outcome_column]
        predicted = df_test[f'{outcome_column}_probability']

        prob_true, prob_pred = calibration_curve(observed, predicted, n_bins=groups)
        plt.plot(prob_pred, prob_true, 's-', label=f'Calibration for {score_column}')

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()

    """
    # deciles
    data = pd.DataFrame({'observed': observed, 'predicted': predicted})
    data['decile'] = pd.qcut(data['predicted'], groups, duplicates='drop')

    # calculate observed and expected frequencies (for each decile)
    obs_freq = data.groupby('decile')['observed'].sum()
    exp_freq = data.groupby('decile')['predicted'].sum()
    total = data.groupby('decile').size()

    # calculate statistic
    hl_stat = ((obs_freq - exp_freq) ** 2 / (exp_freq * (1 - exp_freq / total))).sum() # NOTE: to be checked (!)
    p_value = 1 - chi2.cdf(hl_stat, groups - 2) # NOTE: to be checked (!)
    print(f'Hosmer-Lemeshow test statistic: {hl_stat}')
    print(f'P-value: {p_value}')
    """

def get_score_regression(df:pd.DataFrame, score_column:str, outcome_column:str) -> None:
    """
    Performs logistic regression for a categorical score and its predicted outcome. Prints the results.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_column (str): The column name containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
    
    Returns:
        None
    """
      
    # reference: https://mskcc-epi-bio.github.io/decisioncurveanalysis/dca-tutorial-python.html#Multivariable_Decision_Curve_Analysis
    model = sm.GLM.from_formula(f'{outcome_column} ~ C({score_column})', data=df, family=sm.families.Binomial())
    results = model.fit()

    print(results.summary())

def get_dca(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, y_limits:list=[-0.01, 0.02], thresholds:np.ndarray=np.arange(0, 0.10, 0.01)) -> pd.DataFrame:
    """
    Plot decision curves for a given score and outcome.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        y_limits (list, optional): Y-axis limits for the plot. Defaults to [-0.01, 0.02].
        thresholds (np.ndarray, optional): Array of thresholds for decision curve analysis. Defaults to np.arange(0, 0.10, 0.01).

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """

    df_test = get_probabilities_for_test(df, score_columns[0], outcome_column, test_size) # NOTE: test_size == 0 is handled in get_probabilities_for_test
    df_test.drop(columns=score_columns[0], inplace=True)
    df_test.rename(columns={f'{outcome_column}_probability':f'{score_columns[0]}'}, inplace=True)

    if len(score_columns) > 1:         
        for score_column in score_columns[1:]:             
            df_test[score_column] = get_probabilities_for_test(df, score_column, outcome_column, test_size)[f'{outcome_column}_probability'] 

    # reference: https://mskcc-epi-bio.github.io/decisioncurveanalysis/dca-tutorial-python.html
    df_dca = dca(data=df_test, outcome=outcome_column, modelnames=score_columns, thresholds=thresholds)
    
    # print(df_dca)
    plot_graphs(plot_df=df_dca, graph_type='net_benefit', y_limits=y_limits, color_names=['blue', 'green', 'red', 'black'][:len(score_columns)+2])

    return df_dca

def validate_score(df:pd.DataFrame, score_columns:list, outcome_column:str, test_size:float=0.2, dca_y_limits:list=[-0.01, 0.02], dca_thresholds:np.ndarray=np.arange(0, 0.10, 0.01)):
    """
    Validates a score on all important metrics.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        score_columns (list): The column names containing the scores.
        outcome_column (str): The column name containing the outcome indicators.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        dca_y_limits (list, optional): Y-axis limits for the decision curve analysis (DCA) plot. Defaults to [-0.01, 0.02].
        dca_thresholds (np.ndarray, optional): Array of thresholds for DCA. Defaults to np.arange(0, 0.10, 0.01).

    Returns:
        None: Prints the results
    """

    """
    for score_column in score_columns:
        _ = get_score_regression(df=df, score_column=score_column, outcome_column=outcome_column)
    """

    print('####################')
    print('Performance:\n')

    for score_column in score_columns:
        _ = get_brier(df=df, score_column=score_column, outcome_column=outcome_column, test_size=test_size)

    print('\n')
    print('####################')
    print('Calibration: \n')

    _ = get_calibration(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size)

    print('####################')
    print('Discrimination: \n')

    _ = get_auroc(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size)
    _ = get_auprc(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size)

    print('####################')
    print('Clinical Value: \n')

    _ = get_dca(df=df, score_columns=score_columns, outcome_column=outcome_column, test_size=test_size, y_limits=dca_y_limits, thresholds=dca_thresholds)

# END: Validation