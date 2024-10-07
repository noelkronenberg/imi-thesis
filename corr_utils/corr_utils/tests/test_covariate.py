# imports

import unittest
import pandas as pd
import contextlib
import io
from datetime import datetime

import corr_utils.covariate as covariate

# utils
# reference: https://docs.python.org/3/library/contextlib.html
@contextlib.contextmanager
def suppress_prints():
    with io.StringIO() as buf, contextlib.redirect_stdout(buf): # reference: https://stackoverflow.com/a/22434594
        yield

# tests

# covariate.extract_df_data()
class TestExtractDfData(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'case_id': ['id_001', 'id_002'],
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })

    def test_rename_columns(self):
        col_dict = {'name': 'first_name'}
        with suppress_prints():
            df_result = covariate.extract_df_data(self.df, col_dict=col_dict)
        self.assertTrue('first_name' in df_result.columns)
        self.assertTrue('age' in df_result.columns)

    def test_exact_match_filter(self):
        filter_dict = {'name': ['Alice']}
        with suppress_prints():
            df_result = covariate.extract_df_data(self.df, filter_dict=filter_dict, exact_match=True)
        self.assertEqual(len(df_result), 1)
        self.assertTrue('Alice' in df_result['name'].values)
        self.assertFalse('Bob' in df_result['name'].values)

    def test_non_exact_match_filter(self):
        filter_dict = {'name': ['A', 'B']}
        with suppress_prints():
            df_result = covariate.extract_df_data(self.df, filter_dict=filter_dict, exact_match=False)
        self.assertEqual(len(df_result), 2)
        self.assertTrue('Alice' in df_result['name'].values)
        self.assertTrue('Bob' in df_result['name'].values)

    def test_numeric_filter(self):
        filter_dict = {'age': [25]}
        with suppress_prints():
            df_result = covariate.extract_df_data(self.df, filter_dict=filter_dict, exact_match=False)
        self.assertEqual(len(df_result), 1)
        self.assertTrue('Alice' in df_result['name'].values)
        self.assertFalse('Bob' in df_result['name'].values)

    def test_remove_prefix(self):
        with suppress_prints():
            df_result = covariate.extract_df_data(self.df, remove_prefix=True)
        self.assertEqual(df_result['case_id'].tolist(), ['001', '002'])

# covariate.handle_duplicates()
class TestHandleDuplicates(unittest.TestCase):
    
    def setUp(self):
        self.df_no_duplicates = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })

        self.df_duplicates = pd.DataFrame({
            'A': [1, 2, 3, 3],
            'B': ['w', 'x', 'y', 'z']
        })

        self.df_multiple_duplicates = pd.DataFrame({
            'A': [1, 1, 2, 2, 2],
            'B': ['v', 'w', 'x', 'y', 'z']
        })

    def test_no_duplicates(self):
        with suppress_prints():
            duplicate_count, unique_duplicate_count = covariate.handle_duplicates(df=self.df_no_duplicates, column='A', drop_duplicates=False, testing=True)
        self.assertEqual(duplicate_count, 0)
        self.assertEqual(unique_duplicate_count, 0)

    def test_duplicates(self):
        with suppress_prints():
            duplicate_count, unique_duplicate_count = covariate.handle_duplicates(df=self.df_duplicates, column='A', drop_duplicates=False, testing=True)
        self.assertEqual(duplicate_count, 2)
        self.assertEqual(unique_duplicate_count, 1)

    def test_multiple_duplicates(self):
        with suppress_prints():
            duplicate_count, unique_duplicate_count = covariate.handle_duplicates(df=self.df_multiple_duplicates, column='A', drop_duplicates=False, testing=True)
        self.assertEqual(duplicate_count, 5)
        self.assertEqual(unique_duplicate_count, 2)

    def test_handle_duplicates_drop(self):
        with suppress_prints():
             df_result = covariate.handle_duplicates(self.df_duplicates, 'A', drop_duplicates=True)
        self.assertEqual(len(df_result), 3)
        self.assertEqual(df_result['A'].tolist(), [1, 2, 3])

    def test_handle_duplicates_no_drop(self):
        with suppress_prints():
             df_result = covariate.handle_duplicates(self.df_duplicates, 'A', drop_duplicates=False)
        self.assertEqual(df_result, None)

# covariate.filter_time_between()
class TestFilterTimeBetween(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2],
            'event_time': [datetime(2020, 1, 1, 6, 0),
                           datetime(2020, 1, 1, 6, 0)]
        })

        self.df_time_reference = pd.DataFrame({
            'id': [1, 2],
            'upper_bound_time': [datetime(2020, 1, 1, 7, 0),
                                 datetime(2020, 1, 1, 4, 0)],
            'lower_bound_time': [datetime(2020, 1, 1, 5, 0),
                                 datetime(2020, 1, 1, 3, 0)]
        })

    def test_filter_time_between_upper_bound(self):
        with suppress_prints():
            df_result = covariate.filter_time_between(self.df, 'event_time', self.df_time_reference, 'upper_bound_time', 'id')
        self.assertEqual(len(df_result), 1)
        self.assertEqual(df_result['id'].tolist(), [1])

    def test_filter_time_between_bounds(self):
        with suppress_prints():
            df_result = covariate.filter_time_between(self.df, 'event_time', self.df_time_reference, 'upper_bound_time', 'id', between=True, df_time_reference_column_lower_bound='lower_bound_time')
        self.assertEqual(len(df_result), 1)
        self.assertEqual(df_result['id'].tolist(), [1])

    def test_filter_time_between_drop_columns(self):
        with suppress_prints():
            df_result = covariate.filter_time_between(self.df, 'event_time', self.df_time_reference, 'upper_bound_time', 'id', between=True, df_time_reference_column_lower_bound='lower_bound_time', drop=True)
        self.assertFalse('upper_bound_time' in df_result.columns)
        self.assertFalse('lower_bound_time' in df_result.columns)

# covariate.exclude_rows()
class TestExcludeRows(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })

    def test_exclude_items(self):
        with suppress_prints():
            df_result = covariate.exclude_rows(self.df, 'name', ['Alice'])
        self.assertEqual(len(df_result), 1)
        self.assertEqual(df_result['name'].tolist(), ['Bob'])

    def test_exclude_numeric_items(self):
        with suppress_prints():
            df_result = covariate.exclude_rows(self.df, 'age', [30])
        self.assertEqual(len(df_result), 1)
        self.assertEqual(df_result['name'].tolist(), ['Alice'])
    
    def test_exclude_items_drop_column(self):
        with suppress_prints():
            df_result = covariate.exclude_rows(self.df, 'name', ['Alice'], drop=True)
        self.assertFalse('name' in df_result.columns)

    def test_exclude_no_items(self):
        with suppress_prints():
            df_result = covariate.exclude_rows(self.df, 'name', [])
        self.assertEqual(len(df_result), 2)
        self.assertEqual(df_result['name'].tolist(), ['Alice', 'Bob'])
    
    def test_exclude_all_items(self):
        with suppress_prints():
            df_result = covariate.exclude_rows(self.df, 'name', ['Alice', 'Bob'])
        self.assertEqual(len(df_result), 0)

# covariate.combine_date_time()
class TestCombineDateTime(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2],
            'date': ['20230724', '20230725'],
            'time': ['123000', '101500']
        })

    def test_combine_date_time(self):
        with suppress_prints():
            result_df = covariate.combine_date_time(self.df, 'date', 'time', 'date_time')
        self.assertTrue('date_time' in result_df.columns)
        self.assertFalse('date' in result_df.columns)
        self.assertFalse('time' in result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df['date_time'].iloc[0], pd.Timestamp('2023-07-24 12:30:00'))
        self.assertEqual(result_df['date_time'].iloc[1], pd.Timestamp('2023-07-25 10:15:00'))
