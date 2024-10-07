# author: Noel Kronenberg

# imports
import corr_utils

from typing import Optional
import getpass
import pandas as pd
import os
from impala.dbapi import connect # pip install impyla
import pandas as pd
import pexpect
# from sqlite3 import Connection # for typing

# default key for unqiue rows and merging of DataFrames
global default_key
# default_key:str = corr_utils.default_key

def connect_impala(remote_hostname:str, username:str) -> tuple:
    """
    Connects to Impala on the specified server.

    Parameters:
        remote_hostname (str): Hostname of server to connect to (e.g. hdl-edge01.charite.de).
        username (str): Charité username.

    Returns:
        tuple[Connection, pexpect.exceptions.ExceptionPexpect]: A tuple containing the Impala Connection object (None if the connection is not successful) and the error (None if the connection is successful).
    """

    try:
        # connect to Impala
        password = getpass.getpass(prompt='Enter your password: ')
        conn = connect(host=remote_hostname, port=21057, use_ssl=True, auth_mechanism='PLAIN', http_path='/default', user=username, password=password)
        error = None
    except pexpect.exceptions.ExceptionPexpect as e:
        print(f'Error obtaining access: {e}')
        error = e
        conn = None
    return conn, error

def get_impala_df(database:str, table:str, conn, limit:int=None, where:str=None) -> pd.DataFrame:
    """
    Extracts data from a specified table and database.

    Parameters:
        database (str): Name of the database.
        table (str): Name of the table.
        conn (sqlite3.Connection): Connection object for the database.
        limit (int, optional): Reduce amount of rows to extract. Defaults to None.
        where (str, optional): Custom 'WHERE' query. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the table.
    """

    if where:
        query = f'SELECT * FROM {database}.{table}'
        query += f' WHERE {where}'
        if limit:
            query += f' LIMIT {limit}'

    else:
        if limit != None:
            query = f'SELECT * FROM {database}.{table} LIMIT {limit}'
        else:
            query = f'SELECT * FROM {database}.{table}'

    df = pd.read_sql(query, conn)
    return df

def disconnect_impala(conn) -> None:
    """
    Disconnects from Impala on the server specified in pipeline.py.

    Parameters:
        conn (sqlite3.Connection): Connection object for the database.

    Returns:
        None
    """

    conn.close()