"""Exposes an API for easily downloading tables from the `sclerodata`
database hosted on teasle.

Each table is downloaded in its entirety and returned as a pandas
DataFrame object that can then be manipulated in memory. The tables
are typically so small with this dataset that this should not cause
any performance issues.

"""
import pandas as pd
import pymssql

from getpass import getpass


def available_tables():
    """Returns a list of tables that can be downloaded."""
    return sorted(list(_TABLES))


class Connection:
    """A context manager for a user's access to the `sclerodata` database
    on teasle.

    """
    def __init__(self, user, password=None):
        """Initialize the connection."""
        self.server = 'sclerodata'
        self.database = 'sclerodata'
        self.user = _full_user(user)
        self.password = getpass() if password is None else password 
        self.connection = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exc):
        self._disconnect()

    def fetch_table(self, table_name):
        """Returns a table from as a pandas DataFrame."""
        assert self.connection
        cursor = self.connection.cursor()
        try:
            cursor.execute('select * from {}'.format(table_name))
            results = pd.DataFrame(cursor.fetchall())
            results.columns = self._columns(cursor)
        finally:
            cursor.close()

        return results

    def _connect(self):
        self.connection = pymssql.connect(
            self.server, self.user, self.password, self.database)

    def _disconnect(self):
        self.connection.close()
        self.connection = None

    def _columns(self, cursor):
        return list(zip(*cursor.description))[0]


_TEASLE_DOMAIN = 'WIN-T312MF37MBJ\\'

_TABLES = {
    'tPtData',
    'tVisit',
    'tPFT',
    'tSera',
    'tMeds'}

    
def _full_user(user):
    return _TEASLE_DOMAIN + user
