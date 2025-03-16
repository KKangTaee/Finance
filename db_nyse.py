
import commonHelper as ch
from mysqlConnecter import MySQLConnector
import pymysql
import pandas as pd


class DB_NYSE(MySQLConnector):
    def __init__(self):
        super().__init__()

    def connect(self):
        super().connect(ch.DBName.DB_NYSE)

    def disconnect(self):
        super().disconnect()