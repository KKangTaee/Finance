
import commonHelper as ch
from mysqlConnecter import MySQLConnector
import pymysql
import pandas as pd


class DB_NYSE(MySQLConnector):
    def __init__(self):
        super().__init__()
        self.table_name = 'Stock'
        self.column_name_symbol ='symbolTicker'

    def connect(self):
        super().connect(ch.DBName.DB_NYSE)

    def disconnect(self):
        super().disconnect()


    def getDeleteSymbolList(self):
        quary = f"""
            SELECT {self.column_name_symbol}
            FROM {self.table_name} 
            WHERE BINARY symbol LIKE '%p%'
        """
        df = self.requestToDB(quary)

        return df[self.column_name_symbol].to_list()
    

    def removeDeleteSymbol(self):
        quary = f"""
            DELETE FROM {self.table_name}
            WHERE BINARY symbol LIKE '%p%';
        """
        self.commitToDB(quary)
        print("제거완료")
        

    def getSymbolList(self):
        query = f"""
            SELECT * FROM {self.table_name} ;
        """

        df = self.requestToDB(query)
        symbol_list = df[self.column_name_symbol].to_list()
      
        # '.U'로 끝나지 않는 심볼만 리스트에 포함
        # 주식티커 뒤에 .U가 붙는 경우 일반적으로 특수목적 인수회사의 유닛을 의마한다고 생각하면됨.
        symbol_list = [symbol for symbol in symbol_list if '.' not in symbol and '-' not in symbol]
       
        return symbol_list