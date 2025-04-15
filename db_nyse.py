
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


    def getDeleteSymbolList(self):
        quary = """
            SELECT symbol 
            FROM Symbol_Stock 
            WHERE BINARY symbol LIKE '%p%'
        """
        df = self.requestToDB(quary)

        return df['symbol'].to_list()
    

    def removeDeleteSymbol(self):
        quary = """
            DELETE FROM Symbol_Stock 
            WHERE BINARY symbol LIKE '%p%';
        """
        self.commitToDB(quary)
        print("제거완료")
        

    def getSymbolList(self):
        query = """
            SELECT * FROM Symbol_Stock;
        """

        df = self.requestToDB(query)

        symbol_list = df['symbol'].to_list()

        # '.U'로 끝나지 않는 심볼만 리스트에 포함
        # 주식티커 뒤에 .U가 붙는 경우 일반적으로 특수목적 인수회사의 유닛을 의마한다고 생각하면됨.
        # symbol_list = [symbol for symbol in df['symbol'].to_list() if (not symbol.endswith('.U') and not symbol.endswith('.A')) ]
        symbol_list = [symbol for symbol in symbol_list if '.' not in symbol]

        return symbol_list