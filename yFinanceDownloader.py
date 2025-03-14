import yfinance as yf
import pandas as pd
import commonHelper as ch
import plotly.graph_objects as go
import plotly.io as pio
from db_financialStatement import DB_FinancialStatement
from db_stock import DB_Stock
from tqdm import tqdm  # 진행률
import sys
import json

class YFinanceDownloader:
    def __init__(self):
        pass

    def setCompany(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

        # 주가 데이터가 없는 경우 예외 처리
        if self.stock.history(period="1d").empty:
            print(f"🚨 {ticker} 데이터 없음!")
            self.stock = None  # 유효하지


    def isValidCompany(self, ticker):
        stock = yf.Ticker(ticker)
        return not stock.history(period ="1d").empty
    

    def getSector(self):
        return self.stock.info.get("sector", "정보없음")
    

    def getIndustry(self):
        return self.stock.info.get("industry", "정보없음")


    def getFsData(self, type, dateType):
        if type == ch.EFinancialStatementType.INCOME_STATEMENT:
            if dateType == ch.EDateType.YEAR:
                return self.stock.financials
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_financials
        
        elif type == ch.EFinancialStatementType.BALANCE_SHEET:
            if dateType == ch.EDateType.YEAR:
                return self.stock.balance_sheet
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_balance_sheet
            
        elif type == ch.EFinancialStatementType.CASH_FLOW:
            if dateType == ch.EDateType.YEAR:
                return self.stock.cash_flow
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_cash_flow
    

    def save_fs_to_db(self, conn, type, dateType): 
        infoName = ch.getStrFinancialStatementType(type)
        dateName = ch.getStrFinancialStatementType(dateType)

        tableName = f"{infoName}_{dateName}"
        stockInfo = self

        try:
            df = stockInfo.getFsData(type, dateType)
            df = df.where(pd.notna(df), None)

            if df.empty:
                print(f"❌ {tableName} 데이터 없음")
                return
            
            symbol = stockInfo.stock.info.get('symbol','N/A')
            name = stockInfo.stock.info.get('shortName','N/A')

            df = df.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
            df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
            df['symbol'] = symbol
            df['name'] = name

            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{tableName}'
                """)

                existing_columns = {row['COLUMN_NAME'] for row in cursor.fetchall()}
                
            new_columns = set(df.columns) - existing_columns - {"symbol", "year"}
            if new_columns:
                with conn.cursor() as cursor:
                    for col in new_columns:
                        value = df[col].iloc[0]

                        # 날짜 타입 확인 (문자열이지만 날짜 변환 가능한 경우)
                        if isinstance(value, str):  
                            try:
                                pd.to_datetime(value)  # 변환 가능 여부 체크
                                col_type = "DATE NULL"  # 변환 가능하면 DATE로 설정
                            except ValueError:
                                col_type = "TEXT NULL" if len(value) > 128 else "VARCHAR(128) NULL"
                        
                        # Pandas datetime 타입 (직접 변환)
                        elif isinstance(value, pd.Timestamp):
                            col_type = "DATE NULL"

                        # 숫자형 데이터는 FLOAT, BIGINT 등으로 저장
                        elif isinstance(value, int):
                            col_type = "BIGINT NULL"
                        elif isinstance(value, float):
                            col_type = "FLOAT NULL"
                        
                        # 기본적으로 VARCHAR(128)으로 설정
                        else:
                            col_type = "VARCHAR(128) NULL"  

                        sql = f"ALTER TABLE {tableName} ADD COLUMN `{col}` {col_type};"

                        cursor.execute(sql)
                    
                    conn.commit()
                print(f"🛠️ {symbol} - 누락된 컬럼 추가 완료: {new_columns}")

            with conn.cursor() as cursor:
                columns = ", ".join([f"`{col}`" for col in df.columns])
                placeholders = ", ".join(["%s"] * len(df.columns))
                update_clause = ", ".join([f"`{col}` = VALUES(`{col}`)" for col in df.columns if col not in ["date", "symbol"]])

                insert_sql = f"""
                    INSERT INTO {tableName} ({columns})
                    VALUES ({placeholders})
                    ON DUPLICATE KEY UPDATE {update_clause};
                """

                data = [tuple(row) for row in df.itertuples(index=False, name=None)]
                cursor.executemany(insert_sql, data)
                conn.commit()

            print(f"✅ {symbol} - {tableName} 데이터 삽입 완료")

        except Exception as e:
            print(f"⚠️ {symbol} - {tableName} 오류 발생: {e}")


    def downloadFinancialStatmentAndSaveDB(self, date, symbols=[]):
        db = DB_FinancialStatement()
        db.connect()
        
        if len(symbols) == 0:
            symbols = db.getSymbolList()

        for symbol in tqdm(symbols, desc="Processing Companies"):
            self.setCompany(symbol)
            self.save_fs_to_db(db.conn, ch.EFinancialStatementType.INCOME_STATEMENT, date)
            self.save_fs_to_db(db.conn, ch.EFinancialStatementType.BALANCE_SHEET, date)
            self.save_fs_to_db(db.conn, ch.EFinancialStatementType.CASH_FLOW, date)

        db.disconnect()


#region : Not Self 처리문

    #-------------------
    # 기간별 데이터 가져오기
    #-------------------
    def getStockData(tickers, start=None, end=None):
        """
        여러 종목의 주식 데이터를 가져와서
        Date, Symbol, Open, High, Low, Close, Volume 형태로 반환하는 함수

        Parameters:
        tickers (list): 주식 종목들의 리스트
        start (str): 시작 날짜 (예: '2023-01-01')
        end (str): 끝 날짜 (예: '2023-12-31')

        Returns:
        pd.DataFrame: 정리된 주식 데이터
        """
        # 데이터 다운로드
        data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)

        # 결과를 저장할 빈 데이터프레임
        final_df = pd.DataFrame()

        # 여러 종목이 들어온 경우
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                temp_df = data[ticker].copy()
                temp_df['Symbol'] = ticker
                temp_df['Date'] = temp_df.index
                temp_df = temp_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                final_df = pd.concat([final_df, temp_df], axis=0)
        else:
            # 단일 종목일 때
            data['Symbol'] = tickers[0]
            data['Date'] = data.index
            final_df = data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # 인덱스 초기화
        final_df.reset_index(drop=True, inplace=True)

        return final_df


    #--------------------------------
    # getStockData 데이터를 바탕으로 그래프
    #--------------------------------
    def showStockDataGraph(df, value, title):
        """
        주식 데이터프레임을 기반으로 특정 값을 꺾은선 그래프로 시각화하는 함수

        Parameters:
        df (pd.DataFrame): fetch_stock_data로 생성된 데이터프레임
        value (str): 그래프에 사용할 컬럼명 (예: 'Close', 'Open', 'High', 'Low', 'Volume')
        title (str): 그래프의 제목

        Returns:
        None (그래프 출력)
        """

        prev_renderer = pio.renderers.default
        # Plotly 기본 렌더러 설정 (옵션: 'browser', 'notebook', 'png' 등. 필요에 따라 변경 가능)
        pio.renderers.default = 'browser'
        
        # 그래프 그릴 객체 생성
        fig = go.Figure()

        # 고유한 종목들에 대해 각각 그래프 추가
        for symbol in df['Symbol'].unique():
            symbol_df = df[df['Symbol'] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['Date'],
                    y=symbol_df[value],
                    mode='lines',
                    name=symbol  # 종목명을 라벨로 사용
                )
            )

        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=value,
            legend_title='Symbol',
            hovermode='x unified'
        )

        # 그래프 출력
        fig.show()

        # 원래대로 복귀
        pio.renderers.default = prev_renderer


    #--------------------
    # DB에 저장
    #--------------------
    def insert_stock_data(conn, df):
        """
        fetch_stock_data로 가져온 DataFrame을 MySQL StockPrices 테이블에 저장
        :param df: fetch_stock_data로 가져온 데이터프레임
        :param db_config: MySQL 접속 정보 (host, user, password, db, port)
        """

        # DB 연결
        connection = conn

        # DataFrame을 튜플 리스트로 변환 (adj_close는 Close로 채워넣음)
        data_to_insert = list(
            df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close']].itertuples(index=False, name=None)
        )

        sql = """
            INSERT INTO StockPrices (date, symbol, open, high, low, close, volume, adj_close)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                adj_close = VALUES(adj_close);
        """

        try:
            with connection.cursor() as cursor:
                cursor.executemany(sql, data_to_insert)
                print(f"{cursor.rowcount} rows inserted/updated successfully.")
            connection.commit()
        except Exception as e:
            print("Error occurred:", e)
            connection.rollback()
        finally:
            connection.close()


    #--------------------
    # 시계열 가져와 DB에 저장
    #--------------------
    def downloadStockDataAndSaveDB(start=None, end = None, symbols=[]):

        if len(symbols) == 0:
            fs_db = DB_FinancialStatement()
            fs_db.connect()

            symbols = fs_db.getSymbolList()

            fs_db.disconnect() 
        

        db = DB_Stock()
        db.connect()

        df = YFinanceDownloader.getStockData(symbols, start, end)
        YFinanceDownloader.insert_stock_data(df)

        db.disconnect()

#endregion
        

# 주가수익비율 (PER)
# 주가가 회사의 수익에 비해 얼마나 높은지 보여주는 것
# 예를들어, 주식가격이 100원이고 주당순이익(EPS)이 5원이라면 PER의 20임 이는 주식가격이 순이익의 20배에 해당한다는 것
# P/E가 높으면 주식이 과대평가가 될 수 있고, 낮으면 저평가된 기업이라고 볼 수 있음.

# 자산순이익률 (ROA) - Asset
# 회사의 자산을 이용해 얼마나 효율적으로 이익을 창출하는지를 나타내는 것
# 예를들어, 자산이 1000만원이고 순이익이 100만원이면 ROA는 10%임. 이 수치가 높으면 자산을 잘 활용하고 있다는 뜻

# 자기자본이익률 (ROE) - Equity
# 회사가 자기자본을 바탕으로 얼마만큼 이익을 올렸는지를 나타내는 것
# 예를들어, 자기자본이 500만원이고 순이익이 50만원이면 ROE는 10%임. 높은 ROE는 회사가 자기자본을 효율적으로 사용하고 있다는 것

# 매출총이익 (Gross Profit)
# 매출에서 매출원가를 뺀 금액.
# 예를들어, 매출이 1000만원이고, 매출원가가 600만원이면 매출총이익은 400만원임. 이 금액은 회사가 기본적인 생산활동으로 벌어들인 이익을 보여줌

# 주식순자산비율 (PBR)
# 회사의 시장가치(주식가격)와 장부가치(자산과 부채차이)의 비율을 나타내는 것.
# 예를들어, 회사의 자산가치가 500만원이고, 주식의 시가총액이 1000만원이면 PBR의 비율은 2임
# 이 비율이 1보다 높으면 시장에서 회사의 가치가 장부가치를 초과한다고 볼 수 있음.

# EBITDA
# 이자, 세금, 감가상각비를 제외한 회사의 이익을 나타내는 것
# 예를들어, 이자비용 50만원, 세금 30만원, 감가상각비 20만원을 제외한 이익이 200만원이라면 EBITDA는 200만원임.
# 회사의 기본적인 수익 창출 능력을 평가할때 유용한 지표임.

# EV/EBITDA
# 회사의 기업가치(시가총액+순부채)와 EBITDA의 비율.
# 예를들어, 기업가치가 1000만원이고, EBITDA가 200만원이라면 EV/EBITDA는 5임.
# 이 비율이 낮으면 회사가 저평가 되었을 가능성이 있고, 높으면 고평가일 가능성이 높음.

# 베타(Beta)
# 주식의 변동성이 시장 전체의 변동성에 비해 얼마나 큰지를 나타내는 지표.
# 예를 들어, 배타가 1.5이면 시장이 1% 변동할때 주식은 1.5%변동할 가능성이 있다는 것을 의미함. 1보다 크면 변동성이 크고, 1보다 작으면 뎔변동적임.