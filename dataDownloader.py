import yfinance as yf
import pandas as pd
import commonHelper as ch
import plotly.graph_objects as go
import plotly.io as pio
import time
import json
import random
import concurrent.futures
import sys
import math

from db_financialStatement import DB_FinancialStatement
from db_stock import DB_Stock
from db_nyse import DB_NYSE
from tqdm import tqdm  # 진행률
from mysqlConnecter import MySQLConnector
from yFinanceInfo import YFinanceInfo
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup




class DataDownloader:
    def __new__(cls, *args, **kwargs):
        raise TypeError("이 클래스는 인스턴스를 만들 수 없습니다. (Static class 방식)")
    
  
    #--------------------
    # 재무재표 DB저장
    #--------------------
    def save_fs_to_db(stockInfo, conn, type, dateType): 
        infoName = ch.getStrFinancialStatementType(type)
        dateName = ch.getStrFinancialStatementType(dateType)

        tableName = f"{infoName}_{dateName}"

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


    #-------------------
    # 재무재표 데이터 저장
    #-------------------
    def downloadFinancialStatmentAndSaveDB(date, symbols=[]):
        db = DB_FinancialStatement()
        db.connect()
        
        info = YFinanceInfo()

        if len(symbols) == 0:
            symbols = db.getSymbolList()

        for symbol in tqdm(symbols, desc="Processing Companies"):
            
            info.setCompany(symbol)

            DataDownloader.save_fs_to_db(info, db.conn, ch.EFinancialStatementType.INCOME_STATEMENT, date)
            DataDownloader.save_fs_to_db(info, db.conn, ch.EFinancialStatementType.BALANCE_SHEET, date)
            DataDownloader.save_fs_to_db(info, db.conn, ch.EFinancialStatementType.CASH_FLOW, date)

        db.disconnect()


    #-------------------
    # 기간별 데이터 가져오기
    #-------------------
    def getStockData(tickers, start=None, end=None):
        """
        여러 종목의 주식 데이터를 가져와서
        Date, Symbol, Open, High, Low, Close, Adj Close, Volume 형태로 반환하는 함수

        Parameters:
        tickers (list or str): 주식 종목들의 리스트 또는 단일 종목
        start (str): 시작 날짜 (예: '2023-01-01')
        end (str): 끝 날짜 (예: '2023-12-31')

        Returns:
        pd.DataFrame: 정리된 주식 데이터
        """
        try:
            # auto_adjust=False로 설정해서 Dividends, Stock Splits 등 포함 가능
            data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)

            # 결과를 저장할 빈 데이터프레임
            final_df = pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                # 여러 종목일 때
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        temp_df = data[ticker].copy()
                        temp_df['Symbol'] = ticker
                        temp_df['Date'] = temp_df.index
                        temp_df = temp_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                        final_df = pd.concat([final_df, temp_df], axis=0)
                    else:
                        print(f"[경고] '{ticker}' 데이터 없음 (상장폐지/잘못된 심볼 등)")
            else:
                # 단일 종목일 때
                data['Symbol'] = tickers if isinstance(tickers, str) else tickers[0]
                data['Date'] = data.index
                final_df = data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

            final_df.reset_index(drop=True, inplace=True)

        except Exception as e:
            print(f"애러 발생! : {e}")

        return final_df


    #--------------------------------
    # getStockData 데이터를 바탕으로 그래프
    #--------------------------------
    @staticmethod
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
    # 주가 DB저장
    #--------------------
    @staticmethod
    def insert_stock_data(conn, df):
        """
        fetch_stock_data로 가져온 DataFrame을 MySQL StockPrices 테이블에 저장
        :param df: fetch_stock_data로 가져온 데이터프레임
        :param db_config: MySQL 접속 정보 (host, user, password, db, port)
        """

        # DB 연결
        connection = conn

        # ✅ NaN -> None 변환 (숫자형 컬럼 포함 safe 처리)
        df = df.astype(object).where(pd.notnull(df), None)

        # DataFrame을 튜플 리스트로 변환 (adj_close는 Close로 채워넣음)
        data_to_insert = list(
            df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].itertuples(index=False, name=None)
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


    #--------------------
    # 시계열 가져와 DB에 저장
    #--------------------
    @staticmethod
    def downloadStockDataAndSaveDB(start=None, end=None, symbols=[], batch_size=100, sleep_sec=5, startIndex=0, endIndex=None):
        """
        주식 데이터를 일정 개수씩 나눠서 다운로드하고 저장하는 함수

        Parameters:
        start (str): 시작 날짜 (예: '2023-01-01')
        end (str): 끝 날짜 (예: '2023-12-31')
        symbols (list): 주식 심볼 리스트
        batch_size (int): 한 번에 처리할 심볼의 수 (기본값: 100)
        sleep_sec (int): 각 batch 간 대기 시간 (초)
        startIndex (int): 심볼 리스트에서 시작할 인덱스 (기본 0)
        endIndex (int): 심볼 리스트에서 끝낼 인덱스 (기본 None, 끝까지)
        """

        if len(symbols) == 0:
            fs_db = DB_FinancialStatement()
            fs_db.connect()
            symbols = fs_db.getSymbolList()
            fs_db.disconnect()

        # 심볼 구간 잘라내기
        if endIndex is None or endIndex > len(symbols):
            endIndex = len(symbols)  # 범위 초과 방지

        symbols = symbols[startIndex:endIndex]  # 원하는 구간만 사용

        total = len(symbols)
        total_batches = math.ceil(total / batch_size)

        print(f"총 {total}개의 심볼을 {batch_size}개씩 나눠서 {total_batches}번에 걸쳐 처리합니다. (심볼 인덱스 범위: {startIndex} ~ {endIndex})")

        for i in range(0, total, batch_size):
            batch = symbols[i:i + batch_size]
            print(f"[{i // batch_size + 1}/{total_batches}] {len(batch)}개 심볼 다운로드 중...")

            db = DB_Stock()
            db.connect()

            try:
                df = DataDownloader.getStockData(batch, start, end)
                df = df.where(pd.notnull(df), None)
                if not df.empty:
                    DataDownloader.insert_stock_data(db.conn, df)
                    print(f"[{i // batch_size + 1}/{total_batches}] 데이터 저장 완료.")
                else:
                    print(f"[{i // batch_size + 1}/{total_batches}] 데이터 없음.")
            except Exception as e:
                print(f"[{i // batch_size + 1}/{total_batches}] 에러 발생: {e}")
            finally:
                db.disconnect()

            # 다음 batch를 위해 대기 (점진적 딜레이)
            if i + batch_size < total:
                print(f"{sleep_sec}초 대기 중...")
                time.sleep(sleep_sec)

        print("모든 데이터 처리 완료.")


    #--------------------
    # 미국주식 심볼로드
    #--------------------
    @staticmethod
    def load_nysc_symbol():
        driver = webdriver.Chrome()
        driver.get("https://www.nyse.com/listings_directory/stock")

        # 쿠키 동의 팝업 처리
        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Accept All Cookies")]'))
            )
            accept_button.click()
            print("쿠키 동의 완료")
        except Exception as e:
            print(f"쿠키 동의 버튼 찾기 실패: {e}")

        # 셀레니움에서 페이지 캡쳐한 것
        data = []

        while True:
            # 테이블 데이터가 나타날 때까지 최대 20초 기다림
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".table-data.w-full.table-border-rows tbody tr"))
            )

            while(True):           
                try:
                    html = driver.page_source 
                    soup = BeautifulSoup(html, 'html.parser')
                    tag = soup.select_one(".table-data.w-full.table-border-rows")            
                    infos = tag.select("tbody tr")

                    for info in infos:
                        symbol = info.select_one("td a").text.strip()
                        url = info.select_one("td a")["href"]
                        name = info.select("td")[1].text.strip()
                        data.append([symbol,name,url])
                    print(f"{soup.select_one('.px-2.text-gray-500').text} 다운완료! len : {len(data)}")

                    break

                except Exception as e:
                    print(f"재시도 : {e}")
                    time.sleep(3)
                    continue
                  
            try:
                btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 
                    r"#integration-id-fcc63aa > div.flex-1 > div.px-3.lg\:px-24 > div.flex.flex-col.flex-wrap.gap-x-8.md\:flex-row > div.basis-2\/3 > div.\!text-center > div > ul > li:nth-child(8) > a"))
                )
                btn.click()

                # 새로운 페이지가 로드될 때까지 대기
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".table-data.w-full.table-border-rows tbody tr"))
                )

            except NoSuchElementException:
                print("버튼을 찾을 수 없습니다. 페이지 탐색 종료.")
                break

            except TimeoutException:
                print("❌ [TimeoutException] 버튼 클릭 또는 페이지 로드가 시간 초과되었습니다. 페이지 탐색 종료.")
                break

        df = pd.DataFrame(data, columns=['symbol', 'name', 'url'])
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='all')
        return df


    @staticmethod
    def downloadNYSE_SymbolAndSaveDB():
        df = DataDownloader.load_nysc_symbol()

        if df.empty:
            print(f"df is null!")
            return
        
        db = MySQLConnector()
        db.connect(ch.DBName.DB_NYSE)

        try:
            with db.conn.cursor() as cursor:
                values = [tuple(x) for x in df[['symbol', 'name', 'url']].to_numpy()]

                # ✅ 쿼리 준비 (ON DUPLICATE KEY UPDATE로 upsert)
                sql = """
                INSERT INTO Symbol_Stock (symbol, name, url)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    url = VALUES(url),
                    update_at = CURRENT_TIMESTAMP;
                """
                # ✅ executemany로 일괄 처리
                cursor.executemany(sql, values)
            db.conn.commit()

        except Exception as e:
            print(f"quary error : {e}")

        print(f"NYSE 심볼 다운로드 : {len(df)}")

        db.disconnect()
        
        
        

    #--------------------
    # 기업정보 로드하기
    #--------------------
    @staticmethod
    def save_company_info_to_db(stockInfo, conn):
        try:
            info = stockInfo.stock.info
            symbol = stockInfo.ticker.upper()  # 주식 심볼 가져오기

            # ✅ MySQL 테이블 컬럼 리스트 확인
            with conn.cursor() as cursor:
                cursor.execute("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Company'")
                existing_columns = {row['COLUMN_NAME']: row['DATA_TYPE'] for row in cursor.fetchall()}

            # ✅ MySQL 컬럼과 매칭되는 데이터 추출
            data = {key: info.get(key, None) for key in info.keys()}
            data.pop("companyOfficers", None)  # ❌ companyOfficers 제거
            data["symbol"] = symbol  # 심볼 추가

            # ✅ 추가해야 할 컬럼 찾기 (데이터 타입 구분)
            missing_columns = set(data.keys()) - set(existing_columns.keys())

            if missing_columns:
                with conn.cursor() as cursor:
                    for col in missing_columns:
                        value = data[col]

                        # 숫자형 데이터는 FLOAT, BIGINT 등으로 저장
                        if isinstance(value, int):
                            col_type = "BIGINT NULL"
                        elif isinstance(value, float):
                            col_type = "FLOAT NULL"
                        elif isinstance(value, str) and len(value) > 255:
                            col_type = "TEXT NULL"  # 255자를 초과하는 문자열은 TEXT
                        else:
                            col_type = "VARCHAR(255) NULL"  # 기본적으로 VARCHAR(255)

                        sql = f"ALTER TABLE Company ADD COLUMN `{col}` {col_type};"
                        cursor.execute(sql)
                    conn.commit()
                print(f"🛠️ {symbol} - 누락된 컬럼 추가 완료: {missing_columns}")


            # ✅ 데이터 변환 (dict, list → JSON 문자열)
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)  # JSON 문자열로 변환
                    
            # ✅ 데이터 삽입 또는 업데이트
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            update_clause = ", ".join([f"{key} = VALUES({key})" for key in data.keys() if key != "symbol"])

            with conn.cursor() as cursor:
                sql = f"""
                INSERT INTO Company ({columns})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_clause};
                """
                cursor.execute(sql, tuple(data.values()))
                conn.commit()
            # print(f"✅ {symbol} - 데이터 저장 완료")
            sys.stdout.write(f"✅ {symbol} - 데이터 저장 완료")  # 한 줄 덮어쓰기
            sys.stdout.flush()

        except Exception as e:
            print(f"⚠️ {symbol} - 오류 발생: {e}")


    # 심볼과 함께 실패도 반환
    @staticmethod
    def process_symbol(symbol):
        try:
            time.sleep(random.uniform(0.5, 1.5))  # 요청 전에 랜덤 대기
            info = YFinanceInfo()
            info.setCompany(symbol)
            return symbol, info
        except Exception as e:
            print(f"⚠️ Error in process_symbol for {symbol}: {e}")
            return symbol, None


    @staticmethod
    def downloadCompanyInfoAndSaveDB(symbols=[]):
        db = DB_FinancialStatement()
        db.connect()

        if len(symbols) == 0:
            symbol_db = DB_NYSE()
            symbol_db.connect()

            symbols = symbol_db.getSymbolList()
            
            symbol_db.disconnect()


        # for symbol in tqdm(symbols, total=len(symbols), desc="Processing Companies"):
        #     info = YFinanceInfo()
        #     info.setCompany(symbol)
        #     DataDownloader.save_company_info_to_db(info, db.conn)

        max_workers = min(3, len(symbols))  # 적절한 워커 수 결정 (너무 많은 스레드 방지)

        # 병렬 처리
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(DataDownloader.process_symbol, symbol): symbol for symbol in symbols}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Companies"):
                symbol = futures[future]  # 항상 가져오기
                try:
                    result_symbol, info = future.result()
                    if info and info.stock is not None:
                        DataDownloader.save_company_info_to_db(info, db.conn)
                    else:
                        print(f"🚨 {result_symbol}: No stock data available.")
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")

        db.disconnect()



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