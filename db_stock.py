from copyreg import dispatch_table
from unittest import result
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from IPython.display import display
import assetAllocation
from mysqlConnecter import MySQLConnector
from commonHelper import EDateType, DBName
from assetAllocation import AssetAllocation


class DB_Stock(MySQLConnector):
    def __init__(self):
        super().__init__()

    def connect(self):
        super().connect(DBName.DB_STOCK)

    def disconnect(self):
        super().disconnect()

    def getStockData(self, symbol, start=None, end=None, freq = EDateType.DAY):
        """
        특정 주식 심볼(symbol)의 주가 데이터를 조회하는 함수.

        :param symbol: 조회할 주식 심볼
        :param start: 조회 시작 날짜 (YYYY-MM-DD, 선택 사항)
        :param end: 조회 종료 날짜 (YYYY-MM-DD, 선택 사항)
        :param freq: 'day' (일별, 기본값) 또는 'monthly' (월별 마지막 날)
        :return: Pandas DataFrame
        """

        # 기본 조회 쿼리
        base_query = f"SELECT * FROM StockPrices WHERE symbol = '{symbol}'"
        query_conditions = []

        # 날짜 조건 추가
        if start:
            query_conditions.append(f"date >= '{start}'")
        if end:
            query_conditions.append(f"date <= '{end}'")

        # 쿼리에 날짜 조건 추가
        if query_conditions:
            base_query += " AND " + " AND ".join(query_conditions)

        # 월별 데이터 처리
        if freq == EDateType.MONTHLY:
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY DATE_FORMAT(date, '%Y-%m') ORDER BY date DESC) as rn
                    FROM StockPrices
                    WHERE symbol = '{symbol}'
            """
            if query_conditions:
                query += " AND " + " AND ".join(query_conditions)

            query += ") AS subquery WHERE rn = 1 ORDER BY date ASC;"  # ✅ MySQL에서 실행 가능하도록 수정

        else:
            query = base_query + " ORDER BY date ASC;"

        # 데이터베이스 요청
        df = super().requestToDB(query)
        return df
    

    @staticmethod
    def get_stock_data(symbols, start_date, end_date):
        with DB_Stock() as stock:
            symbols_str =  ",".join(f"'{s}'" for s in symbols)
            query =f"""
                SELECT *
                FROM StockPrices
                WHERE Symbol In ({symbols_str})
                    AND Date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY Symbol ASC, Date ASC
                """
            df = stock.requestToDB(query)
            df = df.drop(columns=["Id"])
            df['Dividends'] = 0.0
            df = df.rename(columns={'Adj_close': 'Adj Close'})
            grouped = df.groupby('Symbol')
            
            symbol_dfs = {}
            for symbol, df in grouped:
                df = df.reset_index()
                df = df.sort_values(by='Date')
                symbol_dfs[symbol] =df

        return symbol_dfs
    



    # MDD 구하기
    def calculate_mdd(self, data):
        """
        주어진 주식 가격 데이터에서 MDD(Maximum Drawdown), recovery_by(회복 시점),
        그리고 underwater_period(최대 낙폭에서 회복까지 걸린 기간)를 계산하는 함수.

        Parameters:
            data (pd.DataFrame): 'date' (날짜)와 'close' (종가) 컬럼을 포함한 데이터프레임.

        Returns:
            pd.DataFrame: MDD, recovery_by, underwater_period 정보가 포함된 데이터프레임.
        """
        try:
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"])  # 날짜 변환
            df["peak"] = df["close"].cummax()  # 최고점 갱신
            df["drawdown"] = df["close"] / df["peak"] - 1  # 드로우다운 계산
            df["mdd"] = df["drawdown"].cummin()  # 최대 낙폭

            # MDD 발생 시점 및 최저점 찾기
            mdd_idx = df["mdd"].idxmin()  # 최대 낙폭 발생 시점
            peak_idx = df.loc[:mdd_idx, "close"].idxmax()  # 최고점 발생 시점
            valley_idx = df.loc[peak_idx:mdd_idx, "close"].idxmin()  # 최저점 발생 시점

            # MDD 발생 시점 및 최저점 날짜
            peak_date = df.loc[peak_idx, "date"]
            valley_date = df.loc[valley_idx, "date"]

            # 회복 시점 찾기 (최고점 회복한 첫 번째 시점)
            recovery_idx = df.loc[mdd_idx:, "close"][df["close"] >= df.loc[peak_idx, "close"]].index.min()
            recovery_date = df.loc[recovery_idx, "date"] if pd.notna(recovery_idx) else None

            # Underwater Period 계산
            if recovery_date:
                underwater_period = recovery_date - peak_date
                recovery_time = recovery_date - valley_date
            else:
                underwater_period = None
                recovery_time = None

            # 결과 출력용 데이터프레임 생성
            result = {
                "Start (Peak Date)": peak_date,
                "End (Valley Date)": valley_date,
                "Length (Drawdown Period)": (valley_date - peak_date),
                "Recovery By": recovery_date,
                "Recovery Time": recovery_time,
                "Underwater Period": underwater_period,
                "Drawdown": df.loc[mdd_idx, "mdd"] * 100
            }

            return pd.DataFrame([result])
        except Exception as e:
            print(f"calculate_mdd 애러 발생 : {e}")
    

    # 해당기간의 복리수익 구하기
    def calculate_cagr(self, df):

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 날짜 순 정렬
        df = df.sort_values('date')

        # 시작/종료 정보
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]

        # 기간 (연 단위)
        num_years = (end_date - start_date).days / 365.25

        # 예외 처리: 기간이 0이거나 시작가가 0이면 계산 불가
        if num_years == 0 or start_price == 0:
            return None

        # CAGR 계산
        cagr = (end_price / start_price) ** (1 / num_years) - 1

        return round(cagr * 100, 2)  # 퍼센트로 반환
    

    def calculate_annual_summary_by_close(self, df, initial_balance=10000):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        # 날짜 기준 정렬
        df = df.sort_values('date')

        # 연도별 마지막 날짜 기준 종가
        yearly = df.drop_duplicates(subset='year', keep='last').copy()

        # 전년도 종가와 비교해 수익률 계산
        yearly['prev_close'] = yearly['close'].shift(1)
        yearly['Return'] = (yearly['close'] / yearly['prev_close']) - 1

        # 첫 해는 전년도 종가가 없으니 수익률 0 처리
        yearly.loc[yearly['prev_close'].isna(), 'Return'] = 0

        # Balance 계산
        balance = initial_balance
        balances = []
        for ret in yearly['Return']:
            balance *= (1 + ret)
            balances.append(balance)

        # 결과 정리
        yearly['Balance'] = balances
        yearly['Yield'] = 0.00
        yearly['Income'] = 0.00

        # 퍼센트 및 반올림
        yearly['Return'] = (yearly['Return'] * 100).round(2)
        yearly['Balance'] = yearly['Balance'].round(2)

        return yearly[['year', 'Return', 'Balance', 'Yield', 'Income']]
    

    
    # 특정 기간의 월별 수익률 구하기
    # 배당 X, close로 집계
    def calculate_monthly_summary_by_close(self, df, initial_balance=10000):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M').astype(str)

        # 날짜 기준 정렬
        df = df.sort_values('date')

        # 월별 마지막 날짜 기준 종가
        monthly = df.drop_duplicates(subset='year_month', keep='last')

        # 전월 종가와 비교해 수익률 계산
        monthly['prev_close'] = monthly['close'].shift(1)
        monthly['Return'] = (monthly['close'] / monthly['prev_close']) - 1

        # 첫 달은 전월이 없으니 수익률 0 처리
        monthly.loc[monthly['prev_close'].isna(), 'Return'] = 0

        # Balance 계산
        balance = initial_balance
        balances = []
        for ret in monthly['Return']:
            balance *= (1 + ret)
            balances.append(balance)

        # 결과 정리
        monthly['Balance'] = balances
        monthly['Yield'] = 0.00
        monthly['Income'] = 0.00

        # 퍼센트 및 반올림
        monthly['Return'] = (monthly['Return'] * 100).round(2)
        monthly['Balance'] = monthly['Balance'].round(2)

        return monthly[['year', 'month', 'Return', 'Balance', 'Yield', 'Income']]

    # 표준편차 구하기
    # 표준편차가 높으면 - 주가 변동성이 크고, 위험성 높음
    # 표준편자가 낮으면 - 주가 변동성이 낮고, 안정적인 수익을 보임
    # 높다 낮다의 기준은? - 섹터별 평군 stdev를 구해서 이것보다 높은지 낮은지 평가.
    def calculate_stdev(self, df, freq=EDateType.DAY):
        """ 
        주어진 freq(일별/월별)에 따라 변동성(Stdev)을 연율화하여 계산
        - daily: 일별 데이터 기준 (252 거래일 연율화)
        - monthly: 월별 데이터 기준 (12개월 연율화)
        """
        df = df.copy()
        df['return'] = df['close'].pct_change()

        if freq == EDateType.DAY:
            annualization_factor = np.sqrt(252)
        elif freq == EDateType.MONTHLY:
            annualization_factor = np.sqrt(12)
        else:
            raise ValueError("freq 값이 잘못되었습니다. 'daily' 또는 'monthly' 중 선택하세요.")

        return np.std(df['return'].dropna()) * annualization_factor
    

    # 샤프지수
    # "위험을 감수한 만큼 얼마나 효율적으로 수익을 냈는가?" 의 지표
    # 샤프지수가 높을수록 위험 대비 수익률이 좋다는 뜻 보통 1이상이면 훌륭하다고 평가됨 (0.5 이하면 그저그럼..)
    def calculate_sharpe_ratio(self, df, risk_free_rate=0.00, freq=EDateType.DAY):
        """ 
        주어진 freq(일별/월별)에 따라 Sharpe Ratio를 연율화하여 계산
        - daily: 일별 데이터 기준 (252 거래일 연율화)
        - monthly: 월별 데이터 기준 (12개월 연율화)
        """
        df = df.copy()
        df['return'] = df['close'].pct_change()

        if freq == EDateType.DAY:
            annualization_factor = 252
        elif freq == EDateType.MONTHLY:
            annualization_factor = 12
        else:
            raise ValueError("freq 값이 잘못되었습니다. 'daily' 또는 'monthly' 중 선택하세요.")

        mean_return = df['return'].mean() * annualization_factor
        stdev = self.calculate_stdev(df, freq=freq)  # Stdev도 freq에 맞게 계산

        return (mean_return - risk_free_rate) / stdev

    def calculate_final_investment(self, df, start_amount):
        return self.calculate_annual_summary_by_close(df)['Balance'].iloc[-1]    
    

    # 얼마를 넣었을 때 수익률을 구하기
    def get_balance(self, df, init_balance):
        df.columns = ['Id', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='all') 

        # 수익률 구하기
        df['Balance'] = None
        df['Profit Ratio'] = None
        for idx in range(len(df)):
            if idx == 0:
                df.at[idx, 'Balance'] = init_balance
            else:
                prev_close = df.at[idx-1, 'Close']
                prev_balance = df.at[idx-1, 'Balance']            
                ratio = ((df.at[idx, 'Close'] - prev_close)/prev_close)

                df.at[idx, 'Profit Ratio'] = (ratio * 100).round(2)
                df.at[idx, 'Balance'] =  int(prev_balance *(1+ratio))

        return df



    def get_performance(self, symbols, start_date=None, end_date=None, init_balance = 10000, freq = EDateType.DAY):
        results = []
        fail_symbols = []
        
        # 딕셔너리 형태로 만듬
        symbol_dfs = {}
        for symbol in symbols:
            try:
                df = self.getStockData(symbol, start_date, end_date, freq=freq)
                if df.empty:
                    fail_symbols.append(symbol)
                    continue

                # 이름명 변경
                df.columns = ['Id', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='all') 
                
                symbol_dfs[symbol] = df

            except Exception as e:
                print(f"{symbol} 애러발생 : {e}")
                print(f"fail_symbol : {fail_symbols}")
                break

        # 월말 데이터만 추출
        symbol_dfs = AssetAllocation.filter_close_last_month(symbol_dfs)
        for symbol, df in symbol_dfs.items():
            df = self.get_balance(df, init_balance)
            display(df)
            df = AssetAllocation.get_performance(df, symbol)
            results.append(df)


        combined_df = pd.concat(results, ignore_index=True)
        return combined_df
    
    
    # 가중치 전략이긴 한데, 이건 그리 효과가 없는듯
    def getPerformanceSummary_Rank(self, symbols, start=None, end=None, start_amount = 10000):
        df = self.getPerformanceSummary(symbols, start, end, freq=EDateType.MONTHLY)
        
        df = df.dropna(subset=['CAGR', 'Sharpe Ratio'])
        df = df[(df['Start Balance'] < df['End Balance']) & (df['MDD'] > -50)]

        # 가중치 설정
        weights = {
            'CAGR': 0.3,          # 성장 중시
            'Sharpe Ratio': 0.2,  # 리스크 조정 수익성
            'MDD': 0.4,           # 최대 손실률
            'Standard Deviation': 0.1  # 변동성
        }

        # 데이터 표준화
        scaler = StandardScaler()

        copy_df = df.copy()

        copy_df[['CAGR', 'Sharpe Ratio', 'MDD', 'Standard Deviation']] = scaler.fit_transform(copy_df[['CAGR', 'Sharpe Ratio', 'MDD', 'Standard Deviation']])

        # 가중치 계산
        df['Total_Score'] = (copy_df['CAGR'] * weights['CAGR'] +
                            copy_df['Sharpe Ratio'] * weights['Sharpe Ratio'] +
                            copy_df['MDD'] * weights['MDD'] +  # MDD는 낮을수록 좋기 때문에 부호를 반전시킵니다.
                            copy_df['Standard Deviation'] * weights['Standard Deviation'])
                            
        # 정렬 및 결과 출력
        df_sorted = df.sort_values(by='Total_Score', ascending=False)
        return df_sorted
      

    def checkStockPricesBySymbol(self, symbols):
        if not symbols:
            return []

        # SQL Injection 방지를 위해 parameterized query 사용
        placeholders = ','.join(['%s'] * len(symbols))  # 심볼 개수만큼 `%s` 생성
        query = f"SELECT symbol FROM StockPrices WHERE symbol IN ({placeholders})"

        # DB에서 존재하는 심볼만 가져오기
        df = self.requestToDB(query, tuple(symbols))  # tuple로 전달

        # DB에 존재하는 심볼 리스트
        existing_symbols = set(df['symbol'].to_list())

        # 존재하지 않는 심볼 찾기
        missing_symbols = [symbol for symbol in symbols if symbol not in existing_symbols]

        return missing_symbols