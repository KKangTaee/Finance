from functools import reduce
from heapq import merge
from symtable import Symbol
from assetAllocation import AssetAllocation
from db_stock import DB_Stock
from mysqlConnecter import MySQLConnector
from commonHelper import EFinancialStatementType, EDateType, EIndustry
from IPython.display import display
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from dateutil.parser import parse

import pymysql
import pandas as pd
import commonHelper as ch
import numpy as np

# 재무재표 DB 데이터 조회
class DB_FinancialStatement(MySQLConnector):
    def __init__(self):
        super().__init__()

    def connect(self):
        super().connect(ch.DBName.DB_FINANCEIAL_STATEMENT)

    def disconnect(self):
        super().disconnect()


    def get_company(self, symbols: list):
        # 심볼 리스트가 비어있으면 빈 DataFrame 반환
        if not symbols:
            return pd.DataFrame()

        # 각 심볼을 작은 따옴표로 감싸서 SQL-safe한 문자열 생성
        formatted_symbols = ', '.join(f"'{symbol}'" for symbol in symbols)

        # 쿼리 문자열 생성 (주의: 직접 문자열 삽입이므로 반드시 escape 필요)
        query = f"""
            SELECT * FROM Company WHERE symbol IN ({formatted_symbols})
        """
        df = super().requestToDB(query)
        return df
    


    def get_symbols_data_existence(self, symbols: list):
        if not symbols:
            return pd.DataFrame()

        base_query = """
            SELECT s.symbol,
                EXISTS (SELECT 1 FROM IncomeStatement_Year     i WHERE i.symbol = s.symbol) AS has_ISY,
                EXISTS (SELECT 1 FROM IncomeStatement_Quarter  i WHERE i.symbol = s.symbol) AS has_ISQ,
                EXISTS (SELECT 1 FROM BalanceSheet_Year        b WHERE b.symbol = s.symbol) AS has_BSY,
                EXISTS (SELECT 1 FROM BalanceSheet_Quarter     b WHERE b.symbol = s.symbol) AS has_BSQ,
                EXISTS (SELECT 1 FROM CashFlow_Year            c WHERE c.symbol = s.symbol) AS has_CFY,
                EXISTS (SELECT 1 FROM CashFlow_Quarter         c WHERE c.symbol = s.symbol) AS has_CFQ
            FROM (
                {symbol_union}
            ) AS s
        """

        symbol_union = ' UNION ALL '.join(f"SELECT '{symbol.strip()}' AS symbol" for symbol in symbols)
        full_query = base_query.format(symbol_union=symbol_union)
        df = super().requestToDB(full_query)
        return df
    


    #---------------------
    # 심볼(티커) 리스트 가져오기
    #---------------------
    def getSymbolList(self):
        symbols = []
        quary = """
            SELECT symbol FROM Company;
        """

        df = super().requestToDB(quary,['symbol'])
        symbols = [row['symbol'] for _, row in df.iterrows()] # iterrows 쓰면, 인덱스랑 데이터 분리되서 나옴
        return symbols
    

    def getSymbolListByFilter(self, min_year = 0):

        if min_year == 0:
            # N년전 1월 1일 반환
            now = datetime.now()
            target_year = now.year - 4 
            years_ago = datetime(year=target_year, month=1,day=1,hour=0,minute=0,second=0)
        else:
            years_ago = datetime(year=min_year, month=1,day=1,hour=0,minute=0,second=0)

        years_ago_ms = int(years_ago.timestamp()*1000)
        print(f"{years_ago} 이전 상장된 기업 추출")

        symbols = []
        query = f"""
            SELECT symbol
            FROM Company
            WHERE isSpec IS NOT NULL
            AND isSpec != 1
            AND firstTradeDateMilliseconds IS NOT NULL
            AND firstTradeDateMilliseconds < {years_ago_ms};
        """

        df = super().requestToDB(query, ['symbol'])
        symbols = [row['symbol'] for _, row in df.iterrows()]
        return symbols


    #---------------------------
    # 시총 높은 순으로 N개의 기업을 조회
    #---------------------------
    def getCompanyByMarketCap(self, count):
        sql = f"""
            SELECT symbol, name, marketCap, industry
            FROM Company
            ORDER BY marketCap DESC
            LIMIT {count};
        """

        df = super().requestToDB(sql)
        return df
        

    #-------------------
    # 모든 산업의 개수 조회
    #-------------------
    def getIndustryCountByAll(self):
        sql = """
                SELECT industry, COUNT(*) AS count
                FROM Company
                GROUP BY industry
                ORDER BY count DESC;
            """
        
        df = super().requestToDB(sql)
        return df
        

    # 시총 높은 순의 기업이 어떤 산업으로 되어 있는지 조회
    def getIndustryCountByMarektCap(self, count):

        sql = f"""
                SELECT industry
                FROM Company
                WHERE industry IS NOT NULL
                ORDER BY marketCap DESC
                LIMIT {count};
            """
        
        df = self.requestToDB(sql)

        # ✅ industry 개수 세기
        industry_counts = df['industry'].value_counts().reset_index()
        industry_counts.columns = ['industry', 'count']

        return industry_counts


    def get_fs(self, symbols:list, type:EFinancialStatementType, dateType: EDateType):
        if not symbols:
            return pd.DataFrame()
        
        infoName = ch.getStrFinancialStatementType(type)
        dateName = ch.getStrDateType(dateType)

        symbols_str = ', '.join(f"'{symbol}'" for symbol in symbols)

        query = f"""
            SELECT *
            FROM {infoName}_{dateName}
            WHERE Symbol IN ({symbols_str})
        """

        df = self.requestToDB(query)

        # 제외할 컬럼
        exclude_cols = ['Id', 'Symbol', 'Date']

        # 제외한 컬럼들만 대상으로 NaN 체크
        cols_to_check = [col for col in df.columns if col not in exclude_cols]

        # 실제 삭제
        df = df.dropna(subset=cols_to_check, how='all')

        return df
    

    def get_fs_all(self, symbols:list, dateType:EDateType, min_year:int = 0):
        
        if not symbols:
            return pd.DataFrame()
        
        is_df = self.get_fs(symbols, EFinancialStatementType.INCOME_STATEMENT, dateType)
        bs_df = self.get_fs(symbols, EFinancialStatementType.BALANCE_SHEET, dateType)
        cf_df = self.get_fs(symbols, EFinancialStatementType.CASH_FLOW, dateType)

        is_df = is_df.drop(columns=['Id'])
        bs_df = bs_df.drop(columns=['Id'])
        cf_df = cf_df.drop(columns=['Id'])

        dfs = [is_df, bs_df, cf_df]
        merged = reduce(lambda left, right: pd.merge(left, right, on=['Symbol', 'Date', 'Name'], how='outer'), dfs)

        # 필수항목이 없으면 제외시킨다

        # Income Statement 항목
        income_statement_cols = [
            "TotalRevenue",
            # "CostOfRevenue", # 매출원가
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            # "DilutedEPS",   # 희석 주당순이익
            # "DepreciationAndAmortization" # 감가상각과 무형자산상각
        ]

        # Balance Sheet 항목
        balance_sheet_cols = [
            "TotalAssets",
            "TotalLiabilitiesNetMinorityInterest",
            "CommonStockEquity",
            # "CashAndCashEquivalents"
        ]

        # Cash Flow 항목
        cash_flow_cols = [
            "OperatingCashFlow",
            "FreeCashFlow",
            "CapitalExpenditure"
        ]

        required_columns = income_statement_cols + balance_sheet_cols + cash_flow_cols
    
        merged = merged.dropna(subset= required_columns)

        if min_year > 0:
            min_date = datetime(min_year, 1, 1).date()
            merged = merged[merged['Date'] >= min_date]

        merged = merged.reset_index(drop=True) # row를 제거했기 때문에 다시 인덱스를 재조정해야함 (안하면 뻑남)

        return merged
    

    

    # 시총구하기
    def get_marketCap(self, df):
        df = df.copy()
        df['MarketCap'] = np.nan # none 보다 nan으로 처리해야지 경고문이 안뜸.

        for idx in range(len(df)):
            date = df.at[idx, 'Date']
            symbol = df.at[idx, 'Symbol']
            ordinarySharesNumber = df.at[idx, 'OrdinarySharesNumber'] # 현재 발행된 보통주의 수량

            first_date, last_date = ch.get_first_and_last_date(date)

            with DB_Stock() as stock:
                try:
                    stock_df = stock.getStockData(symbol, first_date, last_date)
                    symbol_dfs = AssetAllocation.filter_close_last_month({symbol: stock_df})
                
                    close = symbol_dfs[symbol].at[0, 'Close']
                    market_cap = ordinarySharesNumber * close
                    df.at[idx, 'MarketCap'] = market_cap
                except Exception as e:
                    print(f"[ERROR] Market cap 계산 중 예외 발생: {e}")
                    # 주가 데이터가 존재하지 않을 경우 NaN으로 유지
                    df.at[idx, 'MarketCap'] = np.nan

        return df
    

    # PSR
    # 주가가 매출에비해 얼마나 높은지를 평가함
    # PSR < 1 저평가 가능성

    def get_psr(self, df:pd.DataFrame):

        if 'MarketCap' not in df.columns:
            return
        
        if 'TotalRevenue' not in df.columns: # 매출액
            return
        
        df = df.copy()
        df['PSR'] = None

        for idx in range(len(df)):
            marketCap = df.at[idx,'MarketCap']
            totalRevenue = df.at[idx, 'TotalRevenue']
            
            if totalRevenue and totalRevenue != 0:
                df.at[idx, 'PSR'] = marketCap / totalRevenue
            else:
                df.at[idx, 'PSR'] = np.nan
        
        return df
    
    # GP/A
    # 기업이 자산을 활용해 얼마나 효율적으로 매출총이익을 창출하는지 평가하는 지표
    # 높은 GP/A : 자산 대비 매출총이익이 높아 자산 효율성이 우수. 즉, 적은자산으로 높은이익을 창출
    def get_gp_a(self, df:pd.DataFrame):

        if 'GrossProfit' not in df.columns:
            return
        
        if 'TotalAssets' not in df.columns:
            return
        
        df = df.copy()
        df['GP/A'] = None

        for idx in range(len(df)):
            grossProfit = df.at[idx,'GrossProfit']
            totalAssets = df.at[idx, 'TotalAssets']

            # 예외 처리: 값이 None이거나 NaN일 경우 계산 생략
            if pd.isna(grossProfit) or pd.isna(totalAssets) or totalAssets == 0:
                continue
            
            df.at[idx, 'GP/A'] = grossProfit/totalAssets

        return df
        

    # POR
    # 주가가 기업의 영업활동으로 창출된 이익에 비해 얼마나 높은지 평가하는 가치 지표
    def get_por(self, df:pd.DataFrame):

        if 'MarketCap' not in df.columns:
            return
        
        if 'OperatingIncome' not in df.columns:
            return

        df = df.copy()
        df['POR'] = None

        for idx in range(len(df)):
            marketCap = df.at[idx,'MarketCap']
            totalAssets = df.at[idx, 'OperatingIncome']

            df.at[idx, 'POR'] = marketCap/totalAssets

        return df
    

    # EV/EVIT
    # 시가총액 + 총부채 - 현금 및 현금성 자산 = EV(기업가치)
    # EVIT는 세전이익을 말함. 그래서 기업가치를 세전이익으로 나눈것을 말한다.
    def get_ev_ebti(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['MarketCap', 'TotalDebt', 'CashCashEquivalentsAndShortTermInvestments', 'EBIT']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['EV/EBIT'] = None

        # 각 행에 대해 EV/EBIT 계산
        for idx in range(len(df)):
            # 기업가치(EV) 계산: MarketCap + TotalDebt - CashCashEquivalentsAndShortTermInvestments
            market_cap = df.at[idx, 'MarketCap']
            total_debt = df.at[idx, 'TotalDebt']
            cash = df.at[idx, 'CashCashEquivalentsAndShortTermInvestments']
            ebit = df.at[idx, 'EBIT']

            # EV 계산
            enterprise_value = market_cap + total_debt - cash

            # EBIT가 0 또는 음수면 EV/EBIT 계산 불가 (None 유지)
            # if ebit > 0:
            df.at[idx, 'EV/EBIT'] = enterprise_value / ebit

        return df
    
    # PER
    # 시총분에 순이익을 나눈 것
    def get_per(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['MarketCap', 'NetIncome']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['PER'] = None

        # 각 행에 대해 PER 계산
        for idx in range(len(df)):
            market_cap = df.at[idx, 'MarketCap']
            net_income = df.at[idx, 'NetIncome']

            # NetIncome이 0 또는 음수면 PER 계산 불가 (None 유지)
            # if net_income > 0:
            if net_income and net_income != 0:
                df.at[idx, 'PER'] = market_cap / net_income
            else:
                df.at[idx, 'PER'] = np.nan

        return df
    

    # 청산가치
    # 유동자산 - 부채총계 = 청산가치
    # 청산가치는 기업이 청산 시 주주에게 남을 수 있는 가치를 나타냄.
    # 시가총액보다 높으면 주식 수익률이 높을 가능성이 있음.
    def get_liquidation_value(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['CurrentAssets', 'TotalLiabilitiesNetMinorityInterest', 'MarketCap']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['LiquidationValue'] = None
        df['IsLiquidationValueHigher'] = None

        # 각 행에 대해 청산가치 계산
        for idx in range(len(df)):
            current_assets = df.at[idx, 'CurrentAssets']
            total_liabilities = df.at[idx, 'TotalLiabilitiesNetMinorityInterest']
            market_cap = df.at[idx, 'MarketCap']

            # 청산가치 계산
            liquidation_value = current_assets - total_liabilities
            df.at[idx, 'LiquidationValue'] = liquidation_value
        
        return df
    

    # 유동비율
    # 유동자산 / 유동부채 = 유동비율
    def get_current_ratio(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['CurrentAssets', 'CurrentLiabilities']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['CurrentRatio'] = None

        # 각 행에 대해 유동비율 계산
        for idx in range(len(df)):
            current_assets = df.at[idx, 'CurrentAssets']
            current_liabilities = df.at[idx, 'CurrentLiabilities']

            # 유동부채가 0이 아니면 유동비율 계산
            if current_liabilities is not None and current_liabilities > 0:
                df.at[idx, 'CurrentRatio'] = current_assets / current_liabilities

        return df
    

    # PBR
    # 시가총액 / 순자산 = PBR
    # PBR이 낮은 기업은 주가가 장부가치 대비 저평가되어있음
    def get_pbr(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['MarketCap', 'StockholdersEquity']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['PBR'] = None

        # 각 행에 대해 PBR 계산
        for idx in range(len(df)):
            market_cap = df.at[idx, 'MarketCap']
            stockholders_equity = df.at[idx, 'StockholdersEquity']

            # 순자산이 0보다 크면 PBR 계산
            if stockholders_equity is not None and stockholders_equity > 0:
                df.at[idx, 'PBR'] = market_cap / stockholders_equity

        return df
    

    # 차입금비율
    # 차입금비율이 개선되가나, 영업기익이 차입금 대비 성장하는 기업은 주식수익률이 높다.
    def get_debt_to_equity_ratio(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['TotalDebt', 'TotalCapitalization']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['DebtToEquityRatio'] = None

        # 각 행에 대해 차입금비율 및 영업이익/차입금 비율 계산
        for idx in range(len(df)):
            total_debt = df.at[idx, 'TotalDebt']
            total_capitalization = df.at[idx, 'TotalCapitalization']

            # 차입금비율 계산 (TotalCapitalization > 0)
            if total_capitalization is not None and total_capitalization > 0:
                df.at[idx, 'DebtToEquityRatio'] = total_debt / total_capitalization

        return df
    

    def get_pcr(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['MarketCap', 'OperatingCashFlow']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['PCR'] = None

        # 각 행에 대해 PCR 계산
        for idx in range(len(df)):
            market_cap = df.at[idx, 'MarketCap']
            operating_cash_flow = df.at[idx, 'OperatingCashFlow']

            # OperatingCashFlow가 0보다 크면 PCR 계산
            if operating_cash_flow is not None and operating_cash_flow > 0:
                df.at[idx, 'PCR'] = market_cap / operating_cash_flow

        return df
    

    def get_pfcr(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['MarketCap', 'OperatingCashFlow', 'CapitalExpenditure']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['PFCR'] = None

        # 각 행에 대해 PFCR 계산
        for idx in range(len(df)):
            market_cap = df.at[idx, 'MarketCap']
            operating_cash_flow = df.at[idx, 'OperatingCashFlow']
            capital_expenditure = df.at[idx, 'CapitalExpenditure']

            # 잉여현금흐름 계산 (CapitalExpenditure는 음수로 기록될 수 있음)
            free_cash_flow = operating_cash_flow - (capital_expenditure if capital_expenditure is not None else 0)

            # FreeCashFlow가 0보다 크면 PFCR 계산
            if free_cash_flow is not None and free_cash_flow > 0:
                df.at[idx, 'PFCR'] = market_cap / free_cash_flow

        return df
    

    # 배당성향
    def get_dividend_payout_ratio(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_columns = ['CommonStockDividendPaid', 'NetIncome']
        for col in required_columns:
            if col not in df.columns:
                return None
        
        # DataFrame 복사
        df = df.copy()
        df['DividendPayoutRatio'] = None

        # 각 행에 대해 배당성향 계산
        for idx in range(len(df)):
            dividend_paid = df.at[idx, 'CommonStockDividendPaid']
            net_income = df.at[idx, 'NetIncome']

            # 배당금은 음수(지출)로 기록될 수 있으므로 절댓값 사용
            # NetIncome이 0보다 크고, 배당금이 0보다 크면 배당성향 계산
            if (net_income is not None and net_income > 0 and 
                dividend_paid is not None and dividend_paid < 0):  # 음수로 기록된 배당금
                df.at[idx, 'DividendPayoutRatio'] = abs(dividend_paid) / net_income

        return df
    

    def get_sector(self, df:pd.DataFrame):

        df['Sector'] = None
        for idx in range(len(df)):
            symbol = df.at[idx, 'Symbol']

            query = f"""SELECT Sector 
                      FROM Company 
                      Where Symbol ='{symbol}';"""
            sector_df = self.requestToDB(query)
            df.at[idx,'Sector'] = sector_df.iloc[0]['Sector']

        return df
    

    # 
    def get_income_growth(self, df:pd.DataFrame) -> pd.DataFrame:
        
        if 'OperatingIncome' not in df.columns:
            return None
        
        df = df.copy()
        df = df.sort_values(by=['Symbol','Date'], ascending=True)
        df['IncomeGrowth'] = None


        for idx in range(len(df)):
            
            if idx == 0:
                continue

            prev_income = df.at[idx-1, 'OperatingIncome']
            curr_income = df.at[idx, 'OperatingIncome']
            ratio = ((curr_income - prev_income)/prev_income).round(2)

            df.at[idx, 'IncomeGrowth'] = ratio

        return df
    
    def get_gross_profit_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매출총이익률 (Gross Profit Margin) = GrossProfit / Revenue
        Revenue 컬럼은 TotalRevenue 또는 OperatingRevenue 중 하나가 필요함.
        """
        
        # 필요한 컬럼 확인
        required_columns = ['GrossProfit', 'TotalRevenue', 'OperatingRevenue']
        if not any(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['Revenue'] = None
        df['GrossProfitMargin'] = None

        # Revenue 설정 우선순위: TotalRevenue > OperatingRevenue
        for idx in range(len(df)):
            if 'TotalRevenue' in df.columns and pd.notnull(df.at[idx, 'TotalRevenue']):
                df.at[idx, 'Revenue'] = df.at[idx, 'TotalRevenue']
            elif 'OperatingRevenue' in df.columns and pd.notnull(df.at[idx, 'OperatingRevenue']):
                df.at[idx, 'Revenue'] = df.at[idx, 'OperatingRevenue']
            else:
                continue  # 둘 다 없으면 넘어감

            gross_profit = df.at[idx, 'GrossProfit']
            revenue = df.at[idx, 'Revenue']

            if gross_profit is not None and revenue and revenue != 0:
                df.at[idx, 'GrossProfitMargin'] = gross_profit / revenue

        return df
    

    def get_dividend_yield_proxy(self, df:pd.DataFrame) ->pd.DataFrame:
        """
        시가총액 대비 배당금 비율을 의미
        실제 "배당수익률"과 유사하나, 1주당 배당금 / 주가가 아닌 총 배당금 / 시총 형태
        """

        required_columns = ['CommonStockDividendPaid', 'MarketCap']
        if not any(col in df.columns for col in required_columns):
            return None
        
        df["CommonStockDividendPaid"] = df["CommonStockDividendPaid"].fillna(0)
        df["MarketCap"] = df["MarketCap"].fillna(0)
        df["DividendYieldProxy"] = df["CommonStockDividendPaid"] / df["MarketCap"]

        return df
    
    def get_roe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ROE (Return on Equity) = NetIncomeCommonStockholders / CommonStockEquity
        """
        required_columns = ['NetIncomeCommonStockholders', 'CommonStockEquity']
        if not all(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['ROE'] = None

        for idx in range(len(df)):
            net_income = df.at[idx, 'NetIncomeCommonStockholders']
            equity = df.at[idx, 'CommonStockEquity']

            if net_income is not None and equity and equity != 0:
                df.at[idx, 'ROE'] = net_income / equity

        return df
    

    def get_operating_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Operating Margin = OperatingIncome / TotalRevenue
        """
        required_columns = ['OperatingIncome', 'TotalRevenue']
        if not all(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['OperatingMargin'] = None

        for idx in range(len(df)):
            operating_income = df.at[idx, 'OperatingIncome']
            revenue = df.at[idx, 'TotalRevenue']

            if operating_income is not None and revenue and revenue != 0:
                df.at[idx, 'OperatingMargin'] = operating_income / revenue

        return df
    

    def get_free_cash_flow_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Free Cash Flow Margin = FreeCashFlow / TotalRevenue
        """
        required_columns = ['FreeCashFlow', 'TotalRevenue']
        if not all(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['FreeCashFlowMargin'] = None

        for idx in range(len(df)):
            fcf = df.at[idx, 'FreeCashFlow']
            revenue = df.at[idx, 'TotalRevenue']

            if fcf is not None and revenue and revenue != 0:
                df.at[idx, 'FreeCashFlowMargin'] = fcf / revenue

        return df
    

    def get_revenue_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revenue Growth = (Current TotalRevenue - Previous TotalRevenue) / Previous TotalRevenue
        Symbol과 Date 컬럼이 필요.
        """
        required_columns = ['Symbol', 'Date', 'TotalRevenue']
        if not all(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['RevenueGrowth'] = None

        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        for idx in range(1, len(df)):
            if df.at[idx, 'Symbol'] == df.at[idx - 1, 'Symbol']:
                current_rev = df.at[idx, 'TotalRevenue']
                prev_rev = df.at[idx - 1, 'TotalRevenue']

                if current_rev is not None and prev_rev and prev_rev != 0:
                    df.at[idx, 'RevenueGrowth'] = (current_rev - prev_rev) / prev_rev

        return df
    


    # 이자보상배율 : 기업이 영업이익으로 이자비용을 얼마나 잘 갚을 수 있는지 나타내는 지표
    def get_interest_coverage_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interest Coverage Ratio = OperatingIncome / InterestExpense
        """
        required_columns = ['OperatingIncome', 'InterestExpense']
        if not all(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        df['InterestCoverageRatio'] = None

        for idx in range(len(df)):
            operating_income = df.at[idx, 'OperatingIncome']
            interest_expense = df.at[idx, 'InterestExpense'] # 이자비용 : 기업이 빌린 돈(부채)에 대해 지급하는 이자비용. 

            if operating_income is not None and interest_expense and interest_expense != 0:
                df.at[idx, 'InterestCoverageRatio'] = operating_income / interest_expense

        return df


    # SPAC 판별 메서드
    def get_mark_spac(self, df: pd.DataFrame):
        # 필수 컬럼 확인
        required_cols = ['symbol', 'longBusinessSummary', 'industry', 'sector']
        for col in required_cols:
            if col not in df.columns:
                print(f"❗필수 컬럼 누락: {col}")
                return None

        df = df.copy()
        df['IsSPAC'] = False

        for idx in range(len(df)):
            try:
                description = str(df.at[idx, 'longBusinessSummary']).lower()
                industry = str(df.at[idx, 'industry']).lower()
                sector = str(df.at[idx, 'sector']).lower()
                officers = df.at[idx, 'companyOfficers']
                symbol = str(df.at[idx, 'symbol']).upper()

                # 1. 키워드 기반
                keywords = ["spac", "blank check", "special purpose acquisition"]
                keyword_hit = any(kw in description or kw in industry or kw in sector for kw in keywords)

                # 2. industry 기반
                industry_based = industry in ["shell companies", "capital markets", "asset management"]

                # 3. 임원 수 기반
                officer_based = isinstance(officers, list) and len(officers) <= 1

                # 4. 티커 네이밍 기반
                ticker_based = any(x in symbol for x in ["-U", "-WS", "-R"])

                # 최종 판별
                is_spac = keyword_hit or industry_based or officer_based or ticker_based
                is_spac = keyword_hit or industry_based or ticker_based
                df.at[idx, 'IsSPAC'] = is_spac

            except Exception as e:
                print(f"[{symbol}] 판별 실패: {e}")
                df.at[idx, 'IsSPAC'] = None

        return df

    def get_data(self, symbols:list, dateType:EDateType, min_year:int = 0):
        df = self.get_fs_all(symbols=symbols, dateType=dateType, min_year= min_year)
        df = self.get_sector(df)
        df = self.get_marketCap(df)
        df = self.get_psr(df)
        df = self.get_gp_a(df)
        df = self.get_por(df)
        df = self.get_ev_ebti(df)
        df = self.get_per(df)
        df = self.get_current_ratio(df)
        df = self.get_pbr(df)
        df = self.get_debt_to_equity_ratio(df)
        df = self.get_pcr(df)
        df = self.get_pfcr(df)
        df = self.get_gross_profit_margin(df)
        df = self.get_income_growth(df)
        df = self.get_roe(df) # 자기자본이익률
        df = self.get_operating_margin(df) # 영업이익률 : 매출대비 영업이익률
        df = self.get_free_cash_flow_margin(df) # 잉여현금흐름률 : 매출 대비 잉여현금흐름 비율
        df = self.get_revenue_growth(df) # 매출성장률
        df = self.get_interest_coverage_ratio(df) # 이자보상배율 : 영업이익이 이자비용을 몇 배나 감당할 수 있는지
        
        df = df[['Date', 'Symbol', 'Sector', 'MarketCap', 'TotalRevenue', 'NetIncome', 'OperatingIncome', 'GrossProfitMargin', 'IncomeGrowth', 'PSR', 'GP/A', 'EV/EBIT', 'PER','CurrentRatio','PBR', 'DebtToEquityRatio', 'PCR', 'PFCR', 'ROE','OperatingMargin', 'FreeCashFlowMargin','RevenueGrowth','InterestCoverageRatio']]
        df = df.dropna(subset=["MarketCap"])
        df = df.infer_objects(copy=False)

        return df
    

    #----------------
    # 스코어링 리스트 조회 (연간/분기)
    #----------------
    def save_file_fs_score_rank(self, symbols:list):

        # 2. 연간/분기 재무재표 로드
        df_year = self.get_data(symbols, EDateType.YEAR)
        df_year_stats = DB_FinancialStatement.calc_sector_statistics(df_year)
        
        df_quarter = self.get_data(symbols, EDateType.QUARTER)
        df_quarter_stats = DB_FinancialStatement.calc_sector_statistics(df_quarter)

        df_year_score = DB_FinancialStatement.calc_scores(df_year, df_year_stats)
        df_year_score = DB_FinancialStatement.aggregate_weighted_scores(df_year_score)
        
        df_quarter_score = DB_FinancialStatement.calc_scores(df_quarter, df_quarter_stats)
        df_quarter_score = DB_FinancialStatement.aggregate_weighted_scores(df_quarter_score, dateType= EDateType.QUARTER)

        df_year_score.to_csv("df_year_score_rank.csv", index=False)
        df_quarter_score.to_csv("df_quarter_score_rank.csv", index=False)

        display(df_year_score)
        display(df_quarter_score)

    



    #--------------
    # 스태틱 메서드
    #--------------

    @staticmethod
    def calc_sector_statistics(df: pd.DataFrame, value_cols=None, verbose=False):
        """
        연간 재무제표 데이터에서 섹터별 컬럼별 통계와 IQR 기반 상하한선 계산
        - df: 원본 DataFrame (Sector 컬럼 포함)
        - value_cols: 스코어링에 쓸 재무 컬럼 리스트 (없으면 숫자형 컬럼 자동 선택)
        - verbose: True면 컬럼 처리 상태를 출력
        리턴: Sector, ColumnName, Mean, Std, UpperBound, LowerBound DataFrame
        """
        results = []

        # 1) value_cols가 안 주어지면 숫자형 컬럼만 자동 선택
        if value_cols is None or len(value_cols) == 0:
            # 숫자형 컬럼만 포함
            value_cols = df.select_dtypes(include=['number']).columns.tolist()
            if verbose:
                print(f"[INFO] 숫자형 컬럼 자동 선택됨: {value_cols}")

        # 2) 섹터별 그룹화
        grouped = df.groupby('Sector')

        exclude_cols = [
                'MarketCap', 'TotalRevenue', 'NetIncome', 'OperatingIncome',
                'GrossProfitMargin', 'IncomeGrowth', 'Date', 'Symbol'
            ]

        for sector, group in grouped:
            for col in value_cols:
                if verbose:
                    print(f"[INFO] Sector: {sector}, Column: {col}")

                # NaN 제거
                series = group[col].dropna()

                if series.empty:
                    continue

                if col in exclude_cols:
                    continue

                mean = series.mean()
                std = series.std()

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                upper = q3 + 1.5 * iqr
                lower = q1 - 1.5 * iqr

                # PBR/PER 같은 지표라면 하한선 음수 방지
                lower = max(lower, 0)

                results.append({
                    'Sector': sector,
                    'ColumnName': col,
                    'Mean': mean,
                    'Std': std,
                    'UpperBound': upper,
                    'LowerBound': lower
                })

        result_df = pd.DataFrame(results)
        return result_df


    @staticmethod
    def calc_scores(df_financials: pd.DataFrame,
                                df_stats: pd.DataFrame,
                                verbose=False):
        """
        각 연도별로 섹터별 상대평가 스코어를 계산

        Parameters
        ----------
        df_financials : pd.DataFrame
            기업별 연간 재무제표 데이터. (Date, Symbol, Sector, ... 포함)
        df_stats : pd.DataFrame
            섹터별 ColumnName별 평균, 표준편차, UpperBound, LowerBound.
        scoring_columns : list
            점수를 계산할 컬럼 리스트.
        lower_better_columns : list
            낮을수록 좋은 컬럼명 리스트.
        verbose : bool
            처리 과정을 출력할지 여부.

        Returns
        -------
        pd.DataFrame
            연도별 기업별 컬럼별 스코어 포함된 DataFrame.
        """


        # 점수화할 컬럼 리스트 예시
        scoring_columns = [
            'PSR', 'GP/A', 'EV/EBIT', 'PER',
            'CurrentRatio', 'PBR', 'DebtToEquityRatio',
            'PCR', 'PFCR', 'ROE', 'OperatingMargin',
            'FreeCashFlowMargin', 'RevenueGrowth', 'InterestCoverageRatio'
        ]

        # 낮을수록 좋은 컬럼
        lower_better_columns = [
            'PER', 'PBR', 'PSR', 'EV/EBIT', 'PCR', 'PFCR', 'DebtToEquityRatio'
        ]
        
        # 이 2개의 데이터만 결측값일 경우 0으로 처리하면 좋음.
        # InterestCoverageRatio : 영업이익 / 이자비용 -> 이자보상배율
        # 해석 : 이 값이 Nan이면 대부분의 이자비용이 0또는 매우 낮다는 의미이거나, 적자기업이라 음수도 나오는 케이스가 많음.
        # 실무에서는 기업이 이자비용을 감당할 능력이 없다고 보고 보수적으로 0처리
        # FreeCashFlowMargin : 매출 대비 잉여현금 흐름.
        missing_fill_zero_columns = ['InterestCoverageRatio', 'FreeCashFlowMargin']
        
        results = []
        if lower_better_columns is None:
            lower_better_columns = []

        for idx, row in df_financials.iterrows():
            company_data = {
                'Date': row['Date'],
                'Symbol': row['Symbol'],
                'Sector': row['Sector']
            }

            sector = row['Sector']

            for col in scoring_columns:
                # 섹터별 컬럼 평균/표준편차/상하한 가져오기
                stat_row = df_stats[
                    (df_stats['Sector'] == sector) & 
                    (df_stats['ColumnName'] == col)
                ]
                if stat_row.empty:
                    if verbose:
                        print(f"[WARN] No stats for Sector={sector}, Column={col}")
                    continue

                mean = stat_row['Mean'].values[0]
                std = stat_row['Std'].values[0]
                upper = stat_row['UpperBound'].values[0]
                lower = stat_row['LowerBound'].values[0]

                value = row[col]

                # ✔️ NaN 처리: 특정 컬럼은 0으로, 그 외는 섹터 평균으로 대체
                if pd.isna(value):
                    if col in missing_fill_zero_columns:
                        value_filled = 0
                        if verbose:
                            print(f"[INFO] NaN for {col}, fill with ZERO")
                    else:
                        value_filled = mean
                        if verbose:
                            print(f"[INFO] NaN for {col}, fill with MEAN={mean:.4f}")
                else:
                    value_filled = value

                # Winsorizing
                value_clipped = np.clip(value_filled, lower, upper)

                # Z-Score
                z = (value_clipped - mean) / std if std != 0 else 0

                # 낮을수록 좋은 지표는 부호 반전
                if col in lower_better_columns:
                    z = -z

                # Z → CDF → 0~20 점수
                cdf = norm.cdf(z)
                score = cdf * 100

                company_data[f'Score_{col}'] = score

                if verbose:
                    print(f"[OK] {row['Symbol']} {col}: Raw={value}, Filled={value_filled}, Clipped={value_clipped}, Z={z:.2f}, Score={score:.2f}")

            results.append(company_data)

        df_result = pd.DataFrame(results)
        return df_result


                
    @staticmethod
    def aggregate_weighted_scores(df_yearly_scores: pd.DataFrame,
                                recent_weight: float = 0.7,
                                past_weight: float = 0.3,
                                sector_weight_dict: dict = None,
                                verbose: bool = False,
                                dateType : EDateType = EDateType.YEAR):
        """
        연도별 스코어 → 기업별 가중평균 스코어 → 섹터별 중요도 가중치로 Final_Score_Total 포함

        Parameters
        ----------
        df_yearly_scores : pd.DataFrame
            연도별 스코어 데이터 (calc_yearly_scores 결과)
        recent_weight : float
            최근 연도 가중치
        past_weight : float
            과거 연도 평균 가중치
        sector_weight_dict : dict
            섹터별 중요도 Dict. ex)
            {
            "Technology": {
                "PSR": 0.2, "GP/A": 0.1, ...
            },
            "Finance": {
                "PER": 0.3, "PBR": 0.2, ...
            }
            }
        verbose : bool
            True면 진행상황 출력

        Returns
        -------
        pd.DataFrame
            Symbol, Sector, Final_컬럼, Final_Score_Total 포함
        """

        sector_weight_dict = ch.get_sector_weights_dict()

        df = df_yearly_scores.copy()
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        score_cols = [col for col in df.columns if col.startswith('Score_')]

        results = []

        for (symbol, sector), group in df.groupby(['Symbol', 'Sector']):
            recent_year = group['Year'].max()
            recent_data = group[group['Year'] == recent_year]
            past_data = group[group['Year'] < recent_year]

            dateStr = "Year" if dateType == EDateType.YEAR else "Quarter"

            result_row = {'Symbol': symbol, 'Sector': sector, 'DateType' : dateStr}

            total_score = 0.0
            total_weight = 0.0

            for col in score_cols:
                base_col = col.replace('Score_', '')  # ex) PSR

                # 연도별 가중평균 스코어
                recent_score = recent_data[col].values[0] if not recent_data.empty else None
                past_mean = past_data[col].mean() if not past_data.empty else None

                if pd.isna(recent_score) and pd.isna(past_mean):
                    final_score = None
                else:
                    r = recent_score if not pd.isna(recent_score) else 0
                    p = past_mean if not pd.isna(past_mean) else 0
                    final_score = (recent_weight * r) + (past_weight * p)

                result_row[f'Final_Score_{base_col}'] = final_score

                # ✔️ 섹터별 가중치 적용
                if sector_weight_dict and sector in sector_weight_dict:
                    weight = sector_weight_dict[sector].get(base_col, 0)
                else:
                    weight = 1.0  # fallback: 1로 간주

                if final_score is not None:
                    total_score += final_score * weight
                    total_weight += weight

                if verbose:
                    print(f"[OK] {symbol} {base_col}: Recent={recent_score}, PastMean={past_mean} => "
                        f"Final={final_score:.2f if final_score else None}, Weight={weight}")

            # ✔️ 가중치 총합이 0이면 안전 처리
            result_row['Final_Score_Total'] = total_score / total_weight if total_weight != 0 else None

            results.append(result_row)

        df_result = pd.DataFrame(results)
        df_result_sorted = df_result.sort_values(by='Final_Score_Total', ascending=False).reset_index(drop=True)
        return df_result_sorted


    @staticmethod
    def combine_scores(
        df_year_score: pd.DataFrame,
        df_quarter_score: pd.DataFrame,
        annual_weight: float = 0.3,
        quarterly_weight: float = 0.7,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        연간/분기 스코어를 Symbol, Sector 기준으로 병합한 뒤
        기간별 가중평균을 적용해 TopN 결과를 반환합니다.

        Parameters:
        ----------
        df_year_score : pd.DataFrame
            ['Symbol', 'Sector', 'Final_Score_Total'] 포함해야 함
        df_quarter_score : pd.DataFrame
            ['Symbol', 'Sector', 'Final_Score_Total'] 포함해야 함
        annual_weight : float
            연간 데이터 가중치 (기본: 0.3)
        quarterly_weight : float
            분기 데이터 가중치 (기본: 0.7)
        top_n : int
            반환할 상위 기업 수 (기본: 20)

        Returns:
        -------
        pd.DataFrame
            Symbol, Sector, Final_Score_Combined 컬럼으로 정렬된 TopN DataFrame
        """
        # 1) Symbol/Sector 기준 merge
        df_merged = pd.merge(
            df_year_score,
            df_quarter_score,
            on=['Symbol', 'Sector'],
            suffixes=('_Annual', '_Quarterly')
        )

        # 2) Final_Score_Total 기준으로 기간별 가중평균
        df_merged['Final_Score_Combined'] = (
            df_merged['Final_Score_Total_Annual'] * annual_weight +
            df_merged['Final_Score_Total_Quarterly'] * quarterly_weight
        )

        # 3) TopN 정렬
        df_result = df_merged[['Symbol', 'Sector', 'Final_Score_Combined']].copy()
        df_result = df_result.sort_values('Final_Score_Combined', ascending=False).reset_index(drop=True)
        df_result = df_result.head(top_n)

        return df_result
    
    def add_quarter_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Date 컬럼을 기준으로 몇 분기인지 계산하여 'Quarter' 컬럼을 추가하고,
        이 컬럼을 'Date' 컬럼 바로 뒤에 위치시켜 반환합니다.
        """
        # Date 컬럼이 datetime 타입이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # month를 기준으로 Quarter 구하기
        df['Quarter'] = df['Date'].dt.month.map({
            1: 'Q1', 2: 'Q1', 3: 'Q1',
            4: 'Q2', 5: 'Q2', 6: 'Q2',
            7: 'Q3', 8: 'Q3', 9: 'Q3',
            10: 'Q4', 11: 'Q4', 12: 'Q4'
        })

        # 'YYYY-Q#' 형태로 표시하고 싶으면 아래와 같이 수정
        df['Quarter'] = df['Date'].dt.year.astype(str) + '-' + df['Quarter']

        # 컬럼 순서 변경: Date 뒤에 Quarter 오도록 재정렬
        cols = df.columns.tolist()
        date_idx = cols.index('Date')
        # 기존 위치에서 Quarter 빼기
        cols.remove('Quarter')
        # Date 뒤에 Quarter 삽입
        cols.insert(date_idx + 1, 'Quarter')
        # 컬럼 순서 적용
        df = df[cols]

        return df


    def create_quarter_groups(df: pd.DataFrame, window_size: int = 4) -> list:
        """
        Quarter 컬럼을 활용해 window_size 크기만큼 rolling 그룹핑하여,
        마지막 row의 Quarter가 Q4이면 그 그룹은 제외하고,
        각 그룹은 Date, Symbol 오름차순 정렬 후 리스트로 반환합니다.

        Parameters:
            df (pd.DataFrame): 입력 DataFrame (Quarter 컬럼 필수)
            window_size (int): 그룹핑할 분기 수 (기본값=4)

        Returns:
            list: 그룹별 DataFrame 리스트
        """
        # 정렬
        df_sorted = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Quarter 고유 값 순서 유지
        unique_quarters = df_sorted['Quarter'].drop_duplicates().tolist()

        result_groups = []

        for i in range(len(unique_quarters) - (window_size - 1)):
            group_quarters = unique_quarters[i:i+window_size]

            # 마지막 row의 Quarter가 Q4이면 제외
            if group_quarters[-1].endswith('Q4'):
                continue

            # 그룹 필터링
            group_df = df_sorted[df_sorted['Quarter'].isin(group_quarters)].copy()

            # 다시 Symbol, Date 순으로 정렬 보장
            group_df = group_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

            result_groups.append(group_df)

        return result_groups


    # year 년을 기준으로 n_years 전년까지의 데이터 필터링
    def filter_annual_data(df: pd.DataFrame, year: int, n_years: int) -> pd.DataFrame:
        """
        기준 year를 포함해 N년 전까지 연간 데이터만 필터링하여 반환합니다.
        
        Parameters:
            df (pd.DataFrame): 입력 데이터프레임
            year (int): 기준 연도 (포함)
            n_years (int): 포함할 연간 데이터 범위 (기준 연도 포함)

        Returns:
            pd.DataFrame: 필터링된 데이터프레임 (Date, Symbol 오름차순 정렬)
        """
        # Date 컬럼이 datetime 타입이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # 연도 범위 계산
        start_year = year - n_years # (n_years - 1)
        end_year = year

        # Date 컬럼에서 연도 추출
        df['Year'] = df['Date'].dt.year

        # 연도 필터링
        filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

        # 필요하다면 Year 컬럼 drop
        filtered_df.drop(columns=['Year'], inplace=True)

        # 정렬
        filtered_df = filtered_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        return filtered_df
    

    #----------------
    # 날짜 범위 반환
    #----------------
    def generate_quarter_range(start_quarter: str, end_quarter: str) -> list:
        def quarter_to_date(q):
            year, qtr = q.split('-Q')
            month = {'1': '01', '2': '04', '3': '07', '4': '10'}[qtr]
            return parse(f"{year}-{month}-01")

        def date_to_quarter(date_obj):
            month = date_obj.month
            quarter = (month - 1) // 3 + 1
            return f"{date_obj.year}-Q{quarter}"

        start_date = quarter_to_date(start_quarter)
        end_date = quarter_to_date(end_quarter)

        quarters = []
        current = start_date
        while current <= end_date:
            quarters.append(date_to_quarter(current))
            current += relativedelta(months=3)

        return quarters

    def get_symbols_with_quarter_range(df: pd.DataFrame, start_quarter: str, end_quarter: str) -> list:
        required_quarters = set(DB_FinancialStatement.generate_quarter_range(start_quarter, end_quarter))
        print(f"🔍 검사할 분기 범위: {sorted(required_quarters)}")

        valid_symbols = []

        for symbol, group in df.groupby('Symbol'):
            available_quarters = set(group['Quarter'].unique())
            if required_quarters.issubset(available_quarters):
                valid_symbols.append(symbol)
            else:
                missing = sorted(required_quarters - available_quarters)
                # print(f"❌ Symbol '{symbol}' 누락 분기: {missing}")

        if not valid_symbols:
            print("⚠️ 지정된 범위를 모두 만족하는 Symbol이 없습니다.")
        else:
            print(f"✅ 범위 내 모든 분기를 가진 Symbol 수: {len(valid_symbols)}")

        return valid_symbols
    

    #----------------------------
    # 2024-Qn 이런식으로 Quarter 컬럼 구현
    #----------------------------
    def add_quarter_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Date 컬럼을 기준으로 몇 분기인지 계산하여 'Quarter' 컬럼을 추가하고,
        이 컬럼을 'Date' 컬럼 바로 뒤에 위치시켜 반환합니다.
        """
        # Date 컬럼이 datetime 타입이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # month를 기준으로 Quarter 구하기
        df['Quarter'] = df['Date'].dt.month.map({
            1: 'Q1', 2: 'Q1', 3: 'Q1',
            4: 'Q2', 5: 'Q2', 6: 'Q2',
            7: 'Q3', 8: 'Q3', 9: 'Q3',
            10: 'Q4', 11: 'Q4', 12: 'Q4'
        })

        # 'YYYY-Q#' 형태로 표시하고 싶으면 아래와 같이 수정
        df['Quarter'] = df['Date'].dt.year.astype(str) + '-' + df['Quarter']

        # 컬럼 순서 변경: Date 뒤에 Quarter 오도록 재정렬
        cols = df.columns.tolist()
        date_idx = cols.index('Date')
        # 기존 위치에서 Quarter 빼기
        cols.remove('Quarter')
        # Date 뒤에 Quarter 삽입
        cols.insert(date_idx + 1, 'Quarter')
        # 컬럼 순서 적용
        df = df[cols]

        return df
    

    def filter_common_quarters(df: pd.DataFrame, symbols: list) -> pd.DataFrame:
        if not symbols:
            print("⚠️ 유효한 Symbol 리스트가 비어 있습니다. 빈 DataFrame 반환.")
            return df.iloc[0:0].copy()

        df_valid = df[df['Symbol'].isin(symbols)].copy()

        # Symbol별 보유 분기 Set
        symbol_quarters = df_valid.groupby('Symbol')['Quarter'].apply(set)

        # 교집합 도출
        common_quarters = set.intersection(*symbol_quarters)
        common_quarters_sorted = sorted(common_quarters)
        print(f"✅ 교집합 Quarters: {common_quarters_sorted}")

        # 각 Symbol이 보유한 전체 분기 수
        total_quarter_counts = symbol_quarters.apply(len)
        min_count = total_quarter_counts.min()
        narrowing_symbols = total_quarter_counts[total_quarter_counts == min_count].index.tolist()

        print(f"⚠️ 교집합이 줄어든 원인 Symbol(가장 적은 분기 보유): {narrowing_symbols}")
        print(f"📊 이들 Symbol의 보유 분기 수: {min_count}")

        # 최종 필터링
        filtered_df = df_valid[df_valid['Quarter'].isin(common_quarters)].copy()
        filtered_df = filtered_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        return filtered_df
    


    def show_rank_by_fs(start_quarter ='2024-Q2',  end_quarter ='2025-Q1'):
        symbol_columns = []
        column_names = []
        
        with DB_FinancialStatement() as fs:
            quarters = DB_FinancialStatement.generate_quarter_range(start_quarter, end_quarter) # 범위 리스트 구하기
            min_year = int(quarters[3].split('-Q')[0]) - 3 # 필터 된 날짜.

            symbols = fs.getSymbolListByFilter(min_year) # 필터링된 날짜까지만 추출
            print(f"{min_year} 이전 상장 티커 수 : {len(symbols)}")

            df_year = fs.get_data(symbols, EDateType.YEAR, min_year)
            df_quarter = fs.get_data(symbols, EDateType.QUARTER)
            df_quarter = DB_FinancialStatement.add_quarter_column(df_quarter)
            
            valid_symbols = DB_FinancialStatement.get_symbols_with_quarter_range(df_quarter, start_quarter, end_quarter)
            df_quarter = DB_FinancialStatement.filter_common_quarters(df_quarter, valid_symbols)   
            df_quarter_list = DB_FinancialStatement.create_quarter_groups(df_quarter) # 4분기씩 리스트로 구함

            for df_quarter in df_quarter_list:
                year = df_quarter.iloc[-1]['Date'].year
                df_year = DB_FinancialStatement.filter_annual_data(df_year, year-1, 3)

                column_name = df_quarter.iloc[-1]['Quarter']

                df_year_stat = DB_FinancialStatement.calc_sector_statistics(df_year)
                df_year_score = DB_FinancialStatement.calc_scores(df_year, df_year_stat)
                df_year_score_total = DB_FinancialStatement.aggregate_weighted_scores(df_year_score)

                df_quarter_stat = DB_FinancialStatement.calc_sector_statistics(df_quarter)
                df_quarter_score = DB_FinancialStatement.calc_scores(df_quarter, df_quarter_stat)
                df_quarter_score_total = DB_FinancialStatement.aggregate_weighted_scores(df_quarter_score, dateType= EDateType.QUARTER)

                df_result = DB_FinancialStatement.combine_scores(df_year_score_total, df_quarter_score_total)
                
                df_result.to_csv(f'{column_name}_rank.csv', index = True)

                # Symbol만 Series로 추출하고 인덱스 초기화
                symbol_col = df_result['Symbol'].reset_index(drop=True)
                symbol_columns.append(symbol_col)
                column_names.append(column_name)

        if len(symbol_columns) > 0:
            # 루프 끝난 뒤: 컬럼 방향으로 병합
            final_df = pd.concat(symbol_columns, axis=1)
            # 컬럼명 지정
            final_df.columns = column_names
            final_df.to_csv(f'rank_top_qurter_{datetime.now()}.csv', index= True)
            display(final_df)