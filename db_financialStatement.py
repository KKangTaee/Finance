from functools import reduce
from heapq import merge
from symtable import Symbol
from assetAllocation import AssetAllocation
import commonHelper
from db_stock import DB_Stock
from mysqlConnecter import MySQLConnector
from commonHelper import EFinancialStatementType, EDateType, EIndustry
from IPython.display import display
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from dateutil.parser import parse
from typing import Callable

import pymysql
import pandas as pd
import commonHelper as ch
import numpy as np
import warnings

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
    def get_symbol_list(self):
        symbols = []
        quary = """
            SELECT symbol FROM Company;
        """

        df = super().requestToDB(quary,['symbol'])
        symbols = [row['symbol'] for _, row in df.iterrows()] # iterrows 쓰면, 인덱스랑 데이터 분리되서 나옴
        return symbols
    

    def get_symbol_list_with_filter(self, min_year:int = 0):
        symbols = []
        base_query = f"""
            SELECT symbol
            FROM Company
            WHERE isSpec != 1
            AND (country IS NULL OR country != 'china')
        """

        if min_year > 0:
            years_ago = datetime(min_year, 1, 1)
            years_ago_ms = int(years_ago.timestamp()*1000)
            query = base_query + f"\n          AND firstTradeDateMilliseconds < {int(years_ago_ms)}"
        else:
            query = base_query

        query += ";"
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
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            "DilutedEPS",          # 희석 주당순이익
            "DilutedAverageShares" # 희석 가능성 포함한 평균 주식수
        ]

        # Balance Sheet 항목
        balance_sheet_cols = [
            "TotalAssets",
            "TotalLiabilitiesNetMinorityInterest",
            "CommonStockEquity",
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

        df = self.get_stock_price_close(merged) # 각 재무재표 발표날짜의 종가 데이터 반환
        df = self.get_sector(df)
        df = self.get_market_cap(df)

        return df
    

    def get_stock_price_close(self, fs_df:pd.DataFrame):
        fs_df['Date'] = pd.to_datetime(fs_df['Date'])
        fs_df = fs_df.sort_values(['Symbol','Date'])
        
        symbols = list(set(fs_df['Symbol']))

        price_df_list = []
        with DB_Stock() as stock:
            for symbol in symbols:
                # 1. 회계 데이터에서 해당 symbol의 날짜 가져오기
                filter_date = fs_df[fs_df['Symbol'] == symbol]['Date'].sort_values().reset_index(drop=True)
                start_date = filter_date.iloc[0].replace(day=1)
                end_date = filter_date.iloc[-1]

                # 2. 주가 데이터 가져오기
                price_df = stock.getStockData(symbol, start_date, end_date, EDateType.MONTHLY)
                price_df['Date'] = pd.to_datetime(price_df['Date'])

                # 3. 연-월 기준 필터링
                filter_date_month = filter_date.dt.to_period('M').astype(str)
                price_date_month = price_df['Date'].dt.to_period('M').astype(str)
                price_df = price_df[price_date_month.isin(filter_date_month)].sort_values(['Symbol','Date']).reset_index(drop=True)

                try:
                    # 4. 일(day)을 filter_date에서 가져와서 수정
                    #    (연/월 순서가 같다고 가정)
                    price_df['Date'] = price_df['Date'].combine(
                        filter_date, 
                        lambda base, match: pd.Timestamp(base).replace(day=match.day))
                except Exception as e:
                    print(f"{symbol}")


                # 5. 결과 저장
                price_df_list.append(price_df)

        price_df_list = [df for df in price_df_list if not df.empty and not df.isna().all().all()]
        price_df = pd.concat(price_df_list, ignore_index=True)
        price_df_subset = price_df[['Date','Symbol','Close']]
        merged = pd.merge(fs_df, price_df_subset, on=['Symbol','Date'], how='left')
        return merged

    

    def get_market_cap(self, df:pd.DataFrame)-> pd.DataFrame:
        df = df.copy()
        df['MarketCap'] = np.nan # none 보다 nan으로 처리해야지 경고문이 안뜸.
        df['MarketCap'] = df['Close'] * df['DilutedAverageShares']
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
    def get_ev_ebit(self, df: pd.DataFrame):
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
    # 근데, 구할때 단순히 해당 분기의 시총값 / 순이익으로 처리하면 안됨.
    # TTM이라고해서 12월 분을 합산해서 처리해야함.
    # 이게 무슨말이냐면. 종가 / (4분기 EPS를 합한 값) 으로 처리해야함.
    # 지금은 현재 분기 기점으로 이전분기가 부족하기 때문에 둬야 한다.
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
            close = df.at[idx, 'MarketCap']
            netIncome = df.at[idx, 'NetIncome']
            
            if netIncome != 0:
                df.at[idx, 'PER'] = close / netIncome
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
            else:
                df.at[idx, 'CurrentRatio'] = 0.0

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
    

    # 차입금비율(부채비율)
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
            else:
                df.at[idx, 'DebtToEquityRatio'] = 0.0

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
            capital_expenditure = df.at[idx, 'CapitalExpenditure'] # 자본적 지출.

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
    

    # 영업이익 성장률
    def get_operating_income_growth(self, df:pd.DataFrame) -> pd.DataFrame:
        
        required_columns = ['Symbol', 'Date', 'OperatingIncome']
        if not all(col in df.columns for col in required_columns):
            return None
              
        df = df.copy()
        df = df.sort_values(by=['Symbol','Date'], ascending=True)
        df['OperatingIncomeGrowth'] = None

        for idx in range(1, len(df)):
            if df.at[idx, 'Symbol'] == df.at[idx - 1, 'Symbol']:
                current_rev = df.at[idx, 'OperatingIncome']
                prev_rev = df.at[idx - 1, 'OperatingIncome']

                if current_rev is not None and prev_rev and prev_rev != 0:
                    df.at[idx, 'OperatingIncomeGrowth'] = (current_rev - prev_rev) / prev_rev

        return df
    
    def get_net_income_growth(self, df:pd.DataFrame) ->pd.DataFrame:

        required_columns = ['Symbol', 'Date', 'NetIncome']
        if not all(col in df.columns for col in required_columns):
            return None
        
        df = df.copy()
        # df['NetIcomeGrowth'] = None

        # df = df.sort_values(by=['Symbol','Date']).reset_index(drop=True)

        # for idx in range(1, len(df)):
        #     if df.at[idx, 'Symbol'] == df.at[idx - 1, 'Symbol']:
        #         current_rev = df.at[idx, 'NetIncome']
        #         prev_rev = df.at[idx - 1, 'NetIncome']

        #         if current_rev is not None and prev_rev and prev_rev != 0:
        #             df.at[idx, 'NetIcomeGrowth'] = (current_rev - prev_rev) / prev_rev

        df['NetIncomeGrowth'] = df.groupby('Symbol')['NetIncome'].pct_change()
        return df
        
        
    
    def get_gross_profit_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매출총이익률 (Gross Profit Margin) = GrossProfit / Revenue
        Revenue 컬럼은 TotalRevenue 또는 OperatingRevenue 중 하나가 필요함.
        """
        
        # 필요한 컬럼 확인
        required_columns = ['GrossProfit', 'TotalRevenue']
        if not any(col in df.columns for col in required_columns):
            return None

        df = df.copy()
        # df['Revenue'] = None
        df['GrossProfitMargin'] = None

        # Revenue 설정 우선순위: TotalRevenue > OperatingRevenue
        for idx in range(len(df)):
            # if 'TotalRevenue' in df.columns and pd.notnull(df.at[idx, 'TotalRevenue']):
            #     df.at[idx, 'Revenue'] = df.at[idx, 'TotalRevenue']
            # elif 'OperatingRevenue' in df.columns and pd.notnull(df.at[idx, 'OperatingRevenue']):
            #     df.at[idx, 'Revenue'] = df.at[idx, 'OperatingRevenue']
            # else:
            #     continue  # 둘 다 없으면 넘어감

            gross_profit = df.at[idx, 'GrossProfit']
            revenue = df.at[idx, 'TotalRevenue']

            df.at[idx, 'GrossProfitMargin'] = (
                gross_profit / revenue
                if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0
                else 0.0
            )
                

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
    

    def get_roa(self, df:pd.DataFrame) -> pd.DataFrame:
        required_columns = ['NetIncome', 'TotalAssets']
        if not all(col in df.columns for col in required_columns):
            return None
        
        df = df.copy()
        df['ROA'] = None
        for idx in range(len(df)):
            net_income = df.at[idx, 'NetIncome']
            total_assets = df.at[idx, 'TotalAssets']

            if net_income is not None and total_assets and total_assets != 0:
                df.at[idx, 'ROA'] = net_income / total_assets

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
    

    def get_asset_turnover_ratio(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        총자산회전률
        정의 : 기업이 보유한 자산을 얼마나 효율적으로 활용하여 매출을 창출했는지 보여주는 지표
        값이 높다? : 자산을 효율적으로 사용해 매출을 잘 만들고 있다는 뜻
        값이 낮다? : 자산이 비효율적으로 묶여있거나 매출 창출 능력이 떨어진다는 뜻
        보통 유통업이 일반적으로 높고 (재고 회전 빠름), 제조업이 낮다.(실비투자 많음)
        """

        required_columns = ['TotalAssets', 'TotalRevenue']
        if not all(col in df.columns for col in required_columns):
            return None
        

        df = df.copy()
        df['AssetTurnoverRatio'] = None
        for idx in range(len(df)):
            total_revenue = df.at[idx, 'TotalRevenue']
            total_assets = df.at[idx, 'TotalAssets']

            if total_revenue is not None and total_assets and total_assets != 0:
                df.at[idx, 'AssetTurnoverRatio'] =  total_revenue / total_assets

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
    
    # 자산 성장률
    def get_total_assets_growth(self, df:pd.DataFrame) -> pd.DataFrame:
        ##
        required_columns = ['Symbol', 'Date', 'TotalAssets']
        if not all(col in df.columns for col in required_columns):
            return None
        
        df = df.copy()
        df['TotalAssetsGrowth'] = None

        df = df.sort_values(by=['Symbol','Date']).reset_index(drop=True)

        for idx in range(1, len(df)):
            if df.at[idx, 'Symbol'] == df.at[idx - 1, 'Symbol']:
                current_rev = df.at[idx, 'TotalAssets']
                prev_rev = df.at[idx - 1, 'TotalAssets']

                if current_rev is not None and prev_rev and prev_rev != 0:
                    df.at[idx, 'TotalAssetsGrowth'] = (current_rev - prev_rev) / prev_rev

        return df

    # 총차입금 성장률
    # 차입금 : 기업이 실제로 이자 비용을 부담하는 부채
    # 여기서 총부채라 함음 "단기차입금 + 장기차입금 + 리스부채" 인 값을 말함.
    # 즉, 부채중에서도 돈을 빌려온 성격만 집계하는 것
    # 총차입금음 총부채가 아님!!
    # 총부채가 훨씬 포괄적인 말임. (총 부채 안에 총차입금이 있음.)
    # 차입금 비율이 낮을수록 파산가능성이 낮음.
    # 총차입금이 1000억이라는건 단기/장기 차입금이 합처서 1000억이라는거
    # 차입금은 절대 음수로 나올 수 없음.
    def get_total_debt_growth(self, df:pd.DataFrame) -> pd.DataFrame:
        required_columns = ['Symbol', 'Date', 'TotalDebt']
        if not all(col in df.columns for col in required_columns):
            return None
        
        df = df.copy()
        df['TotalDebtGrowth'] = None

        df = df.sort_values(by=['Symbol','Date']).reset_index(drop=True)
        for idx in range(1, len(df)):
            if df.at[idx, 'Symbol'] == df.at[idx - 1, 'Symbol']:
                current_rev = df.at[idx, 'TotalDebt']
                prev_rev = df.at[idx - 1, 'TotalDebt']

                if current_rev is not None and prev_rev and prev_rev != 0:
                    df.at[idx, 'TotalDebtGrowth'] = (current_rev - prev_rev) / prev_rev
        
        return df
    
    # 주식성장률 구하기 (N개월)
    def get_volatility_m(self, df:pd.DataFrame, month:int = 12) -> pd.DataFrame:
        symbol_to_date = df.groupby('Symbol')['Date'].apply(list).to_dict()
        all_dates = sum(symbol_to_date.values(), [])
        all_dates = pd.to_datetime(all_dates)
        oldest = all_dates.min()
        latest = all_dates.max()
        oldest = oldest-relativedelta(years=1)
        df_stock = AssetAllocation.get_stock_data_with_ma(symbol_to_date.keys(), oldest, latest, [10], 'ma_month', True)
        df_stock = AssetAllocation.filter_close_last_month(df_stock)
        
        column_name = f"Volatility{month}M"
        for sym, stock in df_stock.items(): 
            stock["MonthlyReturn"] = stock["Close"].pct_change(fill_method=None)
            stock[column_name] = (
                stock['MonthlyReturn'].rolling(window=12).std(ddof=1)*np.sqrt(12)
            )
            stock = stock.dropna()
            stock = stock[['Date','Symbol','MonthlyReturn', column_name]]
            df_stock[sym] = stock

        df['MonthlyReturn'] = None
        df[column_name] = None

        for i in range(len(df)):
            date = df.at[i,'Date']
            sym = df.at[i, 'Symbol']
            stock = df_stock[sym]
            mask = (stock['Date'].dt.year == date.year) & (stock['Date'].dt.month == date.month)
            row = stock.loc[mask]

            if not row.empty:
                df.at[i, 'MonthlyReturn'] = row['MonthlyReturn'].values[0]
                df.at[i, column_name] = row[column_name].values[0]
            else:
                print(f"{sym}  fs date : {date}")
                df.at[i, 'MonthlyReturn'] = np.nan
                df.at[i, column_name] = np.nan



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

    

    #------------------------
    # 벨류 펙터 데이터 반환 (PER, PBR, PCR, PSR, PFCR, LiquidationValue, EV/EVIT)
    #------------------------
    def get_value_data(self, df):
        df = self.get_per(df)
        df = self.get_pbr(df)
        df = self.get_pcr(df)
        df = self.get_psr(df)
        df = self.get_pfcr(df)
        df = self.get_liquidation_value(df) # 청산가치
        df = self.get_ev_ebit(df)
        df = df[['Date', 'Symbol', 'Sector', 'MarketCap', 'Close', 'NetIncome', 'OperatingIncome', 'PER','PBR','PCR','PSR','PFCR', 'LiquidationValue', 'EV/EBIT']]
        return df


    #-------------------------
    # 퀄리티 데이터
    #-------------------------
    def get_quality_date(self, df):
        df = self.get_gp_a(df)
        df = self.get_roa(df)
        df = self.get_current_ratio(df)
        df = self.get_debt_to_equity_ratio(df) # 차입금비율
        df = self.get_asset_turnover_ratio(df) # 총자산회전률
        df = self.get_gross_profit_margin(df)  # 매출총이익률
        df = self.get_total_assets_growth(df)  # 자산성장률
        # CommonStockIssuance : 보통주 - 이 값이 양수일 경우 신규보통주 발행
        # NetCommonStockIssuance : 순 보통주 발행 - 발행으로 유입된 금액에서 자사주 매입등으로 유출된 금액을 뺀 순액을 말하는거.
        df = df[['Date', 'Symbol', 'Sector', 'MarketCap', 'Close', 'NetIncome', 'OperatingCashFlow', 'CommonStockIssuance', 'NetCommonStockIssuance', 'GP/A', 'ROA', 'CurrentRatio', 'DebtToEquityRatio', 'AssetTurnoverRatio', 'GrossProfitMargin', 'TotalAssetsGrowth']]
        return df
    

    #-------------------------
    # 모멘텀 데이터
    #-------------------------
    def get_momentun_data(self, df):
        df = self.get_operating_income_growth(df)
        df = self.get_net_income_growth(df)
        df = self.get_total_debt_growth(df)
        df = df[['Date', 'Symbol', 'Sector', 'MarketCap', 'Close', 'NetIncome','OperatingIncomeGrowth','NetIncomeGrowth', 'TotalDebtGrowth']]
        return df


    def get_fs_data(self, df):
        df_value = self.get_value_data(df)
        df_quality =self.get_quality_date(df)
        df_momentum = self.get_momentun_data(df)

        df_merged = pd.merge(df_value, df_quality, on=['Date','Symbol','Sector','MarketCap','Close','NetIncome'])
        df_merged = pd.merge(df_merged, df_momentum, on=['Date','Symbol','Sector','MarketCap','Close','NetIncome'])
        return df_merged
    

    

    # 재무재표 흐름
    # Total Revenue (매출액)
    #     └─ (-) COGS (매출원가)
    #         └─> Gross Profit (매출총이익)
    #             └─ (-) Operating Expenses (영업비용(ex. 인건비))
    #                 └─> Operating Income (영업이익)
    #                     └─ (+/-) 기타 수익 및 비용
    #                         └─> Pre-Tax Income (새전이익)
    #                             └─ (-) Taxes 
    #                                 └─> Net Income (순이익)
    
    # PER을 구할때는 DilutedEPS(희석주당순이익)로 구함 (BasicEPS 가 아니라)
    # EPS는 순이익/보통주 한것 -> 즉, 1개의 주식이 얼마의 이익을 창출했냐를 보는 것
    # 근데, 희석주당순이익은 여러가지 옵션들이 들어가서 구해진 주당순이익


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
    

    def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Date 컬럼을 기준으로 'Year' 컬럼을 추가하고,
        이 컬럼을 'Date' 컬럼 바로 뒤에 위치시켜 반환합니다.
        """
        # Date 컬럼이 datetime 타입이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Year 컬럼 추가
        df['Year'] = df['Date'].dt.year

        # 컬럼 순서 변경: Date 뒤에 Year 오도록 재정렬
        cols = df.columns.tolist()
        date_idx = cols.index('Date')
        cols.remove('Year')
        cols.insert(date_idx + 1, 'Year')
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
        missing_rows = []

        for symbol, group in df.groupby('Symbol'):
            available_quarters = set(group['Quarter'].unique())
            if required_quarters.issubset(available_quarters):
                valid_symbols.append(symbol)
            else:
                missing = sorted(required_quarters - available_quarters)
                missing_rows.append({'Symbol': symbol, 'Missing': missing})
                # print(f"❌ Symbol '{symbol}' 누락 분기: {missing}")

        # 루프가 끝난 뒤 한 번에 DataFrame 생성
        if len(missing_rows) > 0:
            missing_df = pd.DataFrame(missing_rows)
            missing_df.to_csv("제외된컬럼.csv", index=False)


        if not valid_symbols:
            print("⚠️ 지정된 범위를 모두 만족하는 Symbol이 없습니다.")
        else:
            print(f"✅ 범위 내 모든 분기를 가진 Symbol 수: {len(valid_symbols)}")

        return valid_symbols
    
    
    #------------------------
    # 넘겨받은 데이터 프레임의 date를 검사해, 나올수 있는 값을 모두 확인
    # 그 다음이, 그 값을 모두 가지고 있는 심볼들의 row만 남기도록 함.
    # ex. TSLA는 2022,2023,2024
    # AMD는 2022,2023,2024
    # A는 2022 있다면?
    # 결과적으로는 TSLA, AMD만 남김
    #-------------------------
    def filter_symbols_with_all_dates(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Date 컬럼의 모든 고유값 추출
        all_dates = set(df['Date'].unique())

        # 2. 심볼별 Date 세트 만들기
        symbol_dates = df.groupby('Symbol')['Date'].apply(set)

        # 3. 모든 날짜를 포함한 심볼만 필터링
        symbols_to_keep = symbol_dates[symbol_dates.apply(lambda dates: all_dates.issubset(dates))].index

        # 4. 해당 심볼만 남긴 DataFrame 반환
        return df[df['Symbol'].isin(symbols_to_keep)].reset_index(drop=True)

    
    
    def filter_remove_same_year(df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Year'] = df['Date'].dt.year

        idx = df.groupby(['Symbol','Year'])['Date'].idxmin()
        df_result = df.loc[idx].copy()
        df_result = df_result.reset_index(drop=True)
        df_result = df_result.sort_values(['Symbol','Year'])
        df_result.drop(columns=['Year'], inplace=True)

        return df_result


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
    

    @staticmethod
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
    



    @staticmethod
    def get_fs_data_static(symbols:list=[], date_type:EDateType = EDateType.QUARTER, min_year:int = 0):
        with DB_FinancialStatement() as fs:
            if len(symbols) == 0:
                symbols = fs.get_symbol_list_with_filter(min_year)

            df = fs.get_fs_all(symbols, date_type)
            df = fs.get_fs_data(df)

            if date_type == EDateType.QUARTER:
                df = DB_FinancialStatement.add_quarter_column(df)
            elif date_type == EDateType.YEAR:
                df = DB_FinancialStatement.add_year_column(df)

            return df
            

    @staticmethod
    def get_f_score_rank_quarter_table(loaded = True):

        csv_file_name = 'f_score_rank_quarter.csv'

        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df
        
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2021)
                df_year = fs.get_fs_all(symbols, EDateType.YEAR)
                df_year = DB_FinancialStatement.filter_remove_same_year(df_year)

                df_year = fs.get_quality_date(df_year)
                df_year['Date'] = df_year['Date'].dt.year

                years = {2021, 2022, 2023, 2024}
                df_year = df_year.groupby('Symbol').filter(lambda x: years.issubset(set(x['Date'])))
                df_year = df_year[df_year['Date'].isin(years)]

                grouped = df_year.groupby('Symbol')
                symbol_dfs = {}
                for symbol, df in grouped:
                    df = df.reset_index(drop= True)
                    dict_score = {
                        'Date' : [],
                        'Symbol' : [],
                        'Sector' : [],
                        'F_Score': []
                    }

                    for idx in range(1, len(df)):
                        
                        score = 0
                        
                        # 전년 당기순이익 0 이상
                        net_income = df.at[idx, 'NetIncome']
                        if net_income > 0:
                            score+=1
                        
                        # 전년 영업현금흐름
                        operating_cash_flow = df.at[idx, 'OperatingCashFlow']
                        if operating_cash_flow > 0:
                            score+=1

                        # ROA 전년 대비 증가
                        roa = df.at[idx, 'ROA']
                        prev_roa = df.at[idx-1, 'ROA']
                        if roa > prev_roa:
                            score +=1

                        # 전년 영업현금흐름 순이익보다 증가
                        if operating_cash_flow > net_income:
                            score +=1

                        # 부채비율 전년대비 감소
                        debt_to_equity_ratio = df.at[idx, 'DebtToEquityRatio']
                        prev_debt_to_equity_ratio = df.at[idx-1, 'DebtToEquityRatio']
                        if debt_to_equity_ratio < prev_debt_to_equity_ratio:
                            score += 1

                        # 유동비율
                        current_ratio = df.at[idx, 'CurrentRatio']
                        prev_current_ratio = df.at[idx-1, 'CurrentRatio']
                        if current_ratio > prev_current_ratio:
                            score += 1

                        # 신규주식발행 없음
                        common_stock_issuance = df.at[idx, 'CommonStockIssuance']
                        if common_stock_issuance == None or common_stock_issuance == 0:
                            score += 1

                        # 매출총이익률
                        gross_profit_margin = df.at[idx, 'GrossProfitMargin']
                        prev_gross_profit_margin = df.at[idx-1, 'GrossProfitMargin']
                        if gross_profit_margin > prev_gross_profit_margin:
                            score += 1
                        
                        # 자산회전률 전년대비 증가
                        asset_turnover_ratio = df.at[idx, 'AssetTurnoverRatio']
                        prev_asset_turnover_ratio = df.at[idx-1, 'AssetTurnoverRatio']
                        if asset_turnover_ratio > prev_asset_turnover_ratio:
                            score += 1

                        dict_score['Date'].append(df.at[idx, 'Date'])
                        dict_score['Symbol'].append(df.at[idx, 'Symbol'])
                        dict_score['Sector'].append(df.at[idx, 'Sector'])
                        dict_score['F_Score'].append(score)

                
                    df = pd.DataFrame(dict_score)
                    df = df[df['F_Score'] >= 3]
                    symbol_dfs[symbol] = df
            

                all_df = pd.concat(symbol_dfs.values(), ignore_index=True)
                all_df['Date'] = all_df['Date'].astype(int)

                all_df = (all_df
                    .groupby(['Date', 'Symbol'], as_index=False)['F_Score']
                    .max())
                
                all_df = all_df.sort_values(['Date', 'F_Score', 'Symbol'],
                                ascending=[True, False, True],
                                kind="mergesort")
                
                year_to_syms = (all_df
                        .groupby('Date')['Symbol']
                        .apply(list)
                        .to_dict())

                # 5) Build a wide DataFrame: columns=years, rows=ranks (0-based then reindex starting at 1)
                years = sorted(year_to_syms.keys())
                max_len = max(len(v) for v in year_to_syms.values()) if years else 0

                data = {}
                for y in years:
                    col = year_to_syms[y]
                    # Pad with None to make equal lengths
                    padded = col + [None] * (max_len - len(col))
                    data[y] = padded

                wide = pd.DataFrame(data)
                wide = wide.head(1000) # 1000개만 추려서 사용
            
                quarters = ['Q1','Q2','Q3','Q4']
                dict_quarter = {}

                def in_quarter_range(value, start, end):
                    def to_key(s):
                        year, q = s.split('-')
                        return int(year), int(q[1])  # Q2 → 2

                    return to_key(start) <= to_key(value) <= to_key(end)

                start_quarter = '2023-Q3'
                end_quarter = '2025-Q2'
                for year in wide.columns:
                    for q in quarters:
                        col = f"{year+1}-{q}"
                        if in_quarter_range(col, start_quarter,end_quarter):
                            dict_quarter[col] = wide[year].to_list()

                df = pd.DataFrame(dict_quarter)
                df.to_csv(csv_file_name, index=False)

            return df
        
    @staticmethod
    def stuff_df_nan(some_dict:dict) -> dict:
        
        max_len = 0
        for key, value in some_dict.items():
            max_len = max(max_len, len(value))

        for key, value in some_dict.items():
            value.extend([np.nan]*(max_len - len(value)))

        return some_dict
        

    # 필터랑 소팅을 통해 분기 데이터를 구함.
    @staticmethod
    def get_rank_table_quarter(label, 
                    preprocess_func=None, 
                    filter_func=None, 
                    rank_key=None, 
                    top_n=20, 
                    loaded=True, 
                    csv_file_name=None,
                    ascending=False):
        """
        범용 랭킹 테이블 생성 함수 (label 중심)
        
        Parameters
        ----------
        label : str
            결과 테이블의 이름 (CSV 저장 파일명에도 사용)
        preprocess_func : function, optional
            DataFrame -> DataFrame 형태의 함수 (새 컬럼 추가/전처리)
        filter_func : function, optional
            DataFrame -> DataFrame 형태의 함수 (행 필터링)
        rank_key : function, required
            DataFrame -> Series 형태의 함수 (정렬에 쓸 값)
        top_n : int, default 20
            최종 랭킹 상위 N개
        loaded : bool, default True
            True면 CSV 파일 로드, False면 새로 생성
        csv_file_name : str, optional
            저장/로드할 CSV 파일 이름 (없으면 label 기반으로 자동 생성)
        ascending : bool, default False
            정렬 방향 (False=내림차순, True=오름차순)
        """
        
        if csv_file_name is None:
            csv_file_name = f"{label}_rank.csv"
        
        if loaded:
            return pd.read_csv(csv_file_name)
        
        # DB에서 데이터 가져오기
        with DB_FinancialStatement() as fs:
            symbols = fs.get_symbol_list_with_filter(2022)
            df = fs.get_fs_all(symbols, commonHelper.EDateType.QUARTER)
            df = fs.get_fs_data(df)
            df = DB_FinancialStatement.add_quarter_column(df)
            
            # 전처리 (새 컬럼 추가 등)
            if preprocess_func:
                df = preprocess_func(df)
            
            # 필터링
            if filter_func:
                df = filter_func(df)
            
            # 정렬용 컬럼
            col_name = f"rank_{label}"
            df[col_name] = rank_key(df) if rank_key else np.nan
            
            # 분기별 그룹화 & 랭킹
            dict_rank = {}
            grouped = df.groupby('Quarter')
            max_len = 0
            for q, group in grouped:
                group = group.sort_values(by=col_name, ascending=ascending)
                group = group[['Date','Quarter','Symbol',col_name]]
                symbol_list = group['Symbol'].tolist()
                dict_rank[q] = symbol_list
                max_len = max(max_len, len(symbol_list))
                display(group.head(10))
            
            # 길이 맞추기
            for key, value in dict_rank.items():
                dict_rank[key] += [np.nan] * (max_len - len(value))
            
            df_rank = pd.DataFrame(dict_rank)
            df_rank = df_rank.head(top_n)
            df_rank.to_csv(csv_file_name, index=False)
            
        return df_rank
    

            
    # NCVA 전략
    # 아래의 조건에 맞는 종목 찾기
    #   1. 유동자산 - 총부채 > 시가총액
    #   2. 분기 수익률 > 0
    # * 1,2번 조건에 맞는 주식들 중에서 (유동자산 - 총부채) / 시가총액 비중이 가장 높은 주식 매수하기
    # 왜? "유동자산 - 총부채 > 시가총액" 이 저평가?
    #  - 기업이 청산되고 남은 현금자산이 시총보다 높다는 이야기
    #  - 즉, 주식시장에서 기업의 가치를 그 기업의 자산보다 낮게 측정했다는 뜻. 
    @staticmethod
    def get_ncva_rank_table(loaded=True) -> pd.DataFrame:
        df_rank = DB_FinancialStatement.get_rank_table_quarter(
            label="Origin_NVAC",
            filter_func=lambda df: df[(df['LiquidationValue'] > df['MarketCap']) & (df['NetIncome'] > 0)],
            rank_key=lambda df: df['LiquidationValue'] / df['MarketCap'],
            ascending=False,
            top_n=20,
            loaded=loaded
        )
        return df_rank

    def get_super_value_rank_table(loaded=True) -> pd.DataFrame:
        csv_file_name = 'super_value_rank.csv'
        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df
        else:
            df_quarter = DB_FinancialStatement.get_fs_data_static(min_year=2022)
            dict_quarter = DB_FinancialStatement.get_rankings_avg_to_dict(df_quarter=df_quarter, selected_metrics=['PER','PBR','PCR','PSR'])
            dict_quarter = DB_FinancialStatement.stuff_df_nan(dict_quarter)
            df = pd.DataFrame(dict_quarter)
            df.to_csv(csv_file_name)
            return df
    

    def get_new_magic_rank_table(loaded=True) -> pd.DataFrame:
        csv_file_name = 'new_magic_rank.csv'
        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df
        else:
            df_quarter = DB_FinancialStatement.get_fs_data_static(min_year=2022)
            dict_quarter = DB_FinancialStatement.get_rankings_avg_to_dict(df_quarter=df_quarter, selected_metrics=['PBR','GP/A'])
            dict_quarter = DB_FinancialStatement.stuff_df_nan(dict_quarter)
            df = pd.DataFrame(dict_quarter)
            df.to_csv(csv_file_name)
            return df
        
    
    def get_f_score_rank_table(loaded = True) -> pd.DataFrame:

        csv_file_name = 'f_score_with_per.csv'
        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df

        else:
            df_rank = DB_FinancialStatement.get_f_score_rank_quarter_table(False)
            symbols = pd.unique(df_rank.values.ravel())
            symbols = [s for s in symbols if pd.notna(s)]

            with DB_FinancialStatement() as fs:
                df_quarter = fs.get_fs_all(symbols, EDateType.QUARTER)
                df_quarter = fs.get_value_data(df_quarter)
                df_quarter = DB_FinancialStatement.add_quarter_column(df_quarter)
                df_quarter['1/PER'] = 1/df_quarter['PER']

                dict_df = {}
                grouped = df_quarter.groupby('Quarter')
                for q, group in grouped:

                    if q not in df_rank:
                        print(f"{q} is not in df_rank")
                        continue

                    check_symbols = df_rank[q].dropna().tolist()
                    df_filter = group[group['Symbol'].isin(check_symbols)]
                    df_filter = df_filter.sort_values(by='1/PER', ascending=False)
                    dict_df[q] = df_filter['Symbol'].tolist()
                
                max_len = max(len(v) for v in dict_df.values())
                padded = {k: v + [None] * (max_len - len(v)) for k, v in dict_df.items()}
                df = pd.DataFrame(padded)
                df = df.head(20)
                df.to_csv(csv_file_name, index=False)
            
            return df
        

    def get_market_cap_rank_table(loaded = True):
        
        csv_file_name = 'market_cap_rank.csv'
        if loaded:
            df = pd.read_csv(csv_file_name)
            return df
        
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2021)
                df = fs.get_fs_all(symbols, commonHelper.EDateType.QUARTER)
                df = fs.get_value_data(df)
                df = DB_FinancialStatement.add_quarter_column(df)

                grouped = df.groupby('Quarter')
                
                grouped_dict = {}
                max_len = 0
                for index, group in grouped:
                    group = group.reset_index(drop=True)
                    group = group.sort_values(by='MarketCap',ascending=True)
                    group_list = group['Symbol'].tolist()
                    grouped_dict[index] = group_list
                    max_len = max(max_len, len(group_list))

                for key, value in grouped_dict.items():
                    value.extend([np.nan]*(max_len - len(value)))

                df = pd.DataFrame(grouped_dict)
                df = df.head(500)
                df.to_csv(csv_file_name, index= False)

            return df
        

    # EV_EBIT 전략
    def get_ev_ebit_rank_table(loaded = True):
        df_ev_ebit = DB_FinancialStatement.get_rank_table_quarter(
            label="EV_EBIT",
            preprocess_func=lambda df: df.assign(**{'1/(EV/EBIT)': 1/df['EV/EBIT'].replace(0, np.nan)}),
            filter_func=lambda df: df[(df['EV/EBIT'] > 0) & (df['EV/EBIT'] < 50)],
            rank_key=lambda df: df['1/(EV/EBIT)'],
            ascending=False,
            top_n=20,
            loaded=loaded
        )

        return df_ev_ebit
        
    # 저 피비알 전략
    def get_low_per_rank_table(loaded =True):
        df_per = DB_FinancialStatement.get_rank_table_quarter(
            label="PER",
            preprocess_func=lambda df: df.assign(**{'1/PER': 1/df['PER'].replace(0, np.nan)}),
            filter_func=lambda df: df[df['PER'] > 0],
            rank_key=lambda df: df['1/PER'],
            ascending=False,
            top_n=20,
            loaded=loaded
        )
        return df_per
    

    # 파마의 고수성 + 저투자 전략
    def get_fama_high_return_rank_table(loaded = True):
        csv_file_name = 'fama_high_return.csv'

        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df

        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2021)
                df = fs.get_fs_all(symbols, commonHelper.EDateType.QUARTER)
                df = fs.get_quality_date(df)
                df = df.dropna(subset='TotalAssetsGrowth')
                df = DB_FinancialStatement.add_quarter_column(df)
                
                grouped = df.groupby('Quarter')
                dict_rank = {}
                for q, group in grouped:
                    group = group[group['TotalAssetsGrowth'] > -0.2]
                    asset_growth = group.sort_values(by='TotalAssetsGrowth', ascending=True)['Symbol'].tolist()
                    gp_a_rank = group.sort_values(by='GP/A', ascending=False)['Symbol'].tolist()

                    rank_asset = {sym:i for i, sym in enumerate(asset_growth)}
                    rank_gp_a = {sym:i for i, sym in enumerate(gp_a_rank)}

                    common_symbol = set(rank_asset) & set(rank_gp_a)
                    avg_rank = []

                    for sym in common_symbol:
                        r1 = rank_asset[sym]
                        r2 = rank_gp_a[sym]
                        avg = (r1 + r2) / 2
                        avg_rank.append((sym, avg))

                    avg_rank_sorted = sorted(avg_rank, key = lambda x:x[1])
                    final_symbols = [sym for sym, _ in avg_rank_sorted]
                    dict_rank[q] = final_symbols

                dict_rank = DB_FinancialStatement.stuff_df_nan(dict_rank)
                df = pd.DataFrame(dict_rank)
                df.to_csv(csv_file_name)
                return df
            

    
    def get_income_momentum_rank_table(loaded = True):
        csv_file_name = 'income_momentum_rank.csv'
        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2022)
                df_year = fs.get_fs_all(symbols, commonHelper.EDateType.YEAR)
                df_year = fs.get_momentun_data(df_year)
                df_year = df_year.dropna(subset=['NetIncomeGrowth'])
                df_year = DB_FinancialStatement.add_year_column(df_year)
                df_quarter = fs.get_fs_all(symbols, commonHelper.EDateType.QUARTER)
                df_quarter = fs.get_momentun_data(df_quarter)
                df_quarter = df_quarter.dropna(subset=['NetIncomeGrowth'])
                df_quarter = DB_FinancialStatement.add_quarter_column(df_quarter)
                
                grouped_year = df_year.groupby('Year')
                grouped_quarter = df_quarter.groupby('Quarter')

                rank_year_dict = {}
                for y, group in grouped_year:
                    y_op = group.sort_values(by='OperatingIncomeGrowth', ascending = False)['Symbol'].tolist()
                    y_net = group.sort_values(by='NetIncomeGrowth', ascending =False)['Symbol'].tolist()

                    if y not in rank_year_dict:
                        rank_year_dict[y] = {}

                    rank_year_dict[y]['OperatingIncomeGrowth'] = {sym:i for i, sym in enumerate(y_op)}
                    rank_year_dict[y]['NetIncomeGrowth'] = {sym:i for i, sym in enumerate(y_net)}

                dict_rank = {}
                for q, group in grouped_quarter:
                    q_op = group.sort_values(by='OperatingIncomeGrowth', ascending = False)['Symbol'].tolist()
                    q_net = group.sort_values(by='NetIncomeGrowth', ascending =False)['Symbol'].tolist()

                    rank_op = {sym:i for i, sym in enumerate(q_op)}
                    rank_net ={sym:i for i, sym in enumerate(q_net)}

                    year = int(q.split('-')[0]) - 1 # 현분기 데이터를 기준이니, 전년도 데이터를 찾아야 함.
                    rank_op_y = rank_year_dict[year]['OperatingIncomeGrowth']
                    rank_net_y = rank_year_dict[year]['NetIncomeGrowth']

                    common_symbol = set(rank_op) & set(rank_net) & set(rank_op_y) & set(rank_net_y)
                    avg_rank = []
                    for sym in common_symbol:
                        r1 = rank_op[sym]
                        r2 = rank_net[sym]
                        r3 = rank_op_y[sym]
                        r4 = rank_net_y[sym]
                        avg = (r1 + r2 + r3 + r4) / 4
                        avg_rank.append((sym, avg))

                    avg_rank_sorted = sorted(avg_rank, key = lambda x:x[1])
                    final_symbols = [sym for sym, _ in avg_rank_sorted]
                    dict_rank[q] = final_symbols

                dict_rank = DB_FinancialStatement.stuff_df_nan(dict_rank)
                df = pd.DataFrame(dict_rank)
                df.to_csv(csv_file_name)
                return df
            

    # 업그레이드 nvca 전략
    def get_upgrade_nvca_rank_table(loaded = True):
        df = DB_FinancialStatement.get_rank_table_quarter(
            label='upgrade_nvca',
            filter_func=lambda df : df[(df['LiquidationValue'] > df['MarketCap']) & 
                                    (df['NetIncome'] > 0) & 
                                    (df['GP/A'] >= df['GP/A'].quantile(0.5)) & 
                                    (df['DebtToEquityRatio'] <= 2) ],
            rank_key = lambda df :  df['LiquidationValue'] / df['MarketCap'],
            ascending = False,
            top_n= 20,
            loaded=loaded
        )
        return df
    
    # 업그레이드 슈퍼가치 전략
    def get_upgrade_super_value_rank_table(loaded=True):
        csv_file_name = 'upgrade_super_value.csv'
        if loaded:
            result_df = pd.read_csv(csv_file_name)
            result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
            return result_df
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2022)
                df_quarter = fs.get_fs_all(symbols, commonHelper.EDateType.QUARTER)
                df_quarter = fs.get_value_data(df_quarter)
                df_quarter = DB_FinancialStatement.add_quarter_column(df_quarter)

                grouped = df_quarter.groupby('Quarter')
                dict_rank = { }
                for q, group in grouped:
                    group['1/PER'] = 1/group['PER']
                    group['1/PBR'] = 1/group['PBR']
                    group['1/PFCR'] = 1/group['PFCR']
                    group['1/PSR'] = 1/group['PSR']

                    rank_per = group.sort_values(by='1/PER', ascending=False)['Symbol'].tolist()
                    rank_pbr = group.sort_values(by='1/PBR', ascending=False)['Symbol'].tolist()
                    rank_pfcr = group.sort_values(by='1/PFCR', ascending=False)['Symbol'].tolist()
                    rank_psr = group.sort_values(by='1/PSR', ascending=False)['Symbol'].tolist()

                    per_dict = {sym:i for i, sym in enumerate(rank_per)}
                    pbr_dict = {sym:i for i, sym in enumerate(rank_pbr)}
                    pfcr_dict = {sym:i for i, sym in enumerate(rank_pfcr)}
                    psr_dict = {sym:i for i, sym in enumerate(rank_psr)}

                    common_symbol = set(per_dict) & set(pbr_dict) & set(pfcr_dict) & set(psr_dict)
                    avg_rank = []
                    for sym in common_symbol:
                        r1 = per_dict[sym]
                        r2 = pbr_dict[sym]
                        r3 = pfcr_dict[sym]
                        r4 = psr_dict[sym]
                        avg = (r1 + r2 + r3 + r4) / 4
                        avg_rank.append((sym, avg))

                    avg_rank_sorted = sorted(avg_rank, key = lambda x:x[1])
                    final_symbols = [sym for sym, _ in avg_rank_sorted]
                    dict_rank[q] = final_symbols
                    
                dict_rank = DB_FinancialStatement.stuff_df_nan(dict_rank)
                df = pd.DataFrame(dict_rank)
                df.to_csv(csv_file_name)
            return df
        

    def get_super_quality_rank_table(loaded=True):
        csv_file_name = 'super_quality_rank.csv'
        if loaded:
                result_df = pd.read_csv(csv_file_name)
                result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
                return result_df
        else:
            with DB_FinancialStatement() as fs:
                symbol = fs.get_symbol_list_with_filter(2022)
                df = fs.get_fs_all(symbol, commonHelper.EDateType.YEAR)
                df = fs.get_fs_data(df)
                df = DB_FinancialStatement.add_year_column(df)

                grouped = df.groupby('Year')
                dict_year = {}
                for y, group in grouped:
                    group = group.fillna(0).infer_objects(copy=False)
                    group = group[(group['CommonStockIssuance'] == 0) &
                                (group['NetIncomeGrowth'] > 0)]
                    dict_year[y] =  [sym for sym in set(group['Symbol'].tolist()) if pd.notna(sym)]
                
                symbol_years = [sym for val in dict_year.values() 
                                    for sym in val]
                
                df = fs.get_fs_all(symbol_years, commonHelper.EDateType.QUARTER)
                df = fs.get_fs_data(df)
                df = fs.get_volatility_m(df)
                df = DB_FinancialStatement.add_quarter_column(df)

                grouped = df.groupby('Quarter')
                dict_quarter = {}
                for q, group in grouped:
                    year = q.split('-')[0]
                    group = group[(group['OperatingCashFlow'] > 0) &
                                (group['Symbol'].isin(dict_year[int(year)-1]))&
                                (group['TotalDebtGrowth'].notna())&
                                (group['TotalDebtGrowth'] != 0)] # 값에 변화가 없을때 제외한다.
                    group = group.copy()
                    group['Income_to_DebtGrowth'] = group['OperatingIncome'] / group['TotalDebtGrowth']

                    if len(group) == 0:
                        continue

                    rank_gp_a = group.sort_values(by='GP/A', ascending=False)['Symbol'].tolist()
                    rank_income_to_debt = group.sort_values(by='Income_to_DebtGrowth', ascending=False)['Symbol'].tolist() # 영업이익/차입금증감율
                    rank_asset_growth = group.sort_values(by='TotalAssetsGrowth')['Symbol'].tolist()
                    rank_volatility =group.sort_values(by='Volatility12M')['Symbol'].tolist()

                    gp_a_dict = {sym:i for i,sym in enumerate(rank_gp_a)}
                    income_to_debt_dict = {sym:i for i,sym in enumerate(rank_income_to_debt)}
                    asset_growth_dict = {sym:i for i,sym in enumerate(rank_asset_growth)}
                    volatility_dict = {sym:i for i,sym in enumerate(rank_volatility)}

                    common_symbol = set(gp_a_dict) & set(income_to_debt_dict) & set (asset_growth_dict) & set(volatility_dict)
                    avg_rank = []
                    for sym in common_symbol:
                        r1 = gp_a_dict[sym]
                        r2 = income_to_debt_dict[sym]
                        r3 = asset_growth_dict[sym]
                        r4 = volatility_dict[sym]
                        avg = (r1 + r2 + r3 + r4) / 4
                        avg_rank.append((sym, avg))

                    avg_rank_sorted = sorted(avg_rank, key = lambda x:x[1])
                    final_symbols = [sym for sym, _ in avg_rank_sorted]
                    dict_quarter[q] = final_symbols
                
                dict_quarter = DB_FinancialStatement.stuff_df_nan(dict_quarter)
                df = pd.DataFrame(dict_quarter)
                df.to_csv(csv_file_name)
                return df
            

    def get_fama_last_weapon_rank_table(loaded = True):
        csv_file_name = 'fama_last_weapon.csv'
        if loaded:
                result_df = pd.read_csv(csv_file_name)
                result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
                return result_df
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2022)
                # symbols = symbols[:100]
                df = fs.get_fs_all(symbols, EDateType.QUARTER)
                df = fs.get_fs_data(df)
                df = DB_FinancialStatement.add_quarter_column(df)

                grouped = df.groupby('Quarter')
                dict_rank = {}
                for q, group in grouped:
                    group = group[['Date','Symbol','MarketCap','PBR', 'GP/A', 'TotalAssetsGrowth']]
                    
                    group = group.copy()
                    group['1/PBR'] = 1 / group['PBR']

                    cutoff = group['MarketCap'].quantile(0.2)
                    group = group[group['MarketCap'] < cutoff]
                    group = group[(group['PBR'] > 0.25)&
                                (group['GP/A'] > 0) &
                                (group['TotalAssetsGrowth'] > -0.2)]
                    
                    rank_pbr = group.sort_values(by='1/PBR', ascending=False)['Symbol'].tolist()
                    rank_gp_a = group.sort_values(by='GP/A', ascending=False)['Symbol'].tolist()
                    rank_asset_growth = group.sort_values(by='TotalAssetsGrowth')['Symbol'].tolist()

                    dict_pbr = {sym:i for i, sym in enumerate(rank_pbr)}
                    dict_gp_a = {sym:i for i, sym in enumerate(rank_gp_a)}
                    dict_asset_growth = {sym:i for i, sym in enumerate(rank_asset_growth)}

                    common_symbol = set(dict_pbr) & set(dict_gp_a) & set(dict_asset_growth)
                    avg_rank = []
                    for sym in common_symbol:
                        r1 = dict_pbr[sym]*0.5
                        r2 = dict_gp_a[sym]*0.25
                        r3 = dict_asset_growth[sym]*0.25
                        avg = (r1+r2+r3)/3
                        avg_rank.append((sym, avg))

                    avg_rank_sorted = sorted(avg_rank, key= lambda x:x[1])
                    final_symbols = [sym for sym, _ in avg_rank_sorted]
                    dict_rank[q] = final_symbols
                        

                dict_rank = DB_FinancialStatement.stuff_df_nan(dict_rank)
                df = pd.DataFrame(dict_rank)
                df.to_csv(csv_file_name)
                return df
            


    #-----------------
    # fscore 필터링기
    #-----------------
    def filter_f_score_to_dict(grouped_df:pd.core.groupby.generic.DataFrameGroupBy, filters: list):

        condition_map = {
            'CommonStockIssuance': lambda df: df['CommonStockIssuance'] == 0,  # 신규 주식 발행 없음
            'NetIncomeGrowth': lambda df: df['NetIncomeGrowth'] > 0,           # 순이익 증가
            'OperatingCashFlow': lambda df: df['OperatingCashFlow'] > 0        # 영업현금흐름 
        }

        result_dict = {}
        for key, df in grouped_df:
            df = df.copy().fillna(0).infer_objects(copy=False)
            mask = pd.Series(True, index=df.index)
            for f in filters:
                if f in condition_map:
                    mask &= condition_map[f](df)

            df_filtered = df[mask]
            result_dict[key] = df_filtered['Symbol'].tolist()

        return result_dict


    #-------------------
    # 각각의 원하는 옵션들을 선택해서 평균을 구하는 함수
    #-------------------
    def get_rankings_avg_to_dict(df_quarter, filter_quarter:dict=None, selected_metrics=None):
        
        if selected_metrics is None:
            selected_metrics =  [
                'PER',
                'PBR',
                'PCR',
                'PSR',
                'GP/A',
                'Income_to_DebtGrowth',
                'TotalAssetsGrowth',
                'Volatility12M'
            ]

        """
        분기별로 심볼 랭킹을 계산하고 평균 랭킹 결과 반환
        
        Parameters:
            df_quarter : DataFrame
            filter_quarter : dict (분기별 심볼 필터)
            selected_metrics : list (선택할 지표 이름들, None이면 전체 8개 사용)
            
        Returns:
            dict_quarter : dict {Quarter: [심볼 랭킹 리스트]}
        """
        dict_quarter = {}
        
        # 랭킹 정의 (지표 이름 → (정렬 기준 컬럼, ascending, key함수))
        ranking_specs = {
            'PER': ('PER', False, lambda x: 1/x),
            'PBR': ('PBR', False, lambda x: 1/x),
            'PCR': ('PCR', False, lambda x: 1/x),
            'PSR': ('PSR', False, lambda x: 1/x),
            'GP/A': ('GP/A', False, None),
            'Income_to_DebtGrowth': ('Income_to_DebtGrowth', False, None),
            'TotalAssetsGrowth': ('TotalAssetsGrowth', True, None),
            'Volatility12M': ('Volatility12M', True, None)
        }
        
        # 선택한 지표 없으면 전체 사용
        if selected_metrics is None:
            selected_metrics = list(ranking_specs.keys())
        
        grouped = df_quarter.groupby('Quarter')
        for q, df in grouped:
            df = df.copy()
            
            if filter_quarter is not None:
                df = df[df['Symbol'].isin(filter_quarter[q])]  # f_score 필터링
            
            df['Income_to_DebtGrowth'] = df['OperatingIncome'] / df['TotalDebtGrowth']
            
            rank_dicts = {}
            for metric in selected_metrics:
                col, asc, key_func = ranking_specs[metric]
                sorted_list = df.sort_values(
                    by=col, ascending=asc, key=key_func
                )['Symbol'].tolist()
                rank_dicts[metric] = {sym: i for i, sym in enumerate(sorted_list)}
            
            # 공통 symbol
            common_symbol = set.intersection(*[set(d.keys()) for d in rank_dicts.values()])
            
            avg_rank = []
            for sym in common_symbol:
                ranks = [rank_dicts[m][sym] for m in selected_metrics]
                avg = sum(ranks) / len(ranks)
                avg_rank.append((sym, avg))
            
            # 평균 순위로 정렬
            avg_rank_sorted = sorted(avg_rank, key=lambda x: x[1])
            final_symbols = [sym for sym, _ in avg_rank_sorted]
            dict_quarter[q] = final_symbols
        
        return dict_quarter  



    def get_super_value_and_quality_rank_table(loaded=True):

        csv_file_name = 'super_value_and_quality_rank.csv'
        if loaded:
                result_df = pd.read_csv(csv_file_name)
                result_df = result_df.drop(columns=['Unnamed: 0'], errors='ignore')
                return result_df
        else:
            with DB_FinancialStatement() as fs:
                symbols = fs.get_symbol_list_with_filter(2022)
                symbols = symbols

                df_year = fs.get_fs_all(symbols, EDateType.YEAR)
                df_year = fs.get_fs_data(df_year)
                df_year = DB_FinancialStatement.add_year_column(df_year)

                df_quarter = fs.get_fs_all(symbols, EDateType.QUARTER)
                df_quarter = fs.get_fs_data(df_quarter)
                df_quarter = fs.get_volatility_m(df_quarter) # 12개월 주가 변동성 (시간오래걸림)
                df_quarter = DB_FinancialStatement.add_quarter_column(df_quarter)

                # 신 F스코어 (3개 처리)
                filter_year = DB_FinancialStatement.filter_f_score_to_dict(df_year.groupby('Year'), ['CommonStockIssuance'])
                filter_quarter = DB_FinancialStatement.filter_f_score_to_dict(df_quarter.groupby('Quarter'), ['NetIncomeGrowth', 'OperatingCashFlow'])
                
                for q, symbols in filter_quarter.items():
                    year = int(q.split('-')[0]) - 1 # 왜냐하면, 연간재무재표는 연간재무재표의 다음해에 사용함. (2024년의 결과를 2025년에 씀)
                    filter_quarter[q] = list(set(symbols) & set(filter_year[year]))

                dict_quarter = DB_FinancialStatement.get_rankings_avg_to_dict(df_quarter, filter_quarter)
                dict_quarter = DB_FinancialStatement.stuff_df_nan(dict_quarter)
                df = pd.DataFrame(dict_quarter)
                df.to_csv(csv_file_name)
            return df



    # 특정 컬럼의 기본값 구하기
    # - 평균 말고 중앙값.
    # - 윈저라이즈 활용 (좌우 몇퍼센트 제거)(비 이상적인 값 제거)
    @staticmethod
    def get_avg_value_by_sector(df_fs, column_name:str):
        from scipy.stats.mstats import winsorize
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

        group_q = df_fs.groupby('Quarter')
        dict_q = {}
        for q, df_q in group_q:
            group_s = df_q.groupby('Sector')
            dict_s = {}
            for s, df_s in group_s:
                if df_s.empty:
                    continue
                pbr_val = df_s[column_name].dropna().to_numpy()
                pbr_winsorized = winsorize(pbr_val, limits=[0.05, 0.05])
                mid = pd.Series(pbr_winsorized).median()
                dict_s[s] = mid
            dict_q[q]= dict_s


        df =pd.DataFrame.from_dict(dict_q, orient='columns')
        df = df.reset_index(names='Selctor')
        return df