from functools import reduce
from assetAllocation import AssetAllocation
from db_stock import DB_Stock
from mysqlConnecter import MySQLConnector
from commonHelper import EFinancialStatementType, EDateType, EIndustry
from IPython.display import display

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
    

    def get_fs_all(self, symbols:list, dateType:EDateType):
        
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

        return merged
    

    # 시총구하기
    def get_marketCap(self, df):
        df = df.copy()
        df['MarketCap'] = None

        for idx in range(len(df)):
            date = df.at[idx, 'Date']
            symbol = df.at[idx, 'Symbol']
            ordinarySharesNumber = df.at[idx, 'OrdinarySharesNumber']

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
                # officers = df.at[idx, 'companyOfficers']
                symbol = str(df.at[idx, 'symbol']).upper()

                # 1. 키워드 기반
                keywords = ["spac", "blank check", "special purpose acquisition"]
                keyword_hit = any(kw in description or kw in industry or kw in sector for kw in keywords)

                # 2. industry 기반
                industry_based = industry in ["shell companies", "capital markets", "asset management"]

                # 3. 임원 수 기반
                # officer_based = isinstance(officers, list) and len(officers) <= 1

                # 4. 티커 네이밍 기반
                ticker_based = any(x in symbol for x in ["-U", "-WS", "-R"])

                # 최종 판별
                # is_spac = keyword_hit or industry_based or officer_based or ticker_based
                is_spac = keyword_hit or industry_based or ticker_based
                df.at[idx, 'IsSPAC'] = is_spac

            except Exception as e:
                print(f"[{symbol}] 판별 실패: {e}")
                df.at[idx, 'IsSPAC'] = None

        return df

    def get_data(self, symbols:list, dateType:EDateType):
        df = self.get_fs_all(symbols=symbols, dateType=dateType)
        df = self.get_sector(df)
        df = self.get_marketCap(df)
        df = self.get_psr(df)
        df = self.get_gp_a(df)
        df = self.get_por(df)
        df = self.get_ev_ebti(df)
        df = self.get_per(df)
        df = self.get_liquidation_value(df)
        df = self.get_current_ratio(df)
        df = self.get_pbr(df)
        df = self.get_debt_to_equity_ratio(df)
        df = self.get_pcr(df)
        df = self.get_pfcr(df)
        df = self.get_dividend_payout_ratio(df)
        df = self.get_income_growth(df)
        df = self.get_gross_profit_margin(df)
        
        df = df[['Date', 'Symbol', 'Sector', 'MarketCap', 'TotalRevenue', 'NetIncome', 'OperatingIncome', 'GrossProfitMargin', 'IncomeGrowth', 'PSR', 'GP/A', 'POR', 'EV/EBIT', 'PER', 'LiquidationValue', 'CurrentRatio','PBR', 'DebtToEquityRatio', 'CommonStockDividendPaid', 'PCR', 'PFCR', 'DividendPayoutRatio']]
        df = df.dropna(subset=["MarketCap"])
        df = df.infer_objects(copy=False)

        return df