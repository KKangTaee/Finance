import pandas as pd
import yfinance as yf
import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta
from time import strptime
from IPython.display import display
from curl_cffi import requests

pd.set_option('display.max_rows', None)  # 모든 행 출
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.expand_frame_repr', None)  # 긴 데이터 프레임 줄바꿈 없이 출력

# 자산배분 클래스
class AssetAllocation:

    #-----------------
    # 배당 데이터 가져오기
    #-----------------
    @staticmethod
    def get_dividends(symbols:list):
        result_dfs = {}

        for symbol in symbols:
            info = yf.Ticker(symbol)
            df = info.actions.reset_index()
            df['Date'] = df['Date'].dt.date # 년월일만 남김
            df['Symbol'] = symbol
            df = df[['Date','Symbol','Dividends']]
            result_dfs[symbol]=df

        return result_dfs
    

    #------------------
    # 주가 데이터 가져오기
    #------------------
    @staticmethod
    def get_stock_data(symbols:list, start_date, end_date):
        
        result_dfs = {}
        
        session = requests.Session(impersonate="safari15_5")
        # 로딩하는건 progress = False 하면 해제됨
        df = yf.download(tickers=symbols, session= session, start=start_date, end=end_date, progress=False, auto_adjust=False)

        # 배당 데이터 가져오기
        dividends_dfs = AssetAllocation.get_dividends(symbols= symbols)

        for symbol in symbols:
            # 특정 심볼 선택
            df_symbol = df.loc[:, df.columns.get_level_values(1) == symbol]

            # 멀티 column의 심볼들 제거 후, 심볼 추가
            df_symbol.columns = df_symbol.columns.droplevel(1)
            df_symbol = df_symbol.reset_index()
            df_symbol .columns.name = None

            # 심볼 추가 및 column 위치 조정
            df_symbol['Symbol'] = symbol
            columns = [col for col in df_symbol.columns if col != 'Date' and col != 'Symbol']
            columns = ['Date', 'Symbol'] + columns 
            df_symbol = df_symbol[columns]

            # Date값을 datetime 으로 변경
            df_symbol['Date'] = pd.to_datetime(df_symbol['Date'])

            df_dividends = dividends_dfs[symbol]
            df_dividends['Date'] = pd.to_datetime(df_dividends['Date'])

            # 배당데이터 병합
            df_symbol = df_symbol.merge(df_dividends, on=['Date','Symbol'], how = 'left')
            df_symbol.fillna(0, inplace=True)
            result_dfs[symbol] = df_symbol

        return result_dfs
    

    #---------------------------
    # 종가 데이터 가져오기 (배당금 포함)
    #---------------------------
    @staticmethod
    def get_stock_data_close(symbols:list, start_date, end_date=None, include_diviend=False, key = 'Close'):
    
        result_dfs = {}

        symbol_dfs = AssetAllocation.get_stock_data(symbols=symbols, start_date=start_date,end_date=end_date)
        
        for symbol, df_symbol in symbol_dfs.items():

            if key == 'Adj Close':
                result_close = df_symbol['Adj Close']
                condition = 'Adj Close'
            
            elif key == 'Close':
                if include_diviend == True:
                    result_close = df_symbol['Close'] + df_symbol['Dividends']
                    condition = 'Close + Dividends'
                else:
                    result_close = df_symbol['Close']
                    condition = 'Close'

            df_symbol['Result Close'] = result_close
            df_symbol['Condition'] = condition
            df_symbol = df_symbol[['Date','Symbol','Adj Close', 'Close', 'Dividends','Result Close','Condition']]

            result_dfs[symbol] = df_symbol

        return result_dfs
    


    #----------------------
    # 이동평균과 함께 데이터 로드
    #----------------------
    @staticmethod
    def get_stock_data_with_ma(symbols:list, start_date, end_date, mas, type='ma_day'):

        result_dfs = {}

        if len(mas) <= 0:
            print('mas count is 0!')
            return
        
        if start_date != None:
            start_date = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        
        if end_date != None:
            end_date = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

        sort_mas = sorted(mas)

        # 일별이동평균
        if type == 'ma_day':
            prev_day = sort_mas[-1]
            prev_date = start_date - relativedelta(days = prev_day*1.5)

        
        # 월별이동평균 - 해당월의 마지막날 데이터의 종가 기준으로 평균 값 구함.
        elif type == 'ma_month':
            prev_day = sort_mas[-1]*25
            prev_date = start_date -relativedelta(days=prev_day*1.5)

        # 월말 데이터 가져오기
        symbol_dfs = AssetAllocation.get_stock_data_close(symbols=symbols, start_date=prev_date, end_date=end_date, include_diviend=True, key= 'Close')

        for symbol, df_symbol in symbol_dfs.items():
            df = df_symbol.copy()

            if type == 'ma_day':
                for ma in mas:
                    df[f'SMA_{ma}'] = df['Result Close'].rolling(window=ma).mean()

            elif type == 'ma_month':
                monthly_df = df.groupby(df['Date'].dt.to_period('M')).apply(lambda x: x.iloc[-1]).reset_index(drop=True)


                new_columes = []
                for ma in mas:
                    col = f'MMA_{ma}'
                    monthly_df[col] = monthly_df['Result Close'].rolling(window=ma).mean()
                    new_columes.append(col)
                    
                
                df = df.merge(monthly_df[['Date', 'Symbol'] + new_columes], on=['Date', 'Symbol'], how='left')
                
                
            # 기간범위 체크
            df = df[df['Date'] >= start_date]
            if end_date is not None:
                df = df[df['Date'] <= end_date]

            result_dfs[symbol] = df

        return result_dfs
    


    #----------------
    # 월말 데이터 가져오기
    #----------------
    def filter_close_last_month(symbol_dfs):
        result_dfs = {}

        for symbol, df_symbol in symbol_dfs.items():
            # ✅ 'Date'가 datetime 타입이 아니면 변환
            if not pd.api.types.is_datetime64_any_dtype(df_symbol['Date']):
                df_symbol['Date'] = pd.to_datetime(df_symbol['Date'], errors='coerce')

            monthly_df = df_symbol.groupby(df_symbol['Date'].dt.to_period('M')).apply(lambda x: x.iloc[-1]).reset_index(drop=True)
            result_dfs[symbol] = monthly_df
        
        return result_dfs
    

    #----------------
    # 평균수익률 계산
    #----------------
    # * intervals에 들어간 값은 해당월을 의미함.
    # * [1, 3, 6] 이렇게 되어 있으면, 1개울 수익률, 3개월 수익률, 6개월 수익률을 말하며, 이것의 평균을 반환함.
    def get_profit_ratio_avg(symbol_dfs:dict, intervals=[1,3,6]):

        result_dfs = {}

        for symbol, df in symbol_dfs.items():
            
            interval_dfs ={}
            for interval in intervals:
                copy_df = df.copy()
                shift_df = copy_df.shift(interval)
                
                copy_df['Interval'] = interval

                profit_value = copy_df['Result Close'] - shift_df['Result Close']

                copy_df['Profit Value'] = profit_value
                copy_df['Profit Ratio'] = ((profit_value/shift_df['Result Close'])*100).round(2)

                interval_dfs[interval] = copy_df
                
            
            avg_profit_ratios = sum(df_interval['Profit Ratio'] for df_interval in interval_dfs.values()) / len(intervals)
            
            new_df = df.copy()
            new_df['Avg Profit Ratio'] = avg_profit_ratios
            new_df['Interval'] = [intervals] * len(df)

            result_dfs[symbol] = new_df

        return result_dfs
    


    #----------------
    # 하나로 병합
    #----------------
    @staticmethod
    def merge_to_dfs(symbol_dfs:dict, except_columns = []):
        all_dfs = []
        for symbol, df_symbol in symbol_dfs.items():
            df = df_symbol.copy()
            all_dfs.append(df)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        group_df = merged_df.groupby('Date').agg(lambda x: list(x)).reset_index()

        for col in group_df.columns:
            group_df[col] = group_df[col].apply(
                lambda x: [round(val, 2) if isinstance(val, (int, float)) else val for val in x] if isinstance(x, list) else x
            )

        if len(except_columns) > 0:
            columns = [col for col in group_df.columns if col not in except_columns]
            group_df = group_df[columns]

        return group_df
    


    #------------------
    # 일별 데이터 병합
    #------------------
    @staticmethod
    def get_performance(df, performance_name):
        # 심볼과 기본 데이터 설정
        symbol = df['Symbol'].iloc[0]
        start_balance = df['Balance'].dropna().iloc[0]
        end_balance = df['Balance'].dropna().iloc[-1]
        start_date = pd.to_datetime(df['Date'].iloc[0])
        end_date = pd.to_datetime(df['Date'].iloc[-1])
        num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # 연복리 성장률 (CAGR) 계산
        years = num_months / 12
        CAGR = ((end_balance / start_balance) ** (1 / years)) - 1

        # 월간 수익률 계산
        balance_series = df['Balance'].dropna().astype(float)
        monthly_returns = balance_series.pct_change().dropna()

        # 무위험 수익률 가정 (연 2%를 월간으로 환산)
        risk_free_rate = 0.03 / 12

        # 연간화된 표준 편차 계산
        annualized_std_dev = monthly_returns.std() * np.sqrt(12)

        # Sharpe Ratio 계산
        sharpe_ratio = (monthly_returns.mean() - risk_free_rate) / monthly_returns.std() * np.sqrt(12)

        # 최대 낙폭 (MDD) 계산
        running_max = np.maximum.accumulate(balance_series)
        drawdown = (balance_series / running_max - 1).min()

        # 결과 데이터프레임 생성
        results = pd.DataFrame({
            "Name" : [performance_name],
            "Symbol": [symbol],
            "Start Date" : [start_date],
            "End Date":[end_date],
            "Start Balance":[start_balance],
            "End Balance": [end_balance],
            "Annualized Return (CAGR)": [CAGR],
            "Standard Deviation": [annualized_std_dev],
            "Sharpe Ratio": [sharpe_ratio],
            "Maximum Drawdown": [drawdown]
        })

        return results


    #-------------------
    # 균등 전략
    #-------------------
    # * interval 마다 균등하게 리벨런싱 처리
    # * 영구포폴, 올웨더에 사용함.
    def strategy_evenly(symbol_dfs: dict, ratios:list, interval:int = 12, init_balance = 10000):
        if len(symbol_dfs) != len(ratios):
            print('심볼이랑, 가중치랑 길이가 다름!')
            return
        
        if sum(ratios) != 100:
            print(f'가중치 값이 다름! : {sum(ratios)}')
            return

        df = AssetAllocation.merge_to_dfs(symbol_dfs)
        columns = [col for col in df.columns if col not in ['Adj Close','Close','Dividends','Condition']]
        df = df[columns]

        total_ratio = sum(ratios)
        start_balances = [round((r / total_ratio) * init_balance, 2) for r in ratios]
        end_balances = [0]*len(ratios)

        # 초기값 설정
        df['End Balance'] = None
        df['Restart Balance'] = None
        df['Balance'] = None
        df['Can Rebalance'] = False

        df.at[0, 'End Balance'] = end_balances
        df.at[0, 'Restart Balance'] = start_balances
        df.at[0, 'Balance'] = init_balance

        for i in range(1, len(df)):
            # interval 값만큼 건너뛰고 리벨런싱
            # 각 종목의 증감률을 Result Close에서 계산
            result_close = df.at[i, 'Result Close']
            change_factors = [result_close[j]/ df.at[i-1,'Result Close'][j] for j in range(len(result_close))]

            end_balances = [round(df.at[i-1, 'Restart Balance'][j] * change_factors[j], 2) for j in range(len(change_factors))]
            df.at[i, 'End Balance'] = end_balances

            next_balance = sum(end_balances)
            df.at[i, 'Balance'] = int(next_balance)
            
            if i % interval == 0:
                restart_balance = [round((r/100)*next_balance,2) for r in ratios]
                df.at[i, 'Restart Balance'] = restart_balance
                df.at[i, 'Can Rebalance'] = True
            else:
                df.at[i, 'Restart Balance'] = end_balances
                

        return df
    

    #---------------------
    # TAA 전략
    #---------------------
    # 선택한 자산군 중 기간월 평균 수익률이 높은 N개를 추출
    # N개를 균등하게 배분
    # N개의 각 자산들의 10개월 이동평균보다 낮다면 현금으로 보유
    # 월 1회 리벨런싱
    def strategy_taa(symbol_dfs: dict, rank = 3, interval = [1,3,6,12], init_balance = 10000):
        
        symbol_dfs = AssetAllocation.get_profit_ratio_avg(symbol_dfs, interval)

        # 'Restart Asset' 생성 함수 (조건 추가)
        def get_top_assets(row):
            profit_ratios = row['Avg Profit Ratio']
            symbols = row['Symbol']
            result_close = row['Result Close']
            mma_10 = row['MMA_10']
            
            # 높은 순으로 인덱스 정렬 후 상위 rank개 추출
            top_indices = sorted(range(len(profit_ratios)), key=lambda i: profit_ratios[i], reverse=True)[:rank]
            
            # 조건에 맞는 symbol만 추가
            selected_assets = [
                symbols[i] for i in top_indices
                if result_close[i] >= mma_10[i]
            ]
            
            return selected_assets


        for symbol, df in symbol_dfs.items():
            symbol_dfs[symbol] = df.dropna(subset=['Avg Profit Ratio'])

        df = AssetAllocation.merge_to_dfs(symbol_dfs, ['Adj Close','Close','Dividends','Condition', 'Interval'])

        df['End Balance'] = None
        df['End Cash'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_top_assets, axis=1) 
        df['Restart Balance'] = None
        
        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']
            
            if idx == 0:
                balance = init_balance
                df.at[idx, 'Balance'] = balance
            else:
                prev_restart_balance = df.at[idx-1, 'Restart Balance']
                prev_restart_asset = df.at[idx-1, 'Restart Asset']
                
                end_balance = []
                end_cash = df.at[idx-1, 'Cash']
                for i in range(len(prev_restart_balance)):
                    symbol_idx = df.at[0, 'Symbol'].index(prev_restart_asset[i])
                    prev_close = df.at[idx-1,'Result Close'][symbol_idx]
                    curr_close = df.at[idx, 'Result Close'][symbol_idx]
                    change_rate = (curr_close - prev_close) / prev_close
                    symbol_val = int(prev_restart_balance[i] *(1+ change_rate))
                    end_balance.append(symbol_val)

                balance = sum([b for b in end_balance]) + end_cash

                df.at[idx, 'End Balance'] = end_balance
                df.at[idx, 'End Cash'] = end_cash
                df.at[idx, 'Balance'] = balance
                
            split_balance = int(balance / rank)
            restart_balance = []
            if len(restart_asset) > 0:
                for _ in  range(len(restart_asset)):
                    restart_balance.append(split_balance)
                    balance -= split_balance

            df.at[idx, 'Restart Balance'] = restart_balance
            df.at[idx, 'Cash'] = balance
                    
        return df
    


    # 자산계산
    def calc_balance(df, idx, init_balance):
        if idx == 0:
            balance = init_balance
            end_balance = []  # 첫 번째 행에는 개별 자산 정보가 없을 수도 있으므로 빈 리스트
        else:
            prev_restart_balance = df.at[idx-1, 'Restart Balance']
            prev_restart_asset = df.at[idx-1, 'Restart Asset']

            end_balance = []
            for i in range(len(prev_restart_balance)):
                symbol_idx = df.at[0, 'Symbol'].index(prev_restart_asset[i])
                prev_close = df.at[idx-1, 'Result Close'][symbol_idx]
                curr_close = df.at[idx, 'Result Close'][symbol_idx]

                change_rate = (curr_close - prev_close) / prev_close
                symbol_val = int(prev_restart_balance[i] * (1 + change_rate))
                end_balance.append(symbol_val)

            balance = sum(end_balance)

        return balance, end_balance


    #----------------------
    # 오리지널 듀얼모멘텀 전략
    #----------------------
    # SPY, EFA(선진국주식), AGG(미국채권) 투자
    # 매월, SPY, EFA, BIL(초단기채권)의 최근 12개월 수익률 계산
    # 수익률이 SPY > BIL 일 경우, SPY, EFA중 수익률 높은 ETF에 투자.
    # BIL < SPY일 경우, AGG에 투자

    def strategy_original_dual_momentum(symbol_dfs:dict, compare_symbols:list, init_balance = 10000):
        
        def get_selected_assets(row):
            symbols = row['Symbol']
            ratios = row['Avg Profit Ratio']

            ratio_dict = dict(zip(symbols, ratios))

            cmp1_ratio = ratio_dict[compare_symbols[0]]
            cmp2_ratio = ratio_dict[compare_symbols[1]]

            if cmp1_ratio > cmp2_ratio:
                selected = ['SPY' if cmp1_ratio >= ratio_dict['EFA'] else 'EFA']
            else:
                selected = ['AGG']
            
            return selected
        

        missing_symbols = [sym for sym in compare_symbols if sym not in symbol_dfs.keys()]

        if missing_symbols:
            print(f"다음 심볼들이 누락되었습니다 : {missing_symbols}")
        if len(compare_symbols) != 2:
            print("비교 심볼이 2개가 아닙니다")

        df =  AssetAllocation.get_profit_ratio_avg(symbol_dfs, [12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Avg Profit Ratio'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval'])
        
        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_selected_assets, axis=1) 
        df['Restart Balance'] = None


        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)

            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance

            split_balance = int(balance / len(restart_asset))
            restart_balance = []
            if len(restart_asset) > 0:
                for _ in  range(len(restart_asset)):
                    restart_balance.append(split_balance)
                    balance -= split_balance

            df.at[idx, 'Restart Balance'] = restart_balance


        return df
        

    #------------------
    # 종합 듀얼 모멘텀
    #------------------
    # 포트폴리오를 4개 파트로 나눔
    # 1. 주식 - SPY, ETF(해외주식)
    # 2. 채권 - LQD(회사채), HYD(미국 하이일드 채권)
    # 3. 부동산 - VNQ(부동산 리츠), REM(모기지 리츠)
    # 4. 불경기 - TLT, GLD
    # 각 파트별 1개씩 투자함 (25% 배분)
    # 최근 12개월 수익률을 비교해서 1개씩 선별함.
    # 근데, 파트의 ETF 수익 모두가 BIL 수익보다 낮으면, BIL로 리벨런싱    
    def strategy_composite_dual_momentum(symbol_dfs:dict, init_balance = 10000):
        

        def get_selected_assets(row):
            symbols = row['Symbol']
            profit_ratios = row['Avg Profit Ratio']
            
            # 종목과 인덱스를 매핑
            symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}
            
            # BIL 수익률
            bil_ratio = profit_ratios[symbol_to_index['BIL']] if 'BIL' in symbol_to_index else None

            if bil_ratio is None:
                return []  # BIL 없으면 처리 안 함

            # 자산군 정의
            asset_groups = {
                'stock': ['SPY', 'EFA'],
                'bond': ['LQD', 'HYG'],
                'real_estate': ['VNQ', 'REM'],
                'recession': ['TLT', 'GLD']
            }

            selected_assets = []

            for group, candidates in asset_groups.items():
                valid_candidates = [
                    (symbol, profit_ratios[symbol_to_index[symbol]])
                    for symbol in candidates
                    if symbol in symbol_to_index
                ]

                # 유효한 후보 없으면 해당 그룹 스킵
                if not valid_candidates:
                    continue

                # 후보 중 수익률이 가장 높은 ETF 찾기
                best_symbol, best_ratio = max(valid_candidates, key=lambda x: x[1])

                # 해당 그룹의 모든 ETF 수익률이 BIL보다 낮은지 확인
                all_lower_than_bil = all(ratio < bil_ratio for _, ratio in valid_candidates)

                if all_lower_than_bil:
                    selected_assets.append('BIL')
                else:
                    selected_assets.append(best_symbol)

            return selected_assets


        df =  AssetAllocation.get_profit_ratio_avg(symbol_dfs, [12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Avg Profit Ratio'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval'])
        
        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_selected_assets, axis=1) 
        df['Restart Balance'] = None


        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)

            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance

            split_balance = int(balance / len(restart_asset))
            restart_balance = []
            if len(restart_asset) > 0:
                for _ in  range(len(restart_asset)):
                    restart_balance.append(split_balance)
                    balance -= split_balance

            df.at[idx, 'Restart Balance'] = restart_balance
        
        
        return df
    


    #----------------    
    # PAA 전략
    #----------------
    # 안전자산 비중 설정
    # 12 ETF중 현재 가격이 12개월 단순이동평균보다 낮은 자산 수를 측정
    # 하락추세 ETF수에 따른 안전자산 비준은 다음과 같이 설정함
    # 0~6개까지 있으며, 6개이상일 경우에는 100퍼센트 안전자산 투자
    # 안전자산은 미국중기국체 IEF
    # 안전자산에 투자하지 않은 금액은 상대 모멘텀으로 6개 ETF에 분산투자 함.
    # 매월 말 각 ETF의 12개월 단순 이동평균을 계싼. (현재가격/12개월 이동평균) -1 이 가장 높은 6개의자산에 투자.
    def strategy_paa(symbol_dfs:dict, init_balance = 10000):
        
        def get_selected_assets(row):
            symbols = row['Symbol']
            prices = row['Result Close']
            mas = row['MMA_10']

            selected = []

            for symbol, price, ma in zip(symbols, prices, mas):
                if symbol == 'IEF':
                    continue  # IEF는 제외
                if price > ma:
                    score = (price / ma) - 1
                    selected.append((symbol, score))

            # 스코어 기준으로 내림차순 정렬
            selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)

            # 심볼만 리스트로 반환
            return [symbol for symbol, _ in selected_sorted]


        df =  AssetAllocation.get_profit_ratio_avg(symbol_dfs, [12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Avg Profit Ratio'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval'])
        
        safe_asset_ratio = [0, 0.16, 0.3333, 0.5, 0.6667, 0.8333, 1]

        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_selected_assets, axis=1) 
        df['Restart Balance'] = None
        df['Safe Asset Ratio'] = None


        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']
            safe_count = len(symbol_dfs) -1 -len(restart_asset)

            if len(safe_asset_ratio) <= safe_count: # 모두 안전자산 비율로
                safe_ratio = 1
            else:
                safe_ratio = safe_asset_ratio[safe_count]

            unsafe_ratio = 1 - safe_ratio

            if safe_ratio == 1:
                df.at[idx, 'Restart Asset'] = ['IEF']
            else:
                df.at[idx, 'Restart Asset'] = df.at[idx, 'Restart Asset'][:6] + ['IEF']

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)

            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance

            restart_balance = []
            if unsafe_ratio > 0:
                unsafe_balance = balance*unsafe_ratio
                for _ in range(6):
                    restart_balance.append(int(unsafe_balance/6))

            if safe_ratio > 0:
                restart_balance.append(int(balance*safe_ratio))
            else:
                restart_balance.append(0)

            df.at[idx, 'Restart Balance'] = restart_balance
            df.at[idx, 'Safe Asset Ratio'] = safe_ratio


        return df
    

    # 모멘텀 스코어 구하기

    def get_momentum_score(symbol_dfs:dict, intervals=[1,3,6]):

        result_dfs = {}
        max_interval = max(intervals)
        

        for symbol, df in symbol_dfs.items():
            
            interval_dfs ={}
            for interval in intervals:
                copy_df = df.copy()
                shift_df = copy_df.shift(interval)
                
                copy_df['Interval'] = interval

                profit_value = copy_df['Result Close'] - shift_df['Result Close']

                copy_df['Profit Value'] = profit_value
                copy_df['Profit Ratio'] = profit_value/shift_df['Result Close']

                interval_dfs[interval] = copy_df
                
            
            weighted_scores = [
                (df_interval['Profit Ratio'] * (max_interval / interval)).round(2)
                for interval, df_interval in interval_dfs.items()
            ]

            # 리스트 형태로 각 row의 interval 점수를 묶기
            momentum_score_list = [list(t) for t in zip(*[s.values for s in weighted_scores])]

            new_df = df.copy()
            new_df['Momentum Score'] = momentum_score_list
            new_df['Total Momentum Score'] = [sum(x) for x in momentum_score_list]
            new_df['Interval'] = str(intervals)

            result_dfs[symbol] = new_df


        return result_dfs
    



    #--------------
    # VAA 공격형
    #--------------
    # 공격형 자산 : SPY, EFA(선진국주식), EEM(개발도상국 주식), AGG(미국혼합 채권)
    # 안전 자산 : LQD(미국회사채), IEF(미국중기채), SHY(미국단기채)
    # 매월말 공격형, 안전자산 모멘텀 스코어 계산
    # 각 자산의 모멘텀 스코어 계산
    # 공격형 자산 4개 모두의 모멘텀 스코어가 0이상일 경우 포트폴리오 전체를 가장 모멘텀 스코어가 높은 공격형 자산에 투자
    # 공격형 자산 4개 모두의 모멘텀 스코어가 0이하일 경우 포트폴리오 전테를 가장 모멘텀 스코어가 낮은 안전자산에 투자
    def strategy_vaa_aggressive(symbol_dfs:dict, init_balance=10000):

        aggressive_assets = ['SPY', 'EFA', 'EEM', 'AGG']
        safe_assets = ['LQD', 'IEF', 'SHY']
        
        def get_selected_assets(row, top_n=1):
            symbols = row['Symbol']
            scores = row['Total Momentum Score']

            score_dict = dict(zip(symbols, scores))

            aggressive_assets = ['SPY', 'EFA', 'EEM', 'AGG']
            safe_assets = ['LQD', 'IEF', 'SHY']

            aggressive_scores = {sym: score_dict[sym] for sym in aggressive_assets}
            safe_scores = {sym: score_dict[sym] for sym in safe_assets}

            if all(score >= 0 for score in aggressive_scores.values()):
                # 공격형 자산 중 상위 top_n 선택
                selected = sorted(
                    aggressive_scores.items(), key=lambda x: x[1], reverse=True
                )[:top_n]
            else:
                # 안전 자산 중 상위 top_n 선택
                selected = sorted(
                    safe_scores.items(), key=lambda x: x[1], reverse=True
                )[:top_n]

            return [sym for sym, score in selected]
        

        df = AssetAllocation.get_momentum_score(symbol_dfs, [1,3,6,12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Total Momentum Score'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval', 'Momentum Score'])


        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_selected_assets, axis=1) 
        df['Restart Balance'] = None
        df['Restart Type'] = None

        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)

            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance

            split_balance = int(balance / len(restart_asset))
            restart_balance = []
            if len(restart_asset) > 0:
                for _ in  range(len(restart_asset)):
                    restart_balance.append(split_balance)
                    balance -= split_balance

            df.at[idx, 'Restart Balance'] = restart_balance


        return df
    

    # VAA 중도형
    # 공격형자산 : SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, GLD, DBC, HYG, LQD, TLT
    # 안전자산 : LQD, IEF, SHY
    # (LQD 중복인데, 이거 확인해봐야할 듯)
    # 공격형자산 중 모멘텀 스코어가 0 이하인 자산의 개수를 측정
    # 하락형 자산이 4개 이상일 경우에는 100% 안전자산 투자
    # 안전자산에 투자하지 않은 자금은 상대모멘텀을 적용해 5개의 ETF에 투자 - 상대모멘텀이 가장 높은 5개

    def strategy_vaa_balance(symbol_dfs:dict, init_balance=10000):

        aggressive_assets = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT']
        safe_assets = ['LQD', 'IEF', 'SHY']

        def get_selected_assets(row, top_n=None):  # top_n=None이면 전체 추출
            symbols = row['Symbol']
            scores = row['Total Momentum Score']

            score_dict = dict(zip(symbols, scores))

            aggressive_scores = {sym: score_dict[sym] for sym in aggressive_assets if score_dict[sym] >= 0}
            safe_scores = {sym: score_dict[sym] for sym in safe_assets}

            selected_scores = []
            if aggressive_scores:
                selected_scores = sorted(aggressive_scores.items(), key=lambda x: x[1], reverse=True)
            # else:
            #    selected_scores = sorted(safe_scores.items(), key=lambda x: x[1], reverse=True)

            # top_n이 None이면 전체, 아니면 top_n개만
            if top_n is not None and top_n > 0:
                selected_scores = selected_scores[:top_n]

            return [sym for sym, score in selected_scores]


        df = AssetAllocation.get_momentum_score(symbol_dfs, [1,3,6,12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Total Momentum Score'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval', 'Momentum Score'])


        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = df.apply(get_selected_assets, axis=1) 
        df['Restart Balance'] = None
        df['Restart Type'] = None


        safe_asset_ratio = [0, 0.25, 0.5, 0.75, 1]

        for idx in range(len(df)):
            restart_asset = df.at[idx, 'Restart Asset']
            safe_count = len(aggressive_assets) - len(restart_asset)
            
            if len(safe_asset_ratio) <= safe_count:
                safe_ratio = 1
            else:
                safe_ratio = safe_asset_ratio[safe_count]

            symbols = df.at[idx, 'Symbol']
            scores  = df.at[idx, 'Total Momentum Score']
            score_dict = dict(zip(symbols, scores))
            safe_scores = {sym: score_dict[sym] for sym in safe_assets}
            selected_scores = sorted(safe_scores.items(), key=lambda x: x[1], reverse=True)
            restart_safe_asset = [sym for sym, score in selected_scores[:1]]

            df.at[idx, 'Safe Ratio'] = safe_ratio

            if safe_ratio == 1:
                df.at[idx, 'Restart Asset'] = restart_safe_asset
            else:
                restart_asset = [unsafe_asset for unsafe_asset in restart_asset[:5]]
                
                if safe_ratio > 0:
                    df.at[idx,'Restart Asset'] = restart_asset + restart_safe_asset
                else:
                    df.at[idx,'Restart Asset'] = restart_asset

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)

            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance
        
            unsafe_balance = balance * (1-safe_ratio)
            safe_balance = balance * safe_ratio

            restart_balance = []
            if unsafe_balance > 0:
                for _ in range(len(restart_asset)):
                    restart_balance.append(int(unsafe_balance/len(restart_asset)))

            if safe_balance > 0:
                restart_balance.append(int(safe_balance))

            df.at[idx, 'Restart Balance'] = restart_balance


        return df
    


    def stragey_daa(symbol_dfs:dict, init_balance = 10000):
        aggressive_assets = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT']
        safe_assets = ['LQD', 'IEF', 'SHY']
        kanaria_assets = ['VWO','BND']

        df = AssetAllocation.get_momentum_score(symbol_dfs, [1,3,6,12])

        for symbol, df_symbol in df.items():
            df[symbol] = df_symbol.dropna(subset=['Total Momentum Score'])

        df = AssetAllocation.merge_to_dfs(df, ['Adj Close','Close','Dividends','Condition', 'Interval', 'Momentum Score'])

        df['End Balance'] = None
        df['Balance'] = None
        df['Restart Asset'] = None
        df['Restart Balance'] = None
        df['Safe Ratio'] = None


        for idx in range(len(df)):
            symbols = df.at[idx, 'Symbol']
            scores  = df.at[idx, 'Total Momentum Score']
            score_dict = dict(zip(symbols, scores))

            safe_score = 0
            if score_dict['VWO'] > 0:
                safe_score += 1
            if score_dict['BND'] > 0:
                safe_score += 1

            if safe_score == 2:
                safe_ratio = 1
            elif safe_score == 1:
                safe_ratio = 0.5
            else:
                safe_ratio = 0

            # 안전비율
            df.at[idx, 'Safe Ratio'] = safe_ratio

            # 안전자산 구하기
            if score_dict['VWO'] >= score_dict['BND']:
                higher_safe_symbol = 'VWO'
            else:
                higher_safe_symbol = 'BND'
            

            safe_assets = []
            unsafe_assets = []

            if safe_ratio > 0:
                safe_assets = [higher_safe_symbol]

            if safe_ratio < 1:
                aggressive_scores = {sym: score_dict[sym] for sym in aggressive_assets}
                selected_scores = []
                if aggressive_scores:
                    selected_scores = sorted(aggressive_scores.items(), key=lambda x: x[1], reverse=True)
                
                unsafe_assets = [sym for sym, score in selected_scores[:5]]


            # 리벨런싱 자산
            df.at[idx, 'Restart Asset'] = unsafe_assets + safe_assets

            balance, end_balance = AssetAllocation.calc_balance(df, idx, init_balance)
            
            df.at[idx, 'End Balance'] = end_balance
            df.at[idx, 'Balance'] = balance

            unsafe_balance = int(balance *(1-safe_ratio))
            safe_balance = balance - unsafe_balance

            restart_balance = []
            if len(unsafe_assets) > 0:
                for _ in range(len(unsafe_assets)):
                    restart_balance.append(int(unsafe_balance/len(unsafe_assets)))

            if len(safe_assets) > 0:
                for _ in range(len(safe_assets)):
                    restart_balance.append(int(safe_balance/len(safe_assets)))

            df.at[idx, 'Restart Balance'] = restart_balance

        return df


