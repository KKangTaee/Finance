import IPython
import IPython.display
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta
from assetAllocation import AssetAllocation
from IPython.display import display




class Portfolio:
    def plot_balance_over_time(df):
        """
        DataFrame의 'Date'를 X축, 'Balance'를 Y축으로 해서 라인 차트를 그립니다.
        
        Parameters:
            df (pd.DataFrame): 'Date'와 'Balance' 컬럼이 포함된 DataFrame
        """
        # 날짜 정렬 및 변환
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        
        # Balance 컬럼이 있는지 확인
        if 'Balance' not in df.columns:
            raise ValueError("'Balance' 컬럼이 DataFrame에 없습니다.")
        
        # 차트 그리기
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Balance'], marker='o', linestyle='-')
        plt.title('Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_multiple_balances_over_time(df_list, labels=None):
        """
        여러 개의 DataFrame을 한 그래프에 Balance Over Time으로 시각화합니다.
        
        Parameters:
            df_list (list of pd.DataFrame): 'Date'와 'Balance' 컬럼이 포함된 DataFrame들의 리스트
            labels (list of str): 각 데이터프레임에 대한 라벨 (범례용). None이면 숫자 인덱스 사용
        """
        plt.figure(figsize=(12, 6))

        for i, df in enumerate(df_list):
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)

            label = labels[i] if labels and i < len(labels) else f"Portfolio {i+1}"
            plt.plot(df['Date'], df['Balance'], marker='o', markersize=2,  linestyle='-', label=label)

        plt.title('Balance Over Time - Multiple Portfolios')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 해리브라운의 영구포트폴리오
    def harry_browne_permanent_portfolio(start_date, end_date):
        df = AssetAllocation.get_stock_data_with_ma(symbols=['VTI','TLT','BIL','GLD'], start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_evenly(df, ratios=[25,25,25,25], interval=12)
        return df

    # 레이달리오 올시즌 포트폴리오
    def ray_dalio_all_seasons(start_date, end_date):
        df = AssetAllocation.get_stock_data_with_ma(symbols=['VTI','TLT','IEF','GSG','GLD'], start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_evenly(df, ratios=[30,40,15,7.5, 7.5], interval=12)
        return df

    # 공격형 TAA전략
    def gtaa3(start_date, end_date):
        # symbols = ['SPY','IWD','IWM', 'IWN','MTUM','EFA','TLT','IEF','LQD','DBC','VNQ','BWX','GLD']
        symbols = ['SPY','IWM','MTUM','EFA','TLT','IEF','LQD','DBC','VNQ','GLD']
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        start_date = date_obj - relativedelta(years=1)
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_taa(df, 3, [1,3,6,12])
        return df


    def original_dual_momentum(start_date, end_date):
        symbols = ['SPY','EFA','AGG','BIL']
        compare_symbols = ['SPY','EFA']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_original_dual_momentum(df, compare_symbols)
        return df


    def composite_dual_momentum(start_date, end_date):
        symbols = ['SPY','EFA','LQD','HYG', 'VNQ', 'REM','TLT','GLD', 'BIL']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_composite_dual_momentum(df)
        return df


    def paa(start_date, end_date):
        symbols = ['SPY','QQQ','IWM','VGK','EWJ','EEM','VNQ','GLD','DBC','HYG','LQD','TLT'] + ['IEF']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_paa(df)
        return df

    def vaa_aggressive(start_date, end_date):
        symbols = ['SPY', 'EFA', 'EEM', 'AGG'] + ['LQD', 'IEF', 'SHY']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_vaa_aggressive(df)
        return df


    def vaa_balance(start_date, end_date):
        symbols = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT'] + ['LQD', 'IEF', 'SHY']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_vaa_balance(df)
        return df


    def daa(start_date, end_date):
        symbols = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT'] + ['LQD', 'IEF', 'SHY'] + ['VWO','BND']
        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=start_date, end_date=end_date, mas=[10], type='ma_month')
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.stragey_daa(df)
        return df



    def show_portfolio():
        start_date = '2013-12-01'
        end_date = '2025-04-21'
        allocations = [
            {'Harry Brown': Portfolio.harry_browne_permanent_portfolio(start_date, end_date)},
            #{'Ray Dailo' : Portfolio.ray_dalio_all_seasons(start_date, end_date)},
            {'GAA3': Portfolio.gtaa3(start_date, end_date)},
            #{'Original DM': Portfolio.original_dual_momentum(start_date, end_date)},
            #{'Composite DM': Portfolio.composite_dual_momentum(start_date, end_date)},
            #{'PAA' : Portfolio.paa(start_date, end_date)},
            #{'VAA_A': Portfolio.vaa_aggressive(start_date, end_date)},
            #{'VAA_B': Portfolio.vaa_balance(start_date, end_date)},
            #{'DAA': Portfolio.daa(start_date, end_date)},
        ]

        Portfolio.plot_multiple_balances_over_time([list(d.values())[0] for d in allocations],
            [list(d.keys())[0] for d in allocations])
        
        for pair in allocations:
            name = list(pair.keys())[0]
            df = list(pair.values())[0]
            display(AssetAllocation.get_performance(df, name))