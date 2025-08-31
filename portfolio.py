import IPython
import IPython.display
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta
from assetAllocation import AssetAllocation
from IPython.display import display

import assetAllocation
import commonHelper
from db_financialStatement import DB_FinancialStatement


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
    

    # NCVA 전략
    # 아래의 조건에 맞는 종목 찾기
    #   1. 유동자산 - 총부채 > 시가총액
    #   2. 분기 수익률 > 0
    # * 1,2번 조건에 맞는 주식들 중에서 (유동자산 - 총부채) / 시가총액 비중이 가장 높은 주식 매수하기
    # * df_ncva 에 랭크 데이터가 들어옴.
    def nvca(start_date, end_date):
        df_ncva_rank = DB_FinancialStatement.get_ncva_rank_table()
        symbols = list(set(val for val in df_ncva_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_ncva_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_ncva_rank, date_dict)

        return df
    

    def super_value(start_date, end_date):
        df_ncva_rank = DB_FinancialStatement.get_super_value_rank_table()
        symbols = list(set(val for val in df_ncva_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_ncva_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_ncva_rank, date_dict)
        
        return df
    

    def new_magic(start_date, end_date):
        df_ncva_rank = DB_FinancialStatement.get_new_magic_rank_table()
        symbols = list(set(val for val in df_ncva_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_ncva_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_ncva_rank, date_dict)
        
        return df
    
    def f_score(start_date, end_date):
        df_rank = DB_FinancialStatement.get_f_score_rank_table()
        symbols = list(set(val for val in df_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_rank, date_dict)
        
        return df
    
    def relative_momentum(start_date, end_date):
        df_rank = DB_FinancialStatement.get_market_cap_rank_table()
        symbols = list(set(val for val in df_rank.values.ravel() if pd.notna(val)))

        quarter_list = df_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)
        dt_oldest = datetime.strptime(oldest, "%Y-%m-%d")
        dt_oldest = dt_oldest - relativedelta(years=1)
        oldest = dt_oldest.strftime("%Y-%m-%d")

        symbols_dfs = AssetAllocation.get_stock_data_with_ma(symbols, oldest, latest, [10], 'ma_month', True)
        symbols_dfs = AssetAllocation.filter_close_last_month(symbols_dfs)
        df = AssetAllocation.strategy_relative_momentum(symbols_dfs, df_rank)

        return df
    
    def magic_multify(start_date, end_date):
        df_rank = DB_FinancialStatement.get_ev_ebit_rank_table()
        symbols = list(set(val for val in df_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_rank, date_dict)
        
        return df
    

    def low_per(start_date, end_date):
        df_rank = DB_FinancialStatement.get_low_per_rank_table()
        symbols = list(set(val for val in df_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_rank, date_dict)
        
        return df
    

    def fama_high_return(start_date, end_date):
        df_rank = DB_FinancialStatement.get_fama_high_return_rank_table()
        df_rank = df_rank.head(20)
        symbols = list(set(val for val in df_rank.values.ravel() if pd.notna(val)))
        quarter_list = df_rank.columns.to_list()

        date_dict = commonHelper.get_date_dict_by_quarter_lazy(quarter_list)
        date_dict = commonHelper.get_trimmed_date_dict(date_dict, start_date, end_date)
        date_dict = commonHelper.adjust_start_data_dict_by_quarter(date_dict, quarter_list[0])
        oldest, latest = commonHelper.get_date_range_from_quarters(date_dict)

        df = AssetAllocation.get_stock_data_with_ma(symbols=symbols, start_date=oldest, end_date=latest, mas=[10], type='ma_month', use_db_stock=True)
        df = AssetAllocation.filter_close_last_month(df)
        df = AssetAllocation.strategy_quarter_rank(df, df_rank, date_dict)

        return df

    def income_mementum(start_date, end_date):
        df_rank = DB_FinancialStatement.get_income_momentum_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df
    
    def upgrade_nvca(start_date, end_date):
        df_rank = DB_FinancialStatement.get_upgrade_nvca_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df
    
    def upgrade_super_value(start_date, end_date):
        df_rank = DB_FinancialStatement.get_upgrade_super_value_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df
    
    def super_quality(start_date, end_date):
        df_rank = DB_FinancialStatement.get_super_quality_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df
    
    def fama_last_weapon(start_date, end_date):
        df_rank = DB_FinancialStatement.get_fama_last_weapon_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df
    
    def super_value_and_quality(start_date, end_date):
        df_rank = DB_FinancialStatement.get_super_value_and_quality_rank_table()
        df = AssetAllocation.calculation_quarter_rank(start_date, end_date, df_rank, 20)
        return df


    def show_portfolio_eft():
        start_date = '2013-12-01'
        end_date = '2025-08-31'
        allocations = [
            {'Harry Brown': Portfolio.harry_browne_permanent_portfolio(start_date, end_date)},
            {'Ray Dailo' : Portfolio.ray_dalio_all_seasons(start_date, end_date)},
            {'GAA3': Portfolio.gtaa3(start_date, end_date)},
            {'Original DM': Portfolio.original_dual_momentum(start_date, end_date)},
            {'Composite DM': Portfolio.composite_dual_momentum(start_date, end_date)},
            {'PAA' : Portfolio.paa(start_date, end_date)},
            {'VAA_A': Portfolio.vaa_aggressive(start_date, end_date)},
            {'VAA_B': Portfolio.vaa_balance(start_date, end_date)},
            {'DAA': Portfolio.daa(start_date, end_date)},
        ]

        Portfolio.plot_multiple_balances_over_time([list(d.values())[0] for d in allocations],
            [list(d.keys())[0] for d in allocations])
        
        for pair in allocations:
            name = list(pair.keys())[0]
            df = list(pair.values())[0]
            display(AssetAllocation.get_performance(df, name))


    def show_portfolio_stock():
        start_date = '2013-12-01'
        end_date = '2025-08-19'
        allocations = [
            #{'NVAC': Portfolio.nvca(start_date, end_date)},
            {'Super Value':Portfolio.super_value(start_date,end_date)},
            # {'New Magic':Portfolio.new_magic(start_date,end_date)},
            # {'F Score' : Portfolio.f_score(start_date, end_date)},
            # {'Relative Momentum' : Portfolio.relative_momentum(start_date, end_date)},
            # {'Magic Multify' : Portfolio.magic_multify(start_date,end_date)},
            # {'Low PER' : Portfolio.low_per(start_date, end_date)},
            {'Fama High Return' : Portfolio.fama_high_return(start_date, end_date)},
            # {'Income Momentum' : Portfolio.income_mementum(start_date, end_date)},
            # {'Upgrade NVAC' : Portfolio.upgrade_nvca(start_date, end_date)},
            {'Upgrade Super Value': Portfolio.upgrade_super_value(start_date, end_date)},
            # {'Super Quality' : Portfolio.super_quality(start_date, end_date)},
            {'Fama Last Weapon' : Portfolio.fama_last_weapon(start_date, end_date)},
            {'Super Value and Quality' : Portfolio.super_value_and_quality(start_date, end_date)},
        ]

        Portfolio.plot_multiple_balances_over_time([list(d.values())[0] for d in allocations],
            [list(d.keys())[0] for d in allocations])
        
        for pair in allocations:
            name = list(pair.keys())[0]
            df = list(pair.values())[0]
            display(AssetAllocation.get_performance(df, name))


    # OperatingCashFlow
    #

    # 신규주식발행(유상증자)
    # 자산회전율