import commonHelper as ch
from mysqlConnecter import MySQLConnector
import pymysql
import pandas as pd

# 재무재표 DB 데이터 조회
class DB_FinancialStatement(MySQLConnector):
    def __init__(self):
        super().__init__()

    def connect(self):
        super().connect(ch.DBName.DB_FINANCEIAL_STATEMENT)

    def disconnect(self):
        super().disconnect()


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



    # 유동부체      -- 단기 지급능력
    # 부채비율      -- 장기 부채 위험 관리
    # FCF         -- 여유 현금 확보
    # OFC         -- 본업으로부터의 지속성 현금창출
    # 순이익        -- 순이익 존재 (적자배제)
    # 영업이익률     -- 본업 이익률 (최소 10%)
    # EBITDA 마진  -- 기본 체력 (EBITDA 기준 마진)
    def getCompanyByRiskCheck(self):
                 
        sql = f"""
            SELECT * FROM Company
            WHERE 
                freeCashflow IS NOT NULL
                AND operatingCashflow IS NOT NULL
                AND netIncomeToCommon IS NOT NULL
                AND operatingMargins IS NOT NULL
                AND ebitdaMargins IS NOT NULL
                AND currentRatio IS NOT NULL
                AND debtToEquity IS NOT NULL
            """
        
        df = self.requestToDB(sql)

        filtered_df = df[
        (df['freeCashflow'] > 0) &
        (df['operatingCashflow'] > 0) &
        (df['netIncomeToCommon'] > 0) &
        (df['operatingMargins'] > 0.1) &
        (df['ebitdaMargins'] > 0.1)
        ].copy()

        final_rows = []

        for _, row in filtered_df.iterrows():
            sector = row['sector']
            debt = row['debtToEquity']
            current = row['currentRatio']

            include = False

            # 업종별 조건 (영어 기준)
            if sector in ['Financial Services', 'Real Estate']:
                if debt <= 2.0 and current >= 1.0:
                    include = True
            elif sector in ['Industrials', 'Consumer Defensive', 'Consumer Cyclical', 'Basic Materials', 'Energy']:
                if debt <= 1.0 and current >= 1.5:
                    include = True
            elif sector in ['Technology', 'Healthcare', 'Communication Services']:
                if debt <= 1.5 and current >= 1.2:
                    include = True
            elif sector in ['Utilities']:
                if debt <= 1.5 and current >= 1.0:
                    include = True
            else:
                # 정의되지 않은 업종: 보수적 기준
                if debt <= 1.0 and current >= 1.5:
                    include = True

            if include:
                final_rows.append(row)

        # 최종 필터링된 DataFrame 반환
        result_df = pd.DataFrame(final_rows)

        return result_df


    # trailingPE, forwardPE 로 저평가된 기업 추출
    # 각 섹터의 중앙값의 1로 설정하고 0.5 ~ 1.5 구간을 추출
    # trailingPE > forwardPE 이고 forwardPE > 0 추출
    def getCompanyByComparePE(self):
        # 1️⃣ 섹터별 중앙값 PE 가져오기
        sector_avg_df = self.getPESectorMedian()
        print("✅ 섹터별 PE 중앙값 데이터 가져오기 완료!")

        # ⚙️ 2️⃣ 섹터별 맞춤 배수 설정 (기본은 0.5 ~ 1.5)
        sector_pe_multipliers = {
            'Technology': (0, 2),
            'Healthcare': (0, 2),
            'Industrials': (0.5, 1.5),
            'Consumer Cyclical': (0.5, 1.5),
            'Energy': (0.5, 1.5),
            'Consumer Defensive': (0.5, 1.5),
            'Basic Materials': (0.5, 1.5),
            'Financial Services': (0.75, 1.25),
            'Real Estate': (0.75, 1.25),
            'Utilities': (0.75, 1.25)
        }

        # 성장주 발굴 (리스크 감수)	0 ~ 2 (테크, 헬스케어 등)
        # 중립적 가치/성장 혼합	0.5 ~ 1.5 (산업재, 소비재, 에너지 등)
        # 배당+안정 투자 (리스크 최소화)	0.75 ~ 1.25 (리츠, 금융, 유틸리티 등)

        # 3️⃣ 배수 적용해서 섹터별 범위 설정
        sector_avg_df['trailingPE_min'] = sector_avg_df.apply(
            lambda row: row['trailingPE'] * sector_pe_multipliers.get(row['sector'], (0.5, 1.5))[0], axis=1)
        sector_avg_df['trailingPE_max'] = sector_avg_df.apply(
            lambda row: row['trailingPE'] * sector_pe_multipliers.get(row['sector'], (0.5, 1.5))[1], axis=1)
        sector_avg_df['forwardPE_min'] = sector_avg_df.apply(
            lambda row: row['forwardPE'] * sector_pe_multipliers.get(row['sector'], (0.5, 1.5))[0], axis=1)
        sector_avg_df['forwardPE_max'] = sector_avg_df.apply(
            lambda row: row['forwardPE'] * sector_pe_multipliers.get(row['sector'], (0.5, 1.5))[1], axis=1)

        print("✅ 섹터별 맞춤 PE 범위 설정 완료!")

        # 4️⃣ Company 테이블 데이터 조회
        query_all_company = """
            SELECT symbol, name, sector, industry, trailingPE, forwardPE
            FROM Company
            WHERE trailingPE IS NOT NULL
            AND forwardPE IS NOT NULL
            AND sector IS NOT NULL
        """
        company_df = self.requestToDB(query_all_company)
        print("✅ Company 테이블 전체 데이터 조회 완료!")

        # 5️⃣ 섹터별 구간 필터링
        filtered_rows = []

        for _, sector_row in sector_avg_df.iterrows():
            sector_name = sector_row['sector']
            tpe_min, tpe_max = sector_row['trailingPE_min'], sector_row['trailingPE_max']
            fpe_min, fpe_max = sector_row['forwardPE_min'], sector_row['forwardPE_max']

            # 조건에 맞는 row 필터링
            sector_filtered = company_df[
                (company_df['sector'] == sector_name) &
                (company_df['trailingPE'] >= tpe_min) & (company_df['trailingPE'] <= tpe_max) &
                (company_df['forwardPE'] >= fpe_min) & (company_df['forwardPE'] <= fpe_max)
            ]
            filtered_rows.append(sector_filtered)

        # 6️⃣ 모든 필터링된 row 합치기
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        print("✅ 섹터 구간 조건 필터링 완료!")

        # 7️⃣ 추가 조건: trailingPE > forwardPE and forwardPE > 0
        filtered_df = filtered_df[
            (filtered_df['trailingPE'] > filtered_df['forwardPE']) &
            (filtered_df['forwardPE'] > 0)
        ].copy()
        print("✅ 추가 조건 (trailingPE > forwardPE, forwardPE > 0) 필터링 완료!")

        # 8️⃣ dropRatioPE (하락률 %) 컬럼 추가
        filtered_df['dropRatioPE'] = -((filtered_df['trailingPE'] - filtered_df['forwardPE']) / filtered_df['trailingPE']) * 100
        print("✅ dropRatioPE(하락률 %) 컬럼 추가 완료!")

        # 9️⃣ 하락률 기준 정렬
        filtered_df = filtered_df.sort_values(by='dropRatioPE', ascending=True).reset_index(drop=True)
        print("✅ 하락률 순 정렬 완료!")

        return filtered_df
    

    # 각 섹터별 trailingPE, forwardPE 의 중앙값을 구함
    # PER 같은 벨류에이션 지표에서는 중앙값보다 평균값을 사용하는 것이 더 효율적임
    def getPESectorMedian(self):
        query= """
             SELECT industry, sector, trailingPE, forwardPE FROM Company
        """
        df = self.requestToDB(query)

        result_df = df.groupby(['sector'], as_index=False)[["trailingPE", "forwardPE"]].median()

        return result_df



    def getCompanyByIndustry(self, industry):
        query = f"""
            SELECT * FROM Company
            WHERE industry = '{industry}'

        """
        df = self.requestToDB(query)

        return df

    