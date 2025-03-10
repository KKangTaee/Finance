import pymysql
import pandas as pd

class MySQLConnector:
    def __init__(self):
        pass

    def connect(self):
        try:
            self.conn = pymysql.connect(
                    host='localhost',
                    user='root',
                    password='1234',
                    database='FinancialStatement',  # 데이터베이스 이름
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
        except Exception as e:
             print(f"⚠️ MySQL 연결 실패: {e}")
             self.conn = None


    def disconnect(self):
        if self.conn:
            self.conn.close()


    def requestToDB(self, sql, columns = None):

        if self.conn is None:
            print("❌ 데이터베이스 연결이 설정되지 않음.")
            return pd.DataFrame()  # ✅ 빈 DataFrame 반환
        
        conn = self.conn

        try:
            with conn.cursor() as cursor:
                
                cursor.execute(sql)
                result = cursor.fetchall()

                # ✅ DataFrame 변환
                df = pd.DataFrame(result)

                if df.empty:
                    print("⚠️ 데이터가 없습니다.")
                    return df

                if columns is not None:
                    # ✅ DataFrame 변환 (딕셔너리 형태의 결과 처리)
                    df = pd.DataFrame(result, columns=columns)

                return df

        except pymysql.err.ProgrammingError as e:
             print(f"⚠️ SQL 문법 오류: {e}")

        except pymysql.err.OperationalError as e:
            print(f"⚠️ MySQL 연결 오류: {e}")

        except Exception as e:
            print(f"⚠️ 알 수 없는 오류 발생: {e}")
            return pd.DataFrame()  # ✅ 오류 발생 시 빈 DataFrame 반환


    # 심볼리스트 가져오기
    def getSymbolList(self):
        symbols = []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT symbol FROM Company")
                result = cursor.fetchall()

                symbols =  [row['symbol'] for row in result]

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

        return symbols


    # 시총 높은 순으로 N개의 기업을 조회
    def getCompanyByMarketCap(self, count):
        if self.conn is None:
            print("❌ 데이터베이스 연결이 설정되지 않음.")
            return pd.DataFrame()  # ✅ 빈 DataFrame 반환
    
        conn = self.conn    
        
        try:
            with conn.cursor() as cursor:
                sql = f"""
                    SELECT symbol, name, marketCap, industry
                    FROM Company
                    ORDER BY marketCap DESC
                    LIMIT {count};
                """
                cursor.execute(sql)
                result = cursor.fetchall()  # ✅ 쿼리 결과 가져오기

                # ✅ DataFrame 변환
                df = pd.DataFrame(result)

                return df

        except pymysql.err.ProgrammingError as e:
            print(f"⚠️ SQL 문법 오류: {e}")

        except pymysql.err.OperationalError as e:
            print(f"⚠️ MySQL 연결 오류: {e}")

        except Exception as e:
            print(f"⚠️ 알 수 없는 오류 발생: {e}")

            return pd.DataFrame()  # ✅ 오류 발생 시 빈 DataFrame 반환
        

    # 모든 산업의 개수 조회
    def getIndustryCountByAll(self):

        if self.conn is None:
            print("❌ 데이터베이스 연결이 설정되지 않음.")
            return pd.DataFrame()  # ✅ 빈 DataFrame 반환

        conn = self.conn

        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT industry, COUNT(*) AS count
                    FROM Company
                    GROUP BY industry
                    ORDER BY count DESC;
                """
                cursor.execute(sql)
                result = cursor.fetchall()  # ✅ 쿼리 실행 후 결과 가져오기

                # ✅ DataFrame 변환
                df = pd.DataFrame(result)

                return df

        except pymysql.err.ProgrammingError as e:
            print(f"⚠️ SQL 문법 오류: {e}")

        except pymysql.err.OperationalError as e:
            print(f"⚠️ MySQL 연결 오류: {e}")

        except Exception as e:
            print(f"⚠️ 알 수 없는 오류 발생: {e}")
            return pd.DataFrame()  # ✅ 오류 발생 시 빈 DataFrame 반환
        

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


    # 유동부체, 부체비율 체크 (> 1.5, <= 1) 
    # FCF, OCF > 0
    # 순이익 > 0 
    def getCompanyByCurrentRatioAndDebtToEquity(self, count):
                 
        sql = f"""
            SELECT * FROM Company
            WHERE currentRatio >= 1.5 
            AND debtToEquity <= 1
            AND freeCashflow > 0
            AND operatingCashflow > 0
            AND netIncomeToCommon > 0
            AND operatingMargins > 0.1
            AND ebitdaMargins > 0.1
            AND state IS NOT NULL
            AND industry <> 'Biotechnology'
            ORDER BY fiftyTwoWeekChangePercent ASC
            LIMIT {count};
            """
        columns = ['symbol', 
                   'name', 
                   'netIncomeToCommon', 
                   'operatingMargins', 
                   'ebitdaMargins', 
                   'fiftyTwoWeekChangePercent', 
                   'regularMarketPrice',
                   'targetMedianPrice', 
                   'industry',]
        
        return self.requestToDB(sql, columns)


    def getPEByIndustrySector(self):
        query = """
            SELECT industry, sector, trailingPE, forwardPE FROM Company
            WHERE industry IS NOT NULL AND sector IS NOT NULL
            AND trailingPE IS NOT NULL AND forwardPE IS NOT NULL
            """
        
        df = self.requestToDB(query)

         # 2️⃣ industry와 sector 기준으로 그룹화하여 평균값 계산
        result_df = df.groupby(["industry", "sector"], as_index=False)[["trailingPE", "forwardPE"]].mean()

        # 4️⃣ trailingPE와 forwardPE 기준으로 정렬 (낮은 값부터)
        # result_df = result_df.sort_values(by=["trailingPE", "forwardPE"], ascending=[True, True])

         # 5️⃣ trailingPE > forwardPE 조건 필터링 후, 완전히 새로운 DataFrame 생성 (경고 방지)
        filtered_df = result_df[result_df["trailingPE"] > result_df["forwardPE"]].copy()

        # 6️⃣ 하락률(%) 계산 컬럼 추가
        filtered_df["drop_percent"] = ((filtered_df["trailingPE"] - filtered_df["forwardPE"]) / filtered_df["trailingPE"]) * 100
        
        # 6️⃣ drop_percent 기준으로 내림차순 정렬 (가장 하락률이 큰 순서)
        filtered_df = filtered_df.sort_values(by="drop_percent", ascending=False)

        print("✅ industry & sector별 PER 계산 완료!")

        return filtered_df
    