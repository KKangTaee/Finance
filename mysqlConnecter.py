import pymysql
import pandas as pd

class MySQLConnector:
    def __init__(self):
        pass

    def connect(self, dbName):
        try:
            self.conn = pymysql.connect(
                    host='localhost',
                    user='root',
                    password='1234',
                    database=dbName,  # 데이터베이스 이름
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
        except Exception as e:
             print(f"⚠️ MySQL 연결 실패: {e}")
             self.conn = None


    def disconnect(self):
        if self.conn:
            self.conn.close()


    def changeDB(self, dbName):
        with self.conn.cursor() as cursor:
            cursor.execute(f"USE {dbName}")
        print(f"데이터베이스가 {dbName}(으)로 변경되었습니다.")


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
                    print(f"⚠️ 데이터가 없습니다. : {sql}")
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
    

    def commitToDB(self, sql):

        if self.conn is None:
            print("❌ 데이터베이스 연결이 설정되지 않음.")
            return pd.DataFrame()  # ✅ 빈 DataFrame 반환
        
        conn = self.conn

        try:
            with conn.cursor() as cursor:
                
                cursor.execute(sql)
                conn.commit()

        except pymysql.err.ProgrammingError as e:
             print(f"⚠️ SQL 문법 오류: {e}")

        except pymysql.err.OperationalError as e:
            print(f"⚠️ MySQL 연결 오류: {e}")

        except Exception as e:
            print(f"⚠️ 알 수 없는 오류 발생: {e}")
            return pd.DataFrame()  # ✅ 오류 발생 시 빈 DataFrame 반환