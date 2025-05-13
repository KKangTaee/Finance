import pymysql
import pandas as pd

class MySQLConnector:
    def __init__(self):
        pass

    def __enter__(self):
        self.connect()
        return self  # 연결된 인스턴스를 리턴

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

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
            self.dbName = dbName
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
        

    def rename_columns_to_capitalize(self, table_names):
        try:
            with self.conn.cursor() as cursor:
                for table_name in table_names:
                    print(f"\n[처리 중] 테이블: {table_name}")

                    # 1. 컬럼 이름 가져오기
                    cursor.execute(f"""
                        SELECT COLUMN_NAME
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = %s
                            AND TABLE_NAME = %s;
                    """, (self.dbName, table_name))
                    
                    result = cursor.fetchall()
                    columns = [row['COLUMN_NAME'] for row in result]

                    # 2. ALTER 쿼리 생성 및 실행
                    for col in columns:
                        new_col = col[0].upper() + col[1:]
                        if col != new_col:
                            alter_query = f"""
                                ALTER TABLE `{table_name}` 
                                RENAME COLUMN `{col}` TO `{new_col}`;
                            """
                            print(f"  실행: {alter_query.strip()}")
                            cursor.execute(alter_query)

            # 전체 커밋
            self.conn.commit()
            print("\n✅ 모든 테이블 컬럼 이름 변경 완료!")

        except Exception as e:
            print(f"❗에러 발생: {e}")
            self.conn.rollback()