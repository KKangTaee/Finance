import yfinance as yf
import commonHelper as ch
from db_financialStatement import DB_FinancialStatement
from db_stock import DB_Stock

class YFinanceInfo:
    def __init__(self):
        pass

    # 기업 설정
    def setCompany(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

        # 1차 fast_info로 빠르게 체크
        fast_info = self.stock.fast_info
        if not fast_info or 'lastPrice' not in fast_info or fast_info['lastPrice'] is None:
            # 2차 info로 좀 더 정확하게 재확인
            info = self.stock.info
            if not info or 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                print(f"🚨 {ticker}: 유효하지 않은 티커입니다. 데이터 없음!")
                self.stock = None  # 무효화
                return False  # 실패 반환
        # print(f"✅ {ticker}: 데이터 확인 완료!")
        return True  # 성공 반환

    # 기업 데이터 유효한지 체크
    def isValidCompany(self, ticker):
        stock = yf.Ticker(ticker)
        return not stock.history(period ="1d").empty
    
    # 섹터 구하기
    def getSector(self):
        return self.stock.info.get("sector", "정보없음")
    
    # 산업 구하기
    def getIndustry(self):
        return self.stock.info.get("industry", "정보없음")
    
    # 재무재표 데이터 조회
    def getFsData(self, type, dateType):
        if type == ch.EFinancialStatementType.INCOME_STATEMENT:
            if dateType == ch.EDateType.YEAR:
                return self.stock.financials
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_financials
        
        elif type == ch.EFinancialStatementType.BALANCE_SHEET:
            if dateType == ch.EDateType.YEAR:
                return self.stock.balance_sheet
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_balance_sheet
            
        elif type == ch.EFinancialStatementType.CASH_FLOW:
            if dateType == ch.EDateType.YEAR:
                return self.stock.cash_flow
            elif dateType == ch.EDateType.QUARTER:
                return self.stock.quarterly_cash_flow