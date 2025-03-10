import yfinance as yf
import pandas as pd
import commonHelper as ch

class FinancialStatement:
    def __init__(self):
        pass

    def setCompany(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

        # 주가 데이터가 없는 경우 예외 처리
        if self.stock.history(period="1d").empty:
            print(f"🚨 {ticker} 데이터 없음!")
            self.stock = None  # 유효하지

    def isValidCompany(self, ticker):
        stock = yf.Ticker(ticker)
        return not stock.history(period ="1d").empty

    def getIncomeStatement(self):
        selected_income_statement = [
            "Total Revenue", # 매출
            "Operating Revenue", # 매출
            "Gross Profit", # 매출 총이익
            "Operating Income", # 영업이익
            "Pretax Income", # 법인세 차감전 순이익
            "Net Income" # 당기순이익
            ]
            # 재무제표 데이터 가져오기
        income_statement = self.stock.financials # 손익계산서

        # 필요한 데이터만 선택하여 새로운 DataFrame 생성
        df_income_statement = income_statement.loc[selected_income_statement]
        return df_income_statement
    
    def getBalanceSheet(self):
        selected_balance_sheet =[
            'Treasury Shares Number', # 자기주식 수량 (이 값이 증가하면 자사주 매입)
            'Common Stock Equity', # 보통주 자본 (보통주나, 주주자본이 감소했다면 자사주 매입)
            'Stockholders Equity', # 주주 자본 (자기자본)
            'Total Assets', # 자산
            'Current Assets', # 유동자산
            'Total Non Current Assets', #비유동자산
            'Total Liabilities Net Minority Interest', # 총 부채
            'Current Liabilities', # 유동부채
            'Total Non Current Liabilities Net Minority Interest', # 비유동부채
            'Capital Stock', # 자본금
            'Retained Earnings', # 이익잉여금 (사업하면서 벌어들인 돈 중에 배당하지 않고 남겨둔 돈) (이게 음수일수도 있음. 자사주를 매입하면)
            ]   
        # 재무제표 데이터 가져오기
        balance_sheet = self.stock.balance_sheet

        # 선택한 항목만 필터링
        selected_balance_sheet = [key for key in selected_balance_sheet if key in balance_sheet.index]

        try:
            df_balance_sheet = balance_sheet.loc[selected_balance_sheet]
        except Exception as e:
            print(f"🚨 재무상태표 데이터 오류: {e}")
            return None

        # 유동비율 & 부채비율 계산
        additional_data = {}

        for year in balance_sheet.columns:
            try:
                # 유동비율 계산
                current_assets = balance_sheet.loc["Current Assets", year] if "Current Assets" in balance_sheet.index else None
                current_liabilities = balance_sheet.loc["Current Liabilities", year] if "Current Liabilities" in balance_sheet.index else None
                current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities else None

                # 부채비율 계산
                total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", year] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None
                stockholders_equity = balance_sheet.loc["Stockholders Equity", year] if "Stockholders Equity" in balance_sheet.index else None
                debt_to_equity_ratio = (total_liabilities / stockholders_equity) if total_liabilities and stockholders_equity and stockholders_equity > 0 else None

                # 딕셔너리에 추가
                additional_data[year] = [current_ratio, debt_to_equity_ratio]

            except Exception as e:
                print(f"🚨 {year} 데이터 오류: {e}")

        # 유동비율 & 부채비율 데이터프레임 생성
        df_ratios = pd.DataFrame(additional_data, index=["Current Ratio", "Debt-to-Equity Ratio"])

        # 기존 데이터프레임에 추가
        df_balance_sheet = pd.concat([df_balance_sheet, df_ratios])

        return df_balance_sheet

    def getCashFlow(self):
        selected_cash_flow = [
            'Free Cash Flow', # 잉여현금흐름 - 기업이 자유롭게 사용할 수 있는 현금
            'Operating Cash Flow', # 영업현금흐름 - 기업이 핵심 영업 활동을 통해 창출한 현금
            'Capital Expenditure', # 자본지출 - 설비, 공장, 기계, 기술 개발등을 위해 투자한 금액 (FCF를 구할때 차감해야함) 
            'Depreciation And Amortization', # 감가상각비 - 유형자산/무형자산의 가치 감소 비용 (현금지출은 아니지만 FCF에 영향을 줌)
            'Repurchase Of Capital Stock',# 자사주 매입
            'Common Stock Payments', # 보통주 지불
            'Net Common Stock Issuance', # 순 보통주 발행 (만약, 음수값이면 자사주를 매입한 것)
            'Common Stock Issuance', # 보통주 발행 (새로운 주식을 발행해서 자금 조달)
            'Cash Dividends Paid' # 현금 배당급 지급
            ]
        
        cash_flow = self.stock.cashflow # 
        selected_cash_flow = [key for key in selected_cash_flow if key in cash_flow.index]

        df_cash_flow = None
        
        try:
             df_cash_flow = cash_flow.loc[selected_cash_flow]
        except Exception as e:
            print(f" 데이터 오류: {e}")

        return df_cash_flow
    
    def getPBR_PER_PCR(self):
        stock = self.stock

        current_price = stock.history(period="1d")['Close'].iloc[-1]

        # 재무제표 데이터 가져오기 (년도별)
        balance_sheet = stock.balance_sheet  # 재무상태표
        cash_flow = stock.cashflow  # 현금흐름표
        income_statement = stock.financials  # 손익계산서

        # 주식수 가져오기
        shares_outstanding = stock.info.get('sharesOutstanding', None)

        # 년도별 데이터 저장할 딕셔너리
        data = {}

        # PBR, PCR, PER 계산
        for year in balance_sheet.columns:
            try:
                # 주당 순자산 (BVPS)
                total_equity = balance_sheet.loc["Total Equity Gross Minority Interest", year] # 자본총계
                bvps = total_equity / shares_outstanding if shares_outstanding else None

                # 주당 현금흐름 (CFPS)
                operating_cash_flow = cash_flow.loc["Operating Cash Flow", year] # 영업현금흐름
                cfps = operating_cash_flow / shares_outstanding if shares_outstanding else None

                # 주당 순이익 (EPS)
                net_income = income_statement.loc["Net Income", year]
                eps = net_income / shares_outstanding if shares_outstanding else None

                # 비율 계산
                pbr = current_price / bvps if bvps else None
                pcr = current_price / cfps if cfps else None
                per = current_price / eps if eps else None

                # 데이터 저장
                data[year] = [pbr, pcr, per]

            except Exception as e:
                print(f"{year} 데이터 오류: {e}")

        # 데이터프레임 생성
        df_ratios = pd.DataFrame(data, index=["PBR", "PCR", "PER"]).T
        return df_ratios.T
            
    def get52WeekHighPrice(self):
        if not self.stock or not self.stock.info:
            print(f"🚨 {self.ticker}의 정보가 없습니다.")
            return None  # 오류 방지

        return self.stock.info.get('fiftyTwoWeekHigh', None)

    def getCurrPrice(self):
        if not self.stock:
            print(f"🚨 {self.ticker}의 주가 정보 없음!")
            return None

        history_data = self.stock.history(period="1d")
        if history_data.empty:
            print(f"🚨 {self.ticker}의 주가 데이터 없음!")
            return None

        return history_data['Close'].iloc[-1]

    def getDropPercent(self):
        curr = self.getCurrPrice()
        high = self.get52WeekHighPrice()

        if curr is None or high is None:  # 값이 없으면 예외 처리
            print(f"🚨 {self.ticker}의 가격 데이터가 없습니다.")
            return None

        percent = ((curr - high) / high) * 100
        return round(percent, 1)
    
    def getSector(self):
        return self.stock.info.get("sector", "정보없음")
    
    def getIndustry(self):
        return self.stock.info.get("industry", "정보없음")
    
    # 거래소 정보 가져오기
    def getExchange(self):
        return

    def getFsData(self, type, dateType):
        if type == ch.FSType.INCOME_STATEMENT:
            if dateType == ch.FSDateType.YEAR:
                return self.stock.financials
            elif dateType == ch.FSDateType.QUARTER:
                return self.stock.quarterly_financials
        
        elif type == ch.FSType.BALANCE_SHEET:
            if dateType == ch.FSDateType.YEAR:
                return self.stock.balance_sheet
            elif dateType == ch.FSDateType.QUARTER:
                return self.stock.quarterly_balance_sheet
            
        elif type == ch.FSType.CASH_FLOW:
            if dateType == ch.FSDateType.YEAR:
                return self.stock.cash_flow
            elif dateType == ch.FSDateType.QUARTER:
                return self.stock.quarterly_cash_flow

    
    
# 주가수익비율 (PER)
# 주가가 회사의 수익에 비해 얼마나 높은지 보여주는 것
# 예를들어, 주식가격이 100원이고 주당순이익(EPS)이 5원이라면 PER의 20임 이는 주식가격이 순이익의 20배에 해당한다는 것
# P/E가 높으면 주식이 과대평가가 될 수 있고, 낮으면 저평가된 기업이라고 볼 수 있음.

# 자산순이익률 (ROA) - Asset
# 회사의 자산을 이용해 얼마나 효율적으로 이익을 창출하는지를 나타내는 것
# 예를들어, 자산이 1000만원이고 순이익이 100만원이면 ROA는 10%임. 이 수치가 높으면 자산을 잘 활용하고 있다는 뜻

# 자기자본이익률 (ROE) - Equity
# 회사가 자기자본을 바탕으로 얼마만큼 이익을 올렸는지를 나타내는 것
# 예를들어, 자기자본이 500만원이고 순이익이 50만원이면 ROE는 10%임. 높은 ROE는 회사가 자기자본을 효율적으로 사용하고 있다는 것

# 매출총이익 (Gross Profit)
# 매출에서 매출원가를 뺀 금액.
# 예를들어, 매출이 1000만원이고, 매출원가가 600만원이면 매출총이익은 400만원임. 이 금액은 회사가 기본적인 생산활동으로 벌어들인 이익을 보여줌

# 주식순자산비율 (PBR)
# 회사의 시장가치(주식가격)와 장부가치(자산과 부채차이)의 비율을 나타내는 것.
# 예를들어, 회사의 자산가치가 500만원이고, 주식의 시가총액이 1000만원이면 PBR의 비율은 2임
# 이 비율이 1보다 높으면 시장에서 회사의 가치가 장부가치를 초과한다고 볼 수 있음.

# EBITDA
# 이자, 세금, 감가상각비를 제외한 회사의 이익을 나타내는 것
# 예를들어, 이자비용 50만원, 세금 30만원, 감가상각비 20만원을 제외한 이익이 200만원이라면 EBITDA는 200만원임.
# 회사의 기본적인 수익 창출 능력을 평가할때 유용한 지표임.

# EV/EBITDA
# 회사의 기업가치(시가총액+순부채)와 EBITDA의 비율.
# 예를들어, 기업가치가 1000만원이고, EBITDA가 200만원이라면 EV/EBITDA는 5임.
# 이 비율이 낮으면 회사가 저평가 되었을 가능성이 있고, 높으면 고평가일 가능성이 높음.

# 베타(Beta)
# 주식의 변동성이 시장 전체의 변동성에 비해 얼마나 큰지를 나타내는 지표.
# 예를 들어, 배타가 1.5이면 시장이 1% 변동할때 주식은 1.5%변동할 가능성이 있다는 것을 의미함. 1보다 크면 변동성이 크고, 1보다 작으면 뎔변동적임.