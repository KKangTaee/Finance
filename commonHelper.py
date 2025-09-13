from enum import Enum
from datetime import datetime, timedelta
from matplotlib.dates import relativedelta

import pandas as pd
import os

# Enum문

class EFinancialStatementType(Enum):
    NONE = 0
    INCOME_STATEMENT = 1 # 손익계산서
    BALANCE_SHEET = 2
    CASH_FLOW = 3

class EDateType(Enum):
    NONE = 0
    YEAR = 1
    QUARTER = 2
    MONTHLY = 3
    DAY = 4

class EIndustry:
    NONE = "none"
    SPECIALTY_RETAIL = "Specialty Retail"
    PHARMACEUTICAL_RETAILERS = "Pharmaceutical Retailers"
    BUSINESS_EQUIPMENT_SUPPLIES = "Business Equipment & Supplies"
    SECURITY_PROTECTION_SERVICES = "Security & Protection Services"
    OTHER_PRECIOUS_METALS_MINING = "Other Precious Metals & Mining"
    URANIUM = "Uranium"
    REIT_OFFICE = "REIT - Office"
    OIL_GAS_INTEGRATED = "Oil & Gas Integrated"
    ENTERTAINMENT = "Entertainment"
    CONFECTIONERS = "Confectioners"
    FOOD_DISTRIBUTION = "Food Distribution"
    OTHER_INDUSTRIAL_METALS_MINING = "Other Industrial Metals & Mining"
    GAMBLING = "Gambling"
    SOFTWARE_APPLICATION = "Software - Application"
    INDUSTRIAL_DISTRIBUTION = "Industrial Distribution"
    UTILITIES_DIVERSIFIED = "Utilities - Diversified"
    STEEL = "Steel"
    AGRICULTURAL_INPUTS = "Agricultural Inputs"
    DRUG_MANUFACTURERS_SPECIALTY_GENERIC = "Drug Manufacturers - Specialty & Generic"
    ALUMINUM = "Aluminum"
    DRUG_MANUFACTURERS_GENERAL = "Drug Manufacturers - General"
    BIOTECHNOLOGY = "Biotechnology"
    ELECTRONIC_COMPONENTS = "Electronic Components"
    DIAGNOSTICS_RESEARCH = "Diagnostics & Research"
    TELECOM_SERVICES = "Telecom Services"
    REIT_RESIDENTIAL = "REIT - Residential"
    TRAVEL_SERVICES = "Travel Services"
    SOFTWARE_INFRASTRUCTURE = "Software - Infrastructure"
    INTERNET_RETAIL = "Internet Retail"
    ASSET_MANAGEMENT = "Asset Management"
    BEVERAGES_WINERIES_DISTILLERIES = "Beverages - Wineries & Distilleries"
    REIT_RETAIL = "REIT - Retail"
    GROCERY_STORES = "Grocery Stores"
    SOLAR = "Solar"
    INFRASTRUCTURE_OPERATIONS = "Infrastructure Operations"
    COMPUTER_HARDWARE = "Computer Hardware"
    COMMUNICATION_EQUIPMENT = "Communication Equipment"
    GOLD = "Gold"
    ADVERTISING_AGENCIES = "Advertising Agencies"
    SEMICONDUCTORS = "Semiconductors"
    HEALTHCARE_PLANS = "Healthcare Plans"
    HEALTH_INFORMATION_SERVICES = "Health Information Services"
    INTERNET_CONTENT_INFORMATION = "Internet Content & Information"
    OIL_GAS_E_P = "Oil & Gas E&P"
    MEDICAL_DEVICES = "Medical Devices"
    CHEMICALS = "Chemicals"
    AEROSPACE_DEFENSE = "Aerospace & Defense"
    APPAREL_MANUFACTURING = "Apparel Manufacturing"
    REIT_HOTEL_MOTEL = "REIT - Hotel & Motel"
    TOOLS_ACCESSORIES = "Tools & Accessories"
    POLLUTION_TREATMENT_CONTROLS = "Pollution & Treatment Controls"
    STAFFING_EMPLOYMENT_SERVICES = "Staffing & Employment Services"
    MEDICAL_DISTRIBUTION = "Medical Distribution"
    AUTO_PARTS = "Auto Parts"
    FOOTWEAR_ACCESSORIES = "Footwear & Accessories"
    AUTO_TRUCK_DEALERSHIPS = "Auto & Truck Dealerships"
    INFORMATION_TECHNOLOGY_SERVICES = "Information Technology Services"
    HOME_IMPROVEMENT_RETAIL = "Home Improvement Retail"
    PACKAGING_CONTAINERS = "Packaging & Containers"
    MEDICAL_INSTRUMENTS_SUPPLIES = "Medical Instruments & Supplies"
    OIL_GAS_EQUIPMENT_SERVICES = "Oil & Gas Equipment & Services"
    CONSUMER_ELECTRONICS = "Consumer Electronics"
    REIT_HEALTHCARE_FACILITIES = "REIT - Healthcare Facilities"
    OIL_GAS_MIDSTREAM = "Oil & Gas Midstream"
    RESORTS_CASINOS = "Resorts & Casinos"
    DISCOUNT_STORES = "Discount Stores"
    MARINE_SHIPPING = "Marine Shipping"
    TRUCKING = "Trucking"
    PACKAGED_FOODS = "Packaged Foods"
    INSURANCE_SPECIALTY = "Insurance - Specialty"
    REAL_ESTATE_SERVICES = "Real Estate Services"
    EDUCATION_TRAINING_SERVICES = "Education & Training Services"
    MORTGAGE_FINANCE = "Mortgage Finance"
    SPECIALTY_INDUSTRIAL_MACHINERY = "Specialty Industrial Machinery"
    SEMICONDUCTOR_EQUIPMENT_MATERIALS = "Semiconductor Equipment & Materials"
    BEVERAGES_BREWERS = "Beverages - Brewers"
    WASTE_MANAGEMENT = "Waste Management"
    OIL_GAS_REFINING_MARKETING = "Oil & Gas Refining & Marketing"
    PERSONAL_SERVICES = "Personal Services"
    RECREATIONAL_VEHICLES = "Recreational Vehicles"
    TOBACCO = "Tobacco"
    SCIENTIFIC_TECHNICAL_INSTRUMENTS = "Scientific & Technical Instruments"
    SPECIALTY_CHEMICALS = "Specialty Chemicals"
    AIRLINES = "Airlines"
    LUXURY_GOODS = "Luxury Goods"
    UTILITIES_REGULATED_ELECTRIC = "Utilities - Regulated Electric"
    INSURANCE_PROPERTY_CASUALTY = "Insurance - Property & Casualty"
    ENGINEERING_CONSTRUCTION = "Engineering & Construction"
    HOUSEHOLD_PERSONAL_PRODUCTS = "Household & Personal Products"
    INTEGRATED_FREIGHT_LOGISTICS = "Integrated Freight & Logistics"
    REIT_MORTGAGE = "REIT - Mortgage"
    INSURANCE_REINSURANCE = "Insurance - Reinsurance"
    BUILDING_PRODUCTS_EQUIPMENT = "Building Products & Equipment"
    REAL_ESTATE_DEVELOPMENT = "Real Estate - Development"
    DEPARTMENT_STORES = "Department Stores"
    RESTAURANTS = "Restaurants"
    INSURANCE_DIVERSIFIED = "Insurance - Diversified"
    BANKS_REGIONAL = "Banks - Regional"
    INSURANCE_BROKERS = "Insurance Brokers"
    TEXTILE_MANUFACTURING = "Textile Manufacturing"
    CREDIT_SERVICES = "Credit Services"
    LEISURE = "Leisure"
    ELECTRONICS_COMPUTER_DISTRIBUTION = "Electronics & Computer Distribution"
    INSURANCE_LIFE = "Insurance - Life"
    AUTO_MANUFACTURERS = "Auto Manufacturers"
    CONSULTING_SERVICES = "Consulting Services"
    ELECTRONIC_GAMING_MULTIMEDIA = "Electronic Gaming & Multimedia"
    LODGING = "Lodging"
    SPECIALTY_BUSINESS_SERVICES = "Specialty Business Services"
    RENTAL_LEASING_SERVICES = "Rental & Leasing Services"
    FARM_HEAVY_CONSTRUCTION_MACHINERY = "Farm & Heavy Construction Machinery"
    BEVERAGES_NON_ALCOHOLIC = "Beverages - Non-Alcoholic"
    APPAREL_RETAIL = "Apparel Retail"
    FINANCIAL_DATA_STOCK_EXCHANGES = "Financial Data & Stock Exchanges"
    UTILITIES_REGULATED_GAS = "Utilities - Regulated Gas"
    OIL_GAS_DRILLING = "Oil & Gas Drilling"
    CONGLOMERATES = "Conglomerates"
    REIT_SPECIALTY = "REIT - Specialty"
    AIRPORTS_AIR_SERVICES = "Airports & Air Services"
    UTILITIES_INDEPENDENT_POWER_PRODUCERS = "Utilities - Independent Power Producers"
    PUBLISHING = "Publishing"
    COKING_COAL = "Coking Coal"
    LUMBER_WOOD_PRODUCTION = "Lumber & Wood Production"
    RAILROADS = "Railroads"
    PAPER_PAPER_PRODUCTS = "Paper & Paper Products"
    FINANCIAL_CONGLOMERATES = "Financial Conglomerates"
    SILVER = "Silver"
    

class DBName:
    DB_FINANCEIAL_STATEMENT = "FinancialStatement"
    DB_STOCK = "Stock"
    DB_NYSE = "NYSE"



def getStrFinancialStatementType(type):
    if type == EFinancialStatementType.INCOME_STATEMENT:
        return "IncomeStatement"
    elif type == EFinancialStatementType.BALANCE_SHEET:
        return "BalanceSheet"
    elif type == EFinancialStatementType.CASH_FLOW:
        return "CashFlow"
    else:
        return None
    
def getStrDateType(type):
    if type == EDateType.YEAR:
        return "Year"
    elif type == EDateType.QUARTER:
        return "Quarter"
    else:
        return None
    


def get_first_and_last_date(date_obj):
    # datetime 또는 pandas Timestamp 타입이 아닌 경우 변환
    if not isinstance(date_obj, (datetime, pd.Timestamp)):
        date_obj = pd.to_datetime(date_obj)

    start = date_obj.replace(day=1)

    if date_obj.month == 12:
        next_month = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
    else:
        next_month = date_obj.replace(month=date_obj.month + 1, day=1)

    end = next_month - timedelta(days=1)
    
    return start.date(), end.date()



def get_sector_weights_dict():
    sector_weights = {
        "Technology": {
            "PSR": 0.20,
            "GP/A": 0.10,
            "EV/EBIT": 0.10,
            "PER": 0.10,
            "CurrentRatio": 0.05,
            "PBR": 0.05,
            "DebtToEquityRatio": 0.05,
            "PCR": 0.10,
            "PFCR": 0.10,
            "ROE": 0.05,
            "OperatingMargin": 0.05,
            "FreeCashFlowMargin": 0.025,
            "RevenueGrowth": 0.025,
            "InterestCoverageRatio": 0.025
        },
        "Healthcare": {
            "PSR": 0.15,
            "GP/A": 0.10,
            "EV/EBIT": 0.10,
            "PER": 0.10,
            "PBR": 0.05,
            "DebtToEquityRatio": 0.05,
            "PCR": 0.05,
            "PFCR": 0.05,
            "ROE": 0.10,
            "OperatingMargin": 0.10,
            "FreeCashFlowMargin": 0.10,
            "RevenueGrowth": 0.10,
            "InterestCoverageRatio": 0.05
        },
        "Financials": {
            "PER": 0.25,
            "PBR": 0.20,
            "PSR": 0.05,
            "GP/A": 0.05,
            "ROE": 0.20,
            "DebtToEquityRatio": 0.10,
            "CurrentRatio": 0.05,
            "OperatingMargin": 0.05,
            "RevenueGrowth": 0.025,
            "InterestCoverageRatio": 0.025
        },
        "ConsumerDiscretionary": {
            "PSR": 0.20,
            "PER": 0.15,
            "PBR": 0.10,
            "GP/A": 0.05,
            "ROE": 0.10,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.10,
            "FreeCashFlowMargin": 0.05,
            "CurrentRatio": 0.05,
            "DebtToEquityRatio": 0.05,
            "InterestCoverageRatio": 0.05
        },
        "ConsumerStaples": {
            "PER": 0.20,
            "PBR": 0.15,
            "GP/A": 0.10,
            "ROE": 0.15,
            "OperatingMargin": 0.15,
            "RevenueGrowth": 0.10,
            "FreeCashFlowMargin": 0.05,
            "DebtToEquityRatio": 0.05,
            "InterestCoverageRatio": 0.05
        },
        "Industrials": {
            "EV/EBIT": 0.20,
            "PER": 0.15,
            "PBR": 0.10,
            "GP/A": 0.10,
            "ROE": 0.10,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.10,
            "FreeCashFlowMargin": 0.05,
            "DebtToEquityRatio": 0.05,
            "InterestCoverageRatio": 0.05
        },
        "Energy": {
            "EV/EBIT": 0.25,
            "PER": 0.15,
            "PBR": 0.10,
            "GP/A": 0.05,
            "ROE": 0.10,
            "OperatingMargin": 0.15,
            "FreeCashFlowMargin": 0.10,
            "RevenueGrowth": 0.05,
            "DebtToEquityRatio": 0.05
        },
        "Utilities": {
            "PER": 0.20,
            "PBR": 0.20,
            "GP/A": 0.10,
            "ROE": 0.15,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.05,
            "DebtToEquityRatio": 0.10,
            "InterestCoverageRatio": 0.10
        },
        "Materials": {
            "EV/EBIT": 0.20,
            "PER": 0.15,
            "PBR": 0.10,
            "GP/A": 0.10,
            "ROE": 0.10,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.10,
            "FreeCashFlowMargin": 0.05,
            "DebtToEquityRatio": 0.05,
            "InterestCoverageRatio": 0.05
        },
        "CommunicationServices": {
            "PSR": 0.15,
            "PER": 0.15,
            "PBR": 0.10,
            "GP/A": 0.05,
            "ROE": 0.15,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.10,
            "FreeCashFlowMargin": 0.10,
            "DebtToEquityRatio": 0.05,
            "InterestCoverageRatio": 0.05
        },
        "RealEstate": {
            "PER": 0.20,
            "PBR": 0.20,
            "GP/A": 0.05,
            "ROE": 0.10,
            "OperatingMargin": 0.05,
            "FreeCashFlowMargin": 0.10,
            "RevenueGrowth": 0.10,
            "DebtToEquityRatio": 0.10,
            "InterestCoverageRatio": 0.10
        }
    }
    return sector_weights



#--------------------
# 분기리스트 넣으면, 딕셔너리로 범위 반환 (일반)
#--------------------
def get_data_dict_by_quarter_normal(quarter_list : list):
        # 분기별 시작일과 종료일 매핑
    quarter_date_map = {
        "Q1": ("01-01", "03-31"),
        "Q2": ("04-01", "06-30"),
        "Q3": ("07-01", "09-30"),
        "Q4": ("10-01", "12-31")
    }

    date_dict = {}
    for q in quarter_list:
        year, quarter = q.split("-")  # 예: "2024-Q1" → year = "2024", quarter = "Q1"
        if quarter in quarter_date_map:
            start_suffix, end_suffix = quarter_date_map[quarter]
            start_date = f"{year}-{start_suffix}"
            end_date = f"{year}-{end_suffix}"
            date_dict[q] = [start_date, end_date]
        else:
            # 예외 처리: 잘못된 quarter 값
            date_dict[q] = [None, None]

    return date_dict



def get_date_dict_by_quarter_lazy(quarter_list: list):
    import calendar
    
    # 분기별 시작일과 종료일 매핑
    # quarter_date_map = {
    #     "Q1": ("06-01", "08-31"),
    #     "Q2": ("09-01", "11-30"),
    #     "Q3": ("12-01", "03-31"),
    #     "Q4": ("04-01", "05-31")
    # }

    quarter_date_map = {
        "Q1": ("05-01", "07-31"),
        "Q2": ("08-01", "10-31"),
        "Q3": ("11-01", "02-29"),  # <-- 문제점: 윤달 처리 필요
        "Q4": ("03-01", "04-30")
    }

    date_dict = {}
    for q in quarter_list:
        year, quarter = q.split("-")  # 예: "2024-Q1" → year = "2024", quarter = "Q1"
        year_int = int(year)

        if quarter in quarter_date_map:
            start_suffix, end_suffix = quarter_date_map[quarter]

            if quarter == 'Q3':
                start_year = year_int
                end_year = year_int + 1
            elif quarter == 'Q4':
                start_year = year_int + 1
                end_year = year_int + 1
            else:
                start_year = year_int
                end_year = year_int

            start_date = f"{start_year}-{start_suffix}"

            # ✅ 윤년 체크 (02-29 → 02-28 보정)
            if end_suffix == "02-29":
                if calendar.isleap(end_year):
                    end_suffix = "02-29"
                else:
                    end_suffix = "02-28"

            end_date = f"{end_year}-{end_suffix}"

            date_dict[q] = [start_date, end_date]
        else:
            date_dict[q] = [None, None]

    return date_dict



def get_date_dict_by_quarter_except_Q4(quarter_list: list):
    quarter_date_map = {
        "Q1": ("06-01", "08-31"),
        "Q2": ("09-01", "11-30"),
        "Q3": ("12-01", "05-31")  # 종료일이 다음 해
    }

    date_dict = {}
    for q in quarter_list:
        year, quarter = q.split("-")  # 예: "2024-Q3"
        year = int(year)
        
        if quarter in quarter_date_map:
            start_suffix, end_suffix = quarter_date_map[quarter]
            start_year = year
            # Q3은 종료 연도가 +1 되어야 함
            end_year = year + 1 if quarter == "Q3" else year

            start_date = f"{start_year}-{start_suffix}"
            end_date = f"{end_year}-{end_suffix}"
            date_dict[q] = [start_date, end_date]
        else:
            date_dict[q] = [None, None]

    return date_dict


#--------------
# date_dict에서 가장 오래된, 가장 최신의 날짜 값 반환
#--------------
def get_date_range_from_quarters(date_dict: dict):
    start_dates = []
    end_dates = []
  
    for start, end in date_dict.values():
        if start and end:
            start_dates.append(
                datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
            )
            end_dates.append(
                datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end
            )

    if not start_dates or not end_dates:
        return None, None  # 유효한 날짜가 없는 경우

    oldest = min(start_dates).strftime("%Y-%m-%d")
    latest = max(end_dates).strftime("%Y-%m-%d")

    return oldest, latest

#---------------
# date 값을 넣으면, 몇년도 몇분기인지 반환하는 코드
#---------------
def get_quarter_by_date(date_input) -> str:
    """
    date_input: 'YYYY-MM-DD' 문자열 또는 datetime 객체
    반환: 'YYYY-Qx' 형식의 문자열
    (분기 구분은 quarter_date_map 기준, Q3/Q4跨연도 고려)
    """
    # 분기별 날짜 매핑 (시작월-시작일, 종료월-종료일)
    quarter_date_map = {
        "Q1": ("06-01", "08-31"),   # 같은 해 6~8월
        "Q2": ("09-01", "11-30"),   # 같은 해 9~11월
        "Q3": ("12-01", "03-31"),   #跨연도: 12월 ~ 다음 해 3월
        "Q4": ("04-01", "05-31")    #跨연도: 다음 해 4~5월
    }
    
    # 문자열 → datetime 변환
    if isinstance(date_input, str):
        dt = datetime.strptime(date_input, "%Y-%m-%d")
    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        raise TypeError("date_input은 str 또는 datetime이어야 합니다.")
    
    year = dt.year
    month = dt.month
    
    # --- 분기 판별 ---
    if month in [5, 6, 7]:          # Q1
        return f"{year}-Q1"
    elif month in [8, 9, 10]:       # Q2
        return f"{year}-Q2"
    elif month in [11, 12]:         # Q3 (같은 해 12월)
        return f"{year}-Q3"
    elif month in [1, 2]:           # Q3 (다음 해 1~3월 → 이전 해 Q3)
        return f"{year-1}-Q3"
    elif month in [3, 4]:           # Q4 (다음 해 4~5월 → 이전 해 Q4)
        return f"{year-1}-Q4"
    
    raise ValueError(f"어떤 분기에도 속하지 않습니다: {date_input}")


#------------------
# start_date, end_date 값을 바탕으로, date_dict에 있는 값을 조정
#------------------
def get_trimmed_date_dict(date_dict, start_date, end_date):

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    result = {}

    for quarter, (start_str, end_str) in date_dict.items():
        q_start = datetime.strptime(start_str, "%Y-%m-%d")
        q_end = datetime.strptime(end_str, "%Y-%m-%d")

        # 현재 분기가 범위와 겹치는지 확인
        if end_dt < q_start or start_dt > q_end:
            continue

        # 교집합 범위 계산
        new_start = max(start_dt, q_start)
        new_end = min(end_dt, q_end)  # ← 여기서 end_dt가 q_end보다 클 경우에도 min을 써서 자름

        result[quarter] = [new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d")]

    # 👉 마지막 분기라면 end_date를 직접 반영
    if result:
        last_key = list(result.keys())[-1]
        last_end = datetime.strptime(result[last_key][1], "%Y-%m-%d")
        if end_dt > last_end:
            result[last_key][1] = end_dt.strftime("%Y-%m-%d")

    return result


#-----------------
# 지정된 쿼터값을 통해 그 쿼터의 start 날짜를 한달 전으로 이동 (시작날짜 처리용도)
#-----------------
def adjust_start_data_dict_by_quarter(date_dict: dict, first_quarter: list) -> dict:
    """
    1. date_dict의 문자열 날짜들을 datetime 객체로 변환
    2. 첫 번째 분기(quarter_list[0])의 시작 날짜를 한 달 앞당김

    Parameters:
        date_dict (dict): {'Q1': ['2023-01-01', '2023-03-31'], ...} 형식의 딕셔너리
        quarter_list (list): 분기 이름 리스트, 예: ['Q1', 'Q2', ...]

    Returns:
        dict: 변환 및 조정된 date_dict
    """
    # 날짜 문자열을 datetime 객체로 변환
    for key, value_list in date_dict.items():
        for i in range(len(value_list)):
            value_list[i] = datetime.strptime(value_list[i], "%Y-%m-%d")

    # 첫 분기의 시작 날짜를 1개월 앞당김
    # first_quarter = quarter_list[0]
    first_start, first_end = date_dict[first_quarter]
    new_first_start = first_start - relativedelta(months=1)
    date_dict[first_quarter] = [new_first_start, first_end]

    return date_dict



# CSV파일로드 및 저장
def load_and_save_csv(folder_path:str, file_name:str, is_load:bool, action):
    if not callable(action):
        raise TypeError("action은 함수(람다 포함)여야 합니다!")

    file_path = os.path.join(folder_path, file_name)

    if is_load:
        return pd.read_csv(file_path)
    else:
        df = action()
        # 폴더 생성 (존재하지 않아도 에러 안 나게)
        os.makedirs(folder_path, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"{file_path} 생성 완료")
        return df