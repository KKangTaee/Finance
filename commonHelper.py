from enum import Enum
from datetime import datetime, timedelta
import pandas as pd

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