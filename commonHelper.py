from enum import Enum

# Enum문

class FSType(Enum):
    NONE = 0
    INCOME_STATEMENT = 1 # 손익계산서
    BALANCE_SHEET = 2
    CASH_FLOW = 3

class FSDateType(Enum):
    NONE = 0
    YEAR = 1
    QUARTER = 2


def getStrFSType(type):
    if type == FSType.INCOME_STATEMENT:
        return "IncomeStatement"
    elif type == FSType.BALANCE_SHEET:
        return "BalanceSheet"
    elif type == FSType.CASH_FLOW:
        return "CashFlow"
    else:
        return None
    
    
def getStrFSDateType(type):
    if type == FSDateType.YEAR:
        return "Year"
    elif type == FSDateType.QUARTER:
        return "Quarter"
    else:
        return None