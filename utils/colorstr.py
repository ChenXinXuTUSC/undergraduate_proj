FORE_BLK = 30
FORE_RED = 31
FORE_GRN = 32
FORE_YLW = 33
FORE_BLE = 34
FORE_PRP = 35
FORE_CYN = 36
FORE_WHE = 37

BACK_BLK = 40
BACK_RED = 41
BACK_GRN = 42
BACK_YLW = 43
BACK_BLE = 44
BACK_PRP = 45
BACK_CYN = 46
BACK_WHE = 47

def get_colorstr(fore:int, back:int, msg:str)->str:
    if fore < 30 or fore > 37:
        fore = 37
    if back < 40 or back > 47:
        back = 40
    return f"\033[1;{fore};{back}m{msg}\033[0m"