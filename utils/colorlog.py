import sys

from . import colorstr

def print_loc():
    print("    ", sys._getframe(2).f_code.co_filename + ':' + str(sys._getframe().f_back.f_lineno))

def log_dbug(msg:str):
    print(colorstr.get_colorstr(colorstr.FORE_WHE, colorstr.BACK_BLE, "[DBUG]"), msg)
    print_loc()

def log_info(msg:str):
    print(colorstr.get_colorstr(colorstr.FORE_WHE, colorstr.BACK_GRN, "[INFO]"), msg)
    print_loc()

def log_warn(msg:str):
    print(colorstr.get_colorstr(colorstr.FORE_WHE, colorstr.BACK_YLW, "[WARN]"), msg)
    print_loc()

def log_erro(msg:str):
    print(colorstr.get_colorstr(colorstr.FORE_WHE, colorstr.BACK_RED, "[ERRO]"), msg)
    print_loc()

def log_fatl(msg:str):
    print(colorstr.get_colorstr(colorstr.FORE_WHE, colorstr.BACK_PRP, "[FATL]"), msg)
    print_loc()