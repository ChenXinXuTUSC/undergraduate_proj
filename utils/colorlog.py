import sys

from . import colorstr

WITH_LOCATION = True

def print_loc():
    print("   ", sys._getframe(2).f_code.co_filename + ':' + str(sys._getframe(1).f_back.f_lineno))
def jget_log():
    return "    " + sys._getframe(2).f_code.co_filename + ':' + str(sys._getframe(1).f_back.f_lineno)

def log_dbug(*msgs, quiet:bool=False):
    log_msg = colorstr.get_colorstr(colorstr.FORE_BLE, colorstr.BACK_ORG, "[DBUG]") + " " + " ".join([str(msg) for msg in [*msgs]])
    log_loc = jget_log()
    if not quiet:
        print(log_msg)
        if WITH_LOCATION:
            print(log_loc)
    if WITH_LOCATION:
        log_msg += '\n' + log_loc
    return log_msg

def log_info(*msgs, quiet:bool=False):
    log_msg = colorstr.get_colorstr(colorstr.FORE_GRN, colorstr.BACK_ORG, "[INFO]") + " " + " ".join([str(msg) for msg in [*msgs]])
    log_loc = jget_log()
    if not quiet:
        print(log_msg)
        if WITH_LOCATION:
            print(log_loc)
    if WITH_LOCATION:
        log_msg += '\n' + log_loc
    return log_msg

def log_warn(*msgs, quiet:bool=False):
    log_msg = colorstr.get_colorstr(colorstr.FORE_YLW, colorstr.BACK_ORG, "[WARN]") + " " + " ".join([str(msg) for msg in [*msgs]])
    log_loc = jget_log()
    if not quiet:
        print(log_msg)
        if WITH_LOCATION:
            print(log_loc)
    if WITH_LOCATION:
        log_msg += '\n' + log_loc
    return log_msg

def log_erro(*msgs, quiet:bool=False):
    log_msg = colorstr.get_colorstr(colorstr.FORE_RED, colorstr.BACK_ORG, "[ERRO]") + " " + " ".join([str(msg) for msg in [*msgs]])
    log_loc = jget_log()
    if not quiet:
        print(log_msg)
        if WITH_LOCATION:
            print(log_loc)
    if WITH_LOCATION:
        log_msg += '\n' + log_loc
    return log_msg

def log_fatl(*msgs, quiet:bool=False):
    log_msg = colorstr.get_colorstr(colorstr.FORE_PRP, colorstr.BACK_ORG, "[FATL]") + " " + " ".join([str(msg) for msg in [*msgs]])
    log_loc = jget_log()
    if not quiet:
        print(log_msg)
        if WITH_LOCATION:
            print(log_loc)
    if WITH_LOCATION:
        log_msg += '\n' + log_loc
    return log_msg

# wrapper test
def with_loc(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        print_loc()
    return wrapper
