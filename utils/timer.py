import time

from .colorlog import*

class timer:
    def __init__(self) -> None:
        self.tic = 0.0
        self.toc = 0.0
    
    def beg(self):
        self.tic = time.time()
        return self.tic

    def end(self, msg=None, quiet=False):
        self.toc = time.time()
        ret_msg = log_info(msg, f"{self.toc - self.tic:.2f}s", quiet=True)
        if not quiet:
            print(ret_msg)
        return self.toc, ret_msg
