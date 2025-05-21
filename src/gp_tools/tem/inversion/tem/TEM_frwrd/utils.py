# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:37:17 2018

@author: lukas aigner at TUWIEN
"""
import re
import numpy as np
import pandas as pd

from scipy.constants import mu_0


# %% function_lib
def get_float_from_string(string):
    numeric_const_pattern = r"""
           [-+]? # optional sign
           (?:
              (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
              |
              (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
           )
           # followed by optional exponent part if desired
           (?: [Ee] [+-]? \d+ ) ?
           """
    
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    numeric_list = rx.findall(string)
    if len(numeric_list) > 1:
        return np.float_(numeric_list)
    elif len(numeric_list) == 1:
        return np.float_(numeric_list)[0]
    else:
        print('error - no numerics found in string')


def round_up(x, level):
    x = int(x)
    shift = x % level
    return x if not shift else x + level - shift

