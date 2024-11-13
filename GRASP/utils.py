import copy
import re
import numpy as np


def drop_trailing_number(s):
    return re.sub(r'\d+$', '', s)

def has_intersection(list1, list2):
    return any(item in list2 for item in list1)

def check_empty(key_groups):
    nump = 0
    for k in key_groups:
        nump += len(key_groups[k])

    if nump == 0:
        return True
    else:
        return False

def check_empty_for_indicator(travesal_indicators):
    nump = 0
    for per_t_group_indicator in travesal_indicators:
        nump += np.sum(per_t_group_indicator)

    if nump == 0:
        return True
    else:
        return False