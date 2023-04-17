#! usr/bin/env python3

# =============================================================================
#
#    Rank
#
# =============================================================================

def rank_cal(data, order = 1):

    """
    This function calculates the rank of numbers.

    :param data: input numbers to be ranked in the format of a vector (list, series, etc.)
    :type data: a vector of numbers

    :param order: an integer with only possible values of 1 and -1, corresponding to ascending and descending order. 
    :type order: int

    :return: a list indluding the rank of input data

    """

    a={}
    rank=1
    if order == -1:
        data = list(map(lambda x: x * -1, data))
    for num in sorted(data):
        if num not in a:
            a[num]=rank
            rank=rank+1
    return[a[i] for i in data]
