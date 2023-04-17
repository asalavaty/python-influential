#! user/bin/env python3

# =============================================================================
#
#    Range Normalization
#
# =============================================================================

def rangeNormalize(data, minimum = 1, maximum = 100):

    """

    This finction range normalizes your input data to the range of min and max provided.

    :param data: A list of input data/numbers to be range normalized.
    :type data: list

    :param min: The minimum number for range normalization.
    type min: Either int or float

    :param max: The maximum number for range normalization.
    :type max: Either int or float

    :return: A list of range normalized data.

    """

    tmp_ranNorm1 = (list(map(lambda x: x - min(data), data)))
    tmp_ranNorm2 = (maximum - minimum)/(max(data) - min(data))
    tmp_ranNorm3 = list(map(lambda x: x * tmp_ranNorm2, tmp_ranNorm1))
    ranNorm_final = list(map(lambda x: minimum + x, tmp_ranNorm3))

    return ranNorm_final
