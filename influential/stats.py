#i user/bin/env python3

# Import the requirements
import pandas as pd
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from .rank import rank_cal

# =============================================================================
#
#    Fast correlation and Mutual Rank Analysis
#
# =============================================================================

def fcor(data, use = "everything", method = "spearman", mutualRank = True, mutualRank_mode = "unsigned", pvalue = False, adjust = "fdr_bh", flat = True):

    """
    This function calculates Pearson/Spearman correlations between all pairs of features in a Pandas dataframe.
    It is also possible to simultaneously calculate mutual rank (MR) of correlations as well as their p-values and adjusted p-values.
    Additionally, this function can automatically combine and flatten the result matrices. Selecting correlated features using an MR-based threshold
    rather than based on their correlation coefficients or an arbitrary p-value is more efficient and accurate in inferring
    functional associations in systems, for example in gene regulatory networks.

    :param data: A numeric Pandas dataframe (features on columns and samples on rows).
    :type data: pandas.core.frame.DataFrame

    :param method: a character string indicating which correlation coefficient is to be computed. One of "pearson", "spearman" (default), or "kendall".
    :type method: str

    :param mutualRank: logical, whether to calculate mutual ranks of correlations or not.
    :type mutualRank: bool

    :param mutualRank_mode: a character string indicating whether to rank based on "signed" or "unsigned" (default) correlation values. 
    In the "unsigned" mode, only the level of a correlation value is important and not its sign (the function ranks the absolutes of correlations). 
    Options are "unsigned", and "signed".
    :type mutualRank_mode: str

    :param pvalue: logical, whether to calculate p-values of correlations or not.
    :type pvalue: bool

    :param adjust: p-value correction method (when pvalue = True), a character string including any of "fdr_bh" (Benjamini/Hochberg; default),
    "bonferroni", "sidak", "holm", "holm-sidak", "hochberg", "simes-hochberg", "hommel", "fdr_by" (Benjamini/Yekutieli), "fdr_tsbh", "fdr_tsbky", or "none".
    :type adjust: str

    :param flat: logical, whether to combine and flatten the result matrices or not.
    :type flat: bool

    :return: Depending on the input data, a Pandas dataframe or a dictionary including cor (correlation coefficients),
    mr (mutual ranks of correlation coefficients), p (p-values of correlation coefficients), and p_adj (adjusted p-values).

    """

    # Set initial NULL values
    corr_coeff = None
    mutR = None
    p_value = None
    padj = None

    # Perform the correlation analysis
    corr_coeff = data.corr(method=method)

    # Calculating P-value
    if pvalue:

        ## Calculate n required for p-value measurement
        n = data.shape[0]

        ## Calculate t required for p-value measurement
        t_stats = (corr_coeff * np.sqrt(n - 2))/np.sqrt(1 - corr_coeff**2)

        ## Calculate p-value
        p_value = -2 * np.expm1(np.log(t.cdf(abs(t_stats), (n - 2))))
        p_value[np.isnan(p_value)] = 1
        p_value[p_value > 1] = 1
        p_value = abs(p_value)
        
        # Calculate adjusted p-value
        if adjust != "none":
            padj = multipletests(p_value.flatten(), method = adjust)[1].reshape(p_value.shape)

    # Calculate Mutual Rank
    ## We set the order= -1 so that higher correlations get higher ranks (highest cor will be first rank)

    if mutualRank:
        if mutualRank_mode == "unsigned":
            r_rank = abs(corr_coeff).apply(func = rank_cal, axis = 0, order = -1) # Fast rank the correlation of each gene with all the other genes
        else:
            r_rank = corr_coeff.apply(func = rank_cal, axis = 0, order = -1)
        mutR = np.sqrt(r_rank*r_rank.transpose())

    if flat:
        # Flatten the results

        ## Reformat the corr_coeff DataFrame
        ut = np.triu(np.ones_like(corr_coeff), k=1).astype(bool)

        corr_row = np.repeat(list(corr_coeff.index), corr_coeff.shape[0]).reshape(corr_coeff.shape)
        corr_col = np.array(list(corr_coeff.index) * corr_coeff.shape[0]).reshape(corr_coeff.shape)

        result = pd.DataFrame({
            'row': corr_row[ut],
            'column': corr_col[ut],
            'cor': corr_coeff.to_numpy()[ut]
            })

        if mutR is not None: 
            result = result.assign(mr = mutR.to_numpy()[ut])
        if p_value is not None: 
            result = result.assign(p = p_value[ut]) 
        if padj is not None: 
            result = result.assign(p_adj = padj[ut])

    else:
        result = dict(cor = corr_coeff, mr = mutR, p = p_value, p_adj = padj)

    return result
