#! user/bin/env python3

# Import the requirements
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import t, norm
from .stats import fcor
import igraph
from .centrality import ivi
from statsmodels.stats.multitest import multipletests

# =============================================================================
#
#    Differential/regression data assembly
#
# =============================================================================

def diff_data_assembly(*dataframes):

    """
    This function assembles a Pandas dataframe required for running the ExIR model. You may provide
    as many differential/regression data as you wish. Also, the datasets should be filtered
    beforehand according to your desired thresholds and, consequently, should only include the significant data.
    Each dataset provided should be a Pandas dataframe with one or two columns.
    The first column should always include differential/regression values
    and the second one (if provided) the significance values. Please also note that the significance (adjusted P-value)
    column is mandatory for differential datasets.

    :param dataframes: desired pandas dataframes containing differential/regression data.
    :type dataframes: pandas.core.frame.DataFrame

    :return: a Pandas dataframe including the collective list of features in rows and all of the
    differential/regression data and their statistical significance in columns with the same
    order provided by the user.

    """

    # Getting the feature names
    feature_names = []
    for i in dataframes:
        feature_names += list(i.index)

    # Creating the Diff_data dataframe
    Diff_data = pd.DataFrame({
    }, index= feature_names)

    for i in range(len(dataframes)):

        feature_names_index = list(map(lambda x: list(Diff_data.index).index(x), list(dataframes[i].index)))

        if dataframes[i].shape[1] == 2:

            # Add the differential value
            Diff_data.insert(loc= Diff_data.shape[1], column= 'Diff_value' + str(i + 1), value=0)
            Diff_data.iloc[feature_names_index, Diff_data.shape[1] - 1] = dataframes[i].iloc[:,0]

            # Add the significance value
            Diff_data.insert(loc = Diff_data.shape[1], column = 'Sig_value' + str(i + 1), value = 1)
            Diff_data.iloc[feature_names_index, Diff_data.shape[1] - 1] = dataframes[i].iloc[:,1]

        elif dataframes[i].shape[1] == 1:

            Diff_data.insert(loc= Diff_data.shape[1], column= 'Diff_value' + str(i + 1), value=0)
            Diff_data.iloc[feature_names_index, Diff_data.shape[1] - 1] = dataframes[i].iloc[:,0]

    return Diff_data

# =============================================================================
#
#    Calculation of ExIR
#
# =============================================================================

def exir_model(Diff_data, Diff_value, Sig_value,
               Exptl_data, Condition_colname, Desired_list = None, 
               Regr_value = None, Normalize = False,
               cor_thresh_method = "mr", r = 0.5, mr = 20,
               max_connections = 50000, alpha = 0.05,
               num_trees = 10000, mtry = 'sqrt',
               inf_const = 10**10, seed = 1234, verbose = True):
    
    """
    This function runs the Experimental data-based Integrated Ranking (ExIR)
    model for the classification and ranking of top candidate features. The input
    data could come from any type of experiment such as transcriptomics and proteomics.
    A shiny app has also been developed for Running the ExIR model, visualization of its results as well as computational
    simulation of knockout and/or up-regulation of its top candidate outputs, which is accessible online at https://influential.erc.monash.edu/.

    :param Desired_list: (optional) A string list of your desired features. This list could be, for
    instance, a list of features obtained from cluster analysis, time-course analysis,
    or a list of dysregulated features with a specific sign.
    :type Desired_list: list

    :param Diff_data: a pandas dataframe of all significant differential/regression data and their
    statistical significance values (p-value/adjusted p-value). Note that the differential data
    should be in the log fold-change (log2FC) format.
    You may have selected a proportion of the differential data as the significant ones according
    to your desired thresholds. A function, named `diff_data_assembly` has also been
    provided for the convenient assembling of the Diff_data dataframe.
    :type Diff_data: pandas.core.frame.DataFrame

    :param Diff_value: an integer list containing the column number(s) of the differential
    data in the Diff_data dataframe. The differential data could result from any type of
    differential data analysis. One example could be the fold changes (FCs) obtained from differential
    expression analyses. The user may provide as many differential data as he/she wish.
    :type Diff_value: list

    :param Regr_value: (optional) an integer list containing the column number(s) of the regression
    data in the Diff_data dataframe. The regression data could result from any type of regression
    data analysis or other analyses such as time-course data analyses that are based on regression models.
    :type Regr_value: list

    :param Sig_value: an integer list containing the column number(s) of the significance values (p-value/adjusted p-value) of
    both differential and regression data (if provided). Providing significance values for the regression data is optional.
    :type Sig_value: list

    :param Exptl_data: a pandas dataframe containing all of the experimental data including a column for specifying the conditions.
    The features/variables of the dataframe should be on the columns and the samples should come on the rows.
    The condition column should be of the string class. For example, if the study includes several replicates of
    cancer and normal samples, the condition column should include "cancer" and "normal" as the conditions of different samples.
    Also, the prior normalization of the experimental data is highly recommended. Otherwise,
    the user may set the `Normalize` argument to True for a simple log2 transformation of the data.
    The experimental data could come from a variety sources such as transcriptomics and proteomics assays.
    :type Exptl_data: pandas.core.frame.DataFrame
    
    :param Condition_colname: a string or character vector specifying the name of the column "condition" of the Exptl_data dataframe.
    type Condition_colname: str

    :param Normalize: whether the experimental data should be normalized or not (default is False). If True, the
    experimental data will be log2 transformed.
    :type Normalize: bool

    :param cor_thresh_method: a character string indicating the method for filtering the correlation results, either
    "mr" (default; Mutual Rank) or "cor.coefficient".
    :type cor_thresh_method: str

    :param mr: an integer determining the threshold of mutual rank for the selection of correlated features (default is 20). Note that
    higher mr values considerably increase the computation time.
    :type mr: int

    :param r: the threshold of Spearman correlation coefficient for the selection of correlated features (default is 0.5).
    :type r: float
    
    :param max_connections: the maximum number of connections to be included in the association network.
    Higher max_connections might increase the computation time, cost, and accuracy of the results (default is 50,000).
    :type max_connections: int

    :param alpha: the threshold of the statistical significance (p-value) used throughout the entire model (default is 0.05)
    :type alpha: float

    :param num_trees: number of trees to be used for the random forests classification (supervised machine learning). Default is set to 10000.
    :type num_trees: int

    :param mtry: mumber of features to possibly split at in each node. Default ('sqrt') is the (rounded down) square root of the
    number of variables. Possible options include "sqrt", "log2", int or float.

    :param inf_const: The constant value to be multiplied by the maximum absolute value of differential (logFC)
    values for the substitution with infinite differential values. This results in noticeably high biomarker values for features
    with infinite differential values compared with other features. Having said that, the user can still use the
    biomarker rank to compare all of the features. This parameter is ignored if no infinite value
    is present within Diff_data. However, this is used in the case of sc-seq experiments where some genes are uniquely
    expressed in a specific cell-type and consequently get infinite differential values. Note that the sign of differential
    value is preserved (default is 10**10).
    :type inf_const: int

    :param seed: the seed to be used for all of the random processes throughout the model (default is 1234).
    :type seed: int

    :param verbose: whether the accomplishment of different stages of the model should be printed (default is True).
    :type verbose: bool
    
    :return: a dictionary of one graph and one to four pandas dataframes including:
    - Driver table: Top candidate drivers
    - DE-mediator table: Top candidate differentially expressed/abundant mediators
    - nonDE-mediator table: Top candidate non-differentially expressed/abundant mediators
    - Biomarker table: Top candidate biomarkers
    The number of returned tables depends on the input data and specified arguments.

    """

    # ProgressBar: Preparing the input data
    if verbose:
      pbar = trange(100)
      pbar.update(1)
      pbar.set_description("Preparing the input data")

      ## Make a copy of the data so that the original input data is not changed
      Exptl_data = deepcopy(Exptl_data)
      Diff_data = deepcopy(Diff_data)

    # Change the colnames of Diff_data
    Diff_data.columns = list(map(lambda x: x + '_source', list(Diff_data.columns)))

    # Change the Inf/-Inf diff values (applicable to sc-Data)
    for i in range(len(Diff_value)):

        ## Get the maximum of the absolute values of the current column while ignoring inf values
        tmp_max_abs = np.ma.masked_invalid(np.absolute(Diff_data.iloc[:,Diff_value[i]])).max()

        ## Replace inf values
        Diff_data.iloc[:,Diff_value[i]].replace({np.inf: inf_const*tmp_max_abs, -np.inf: -1*inf_const*tmp_max_abs}, inplace = True)

    # Get the column number of condition column
    condition_index = list(Exptl_data.columns).index(Condition_colname)

    # Correct the type of columns
    condition_less_columns = Exptl_data.columns.difference([Condition_colname])
    Exptl_data[condition_less_columns] = Exptl_data[condition_less_columns].apply(pd.to_numeric, axis = 1)

    # Normalize the experimental data (if required)
    if Normalize:
      Exptl_data[condition_less_columns] = np.log2(Exptl_data[condition_less_columns]+1)

    if verbose:
        pbar.update(4)
        pbar.set_description("Calculating the differential score")

    #1 Calculate differential score
    Diff_data['sum_Diff_value'] = np.absolute(Diff_data.iloc[:, Diff_value].apply(np.sum, axis = 1))

    #range normalize the differential score
    Diff_data['sum_Diff_value'] = rangeNormalize(data = Diff_data['sum_Diff_value'], minimum = 1, maximum = 100)

    if verbose:
        pbar.update(5)
        pbar.set_description("Calculating the regression/time-course R-squared score")

    #2 Calculate regression/time-course R-squared score (if provided)
    if Regr_value is not None:
        Diff_data['sum_Regr_value'] = Diff_data.iloc[:, Regr_value].apply(np.sum, axis = 1)

        #range normalize the R-squared score
        Diff_data['sum_Regr_value'] = rangeNormalize(data = Diff_data['sum_Regr_value'], minimum = 1, maximum = 100)

    if verbose:
        pbar.update(5)
        pbar.set_description("Calculating the collective statistical significance of differential/regression factors")

    #3 Calculate the statistical significance of differential/regression factors
    if max(Diff_data.iloc[:, Sig_value].max()) > 1 or min(Diff_data.iloc[:, Sig_value].min()) < 0:
        raise ValueError('input Sig-values (p-value/padj) must all be in the range 0 to 1!')
        
    for i in range(len(Sig_value)):
        if Diff_data.iloc[:, Sig_value[i]].min() == 0:

            # range normalize the primitive Sig_value
            temp_min_Sig_value = np.sort(Diff_data.iloc[:,Sig_value[i]].unique())[1]
            temp_max_Sig_value = Diff_data.iloc[:, Sig_value[i]].max()

            Diff_data.iloc[:,Sig_value[i]] = rangeNormalize(data = Diff_data.iloc[:,Sig_value[i]], 
                                                            minimum = temp_min_Sig_value, 
                                                            maximum = temp_max_Sig_value)
    Diff_data.iloc[:,Sig_value] = -np.log10(Diff_data.iloc[:,Sig_value])
            
    Diff_data['sum_Sig_value'] = Diff_data.iloc[:,Sig_value].apply(np.sum, axis = 1)

    # Range normalize the statistical significance
    Diff_data['sum_Sig_value'] = rangeNormalize(data = Diff_data['sum_Sig_value'], minimum = 1, maximum = 100)

    if verbose:
        pbar.update(5)
        pbar.set_description("Performing the random forests classification (supervised machine learning)")
    
    #4 Calculation of the Integrated Value of Influence (IVI)

    # a Separate a transcriptomic profile of diff features
    if Desired_list is not None:
        sig_diff_index = list(map(lambda x: list(Exptl_data.columns).index(x), Desired_list))
    else:
        sig_diff_index = list(map(lambda x: list(Exptl_data.columns).index(x), list(Diff_data.index)))
    
    exptl_for_super_learn = Exptl_data.iloc[:, sig_diff_index]
    exptl_for_super_learn['condition'] = Exptl_data.iloc[:, condition_index]
    
    # b Perform random forests classification
    np.random.seed(seed=seed)

    ## Train a random forest classifier:
    rf_diff_exptl = RandomForestClassifier(n_estimators= num_trees, max_features= mtry, criterion="entropy", random_state= seed)

    rf_diff_exptl.fit(X= exptl_for_super_learn.drop('condition', axis=1), y= exptl_for_super_learn['condition'])

    if verbose:
        pbar.update(5)

    ## Compute feature importances
    rf_diff_exptl_imp = rf_diff_exptl.feature_importances_

    ## Calculate p-values

    ### Calculate n required for p-value measurement
    n = Exptl_data.shape[0]

    ### Calculate t required for p-value measurement
    t_stats = (rf_diff_exptl_imp * np.sqrt(n - 2))/np.sqrt(1 - rf_diff_exptl_imp**2)

    ### Calculate p-value
    p_value = -2 * np.expm1(np.log(t.cdf(abs(t_stats), (n - 2))))
    p_value[np.isnan(p_value)] = 1
    p_value[p_value > 1] = 1
    p_value = abs(p_value)

    if verbose:
        pbar.update(5)

    rf_diff_exptl_pvalue = pd.DataFrame({'feature': exptl_for_super_learn.drop("condition", axis=1).columns,
                                         'importance': rf_diff_exptl_imp,
                                         'p_value': p_value})
    
    ## Filtering the RF output data
    if Desired_list is not None:
        select_number = round(len(Desired_list)/2)
    else:
        select_number = 100

    sig_rf_index = [i for i, x in enumerate(list(rf_diff_exptl_pvalue['p_value'] < alpha)) if x]
    non_sig_rf_index = [i for i, x in enumerate(list(rf_diff_exptl_pvalue['p_value'] < alpha)) if not x]
    if len(sig_rf_index) >= select_number:
        rf_diff_exptl_pvalue = rf_diff_exptl_pvalue.iloc[sig_rf_index]
    else:
        required_pos_importance = select_number - len(sig_rf_index)
        temp_rf_diff_exptl_pvalue = rf_diff_exptl_pvalue.iloc[non_sig_rf_index]
        rf_importance_select = list(np.argsort(temp_rf_diff_exptl_pvalue['importance'])[-required_pos_importance:])
        temp_rf_diff_exptl_pvalue = temp_rf_diff_exptl_pvalue.iloc[rf_importance_select]

        ### Combine pvalue-based and importance-based tables
        rf_diff_exptl_pvalue = pd.concat([rf_diff_exptl_pvalue.iloc[sig_rf_index], temp_rf_diff_exptl_pvalue], axis=0)

    # Negative importance values could be considered as 0
    neg_imp_index = [i for i,x in enumerate(list(rf_diff_exptl_pvalue['importance'] < 0)) if x]
    if len(neg_imp_index) > 0:
        rf_diff_exptl_pvalue['importance'].iloc[neg_imp_index] = 0

    # Taking care of zero p-values
    if min(rf_diff_exptl_pvalue['p_value']) == 0:
      
      ## Range normalize the primitive pvalue
      temp_min_pvalue = sorted(rf_diff_exptl_pvalue["pvalue"].unique())[1]
      temp_max_pvalue = max(rf_diff_exptl_pvalue["pvalue"])
      rf_diff_exptl_pvalue["pvalue"] = rangeNormalize(data = rf_diff_exptl_pvalue["pvalue"], minimum = temp_min_pvalue, maximum = temp_max_pvalue)

    if verbose:
        pbar.update(5)
        pbar.set_description("Performing PCA (unsupervised machine learning)")

    # 5 Unsupervised machine learning (PCA)
    Exptl_data_for_PCA_index = list(map(lambda x: list(Exptl_data.columns).index(x), rf_diff_exptl_pvalue.feature))
    temp_Exptl_data_for_PCA = Exptl_data.iloc[:,Exptl_data_for_PCA_index]

    temp_PCA = PCA()
    temp_PCA.fit(temp_Exptl_data_for_PCA)

    tmp_PCA_r = np.abs(temp_PCA.components_[0])

    ## Range normalize the rotation values
    tmp_PCA_r = rangeNormalize(data = tmp_PCA_r, minimum = 1, maximum = 100)
    
    if verbose:
        pbar.update(5)
        pbar.set_description("Performing the first round of association analysis")
    
    #c Performing correlation analysis
    if cor_thresh_method == "mr":
        mutualRank_mode = True
    else:
        mutualRank_mode = False
    
    temp_corr = fcor(data = Exptl_data[condition_less_columns], method = "spearman", mutualRank = mutualRank_mode)

    ## Save a second copy of all cor data
    temp_corr_for_sec_round = deepcopy(temp_corr)

    ###################################################

    # Check which items from subject satisfy the operator function respective to the source
    def which(subject, source, operator = 'in'):

        ## For each element in subject check if that satisfies the operator function respective to the source
        if operator == 'in':
            bool_index = list(map(lambda x: x in source, subject))
        elif operator == 'not in':
            bool_index = list(map(lambda x: x not in source, subject))
        elif operator == '>=':
            bool_index =  list(map(lambda x: x >= source, subject))
        elif operator == '>':
            bool_index =  list(map(lambda x: x > source, subject))
        elif operator == '<=':
            bool_index =  list(map(lambda x: x <= source, subject))
        elif operator == '<':
            bool_index =  list(map(lambda x: x < source, subject))

        ## Get the indices of true elements
        match_index = [i for i, x in enumerate(bool_index) if x]

        ## Get the values of true elements
        match_values = list(map(lambda x: subject[x], match_index))

        ## Return a dictionary including match indices as keys and their corresponding values as the values
        return dict(zip(match_index, match_values))
    
    ###################################################


    # Filter corr data for only those corr between diff features and themselves/others
    filter_corr_index = list(which(subject = list(temp_corr.row), source = list(rf_diff_exptl_pvalue.feature), operator = 'in').keys()) + list(which(subject = list(temp_corr.column), source = list(rf_diff_exptl_pvalue.feature), operator = 'in').keys())
    filter_corr_index = list(set(filter_corr_index))
    temp_corr = temp_corr.iloc[filter_corr_index]

    ## Filtering low level correlations
    cor_thresh = deepcopy(r)
    mr_thresh = deepcopy(mr)

    if cor_thresh_method == "mr":
        temp_corr = temp_corr[temp_corr['mr'] < mr_thresh]

        if temp_corr.shape[0] > (max_connections*0.95):
            temp_corr_select_index = list(np.argsort(temp_corr.mr)[:round(max_connections*0.95)])
            temp_corr = temp_corr.iloc[temp_corr_select_index]
    elif cor_thresh_method == "cor.coefficient":
        temp_corr = temp_corr[abs(temp_corr['cor']) > cor_thresh]

        if temp_corr.shape[0] > (max_connections*0.95):
            temp_corr_select_index = list(np.argsort(temp_corr.cor)[-round(max_connections*0.95):])
            temp_corr = temp_corr.iloc[temp_corr_select_index]

    diff_only_temp_corr = deepcopy(temp_corr)

    # Getting the list of diff features and their correlated features
    diff_plus_corr_features = list(temp_corr.row) + list(temp_corr.column)
    diff_plus_corr_features = list(set(diff_plus_corr_features))

    # Find the diff features amongst diff_plus_corr_features
    non_diff_only_features = list(which(subject = diff_plus_corr_features, source = list(rf_diff_exptl_pvalue.feature), operator = 'not in').values())

    if verbose:
        pbar.update(10)
        pbar.set_description("Performing the first round of association analysis")

    if len(non_diff_only_features) > 0:

        ## Redo correlation analysis
        temp_corr = deepcopy(temp_corr_for_sec_round)
        del temp_corr_for_sec_round

        ## Filter corr data for only those corr between non_diff_only_features and themselves/others
        filter_corr_index = list(which(subject = temp_corr.row, source = non_diff_only_features, operator = 'in').keys()) + list(which(subject = temp_corr.column, source = non_diff_only_features, operator = 'in').keys())
        filter_corr_index = list(set(filter_corr_index))
        
        temp_corr = temp_corr.iloc[filter_corr_index]

        ## Filtering low level correlations
        cor_thresh = deepcopy(r)
        mr_thresh = deepcopy(mr)

        if cor_thresh_method == "mr":
            temp_corr = temp_corr[temp_corr['mr'] < mr_thresh]
        elif cor_thresh_method == "cor.coefficient":
                temp_corr = temp_corr[abs(temp_corr['cor']) > cor_thresh]

        ## Separate non_diff_only features
        temp_corr_diff_only_index = list(which(subject = temp_corr.row, source = rf_diff_exptl_pvalue.feature, operator = 'in').keys()) + list(which(subject = temp_corr.column, source = rf_diff_exptl_pvalue.feature, operator = 'in').keys())
        temp_corr_diff_only_index = list(set(temp_corr_diff_only_index))

        if len(temp_corr_diff_only_index) > 0:
            temp_corr = temp_corr.drop(temp_corr.index[temp_corr_diff_only_index], axis = 0)

        if temp_corr.shape[0] > (max_connections - diff_only_temp_corr.shape[0]):
            if cor_thresh_method == "mr":
                temp_corr_select_index = list(np.argsort(temp_corr.mr)[:(max_connections - diff_only_temp_corr.shape[0])])
            elif cor_thresh_method == "cor.coefficient":
                temp_corr_select_index = list(np.argsort(temp_corr.cor)[-(max_connections - diff_only_temp_corr.shape[0]):])

            temp_corr = temp_corr.iloc[temp_corr_select_index]

        ## Recombine the diff_only_temp_corr data and temp_corr
        temp_corr = pd.concat([temp_corr, diff_only_temp_corr], axis=0)

    else:
        temp_corr = deepcopy(diff_only_temp_corr)
        del diff_only_temp_corr

    if verbose:
        pbar.update(10)
        pbar.set_description("Network reconstruction")
    
    #d Graph reconstruction
    temp_corr_graph = igraph.Graph.DataFrame(edges = temp_corr.iloc[:,[0,1]], directed = False, use_vids=False)

    if verbose:
        pbar.update(5)
        pbar.set_description("Calculation of the integrated value of influence (IVI)")

    #e Calculation of IVI
    temp_corr_ivi = ivi(temp_corr_graph)

    if verbose:
        pbar.update(5)
        pbar.set_description("Calculation of the primitive driver score")

    ## Driver score and ranking

    #a Calculate first level driver score based on #3 and #4

    Diff_data_IVI_index = list(which(subject= list(temp_corr_ivi.Node_name), source= list(Diff_data.index), operator = 'in').values())

    if len(Diff_data_IVI_index) > 0:
        Diff_data['IVI'] = 0
        Diff_data.loc[Diff_data_IVI_index, ['IVI']] = list(temp_corr_ivi.IVI.iloc[list(which(subject= temp_corr_ivi.Node_name, source = Diff_data_IVI_index, operator = 'in').keys())])

        ## Range normalize the IVI
        Diff_data.IVI = rangeNormalize(data = Diff_data.IVI, minimum = 1, maximum = 100)

    else:
        Diff_data['IVI'] = 1
    
    Diff_data['first_Driver_Rank'] = 1

    for i in range(Diff_data.shape[0]):
        if any(Diff_data.iloc[i, Diff_value] < 0) and any(Diff_data.iloc[i, Diff_value] > 0):
            Diff_data['first_Driver_Rank'][i] = 0
        else:
            Diff_data['first_Driver_Rank'][i] = Diff_data.sum_Sig_value[i]*Diff_data.IVI[i]

    # Range normalize the first driver rank
    if any(Diff_data.first_Driver_Rank == 0):
        Diff_data.first_Driver_Rank = rangeNormalize(data = Diff_data.first_Driver_Rank, minimum = 0, maximum = 100)
    else:
        Diff_data.first_Driver_Rank = rangeNormalize(data = Diff_data.first_Driver_Rank, minimum = 1, maximum = 100)

    if verbose:
        pbar.update(5)
        pbar.set_description("Calculation of the neighborhood driver score")

    #b (#6) calculate neighborhood score

    ## Get the list of network nodes
    network_nodes = list(temp_corr_ivi.Node_name)

    neighborehood_score_table = pd.DataFrame({'node': network_nodes, 'N_score': 0})
    for i in range(neighborehood_score_table.shape[0]):

        first_neighbors = temp_corr_graph.neighborhood(vertices = neighborehood_score_table.node[i], order = 1, mode = 'all')
        ## Removing the node index of each node from the list of its neighborhood
        first_neighbors = first_neighbors[1:]
        first_neighbors = list(temp_corr_ivi.Node_name[first_neighbors])

        first_neighbors_index = list(which(subject = list(Diff_data.index), source = first_neighbors, operator = 'in').keys())
        neighborehood_score_table.N_score[i] = sum(Diff_data.first_Driver_Rank[first_neighbors_index])

    if verbose:
        pbar.update(5)
        pbar.set_description("Preparation of the driver table")

    Diff_data['N_score'] = 0
    Diff_data_N_score_index = list(which(subject = list(neighborehood_score_table.node), source = list(Diff_data.index), operator = 'in').values())
    Diff_data.loc[Diff_data_N_score_index, ['N_score']] = list(neighborehood_score_table.N_score[list(which(subject = list(neighborehood_score_table.node), source = Diff_data_N_score_index, operator = 'in').keys())])

    ## Range normalize (1,100) the neighborhood score
    Diff_data.N_score = rangeNormalize(data = Diff_data.N_score, minimum = 1, maximum = 100)

    #c calculate the final driver score
    Diff_data['final_Driver_score'] = (Diff_data.first_Driver_Rank)*(Diff_data.N_score)
    Diff_data.final_Driver_score[Diff_data.final_Driver_score==0] = np.nan

    # Create the Drivers table
    Driver_table = deepcopy(Diff_data)

    ## Remove the rows/features with NA in the final driver score
    Driver_table = Driver_table[Driver_table.final_Driver_score != np.nan]

    ## Filter the driver table by the desired list (if provided)
    if Desired_list is not None:
        Driver_table_row_index = list(which(subject = list(Driver_table.index), source = Desired_list, operator = 'in').keys())
        Driver_table = Driver_table.iloc[Driver_table_row_index]

    if pd.DataFrame(Driver_table).shape[0] == 0:
        Driver_table = None
    else:
        ### Range normalize final driver score
        if len(Driver_table.final_Driver_score.unique()) > 1:
            Driver_table.final_Driver_score = rangeNormalize(data = Driver_table.final_Driver_score, minimum = 1, maximum = 100)
        else:
            Driver_table.final_Driver_score = 1
        
        #add Z.score
        Driver_table['Z_score'] = ((Driver_table.final_Driver_score - np.mean(Driver_table.final_Driver_score)) / np.std(Driver_table.final_Driver_score))

        #add driver rank
        Driver_table['rank'] = rank_cal(list(Driver_table.final_Driver_score), order = -1)

        #add P-value
        Driver_table['p_value'] = 1 - norm.cdf(list(Driver_table.Z_score))

        #add adjusted pvalue
        Driver_table['padj'] = multipletests(pvals = Driver_table.p_value, method = 'fdr_bh')[1]

        #add driver type
        Driver_table['driver_type'] = ""

        for i in range(Driver_table.shape[0]):
            if sum(Driver_table.iloc[i,Diff_value]) < 0:
                Driver_table.driver_type[i] = "Decelerator"
            elif sum(Driver_table.iloc[i,Diff_value]) > 0:
                Driver_table.driver_type[i] = "Accelerator"
            else:
                Driver_table.driver_type[i] = np.nan

        Driver_table = Driver_table[Driver_table.driver_type != np.nan]

        #remove redundent columns
        Driver_table = Driver_table[["final_Driver_score",
                                      "Z_score",
                                      "rank",
                                      "p_value",
                                      "padj",
                                      "driver_type"]]
        
        #rename column names
        Driver_table = Driver_table.set_axis(["Score", "Z_score", "Rank", "P_value", "P_adj", "Type"], axis = 1)

        if pd.DataFrame(Driver_table).shape[0] == 0:
            Driver_table = None

    if verbose:
        pbar.update(5)
        pbar.set_description("Preparation of the biomarker table")

    # Create the Biomarker table

    Biomarker_table = deepcopy(Diff_data)

    ## Remove the rows/features with NA in the final driver score
    Biomarker_table = Biomarker_table[Biomarker_table.final_Driver_score != np.nan]

    ## Filter the biomarker table by the desired list (if provided)
    if Desired_list is not None:
        Biomarker_table_row_index = list(which(subject = list(Biomarker_table.index), source = Desired_list, operator= 'in').keys())
        Biomarker_table = Biomarker_table.iloc[Biomarker_table_row_index]

    if pd.DataFrame(Biomarker_table).shape[0] == 0:
        Biomarker_table = None
    else:

        #add RF importance score and p-value
        Biomarker_table['rf_importance'] = 0
        Biomarker_table['rf_pvalue'] = 1

        Biomarker_table_rf_index = list(which(subject = list(rf_diff_exptl_pvalue.feature), source = list(Biomarker_table.index), operator = 'in').values())

        rf_for_Biomarker_table = list(which(subject = list(rf_diff_exptl_pvalue.feature), source = Biomarker_table_rf_index, operator = 'in').keys())

        Biomarker_table.loc[Biomarker_table_rf_index, ['rf_importance']] = list(rf_diff_exptl_pvalue.importance.iloc[rf_for_Biomarker_table])
        Biomarker_table.loc[Biomarker_table_rf_index, ['rf_pvalue']] = list(rf_diff_exptl_pvalue.p_value.iloc[rf_for_Biomarker_table])

        ## Range normalize rf_importance and rf_pvalue
        Biomarker_table.rf_importance = rangeNormalize(data = Biomarker_table.rf_importance, minimum = 1, maximum = 100)
        Biomarker_table.rf_pvalue = rangeNormalize(data = Biomarker_table.rf_pvalue, minimum = 1, maximum = 100)

        ## Add rotation values
        Biomarker_table['rotation'] = 0
        Biomarker_table.loc[Biomarker_table_rf_index, ['rotation']] = list(map(lambda x: tmp_PCA_r[x], rf_for_Biomarker_table))

        ## Range normalize rotation values
        Biomarker_table.rotation = rangeNormalize(data = Biomarker_table.rotation, minimum = 1, maximum = 100)

        ## Calculate biomarker score
        if Regr_value is not None:
            Biomarker_table['final_biomarker_score'] = Biomarker_table.sum_Diff_value * Biomarker_table.sum_Regr_value * Biomarker_table.sum_Sig_value * Biomarker_table.rf_pvalue * Biomarker_table.rf_importance * Biomarker_table.rotation
        else:
            Biomarker_table['final_biomarker_score'] = Biomarker_table.sum_Diff_value * Biomarker_table.sum_Sig_value * Biomarker_table.rf_pvalue * Biomarker_table.rf_importance * Biomarker_table.rotation
        
        if pd.DataFrame(Biomarker_table).shape[0] == 0:
            Biomarker_table = None
        else:
            ### Range normalize final biomarker score
            if len(Biomarker_table.final_biomarker_score.unique()) > 1:
                Biomarker_table.final_biomarker_score = rangeNormalize(data = Biomarker_table.final_biomarker_score, minimum = 1, maximum = 100)
            else:
                Biomarker_table.final_biomarker_score = 1

        #add Z.score
        Biomarker_table['Z_score'] = ((Biomarker_table.final_biomarker_score - np.mean(Biomarker_table.final_biomarker_score)) / np.std(Biomarker_table.final_biomarker_score))

        #add biomarker rank
        Biomarker_table['rank'] = rank_cal(list(Biomarker_table.final_biomarker_score), order = -1)

        #add P-value
        Biomarker_table['p_value'] = 1 - norm.cdf(list(Biomarker_table.Z_score))

        #add adjusted pvalue
        Biomarker_table['padj'] = multipletests(pvals = Biomarker_table.p_value, method = 'fdr_bh')[1]

        #add biomarker type
        Biomarker_table['type'] = ""

        for i in range(Biomarker_table.shape[0]):
            if sum(Biomarker_table.iloc[i,Diff_value]) < 0:
                Biomarker_table.type[i] = "Down-regulated"
            elif sum(Biomarker_table.iloc[i,Diff_value]) > 0:
                Biomarker_table.type[i] = "Up-regulated"
            else:
                Biomarker_table.type[i] = np.nan

        Biomarker_table = Biomarker_table[Biomarker_table.type != np.nan]

        ## Remove redundent columns
        Biomarker_table = Biomarker_table[["final_biomarker_score",
                                            "Z_score", "rank",
                                            "p_value", "padj", "type"]]
        
        Biomarker_table = Biomarker_table.set_axis(["Score", "Z_score", "Rank", "P_value", "P_adj", "Type"], axis = 1)

        if pd.DataFrame(Biomarker_table).shape[0] == 0:
            Biomarker_table = None

    if verbose:
        pbar.update(5)
        pbar.set_description("Preparation of the DE-mediator table")

    # Create the DE mediators table

    DE_mediator_table = deepcopy(Diff_data)

    ## Include only rows/features with NaN in the final driver score (which are mediators)
    DE_mediator_table_index = list(i for i, x in enumerate(list(np.isnan(DE_mediator_table.final_Driver_score))) if x)
    
    DE_mediator_table = DE_mediator_table.iloc[DE_mediator_table_index]

    DE_mediator_table['DE_mediator_score'] = DE_mediator_table.sum_Sig_value * DE_mediator_table.IVI * DE_mediator_table.N_score

    ## Filter the DE mediators table by either the desired list or the list of network nodes
    DE_mediator_row_index = list(which(subject = list(DE_mediator_table.index), source = list(temp_corr_ivi.Node_name), operator= 'in').keys())

    if Desired_list is not None:
        desired_DE_mediator_row_index = list(which(subject = list(DE_mediator_table.iloc[DE_mediator_row_index].index), source = Desired_list, operator= 'in').keys())
        DE_mediator_row_index = list(map(lambda x: DE_mediator_row_index[x], desired_DE_mediator_row_index))

    DE_mediator_table = DE_mediator_table.iloc[DE_mediator_row_index]

    if pd.DataFrame(DE_mediator_table).shape[0] == 0:
        DE_mediator_table = None
    else:
        ### Range normalize DE mediators score
        if len(DE_mediator_table.DE_mediator_score.unique()) > 1:
            DE_mediator_table.DE_mediator_score = rangeNormalize(data = DE_mediator_table.DE_mediator_score, minimum = 1, maximum = 100)
        else:
            DE_mediator_table.DE_mediator_score = 1

        #add Z.score
        DE_mediator_table['Z_score'] = ((DE_mediator_table.DE_mediator_score - np.mean(DE_mediator_table.DE_mediator_score)) / np.std(DE_mediator_table.DE_mediator_score))

        #add biomarker rank
        DE_mediator_table['rank'] = rank_cal(list(DE_mediator_table.DE_mediator_score), order = -1)

        #add P-value
        DE_mediator_table['p_value'] = 1 - norm.cdf(list(DE_mediator_table.Z_score))

        #add adjusted pvalue
        DE_mediator_table['padj'] = multipletests(pvals = DE_mediator_table.p_value, method = 'fdr_bh')[1]

        ## Remove redundent columns
        DE_mediator_table = DE_mediator_table[["DE_mediator_score", "Z_score", "rank", "p_value", "padj"]]

        ## Rename column names
        DE_mediator_table = DE_mediator_table.set_axis(["Score", "Z_score", "Rank", "P_value", "P_adj"], axis = 1)

        ## Filtering redundant (NaN) results
        DE_mediator_table = DE_mediator_table[DE_mediator_table.Score != np.nan]

        if pd.DataFrame(DE_mediator_table).shape[0] == 0:
            DE_mediator_table = None

    if verbose:
        pbar.update(5)
        pbar.set_description("Preparation of the nonDE-mediator table")

    # Create the non-DE mediators table
    non_DE_mediators_index = list(which(subject = list(neighborehood_score_table.node), source = list(Diff_data.index), operator= 'in').keys())
    non_DE_mediators_index = list(set(non_DE_mediators_index))

    non_DE_mediators_table = neighborehood_score_table.drop(neighborehood_score_table.index[non_DE_mediators_index], axis = 0)

    if pd.DataFrame(non_DE_mediators_table).shape[0] == 0:
        non_DE_mediators_table = None
    else:
        ## Filter the non-DE mediators table by either the desired list
        if Desired_list is not None:
            non_DE_mediator_row_index = list(which(subject = list(non_DE_mediators_table.node), source = Desired_list, operator = 'in').keys())
            non_DE_mediators_table = non_DE_mediators_table.iloc[non_DE_mediator_row_index]

    if pd.DataFrame(non_DE_mediators_table).shape[0] == 0:
        non_DE_mediators_table = None
    else:
        non_DE_mediators_table.index = list(non_DE_mediators_table.node)

        non_DE_mediators_table['ivi'] = 1
        non_DE_mediators_ivi_index = list(which(subject = list(temp_corr_ivi.Node_name), source = list(non_DE_mediators_table.index), operator = 'in').values())
        
        non_DE_mediators_table.loc[non_DE_mediators_ivi_index, 'ivi'] = list(temp_corr_ivi.IVI.iloc[list(which(subject = list(temp_corr_ivi.Node_name), source = non_DE_mediators_ivi_index, operator = 'in').keys())])
        
        non_DE_mediators_table['non_DE_mediator_score'] = non_DE_mediators_table.N_score * non_DE_mediators_table.ivi

        ### Range normalize non-DE mediators score
        if len(non_DE_mediators_table.non_DE_mediator_score.unique()) > 1:
            non_DE_mediators_table.non_DE_mediator_score = rangeNormalize(data = non_DE_mediators_table.non_DE_mediator_score, minimum = 1, maximum = 100)
        else:
            non_DE_mediators_table.non_DE_mediator_score = 1

        #add Z.score
        non_DE_mediators_table['Z_score'] = ((non_DE_mediators_table.non_DE_mediator_score - np.mean(non_DE_mediators_table.non_DE_mediator_score)) / np.std(non_DE_mediators_table.non_DE_mediator_score))

        #add biomarker rank
        non_DE_mediators_table['rank'] = rank_cal(list(non_DE_mediators_table.non_DE_mediator_score), order = -1)

        #add P-value
        non_DE_mediators_table['p_value'] = 1 - norm.cdf(list(non_DE_mediators_table.Z_score))

        #add adjusted pvalue
        non_DE_mediators_table['padj'] = multipletests(pvals = non_DE_mediators_table.p_value, method = 'fdr_bh')[1]

        ## Remove redundent columns
        non_DE_mediators_table = non_DE_mediators_table[["non_DE_mediator_score", "Z_score", "rank", "p_value", "padj"]]

        ## Rename column names
        non_DE_mediators_table = non_DE_mediators_table.set_axis(["Score", "Z_score", "Rank", "P_value", "P_adj"], axis = 1)

        ## Filtering redundant (NaN) results
        non_DE_mediators_table = non_DE_mediators_table[non_DE_mediators_table.Score != np.nan]

        if pd.DataFrame(non_DE_mediators_table).shape[0] == 0:
            non_DE_mediators_table = None

    final_Results = {"Driver table": Driver_table,
                     "DE-mediator table": DE_mediator_table,
                     "nonDE-mediator table": non_DE_mediators_table,
                     "Biomarker table": Biomarker_table,
                     "Graph": temp_corr_graph}
    
    final_Results = {k: v for k, v in final_Results.items() if v is not None}

    if verbose:
        pbar.update(5)
        pbar.close()

    return final_Results
