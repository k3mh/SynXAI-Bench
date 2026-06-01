## Create Model
import sys
import os
import gc
import logging
sys.path.append(os.getcwd())
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import importlib
import DataSetGen_v2 as dg
import Evaluation
import Explaination
import Evaluation_plots
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from xgboost import XGBClassifier
import time

from sklearn.metrics import accuracy_score, roc_auc_score
import lime.lime_tabular
from anchor import anchor_tabular
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(format='%(asctime)s: %(module)s: %(message)s')

# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info("=================================================================================")
logger.info("====================================== Start ====================================")


importlib.reload(dg)
importlib.reload(Evaluation)
importlib.reload(Explaination)
importlib.reload(Evaluation_plots)

from numba import njit, prange
import numba


###### for all datasets
def dataset_signals(count, dsts_lst, dsts_prop):
    assert np.sum(dsts_prop) <= 1  # make sure proportions percentages are sum to 1 or less  .
    assert len(dsts_prop) == len(dsts_lst)  # make sure that percentages are the same number as the datasets.

    ds = dg.dataset()
    funs = ds.get_gen_fn_list()
    generated_ds = pd.DataFrame()
    metadata_df = pd.DataFrame()

    for iter_, dataset_index in enumerate(dsts_lst):


        func_= funs[dataset_index]
        prop_size = int(dsts_prop[iter_] * count)

        generated_ds = pd.concat([generated_ds, func_(prop_size).get_data()])
        metadata_df  = pd.concat([ metadata_df, ds.meta_data])


        #print(generated_ds.head())
        logger.info(ds.meta_data.shape)
        features_names = ds.features_names

    return  generated_ds, metadata_df

dataset_size= 300
Dataframe_list = []
accuracy_lst = []
auc_list = []
features_names = dg.dataset.features
target_name = "y"
results_df = pd.DataFrame(columns=["dataset", "metric", "lib", "score", "score_list"])
random_state = 19

load_datasets = False
save_datasets = True

skip_explanations = False
load_explanations = False
save_explanations = True

skip_iter_evaluation = False
load_iter_evaluation = False
save_iter_evaluation = True

load_final_result = False
save_final_result = True

get_score = True

generate_plots = True



unique_id = datetime.now().strftime("%d%b%Y_%Hh%Mm%Ss")
#unique_id = '13Apr2020_22h57m27s'

# parent_dir = "Dataset\\" + unique_id + "\\"
parent_dir = os.path.join("Dataset", unique_id, "")
Path(parent_dir).mkdir(parents=True, exist_ok=True)

if not load_final_result:

    #datasets_seq_lst = [
                         # [0],
                  #       [1,1],
                  #       [2],
                  #       [3],
                  #       [4],
                  #       [5],
                  #       [6],
                  #       [7],
                  #       [8],
                  #       [9],
                   #      [10],
                   ##      [11],
                   #      [12,11],
                         # [0],
                         # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    # ]
    # datasets_seq_lst =                    list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 5))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 6))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 7))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 8))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 9))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 10))
    # datasets_seq_lst = datasets_seq_lst + list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 11))


    # selected based on 10s percentiles
    datasets_seq_lst = [
    [1, 3],
    [3, 9, 10],
    [1, 5, 6, 11, 12],
    [1, 3, 7, 9],
    [2, 3, 4, 5, 6, 8, 10, 12],
    [2, 3, 5, 6, 8, 9, 10, 11],
    [1, 2, 3, 5, 6, 8, 9, 10, 11, 12],
    [1, 4, 5, 6, 7, 8, 10, 11, 12],
    [2, 3, 4, 5, 7, 8, 10],
    [4, 5, 6, 7, 8, 10, 11, 12],
    [5, 6, 7, 9, 10]
    ]



    datasets_prop_lst = []

    data_lst_filename = parent_dir +  "datasets_seq_lst_" + str(dataset_size) + ".txt"
    datasets_seq_file = Path(data_lst_filename)
    if not datasets_seq_file.is_file():
        pd.DataFrame(datasets_seq_lst).to_csv(data_lst_filename)


    # equal proportional parts
    for i in datasets_seq_lst:

        datasets_prop_lst.append([np.floor((1/len(i)) * 1000) / 1000] * len(i))


    for iter_ in  range(len(datasets_seq_lst)):
        logger.info("==================== Dataset start =======================")
        logger.info("Dataset :{} out of {}".format(iter_+1, len(datasets_seq_lst)))
        gc.collect()
        dsts_lst = datasets_seq_lst[iter_]
        dsts_prop = datasets_prop_lst[iter_]
        precision_lst1 = []
        precision_lst2 = []
        FPR_lst1 = []
        FPR_lst2 = []
        sens_lst1 = []
        sens_lst2 = []
        results_df_iter = pd.DataFrame(columns=["dataset", "metric", "lib", "score", "score_list"])

        if load_datasets:
            dataset = pd.read_pickle(parent_dir + "dataset_" + str(iter_) + ".pkl")
            metadata = pd.read_pickle(parent_dir + "metadata_" + str(iter_) + ".pkl")
        else:
            dataset, metadata = dataset_signals(dataset_size, dsts_lst, dsts_prop)
            dataset.reset_index(drop=True, inplace=True)
            metadata.reset_index(drop=True, inplace=True)

        if save_datasets:
            dataset.to_pickle(parent_dir + "dataset_" + str(iter_) + ".pkl")
            metadata.to_pickle(parent_dir + "metadata_" + str(iter_) + ".pkl")

        Dataframe_list.append(dataset)

        train, test, labels_train, labels_test = train_test_split(Dataframe_list[iter_][features_names],
                                                              Dataframe_list[iter_][target_name], train_size=0.90, random_state= random_state, stratify=Dataframe_list[iter_][target_name])

        meta_train = metadata.loc[train.index]
        meta_test = metadata.loc[test.index]


        ## Training
        rf = RandomForestClassifier(n_jobs=50)
        # rf = ExtraTreesClassifier()
        # rf = XGBClassifier()
        rf.fit(train.values, labels_train)
        accuracy_lst.append(accuracy_score(labels_test, rf.predict(test.values)))
        auc_list.append(roc_auc_score(labels_test, rf.predict_proba(test.values)[:, 1]))
        logger.info(f"accuracy_list: {np.round(accuracy_lst, 4)}")
        logger.info(f"auc_list     : {np.round(auc_list, 4)}")

        ########## explanations
        if not skip_explanations:
            # Lime explainer
            if load_explanations:
                explaination_lime   = pd.read_pickle(parent_dir + "lime_exp_" + str(iter_) + ".pkl")
                explaination_anchor = pd.read_pickle(parent_dir + "anchor_exp_" + str(iter_) + ".pkl")

            else:
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(train.values, feature_names=features_names,
                                                                        class_names=target_name, discretize_continuous=True)
                start = time.time()
                # explaination_lime = Explaination.get_exp_lime(test, lime_explainer, features_names, rf)
                explaination_lime = Explaination.get_exp_lime_paralell(test, lime_explainer, features_names, rf)

                end = time.time()
                logger.info("Lime time elapsed")
                logger.info(end - start)
                #explaination_lime.to_pickle(lime_explanation_file)



                # Anchor explainer

                # Note: old, anchor_explainer preparation, not sure if working
                # anch_explainer = anchor_tabular.AnchorTabularExplainer(class_names=target_name, feature_names=features_names,
                #                                                        data=train, categorical_names={})

                # anch_explainer.fit(train.values, labels_train, test.values, labels_test)
                # predict_fn = lambda x: rf.predict(anch_explainer.encoder.transform(x))
                # Note end


                anch_explainer = anchor_tabular.AnchorTabularExplainer(class_names=labels_test.unique(),
                                                                       feature_names=features_names, train_data=train.values)


                start = time.time()
                explaination_anchor = Explaination.get_exp_anchor_parallel(test, anch_explainer, 0.8, rf)
                # explaination_anchor = Explaination.get_exp_anchor(test, anch_explainer, 0.8, rf)

                end = time.time()
                logger.info("Anchor time elapsed")
                logger.info(end - start)

                #explaination_anchor.to_pickle(anchor_explanation_file)

            if save_explanations:
                explaination_lime.to_pickle(parent_dir + "lime_exp_" + str(iter_) + ".pkl")
                explaination_anchor.to_pickle(parent_dir + "anchor_exp_" + str(iter_) + ".pkl")

        ########### evaluation
        if not skip_iter_evaluation:
            importlib.reload(Evaluation)
            if load_iter_evaluation:
                results_df_iter =  pd.read_pickle(parent_dir + "evaluation_" + str(iter_) + ".pkl")
            else:
                # precision_lst1.append(Evaluation.Precision(meta_test, explaination_lime, True))
                recall_, recall_list = Evaluation.Recall(meta_test, explaination_lime, True)

                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["recall_partial"], "lib": ["lime"], "score": [recall_],
                     "score_list": [recall_list], "RGS": [meta_test.RGS.to_list()] })])

                # precision_lst2.append(Evaluation.Precision(meta_test, explaination_anchor, True))
                recall_, recall_list = Evaluation.Recall(meta_test, explaination_anchor, True)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["recall_partial"], "lib": ["anchor"], "score": [recall_],
                     "score_list": [recall_list], "RGS": [meta_test.RGS.to_list()] })])

                # precision_lst1.append(Evaluation.Precision(meta_test, explaination_lime, False))
                recall_, recall_list = Evaluation.Recall(meta_test, explaination_lime, False)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["recall"], "lib": ["lime"], "score": [recall_],
                     "score_list": [recall_list], "RGS": [meta_test.RGS.to_list()] })])

                # precision_lst2.append(Evaluation.Precision(meta_test, explaination_anchor, False))
                recall_, recall_list = Evaluation.Recall(meta_test, explaination_anchor, False)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["recall"], "lib": ["anchor"], "score": [recall_],
                     "score_list": [recall_list], "RGS": [meta_test.RGS.to_list()] })])

                # FPR all list
                # FPR_lst1.append(Evaluation.FPR(meta_test, explaination_lime, ds2.features_names))
                FPR_, fpr_list = Evaluation.FPR(meta_test, explaination_lime, features_names)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["FPR"], "lib": ["lime"], "score": [FPR_], "score_list": [fpr_list], "RGS": [meta_test.RGS.to_list()] })])

                # FPR_lst2.append(Evaluation.FPR(meta_test, explaination_anchor, ds2.features_names))
                FPR_, fpr_list= Evaluation.FPR(meta_test, explaination_anchor, features_names)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["FPR"], "lib": ["anchor"], "score": [FPR_], "score_list": [fpr_list], "RGS": [meta_test.RGS.to_list()] })])

                # Sensitivity
                sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_lime, 4)
                # sens_lst1.append(sens_summ)
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["sensitivity"], "lib": ["lime"], "score": [sens_summ],
                     "score_list": [sens_df], "RGS": [meta_test.RGS.to_list()] })])

                logger.info(sens_summ)
                sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_anchor, 4)
                #sens_lst2.append(sens_summ)
                #temp_df =
                results_df_iter = pd.concat([results_df_iter, pd.DataFrame(
                    {"dataset": [iter_ + 1], "metric": ["sensitivity"], "lib": ["anchor"], "score": [sens_summ],
                     "score_list": [sens_df], "RGS": [meta_test.RGS.to_list()] })])
                logger.info(sens_summ)

                if save_iter_evaluation :
                    results_df_iter.to_pickle(parent_dir + "evaluation_" + str(iter_) + ".pkl")

                results_df = pd.concat([results_df, results_df_iter])
else:
    results_df = pd.read_pickle(parent_dir + "evaluation_" + "all" + ".pkl")
    temp_df = pd.read_csv(parent_dir + "model_performance.csv")
    accuracy_lst = temp_df.accuracy.to_list()
    auc_list = temp_df.auc.to_list()

if save_final_result:
    results_df.to_pickle(parent_dir + "evaluation_" + "all" + ".pkl")
    pd.DataFrame({"accuracy": accuracy_lst, "auc":auc_list, "datasets_list":datasets_seq_lst}).to_csv(parent_dir + "model_performance.csv")

results_df_str = results_df.astype("str")
if generate_plots:
    importlib.reload(Evaluation_plots)
    Evaluation_plots.plot_results(parent_dir)

if get_score:
    importlib.reload(Evaluation)

    scaled_complexity = [x for x  in range(1,12)]

    score_lime_all   , score_lime_lst   = Evaluation.overall_score(results_df.loc[results_df.lib == 'lime'], scaled_complexity, accuracy_lst)
    score_anchor_all , score_anchor_lst = Evaluation.overall_score(results_df.loc[results_df.lib == 'anchor'], scaled_complexity, accuracy_lst)




# test 1 : FPR (1)
# test 2 : FP + univariate (1,2)
# test 3 : univariate + two independent variable (2,3)
# test 4 : univariate + two independent variable + non linear relation (2,3,4)
# test 5 : univariate + categorical (2,5)
# test 6 : univariate + categorical and cont (2,6)
# test 7 : univariate + condotional relation (2,7)
# test 8 : categorical + categorical cont (5,6)
# test 9 : categorical + categorical cont + condotional relation(5,6,7)
# test 10: categorical + non uniform categorical cont (5,8)
# test 11: univariate + complex conditions 4 variables (2,9)
# test 12: univariate + complex conditions 5 variables (2,10)
# test 13: univariate + conditional 5 variables (2,11)
