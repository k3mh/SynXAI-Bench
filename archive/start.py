## Create Model
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import importlib
import DataSetGen as dg
import Evaluation
import Explaination
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lime.lime_tabular
from anchor import anchor_tabular
importlib.reload(dg)
importlib.reload(Evaluation)
importlib.reload(Explaination)

from numba import njit, prange
import numba

# ds1 = dg.dataset()
# df1 = ds1.generate_ds1(1000).get_data()
# feature_names = ds1.features_names
# target = "y"
# target_names = np.array(["0", "1"])
# train, test, labels_train, labels_test = train_test_split(df1[feature_names], df1[target], train_size=0.80)
# meta_train = ds1.meta_data.loc[train.index]
# meta_test  = ds1.meta_data.loc[test.index]
# rf = RandomForestClassifier(n_estimators=500)
# rf.fit(train, labels_train)
# accuracy_score(labels_test, rf.predict(test))
#
# # Lime explainer
# lime_explainer = lime.lime_tabular.LimeTabularExplainer(train.values, feature_names=feature_names, class_names=target_names, discretize_continuous=True)
# explaination_df = Explaination.get_exp_lime(test, lime_explainer, feature_names,  rf)
#
# # Anchor explainer
# anch_explainer = anchor_tabular.AnchorTabularExplainer(class_names=target_names, feature_names=feature_names, data=train, categorical_names={})
# anch_explainer.fit(train.values, labels_train, test.values, labels_test)
# predict_fn = lambda x: rf.predict(anch_explainer.encoder.transform(x))
#
# explaination_df2 = Explaination.get_exp_anchor(test, anch_explainer, 0.8, rf)
#
#
#### Evaluation
# TPR partial
# importlib.reload(Evaluation)
#
# Evaluation.Precision(meta_test, explaination_df, True)
# Evaluation.Precision(meta_test, explaination_df2, True)
#
# # FPR all list
# Evaluation.FPR(meta_test, explaination_df, ds1.features_names)
# Evaluation.FPR(meta_test, explaination_df2, ds1.features_names)
#
# # Sensitivity
# sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_df, 4)
# print(sens_summ)
# sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_df2, 4)
# print(sens_summ)
#

###### for all datasets




importlib.reload(dg)
exp_path = "./expDatasets/"
dataset_path = "./Dataset/"
results_path = "./results/"
read_data_from_disk = False
dataset_size = 30000
ds2 = dg.dataset()
Dataframe_list = []
funs = ds2.get_gen_fn_list()
precision_lst1 = []
precision_lst2 = []
FPR_lst1 =  []
FPR_lst2 =  []
sens_lst1 = []
sens_lst2 = []
target_names = np.array(["0", "1"])
target = 'y'
results_df = pd.DataFrame(columns=["dataset", "metric", "lib", "score", "score_list"])
train, test, labels_train, labels_test = [None] * 4


for iter, fun_ in enumerate(funs):
    print("===============" + str(iter) + "=================")

    # lime_explanation_file   = exp_path + 'explanation_lime_'   + str(dataset_size) + "_" + str(iter) + '.csv'
    # anchor_explanation_file = exp_path + 'explanation_anchor_' + str(dataset_size) + "_" + str(iter) + '.csv'
    # train_dataset_file      = dataset_path + "dataset_train"   + str(dataset_size) + "_" + str(iter) + ".csv"
    # test_dataset_file       = dataset_path + "dataset_test"    + str(dataset_size) + "_" + str(iter) + ".csv"
    # labels_train_file       = dataset_path + "labels_train"    + str(dataset_size) + "_" + str(iter) + ".csv"
    # labels_test_file        = dataset_path + "labes_test"      + str(dataset_size) + "_" + str(iter) + ".csv"
    # meta_train_file         = dataset_path + "meta_train"      + str(dataset_size) + "_" + str(iter) + ".csv"
    # meta_test_file          = dataset_path + "meta_test"       + str(dataset_size) + "_" + str(iter) + ".csv"

    lime_explanation_file   = exp_path + 'explanation_lime_'   + str(dataset_size) + "_" + str(iter) + '.pkl'
    anchor_explanation_file = exp_path + 'explanation_anchor_' + str(dataset_size) + "_" + str(iter) + '.pkl'
    train_dataset_file      = dataset_path + "dataset_train"   + str(dataset_size) + "_" + str(iter) + ".pkl"
    test_dataset_file       = dataset_path + "dataset_test"    + str(dataset_size) + "_" + str(iter) + ".pkl"
    labels_train_file       = dataset_path + "labels_train"    + str(dataset_size) + "_" + str(iter) + ".pkl"
    labels_test_file        = dataset_path + "labes_test"      + str(dataset_size) + "_" + str(iter) + ".pkl"
    meta_train_file         = dataset_path + "meta_train"      + str(dataset_size) + "_" + str(iter) + ".pkl"
    meta_test_file          = dataset_path + "meta_test"       + str(dataset_size) + "_" + str(iter) + ".pkl"
    features_names_file     = dataset_path + "features_names"  + str(dataset_size) + "_" + str(iter) + ".txt"
    results_file            = results_path + "results"         + str(dataset_size) +".plk"

    if os.path.isfile(lime_explanation_file)   and os.path.isfile( anchor_explanation_file) and \
            os.path.isfile(train_dataset_file) and os.path.isfile(test_dataset_file) and \
            os.path.isfile(labels_train_file)  and os.path.isfile(labels_test_file) and\
            os.path.isfile(meta_train_file)    and os.path.isfile(meta_test_file):
        print( " Mode : Read data from disk")
        read_data_from_disk = True

    else :
        print( " Mode : Calculate data")
        read_data_from_disk = False

    if read_data_from_disk:

        # with open(train_dataset_file, 'r') as filehandle:
        #     for line in filehandle:
        #         train = line[:-1]

        # train = pd.read_csv(train_dataset_file).set_index("row_index")
        # test = pd.read_csv(test_dataset_file).set_index("row_index")
        #
        # labels_train = pd.read_csv(labels_train_file).set_index("row_index")
        # labels_test = pd.read_csv(labels_test_file).set_index("row_index")
        #
        # meta_train = pd.read_csv(meta_train_file).set_index("row_index")
        # meta_test  = pd.read_csv(meta_test_file).set_index("row_index")


        train = pd.read_pickle(train_dataset_file)
        test = pd.read_pickle(test_dataset_file)

        labels_train = pd.read_pickle(labels_train_file)
        labels_test = pd.read_pickle(labels_test_file)

        meta_train = pd.read_pickle(meta_train_file)
        meta_test  = pd.read_pickle(meta_test_file)

        with open(features_names_file, 'r') as filehandle:
            features_names= []
            for line in filehandle:
                features_names.append(line[:-1])

    else:
        Dataframe_list.append(fun_(dataset_size).get_data())
        features_names = ds2.features_names
        train, test, labels_train, labels_test = train_test_split(Dataframe_list[iter][features_names],
                                                                  Dataframe_list[iter][target], train_size=0.80)

        meta_train = ds2.meta_data.loc[train.index]
        meta_test = ds2.meta_data.loc[test.index]

        # with open(test_dataset_file, 'w') as filehandle:
        #     for listitem in test:
        #         filehandle.write('%s\n' % listitem)

        # train.to_csv(train_dataset_file, index_label= "row_index")
        # test.to_csv(test_dataset_file, index_label="row_index")
        #
        # labels_train.to_csv(labels_train_file, index_label= "row_index")
        # labels_test.to_csv(labels_test_file, index_label= "row_index")
        #
        # meta_train.to_csv(meta_train_file, index_label= "row_index")
        # meta_test.to_csv(meta_test_file, index_label= "row_index")


        train.to_pickle(train_dataset_file)
        test.to_pickle(test_dataset_file)

        labels_train.to_pickle(labels_train_file)
        labels_test.to_pickle(labels_test_file)

        meta_train.to_pickle(meta_train_file)
        meta_test.to_pickle(meta_test_file)

        with open(features_names_file, 'w') as filehandle:
            for listitem in features_names:
                filehandle.write('%s\n' % listitem)


    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train, labels_train)
    accuracy_score(labels_test, rf.predict(test))
    ########## explanations
    # Lime explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(train.values, feature_names=features_names,
                                                            class_names=target_names, discretize_continuous=True)

    if read_data_from_disk:
        print ("File exist")
        #explaination_lime  = pd.read_csv(lime_explanation_file).set_index("row_index")
        explaination_lime = pd.read_pickle(lime_explanation_file)
    else:
        print ("File not exist")
        explaination_lime = Explaination.get_exp_lime(test, lime_explainer, features_names, rf)
        #explaination_lime.to_csv(lime_explanation_file, index_label= "row_index")
        explaination_lime.to_pickle(lime_explanation_file)

    # Anchor explainer
    anch_explainer = anchor_tabular.AnchorTabularExplainer(class_names=target_names, feature_names=features_names,
                                                           data=train, categorical_names={})
    anch_explainer.fit(train.values, labels_train, test.values, labels_test)
    predict_fn = lambda x: rf.predict(anch_explainer.encoder.transform(x))


    if read_data_from_disk:
        print ("File exist")
        #explaination_anchor  = pd.read_csv(anchor_explanation_file).set_index("row_index")
        explaination_anchor = pd.read_pickle(anchor_explanation_file)

    else:
        print ("File not exist")
        explaination_anchor = Explaination.get_exp_anchor(test, anch_explainer, 0.8, rf)
        #explaination_anchor.to_csv(anchor_explanation_file, index_label= "row_index")
        explaination_anchor.to_pickle(anchor_explanation_file)


    importlib.reload(Evaluation)
    ########### evaluation

    #precision_lst1.append(Evaluation.Precision(meta_test, explaination_lime, True))
    percision_,  percision_list = Evaluation.Recall(meta_test, explaination_lime, True)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["percision_partial"], "lib": ["lime"], "score" : [percision_],  "score_list" : [percision_list]})])

    #precision_lst2.append(Evaluation.Precision(meta_test, explaination_anchor, True))
    percision_, percision_list = Evaluation.Recall(meta_test, explaination_anchor, True)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["percision_partial"], "lib": ["anchor"], "score" : [percision_],  "score_list" : [percision_list]})])

    #precision_lst1.append(Evaluation.Precision(meta_test, explaination_lime, False))
    percision_,  percision_list = Evaluation.Recall(meta_test, explaination_lime, False)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["percision"], "lib": ["lime"], "score" : [percision_],  "score_list" : [percision_list]})])

    #precision_lst2.append(Evaluation.Precision(meta_test, explaination_anchor, False))
    percision_, percision_list = Evaluation.Recall(meta_test, explaination_anchor, False)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["percision"], "lib": ["anchor"], "score" : [percision_],  "score_list" : [percision_list]})])


    # FPR all list
    #FPR_lst1.append(Evaluation.FPR(meta_test, explaination_lime, ds2.features_names))
    FPR_, fpr_list = Evaluation.FPR(meta_test, explaination_lime, features_names)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["FPR"], "lib": ["lime"], "score" : [FPR_], "score_list": [fpr_list]})])

    #FPR_lst2.append(Evaluation.FPR(meta_test, explaination_anchor, ds2.features_names))
    FPR_ = Evaluation.FPR(meta_test, explaination_anchor, features_names)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["FPR"], "lib": ["anchor"], "score" : [FPR_], "score_list": [fpr_list]})])


    # Sensitivity
    sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_lime, 4)
    #sens_lst1.append(sens_summ)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["sensitivity"], "lib": ["lime"], "score" : [sens_summ], "score_list" :[sens_df]})])

    print(sens_summ)
    sens_summ, sens_df = Evaluation.sensetivity(meta_test, explaination_anchor, 4)
    sens_lst2.append(sens_summ)
    results_df = pd.concat([results_df, pd.DataFrame({"dataset": [iter + 1], "metric" : ["sensitivity"], "lib": ["anchor"], "score" : [sens_summ], "score_list" :[sens_df]})])
    print(sens_summ)

if not read_data_from_disk:
    results_df.to_pickle(results_file)