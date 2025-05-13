# from __future__ import print_function
import importlib
import DataSetGen
import Evaluation
import asyncio
importlib.reload(DataSetGen)
importlib.reload(Evaluation)

import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np

import lime
import lime.lime_tabular

from anchor import utils
from anchor import anchor_tabular

import collections
from numba import njit, prange
import numba

import asyncio
from joblib import Parallel, delayed
import time
import multiprocessing

import logging

logger = logging.getLogger()



def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


#@numba.jit(parallel=True)
# @background
def get_exp_lime(data, explainer, feature_names,  model, num_feature=2, top_labels=1):
    num_feature=len(feature_names)

    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])
    explainer_lib = "lime"
    for count, i in enumerate(data.index.to_list()):
        logger.info(f"{count}, {i}")
        exp = explainer.explain_instance(data.loc[i], model.predict_proba, num_features=num_feature, top_labels=top_labels)
        logger.info(f"{exp.as_map().items()}")
        for exp_list in exp.as_map().items():
            explaination_df =  pd.concat([explaination_df, pd.DataFrame({"explainer_lib": [explainer_lib], "instance": [i], "features": [[feature_names[item[0]] for item in exp_list[1]]], "importance":  [[item[1] for item in exp_list[1]]] })])

    return explaination_df

def get_exp_lime_paralell(data, explainer, feature_names,  model, num_feature=2, top_labels=1, feature_selection_ = "auto"):
    logger.info(" get_exp_lime_paralell function start")
    num_feature=len(feature_names)
    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])
    explainer_lib = "lime"

    def inner_exp(data_, model, num_feature, top_labels, i_):

        e_ = explainer.explain_instance(data_, model.predict_proba,  top_labels=top_labels)
        # print("i", i_)
        return e_, i_


    # exp_lst = Parallel(n_jobs=-5, max_nbytes=None, backend="multiprocessing")(delayed(inner_exp)(data.loc[i], model, num_feature, top_labels, i) for i in data.index.to_list())
    exp_lst = Parallel(n_jobs=-2)(delayed(inner_exp)(data.loc[i], model, num_feature, top_labels, i) for i in data.index.to_list())


    for exp, i in exp_lst:
        for exp_list in exp.as_map().items():
            explaination_df =  pd.concat([explaination_df, pd.DataFrame({"explainer_lib": [explainer_lib], "instance": [i], "features": [[feature_names[item[0]] for item in exp_list[1]]], "importance":  [[item[1] for item in exp_list[1]]] })])
    logger.info(" get_exp_lime_paralell function end")

    return explaination_df




# Anchor explainer

# @background
def get_exp_anchor(data, explainer, precision=.95, model=None):

    explainer_lib = "anchor"
    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])

    for count, i in enumerate(data.index.to_list()):
        # print(count, i)
        explanation = explainer.explain_instance(data.loc[i].values, model.predict, threshold=precision)
        # vars = list(map((lambda x: x.split()[0]), explanation.names()))
        vars = [i   for x in explanation.names() for i in  x.split() if i in data.columns.to_list()  ]
        # print(explanation.names(), vars)
        explaination_df = pd.concat([explaination_df, pd.DataFrame(
            {"explainer_lib": [explainer_lib], "instance": [i], "features": [vars],
             "importance": [list(range(1, len(vars) + 1)) ] })])
    return  explaination_df


def get_exp_anchor_parallel(data, explainer, precision=.95, model=None):
    logger.info("get_exp_anchor_parallel function start")
    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])
    explainer_lib = "anchor"

    def inner_exp(data_, model_,  i_):
        e_ = explainer.explain_instance(data_.values, model_.predict, threshold=precision)

        # print("i", i_)
        return e_, i_

    # exp_lst = Parallel(n_jobs=-5, max_nbytes=None, backend="multiprocessing")(delayed(inner_exp)(data.loc[i], model,  i) for i in data.index.to_list())
    exp_lst = Parallel(n_jobs=4)(delayed(inner_exp)(data.loc[i], model,  i) for i in data.index.to_list())


    for exp, i in exp_lst:
        # vars = list(map((lambda x: x.split()[0]), explanation.names()))
        vars = [i for x in exp.names() for i in x.split() if i in data.columns.to_list()]
        logger.debug(f"exp.names():{exp.names()}, vars {vars}")
        explaination_df = pd.concat([explaination_df, pd.DataFrame(
            {"explainer_lib": [explainer_lib], "instance": [i], "features": [vars],
             "importance": [list(range(1, len(vars) + 1))]})])

    logger.info("get_exp_anchor_parallel function end")
    return  explaination_df

