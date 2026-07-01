#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 29 June 2026

@author Sergio Chávez
"""
"""
Function below runs multiple NMF algorithms and clusters the resulting /
signatures of each run. The mean of each cluster is the final signature 
"""


import pandas as pd
import os
import sys

from surgeNMF.surgeKorax import Crows

def Anhinga():
    Data = pd.read_csv("data.txt", sep="\t", header=0, index_col=0)
    Data.to_csv("data_no_names.txt", sep="\t", header=False, index=False)

    Crows(
        "data_no_names.txt",
        "Dummy_result",
        min_num_sigs=3,
        max_num_sigs=5,
        num_trials=20,
        num_cores=20,
        which_nmf="St Frob",
        Lambdas=[1, 0.5],
        seed=None,
        cosmic=False
    )


if __name__ == "__main__":

    Anhinga()