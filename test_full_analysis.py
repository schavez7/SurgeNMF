#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 4 June 2026

@author Sergio Chávez
"""
"""
Function below runs multiple NMF algorithms and clusters the resulting /
signatures of each run. The mean of each cluster is the final signature 
"""


import numpy as np 
import pandas as pd
import string
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import time

from surgeNMF import NMF
from surgePeacock import plot_signatures
from surgeCrows import Crows

import os
import sys
from multiprocessing import Pool
from functools import partial



def main_function(): 
    # ----Performs the entire analysis on a data set---- #

    # -------------------------------------------------------------------------------- #
    # INITIAL INPUTS CHECK
    if len(sys.argv) < 5:
        print("Utilisation: python surgeParallel.py <fichier info> <dir data> <dir results Gauss> <dir results Poisson>")
        sys.exit(1)


    # -------------------------------------------------------------------------------- #
    # SYSTÈME
    fichier_info       = sys.argv[1]
    dir_data           = sys.argv[2]
    dir_resultats_Gaus = sys.argv[3] 
    dir_resultats_Pois = sys.argv[4]

    # Télécharger les informations du fichier txt
    info = pd.read_csv(fichier_info, sep="\t", header=None, index_col=0)
    seed     = int(info.loc["seed"].values[0])
    num_obs  = int(info.loc["num_obs"].values[0])
    num_sigs = int(info.loc["num_sigs"].values[0])
    if num_sigs < 2: 
        print(f"   {num_sigs} n'est pas un entier positif supérieur à 1")
        sys.exit(1)

    # Fichier de data 
    fichier_data = os.path.join(dir_data, "data_Gaussian_distr.txt")

    nmf_results = Crows(
        fichier_data, 
        dir_resultats_Gaus, 
        min_num_sigs=3, 
        max_num_sigs=5, 
        num_trials=20,
        num_cores=20,
        which_nmf="VR Gaussian",
        # which_nmf="St Frob",
        seed=seed
    )

    # # Les noms des denovo signatures
    # alphabeta   = string.ascii_uppercase
    # nomsColonnes_temp = [f"Sig {i}" for i in alphabeta[:num_sigs]]

    # num_trials = 20
    # which_nmf = "St Frob"
    # Lambdas = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3,1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    # lam = Lambdas[0]
    # # W, score_temp = nmf_clustering(seed, num_sigs, num_trials, which_nmf=which_nmf, lam=lam, num_cores=20, datafilepath=fichier_data)
    
    print("Ça marche!")


if __name__ == "__main__":

    main_function()

    # # File paths (data, parameters,)
    # datafilepath  = sys.argv[1]
    # infopath  = sys.argv[2]
    # plotspath = sys.argv[3]

    # # Other passed parameters
    # seed       = int(sys.argv[4])
    # threshold  = float(sys.argv[5]) # Typically equal or larger than 0.85
    # num_trials = int(sys.argv[6])   # Typically equal or larger than 20

    # # # Plot some catalogues to understand how they look 
    # # data_original_df = pd.read_csv(datafilepath, sep="\t", index_col=0)
    # # data_original    = data_original_df.to_numpy(dtype=float)
    # # plotfilename     = os.path.join(plotspath,"Sample_catalogues")
    # # plot_signatures(data_original[:,:4],save_loc=plotfilename)

    # # Signature search through these numbers and lambda values
    # Sig_Range = range(3,6)
    # Lambdas = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3,1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    # # which_nmf = "St Frob"
    # # for num_sigs in range(2,3):
    # #     print("Number of signatures:",num_sigs,"------------------------------")
    # #     W = nmf_clustering(seed, num_sigs, num_trials, which_nmf=which_nmf, lam=None, num_cores=8)

    # # which_nmf = "VR Gaussian"
    # which_nmf = "VR Poisson"

    # Scores = np.zeros((len(Sig_Range),len(Lambdas)))
    # for num_sigs in Sig_Range:
    #     print("\nNumber of signatures:",num_sigs,"------------------------------")
    #     for k in range(len(Lambdas)):
    #         lam = Lambdas[k]
    #         W, score_temp = nmf_clustering(seed, num_sigs, num_trials, which_nmf=which_nmf, lam=lam, num_cores=8, datafilepath=datafilepath)
    #         Scores[num_sigs-Sig_Range[0],k] = score_temp 

    # # *** Next: plot the winning signatures. This might have to go somewhere below


    # # Default to best lambda in first signature try if no score is above threshold
    # num_sigs_temp = 0
    # row_max = Scores[0,:].max()
    # lam_indx_temp = np.where(Scores[0,:] == row_max)[0][0]
    # # print("  Best lambda value",Lambdas[lam_indx_temp],"is at index",lam_indx_temp)

    # # Now test each other signature 
    # for sig in range(1,len(Scores)):
    #     # Take just one line and find its max 
    #     ligne = Scores[sig]
    #     max_val = max(ligne) 

    #     # Next find its location relative to the line and pull out its value 
    #     ind = np.where(ligne==max_val)[0][0]
    #     val = ligne[ind] 
        
    #     # Check and if still larger than the threshold, change it for that one
    #     if (val >= threshold):
    #         num_sigs_temp = sig 
    #         lam_indx_temp = ind 

    # # Collect and print into info.txt
    # num_sigs = num_sigs_temp + Sig_Range[0]
    # lam = Lambdas[lam_indx_temp]
    # with open(infopath, "a") as f: 
    #     f.write(f"num_sigs\t{num_sigs}\n")
    #     f.write(f"lambda\t{lam}\n")

    # print("\n")
    # print("  Best signature is at",num_sigs)
    # print("  Best lambda value",lam,"\n")

    # # After obtaining a good lambda value rerun at more bootstraps
    # W, _ = nmf_clustering(seed, num_sigs, 100, which_nmf=which_nmf, lam=lam, num_cores=8, datafilepath=datafilepath)
    # plotfilename = os.path.join(plotspath,"GSNMF_VR_"+which_nmf)
    # # plot_signatures(W,save_loc=plotfilename)
