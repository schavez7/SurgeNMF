#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 25 June 2026

@author Sergio Chávez
"""
"""
Function below runs multiple NMF algorithms and clusters the resulting /
signatures of each run. The mean of each cluster is the final signature 
"""


import pandas as pd
# from sklearn.metrics import silhouette_score, silhouette_samples

from surgeNMF.surgeKorax import Crows

import os
import sys



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
        max_num_sigs=6, 
        num_trials=20,
        num_cores=20,
        # which_nmf="VR Poisson",
        # which_nmf="VR Gaussian",
        which_nmf="St Frob",
        # which_nmf="St KLd",
        Lambdas=[1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3],#, 1e-3, 5e-4, 1e-4],
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