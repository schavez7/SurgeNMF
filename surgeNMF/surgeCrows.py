#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 24 June 2026

@author Sergio Chávez
"""
"""
Contains the functions needed to run the parallel NMF runs, for clustering, 
and for choosing the best solution based on the best avg silhouette width 
"""

import numpy as np 
import pandas as pd
import time
import string

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, silhouette_samples

from surgeNMF.surgeNMF import NMF
from surgeNMF.surgePeacock import plot_signatures, plot_against_cosmic

import os
import sys
from multiprocessing import Pool
from functools import partial


def bootstrap_data(
        X_data,
        seed
):
    '''
    Creates a new data matrix based on the input matrix where each new column 
    is due to a multinomial distribution from the original column. 
    '''

    # Ensure using a unique seed from the other bootstraps
    np.random.seed(seed)
    
    _, num_obs = X_data.shape
    # Iterate through each column of input matrix
    for k in range(num_obs):
        # Create a pmf due to the column 
        num_total_mutations = X_data[:,k].sum()
        pmf_of_col  = X_data[:,k] / num_total_mutations

        # Randomly create a new column
        X_data[:,k] = np.random.multinomial(num_total_mutations, pmf_of_col)

    return X_data


def one_nmf_run(
        clustering_iter, 
        seed, 
        lam, 
        num_sigs, 
        max_iterations_general, 
        max_iterations_D,
        max_iterations_S, 
        which_nmf, 
        datafilepath
):
    # Seed pour cet exécution
    seed_ici = (seed + num_sigs * 10000 + clustering_iter * 100) % (2**32 - 1)
    
    # Load data 
    data_original_df = pd.read_csv(datafilepath, sep="\t", index_col=0)
    data_original = data_original_df.to_numpy(dtype=float)
    data_bootstrp = bootstrap_data(data_original, seed_ici)

    nmf_results = NMF(
        data_bootstrp, 
        num_sigs=num_sigs,
        seed=seed_ici
    )
    
    if (which_nmf == "St Frob"):
        # I. RUN STANDARD NMF WITH FROBENIUS NORM
        nmf_results.standard_NMF_Frobenius(
            max_iterations=max_iterations_general,
            tolerance=1e-8
            )
        return clustering_iter, nmf_results.St_NMF_Frobenius["W_final"]
    if (which_nmf == "St KLd"):
        # II. RUN STANDARD NMF WITH KULLBACK-LEIBLER DIVERGENCE
        nmf_results.standard_NMF_Kullback_Leibler(
            max_iterations=max_iterations_general,
            tolerance=1e-8
        )
        return clustering_iter, nmf_results.St_NMF_KullbackLeibler["W_final"]
    if (which_nmf == "VR Gaussian"):
        # III. RUN VOLUME REGULARISATION NMF (w/ COVARIANCE FORM) GAUSSIAN RVs
        nmf_results.standard_NMF_Frobenius(
            max_iterations=max_iterations_general,
            tolerance=1e-3
            )
        
        nmf_results.volume_regularisation_NMF_Gaussian(
            vol_reg_val=lam, 
            num_singular_vals=num_sigs,
            max_iterations_general=max_iterations_general,
            max_iterations_D=max_iterations_D,
            max_iterations_S=max_iterations_S,
            tolerance=1e-8
            # W=nmf_results.St_NMF_Frobenius["W_final"],
            # H=nmf_results.St_NMF_Frobenius["H_final"]
        )
        return clustering_iter, nmf_results.VR_NMF_Gaussian["W_final"]
    if (which_nmf == "VR Poisson"):
        # IV. RUN VOLUME REGULARISATION NMF (w/ COVARIANCE FORM) POISSON RVs
        nmf_results.volume_regularisation_NMF_Poisson(
            vol_reg_val=lam, 
            num_singular_vals=num_sigs,
            max_iterations_general=max_iterations_general,
            max_iterations_D=max_iterations_D,
            max_iterations_S=max_iterations_S,
            tolerance=1e-8
            # W=nmf_results.St_NMF_KullbackLeibler["W_final"],
            # H=nmf_results.St_NMF_KullbackLeibler["H_final"]
        )
        return clustering_iter, nmf_results.VR_NMF_Poisson["W_final"]


def silhouette_finder(
        clustering_iter,
        W_all, 
        seed, 
        num_sigs,  
        num_trials
):
    # Seed pour cette exécution
    seed_ici = (seed + num_sigs * 10000 + clustering_iter * 100) % (2**32 - 1)
    rng = np.random.default_rng(seed_ici)
    
    # Initialiser les moyennes de chaque cluster 
    avgSilVal_pre = avgSilVal = -1.0
    cluster_means = rng.random(size=(W_all.shape[0], num_sigs))
    
    count = 0
    while True:
        # Initialiser le vecteur d'indices et la matrice qui contient les colonnes clustées
        cluster_indices = list(range(0, W_all.shape[1], num_trials))
        Clusters = np.zeros(W_all.shape)
        for k in range(num_trials):
            # Trouver les distances entres les colonnes de cluster_means et chaque W (cosine similarity)
            W = W_all[:,num_sigs*k:num_sigs*(k+1)]
            dist_mat = cdist(cluster_means.T, W.T, metric="cosine")
            
            # Trouver les meilleur combination sans répètes 
            idx_clust_mean, idx_W = linear_sum_assignment(dist_mat)

            # Ça place les colonnes de chaque W dans leur cluster 
            for i in range(num_sigs):
                Clusters[:, cluster_indices[idx_clust_mean[i]]] = W[:, idx_W[i]]
                cluster_indices[idx_clust_mean[i]] += 1

        # Determiner les silhouette coefficients 
        silhouettes_vect = metrics.silhouette_samples(
                                Clusters.T, 
                                np.repeat(range(num_sigs), num_trials),
                                metric="cosine"
        )
        
        # Mettre à jour les informations importantes et nouvelles colonnes
        avgSilVal_pre = avgSilVal
        avgSilVal     = np.mean(silhouettes_vect)
        avgSilValParCluster = np.zeros(num_sigs)
        for k in range(num_sigs):
            avgSilValParCluster[k] = np.mean(silhouettes_vect[num_trials*k:num_trials*(k+1)])
            cluster_means[:,k] = np.mean(Clusters[:, num_trials*k:num_trials*(k+1)], axis=1) 

        count += 1
        # SigProfilerExtractor s'arrête à 10
        if (count > 9):
            break
        if (np.isclose(avgSilVal, 1.0, atol=1e-2)):
            break
        if (np.isclose(avgSilVal_pre, avgSilVal, atol=1e-3)):
            break

    return avgSilVal, avgSilValParCluster, cluster_means


def nmf_clustering(
    seed: int,
    num_sigs: int,
    num_trials: int,
    which_nmf: str,
    lam=None,
    num_cores: int=int(4),
    datafilepath=None
):
    """
    Docstring for nmf_clustering

    This function performs multiple NMF, each on a different bootstrapped data, then performs 
    the clustering of all the columns of W with the number of clusters matching the number of 
    signatures. Each of these parts will use parallel computing. 

    *Deux choses que voir plus tard est de changer les num_trials pour chaque partie et de 
    authoriser l'option de mettres les max_iterations dans cette fonction. 

    :param seed: [positive integer] Randomisation seed, default is None
    :param num_sigs: [positive integer] Number of signatures to look for 
    :param num_trials: [positive integer] Number of trials each cluster 
    :param which_nmf: [string] Which NMF vr_version to use
    :param lam: [positive float] This is the volume regularisation parameter 
    :param num_cores: [positive integer] Number of cores for parallel computing. Default 4
    :param datafilepath: [txt file]
    """

    # I) NMF
    # Prepare the parameters
    g = partial(one_nmf_run,
                # Parameters
                seed=seed,
                lam=lam,
                num_sigs=num_sigs,
                max_iterations_general=1000,
                max_iterations_D=5,
                max_iterations_S=5,
                which_nmf=which_nmf,
                datafilepath=datafilepath
    )
    
    # Exécuter le process parallèle
    with Pool(processes=num_cores) as pool:
        nmf_results_opt_val = pool.map(g, range(num_trials))

    # Unload information on a super large W matrix
    num_vars   = 96
    W_all = np.zeros((num_vars, num_sigs * num_trials))
    for item in nmf_results_opt_val:
        k = item[0]
        W_all[:,num_sigs*k:num_sigs*(k+1)] = item[1]


    # II) CLUSTERING
    # Preparer les paramètres 
    g = partial(silhouette_finder,
                # Paramètres, 
                W_all=W_all,
                seed=seed,
                num_sigs=num_sigs,
                num_trials=num_trials
    )

    # Exécuter le process parallèle 
    with Pool(processes=num_cores) as pool:
        multi_clustering = pool.map(g, range(num_trials))

    # Extraire les résultats 
    avgSilVal_alltrials = [] 
    for item in multi_clustering:
        avgSilVal_alltrials.append(item[0])
    avgSilVal_meilleur_idx = np.argmax(avgSilVal_alltrials)
    avgSilVal_meilleur = multi_clustering[avgSilVal_meilleur_idx][0]
    avgSilValParClust_meilleur = multi_clustering[avgSilVal_meilleur_idx][1]
    Signatures = multi_clustering[avgSilVal_meilleur_idx][2]

    return avgSilVal_meilleur, avgSilValParClust_meilleur, Signatures


def save_info(
        W,          
        num_sigs,
        avgSilWidth,
        dir_results,
        lam=None,
        cosmic=False
):
    # D'abord, créer le sous-répertoire pour ce résultat
    if cosmic:
        dir_sous = os.path.join(dir_results, "Optimal_Solution")
    else:
        dir_sous = os.path.join(dir_results, f"{num_sigs}")
    os.makedirs(dir_sous, exist_ok=True)

    # Sauvegarder le valeur de avgSilWidth
    fichier = os.path.join(dir_sous, "systeme_info.txt")
    with open(fichier, "w") as f:
        if lam is not None:
            f.write(f"lam\t{lam}\n")
        f.write(f"avgSilhouetteWidth\t{avgSilWidth}\n")

    # Sauvegarder les denovo signatures
    fichier = os.path.join(dir_sous, "denovo_sigs_mat.txt")
    W.to_csv(fichier, sep="\t")

    fichier = os.path.join(dir_sous, "denovo_sigs")
    plot_signatures(np.array(W),save_loc=fichier)

    if cosmic:
        fichier = os.path.join(dir_sous, "denovo_compare")
        plot_against_cosmic(np.array(W), save_loc=fichier)


class Crows:
    """ 
    Docstring for Crows

    This class will allow the complete analysis of the MV-NMF scheme thru:
    - Multiple NMF solutions via bootstrapped data 
    - Cluster the multiple solutions to obtain the best estimate of signatures 
    - Grid search of the best number of signatures and volume-regularisation hyperperameter 
    - The best solution is determined by the average silhouette width
    """

    def __init__(
            self, 
            file_data_txt,
            dir_results,
            min_num_sigs,
            max_num_sigs,
            num_trials=20,
            num_cores=4,
            which_nmf="St Frob",
            Lambdas=None,
            seed=None
    ):
        """
        Docstring for __init__

        :param file_data_txt:
        :param min_num_sigs:
        :param max_num_sigs:
        :param num_trials:
        :param num_cores:
        :param which_nmf:
        :param Lambdas: 
        :param seed:
        """
        # Data txt file
        self.file_data_txt = file_data_txt
        self.dir_results   = dir_results

        # Obtenir les index de la data
        info = pd.read_csv(self.file_data_txt, sep="\t", header=0, index_col=0)
        self.data_index = info.index.tolist()

        # Preliminary tests and assignments
        try: 
            self.min_num_sigs = int(min_num_sigs)
            self.max_num_sigs = int(max_num_sigs)
            self.num_trials   = int(num_trials)
            self.num_cores    = int(num_cores)
        except (TypeError, ValueError):
            # if not isinstance(num_sigs, int) or (num_sigs < 1):
            raise TypeError("Input num_sigs must be convertible to an integer")
        if (self.min_num_sigs < 1) or (self.max_num_sigs < 1) or (self.min_num_sigs > self.max_num_sigs):
            raise ValueError("Inputs min_num_sigs <= max_num_sigs must be postive integers")
        if seed is None:
            self.seed = int(time.time() * 1000) 
            self.rng = np.random.default_rng(self.seed)
        else:
            try:  
                self.seed = int(seed)
                self.rng = np.random.default_rng(self.seed)
            except (TypeError, ValueError):
                raise TypeError("Input seed must be convertible to an integer")

        # Which method of NMF: {"St Frob", "St KLd", "VR Gaussian", "VR Poisson"}
        self.which_nmf = which_nmf

        # Lambda values to search 
        if Lambdas is None:
            self.Lambdas = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        else:
            self.Lambdas = Lambdas

        # Stability parameter (This is the delta within the determinant mean to avoid log (zero) )
        self.epsilon = 1.

        # Run the clustering 
        self.__run()
    

    def __choose_optimal(
            self,
            Wall, 
            vr_version,
            Scores,
    ):
        # Pour donner les noms denovo alphabetisé
        alphabeta  = string.ascii_uppercase
        sig_range = list(dict.fromkeys(k if not isinstance(k, tuple) else k[0] for k in Wall.keys()))
        
        # ----------------------------------------------------------------- #
        # CAS STANDARD (St Frob, St KLd) — Scores est 1D
        if not vr_version:
            # Sauvegarder chaque num_sigs
            for num_sigs, W in Wall.items():                
                which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:num_sigs]]
                Wdf = pd.DataFrame(W, index=self.data_index, columns=which_sigs)
                save_info(Wdf, num_sigs, Scores, dir_results=self.dir_results)
            
            # Ça nous donne l'index qui a la valeur la plus à droite qui est supérieure à 0.8
            candidats = np.where(Scores > 0.8)[0]
            if len(candidats) == 0:
                optimal_index = np.argmax(Scores)
            else:
                optimal_index = candidats[-1]

            optimal_score    = Scores[optimal_index]
            optimal_num_sigs = sig_range[optimal_index]
        
            # Sauvegarder la meilleur 
            which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:optimal_num_sigs]]
            Wdf = pd.DataFrame(Wall[optimal_num_sigs], index=self.data_index, columns=which_sigs)
            save_info(Wdf, optimal_num_sigs, optimal_score, dir_results=self.dir_results, cosmic=True)
            print(f"\n Optimal St: num_sigs={optimal_num_sigs}, score={optimal_score:.4f}")

        # ----------------------------------------------------------------- #
        # CAS VOLUME-REGULARISATION (VR Gaussian, VR Poisson) — Scores est 2D
        else:
            best_W_per_sig     = {}
            best_score_per_sig = {}
            best_lam_per_sig   = {}

            print("\n")
            for i, num_sigs in enumerate(sig_range):
                # Trouver le lam optimal pour ce num_sigs 
                scores_for_this_sig = Scores[i,:] 
                best_lam_index      = np.argmax(scores_for_this_sig)
                best_lam            = self.Lambdas[best_lam_index]
                best_score          = scores_for_this_sig[best_lam_index]

                best_W_per_sig[num_sigs]     = Wall[num_sigs, best_lam]
                best_score_per_sig[num_sigs] = best_score 
                best_lam_per_sig[num_sigs]   = best_lam 

                # Sauvegarder la meilleure solution pour ce num_sigs 
                which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:num_sigs]]
                Wdf = pd.DataFrame(best_W_per_sig[num_sigs], index=self.data_index, columns=which_sigs)
                save_info(Wdf, num_sigs, best_score, dir_results=self.dir_results, lam=best_lam)
                print(f" For num_sigs={num_sigs}: optimal lam={best_lam}, score={best_score:.4f}")

            # Trouver la solution globalement optimale 
            best_scores_array = np.array([best_lam_per_sig[n] for n in sig_range])
            candidats         = np.where(best_scores_array > 0.8)[0] 
            if len(candidats) == 0:
                optimal_index = np.argmax(best_scores_array)
            else:
                optimal_index = candidats[-1] 

            optimal_num_sigs = sig_range[optimal_index]
            optimal_score    = best_score_per_sig[optimal_num_sigs] 
            optimal_lam      = best_lam_per_sig[optimal_num_sigs]

            # Sauvegarder la solution globalement optimale
            which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:optimal_num_sigs]]
            Wdf = pd.DataFrame(best_W_per_sig[optimal_num_sigs], index=self.data_index, columns=which_sigs)
            save_info(Wdf, optimal_num_sigs, optimal_score, dir_results=self.dir_results, lam=optimal_lam, cosmic=True)
            print(f"\n  Optimal VR: num_sigs={optimal_num_sigs}, lam={optimal_lam}, score={optimal_score:.4f}")


    def __run(
            self
    ):
        # Grid search for number of signatures and lambda value 
        sig_range = range(self.min_num_sigs, self.max_num_sigs + 1)

        # If only doing Standard NMF only need silhouettes for each number of sigs, otherwise need two dimensions 
        if self.which_nmf in {"St Frob", "St KLd"}:
            Scores = np.zeros(len(sig_range))
        if self.which_nmf in {"VR Gaussian", "VR Poisson"}:
            Scores = np.zeros((len(sig_range), len(self.Lambdas)))


        Wall_dict = {}
        for num_sigs in sig_range:
            print("\nNombre de signatures:", num_sigs, "----------------------------")
            
            # For traditional Standard NMF
            if self.which_nmf in {"St Frob", "St KLd"}:
                vr_version = False
                score_temp, _, W = nmf_clustering(
                    self.seed, 
                    num_sigs, 
                    self.num_trials, 
                    which_nmf=self.which_nmf, 
                    lam=0, 
                    num_cores=self.num_cores, 
                    datafilepath=self.file_data_txt
                )
                # print("  Score for",num_sigs,"number of signatures is",score_temp)

                Wall_dict[num_sigs] = W
                Scores[num_sigs-sig_range[0]] = score_temp
            
            # **** 
            # Determine what to do here! 
            # ****
            exit()

            # For the volume-regularisation NMF
            if self.which_nmf in {"VR Gaussian", "VR Poisson"}:
                vr_version = True
                for k in range(len(self.Lambdas)):
                    lam = self.Lambdas[k]
                    print("   Lambda values of", lam)
                    score_temp, _, W = nmf_clustering(
                        self.seed, 
                        num_sigs, 
                        self.num_trials, 
                        which_nmf=self.which_nmf, 
                        lam=lam, 
                        num_cores=self.num_cores, 
                        datafilepath=self.file_data_txt
                    )
                    # print("   Score for",num_sigs,"number of signatures and lambda",lam,"is",score_temp)

                    Wall_dict[num_sigs,lam] = W
                    Scores[num_sigs-sig_range[0], k] = score_temp

        # self.__choose_optimal(Wall_dict, vr_version, Scores)

        # Save each vr_version ?
        print("\nGood here")
    

if __name__ == "__main__":

    print("   Fait autre chose!   ")