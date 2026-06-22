#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 2 June 2026

@author Sergio Chávez
"""
"""
Purpose: This runs a single NMF run (mainly not parallel) for verification purposes

Tasks for Serge: 
    - For today (2 June)
        Quelque chose
"""

# -------------------------------------------------------------------------------- #
# LIBRARIES
import pandas as pd
import string

import os
import sys
import time
# use: 
    # t0 = time.perf_counter()
    # stuff
    # t1 = time.perf_counter()
    # print("np.delete:",t1-t0)

from surgeNMF import NMF
from surgePeacock import plot_signatures


# -------------------------------------------------------------------------------- #
# INITIAL INPUTS CHECK
if len(sys.argv) < 5:
    print("Utilisation: python surgeNMF.py <fichier info> <dir data> <dir results Gauss> <dir results Poisson>")
    exit(1)


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

# Les noms des denovo signatures
alphabeta   = string.ascii_uppercase
nomsColonnes_temp = [f"Sig {i}" for i in alphabeta[:num_sigs]]

# Load data 
fichier_true = os.path.join(dir_data, "vrai_W.txt")
fichier_Gaus = os.path.join(dir_data, "data_Gaussian_distr.txt")
fichier_Pois = os.path.join(dir_data, "data_Poisson_distr.txt")
data_Gaus    = pd.read_csv(fichier_Gaus, sep="\t", index_col=0)
data_Pois    = pd.read_csv(fichier_Pois, sep="\t", index_col=0)
data_Gaus_np = data_Gaus.to_numpy(dtype=float)
data_Pois_np = data_Pois.to_numpy(dtype=float)

# Load a reference so that we can extract its row names
ref = pd.read_csv(f"/data/{os.getenv('USER')}/surgeNMF/Reference_SBS96_COSMIC_Catalogue_Ordered.txt", sep="\t", index_col=0)
nomsLignes   = ref.index

# Load true W used in data
vrai_W = pd.read_csv(fichier_true, sep="\t", index_col=0, header=0)
nomsColonnes = vrai_W.columns

# Outpath for plots
output_sigs_dir  = os.path.join(dir_resultats_Gaus, "DenovoSigs")
os.makedirs(output_sigs_dir, exist_ok=True)


# -------------------------------------------------------------------------------- #
# PARAMÈTRES
lam = 0.0 
max_iterations_general = 5000
max_iterations_D = 5
max_iterations_S = 5
tolerance = 1e-8


# -------------------------------------------------------------------------------- #
# EXÉCUTER CHAQUE MÉTHODE

# Set up NMF for Gaussian data
nmf_results = NMF(
    data_Gaus_np, 
    num_sigs=num_sigs, 
    seed=seed
)

# --- Ἄσκησις α´: Standard NMF w/ Frobenius norm for Gaussian data --- #
nmf_results.standard_NMF_Frobenius(
    max_iterations=max_iterations_general,
    tolerance=1e-8
)

# Save the results
nom_travail = "Std-NMF-Frob"
print(">> St-NMF (Frobenious) finished at iteration", nmf_results.St_NMF_Frobenius["iter"],"out of",max_iterations_general)
denovoSigsFilename = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs.txt")
Sigs_df = pd.DataFrame(nmf_results.St_NMF_Frobenius["W_final"], index=nomsLignes, columns=nomsColonnes_temp)
Sigs_df.to_csv(denovoSigsFilename, sep="\t")

# Sauvegarder l'information du système
with open(os.path.join(output_sigs_dir, nom_travail+"_systeme_info.txt"), "w") as f:
    f.write(f"seed\t{seed}\n")
    f.write(f"num_sigs\t{num_sigs}\n")
    f.write(f"num_obs\t{num_obs}\n")
    f.write(f"finished_iter\t{nmf_results.St_NMF_Frobenius['iter']}\n")
    f.write(f"max_iterations\t{max_iterations_general}\n")
    f.write(f"err\t{nmf_results.St_NMF_Frobenius['err']}\n")
    f.write(f"tolerance\t{nmf_results.St_NMF_Frobenius['tolerance']}\n")
    f.write(f"H_exposure_mat_vol\t{nmf_results.St_NMF_Frobenius['H_exposure_mat_vol']}\n")
    f.write(f"optimisation_func_val\t{nmf_results.St_NMF_Frobenius['optimisation_func_val']}\n")

# Save a plot 
plot_signatures(
    nmf_results.St_NMF_Frobenius["W_final"],
    which_sigs = None,
    W_alt = vrai_W,
    title = "Standard NMF (Frobenius Norm) Gaussian data",
    set1_name = "de novo sigs",
    set2_name = "COSMIC sigs",
    save_loc = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs"),
    show_plot = False
)


# -------------------------------------------------------------------------------- #
# --- Ἄσκησις β´: Standard NMF w/ KL divergence for Gaussian data --- #
nmf_results.standard_NMF_Kullback_Leibler(
    max_iterations=max_iterations_general,
    tolerance=1e-8
)

# Sauvegarder les résultats
nom_travail = "Std-NMF-KLd"
print(">> St-NMF (Kullback-Leibler divergence) finished at iteration", nmf_results.St_NMF_KullbackLeibler["iter"],"out of",max_iterations_general)
denovoSigsFilename = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs.txt")
Sigs_df = pd.DataFrame(nmf_results.St_NMF_KullbackLeibler["W_final"], index=nomsLignes, columns=nomsColonnes_temp)
Sigs_df.to_csv(denovoSigsFilename, sep="\t")

# Sauvegarder l'information du système
with open(os.path.join(output_sigs_dir, nom_travail+"_systeme_info.txt"), "w") as f:
    f.write(f"seed\t{seed}\n")
    f.write(f"num_sigs\t{num_sigs}\n")
    f.write(f"num_obs\t{num_obs}\n")
    f.write(f"finished_iter\t{nmf_results.St_NMF_KullbackLeibler['iter']}\n")
    f.write(f"max_iterations\t{max_iterations_general}\n")
    f.write(f"err\t{nmf_results.St_NMF_KullbackLeibler['err']}\n")
    f.write(f"tolerance\t{nmf_results.St_NMF_KullbackLeibler['tolerance']}\n")
    f.write(f"H_exposure_mat_vol\t{nmf_results.St_NMF_KullbackLeibler['H_exposure_mat_vol']}\n")
    f.write(f"optimisation_func_val\t{nmf_results.St_NMF_KullbackLeibler['optimisation_func_val']}\n")

# Sauvegarder les plots 
plot_signatures(
    nmf_results.St_NMF_KullbackLeibler["W_final"],
    which_sigs = None,
    W_alt = vrai_W,
    title = "Standard NMF (Kullback-Leibler divergence) Gaussian data",
    set1_name = "de novo sigs",
    set2_name = "COSMIC sigs",
    save_loc = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs"),
    show_plot = False
)


# -------------------------------------------------------------------------------- #
# --- Ἄσκησις γ´: Minimum-Volume NMF w/ Frobenius Norm assuming Gaussian data for Gaussian data
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

# Sauvegarder les résultats 
nom_travail = "MV-NMF-Gaussian"
print(">> MV-NMF for Gaussian data finished at iteration", nmf_results.VR_NMF_Gaussian["iter_general"], "out of", max_iterations_general)
denovoSigsFilename = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs.txt")
Sigs_df = pd.DataFrame(nmf_results.VR_NMF_Gaussian["W_final"], index=nomsLignes, columns=nomsColonnes_temp)
Sigs_df.to_csv(denovoSigsFilename, sep="\t")

# Sauvegarder l'information du système
with open(os.path.join(output_sigs_dir, nom_travail+"_systeme_info.txt"), "w") as f:
    f.write(f"seed\t{seed}\n")
    f.write(f"num_sigs\t{num_sigs}\n")
    f.write(f"num_obs\t{num_obs}\n")
    f.write(f"finished_iter\t{nmf_results.VR_NMF_Gaussian['iter_general']}\n")
    f.write(f"max_iterations\t{max_iterations_general}\n")
    f.write(f"max_iterations_D\t{max_iterations_D}\n")
    f.write(f"max_iterations_S\t{max_iterations_S}\n")
    f.write(f"err\t{nmf_results.VR_NMF_Gaussian['err_general'][-1]}\n")
    f.write(f"tolerance\t{nmf_results.VR_NMF_Gaussian['tolerance']}\n")
    f.write(f"H_exposure_mat_vol\t{nmf_results.VR_NMF_Gaussian['H_exposure_mat_vol'][-1]}\n")
    f.write(f"optimisation_func_val\t{nmf_results.VR_NMF_Gaussian['optimisation_func_val'][-1]}\n")

# Sauvegarder les plots 
plot_signatures(
    nmf_results.VR_NMF_Gaussian["W_final"],
    which_sigs = None,
    W_alt = vrai_W,
    title = "Min-Vol NMF (for Gaussian data) Gaussian data",
    set1_name = "de novo sigs",
    set2_name = "COSMIC sigs",
    save_loc = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs"),
    show_plot = False
)


# -------------------------------------------------------------------------------- #
# --- Ἄσκησις δ´: Minimum-Volume NMF w/ Frobenius Norm assuming Poisson data but for Gaussian data
nmf_results.volume_regularisation_NMF_Poisson(
    vol_reg_val=lam, 
    num_singular_vals=num_sigs,
    max_iterations_general=max_iterations_general,
    max_iterations_D=max_iterations_D,
    max_iterations_S=max_iterations_S,
    tolerance=1e-8
    # W=nmf_results.St_NMF_Frobenius["W_final"],
    # H=nmf_results.St_NMF_Frobenius["H_final"]
)

# Sauvegarder les résultats 
nom_travail = "MV-NMF-Poisson"
print(">> MV-NMF for Poisson data finished at iteration", nmf_results.VR_NMF_Poisson["iter_general"], "out of", max_iterations_general)
denovoSigsFilename = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs.txt")
Sigs_df = pd.DataFrame(nmf_results.VR_NMF_Poisson["W_final"], index=nomsLignes, columns=nomsColonnes_temp)
Sigs_df.to_csv(denovoSigsFilename, sep="\t")

# Sauvegarder l'information du système
with open(os.path.join(output_sigs_dir, nom_travail+"_systeme_info.txt"), "w") as f:
    f.write(f"seed\t{seed}\n")
    f.write(f"num_sigs\t{num_sigs}\n")
    f.write(f"num_obs\t{num_obs}\n")
    f.write(f"finished_iter\t{nmf_results.VR_NMF_Poisson['iter_general']}\n")
    f.write(f"max_iterations\t{max_iterations_general}\n")
    f.write(f"max_iterations_D\t{max_iterations_D}\n")
    f.write(f"max_iterations_S\t{max_iterations_S}\n")
    f.write(f"err\t{nmf_results.VR_NMF_Poisson['err_general'][-1]}\n")
    f.write(f"tolerance\t{nmf_results.VR_NMF_Poisson['tolerance']}\n")
    f.write(f"H_exposure_mat_vol\t{nmf_results.VR_NMF_Poisson['H_exposure_mat_vol'][-1]}\n")
    f.write(f"optimisation_func_val\t{nmf_results.VR_NMF_Poisson['optimisation_func_val'][-1]}\n")

# Sauvegarder les plots 
plot_signatures(
    nmf_results.VR_NMF_Poisson["W_final"],
    which_sigs = None,
    W_alt = vrai_W,
    title = "Min-Vol NMF (for Poisson data) Gaussian data",
    set1_name = "de novo sigs",
    set2_name = "COSMIC sigs",
    save_loc = os.path.join(output_sigs_dir, nom_travail+"_denovo-sigs"),
    show_plot = False
)

