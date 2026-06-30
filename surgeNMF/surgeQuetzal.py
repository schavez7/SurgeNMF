# ----------------------------------------------------------------- #
# surgeQuetzal.py
#   
#   Créer les plots des signatures et comparation contres ceux de 
#       COSMIC. Maintenatn, seulement pour COSMIC SBS96
# 
#   Tâches: 
#      - Ajouter l'option pour COSMIC ID83, et pour quiconque style des signatures
# 
# Last modified: Le 29 Juin 2026
# ----------------------------------------------------------------- #

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import string
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



def cosine_similarity_matrix(A, B):
    """
    Calcule la similarité cosinus entre chaque colonne de A et chaque colonne de B.
    :param A: np.array de taille (num_vars, num_sigs_denovo)
    :param B: np.array de taille (num_vars, num_sigs_ref)
    :return:  matrice de taille (num_sigs_denovo, num_sigs_ref)
    """
    # Normaliser chaque colonne
    A_norm = A / np.linalg.norm(A, axis=0, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=0, keepdims=True)
    return A_norm.T @ B_norm


def plot_signatures(
        W,
        which_sigs = None,
        W_alt = None,
        title = None,
        set1_name = None,
        set2_name = None,
        save_loc = None,
        show_plot = False

):
    """
    Docstring for plot_signatures
    
    :param W: [array of arrays or ndarray, no pandas.DataFrame] the signatures (as columns) within a matrix 
    :param noms_denovo: [array of positive intergers] that indicate which signatures to plot 
    :param W_alt: [array of arrays or ndarray] another set of signatures to compare with
    :param set1_name: [string] name of set W of signatures 
    :param set2_name: [string] name of set W_alt of signatures
    :param save_plot: [string] address of file to save
    :param show_plot: [boolean]
    """

    # Hauteur fixe par barplot (en pouces)
    hauteur_par_plot = 2.5
    largeur          = 10

    # Dimension parameters
    num_vars, num_sigs = np.shape(W)
    if which_sigs is None: 
        # Les noms des denovo signatures
        alphabeta   = string.ascii_uppercase
        which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:num_sigs]]
    num_plots = len(which_sigs)

    # For color purposes
    if ( (num_vars % 6) == 0):
        n = int(num_vars / 6)        
        colors = (["#D81B60"] * n) + (["#BFB3C3"] * n) + (["#1E88E5"] * n) + (["#004D40"] * n) + (["#FFC107"] * n) + (["#723838"] * n)
    else:
        colors = (["#008080"] * num_vars)

    # Hauteur totale de la figure selon le nombre de plots
    hauteur_totale = hauteur_par_plot * num_plots
    

    # ----------------------------------------------------------------- #
    # Sans W_alt: Depending on if comparing two sets of signatures or not
    if W_alt is None: 
        figure, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(largeur, hauteur_totale))
        if num_plots == 1:
            axes = [axes]  # s'assurer que axes est toujours une liste
        if title is not None:
            figure.suptitle(title, fontsize=14)

        for i, sig in enumerate(which_sigs):
            axes[i].bar(range(1, num_vars+1), W[:, i], color=colors, alpha=0.9)
            axes[i].set_title(sig)
            if i < num_plots - 1:
                axes[i].set_xticks([])
            else:
                axes[i].set_xlabel("Variables")

        plt.tight_layout()

        if save_loc is not None:
            with PdfPages(save_loc + ".pdf") as pdf:
                pdf.savefig(figure, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        plt.close()


    # ----------------------------------------------------------------- #
    # Avec W_alt — comparaison de deux ensembles de signatures
    else:
        Walt_np = W_alt.to_numpy()

        # Re-order the results based on who they are closest to 
        cosineDist_mat = cdist(W.T, W_alt.T, "cosine")
        row_ind, col_ind = linear_sum_assignment(cosineDist_mat)

        figure, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(largeur * 2, hauteur_totale))
        if num_plots == 1:
            axes = [axes]  # s'assurer que axes est toujours une liste
        figure.suptitle("Signatures comparées", fontsize=14)

        for i in range(num_plots):
            axes[i][0].bar(range(1, num_vars+1), W[:, row_ind[i]], color=colors, alpha=0.9)
            axes[i][1].bar(range(1, num_vars+1), Walt_np[:, col_ind[i]], color=colors, alpha=0.9)
            axes[i][0].set_title(f"{set1_name} {i+1}" if set1_name else f"De novo {i+1}")
            axes[i][1].set_title(W_alt.columns[col_ind[i]])
            if i < num_plots - 1:
                axes[i][0].set_xticks([])
                axes[i][1].set_xticks([])
            else:
                axes[i][0].set_xlabel("Variables")
                axes[i][1].set_xlabel("Variables")

        plt.tight_layout()

        if save_loc is not None:
            with PdfPages(save_loc + ".pdf") as pdf:
                pdf.savefig(figure, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        plt.close()

    

def plot_against_cosmic(
        W,
        save_dir
):
    """
    Docstring for plot_signatures
    
    :param W: [array of arrays or ndarray, no pandas.DataFrame] the signatures (as columns) within a matrix 
    :param save_dir: [string] directory of where to save files
    """

    # Hauteur fixe par barplot (en pouces)
    hauteur_par_plot = 2.5
    largeur          = 10

    # Dimension parameters
    num_vars, num_sigs = np.shape(W)
    # Les noms des denovo signatures
    alphabeta   = string.ascii_uppercase
    noms_denovo = [f"Denovo Sig {i}" for i in alphabeta[:num_sigs]]
    num_plots = len(noms_denovo)

    # For color purposes
    if ( (num_vars % 6) == 0):
        n = int(num_vars / 6)        
        colors = (["#D81B60"] * n) + (["#BFB3C3"] * n) + (["#1E88E5"] * n) + (["#004D40"] * n) + (["#FFC107"] * n) + (["#723838"] * n)

    # Hauteur totale de la figure selon le nombre de plots
    hauteur_totale = hauteur_par_plot * num_plots


    # ----------------------------------------------------------------- #
    # Télécharger les signatures cosmic 
    ref = pd.read_csv(f"/data/{os.getenv('USER')}/SurgeNMF/surgeNMF/Reference_SBS96_COSMIC_Catalogue_Ordered.txt", sep="\t", index_col=0)
    noms_ref = list(ref.columns)

    # Déterminer les distances entre chaque colonne
    dist_mat = cdist(W.T, ref.T, "cosine")
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    # Trouver les signatures de ref
    Walt    = ref.iloc[:, col_ind]
    Walt_np = Walt.to_numpy()

    # Trouver les distance entre chaque pair de signatures
    sim_matrix = cosine_similarity_matrix(W, ref)


    # ----------------------------------------------------------------- #
    # I. Créer les signatures en deux colonnes
    figure, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(largeur * 2, hauteur_totale))
    if num_plots == 1:
        axes = [axes]  # s'assurer que axes est toujours une liste
    figure.suptitle("Signatures de novo vs COSMIC", fontsize=14)

    for i in range(num_plots):
        axes[i][0].bar(range(1, num_vars+1), W[:,row_ind[i]], color=colors, alpha=0.9)
        axes[i][1].bar(range(1, num_vars+1), Walt_np[:,i], color=colors, alpha=0.9)
        axes[i][0].set_title(noms_denovo[row_ind[i]])
        axes[i][1].set_title(Walt.columns[i])
        if i < num_plots - 1:
            axes[i][0].set_xticks([])
            axes[i][1].set_xticks([])
        else:
            axes[i][0].set_xlabel("Variables")
            axes[i][1].set_xlabel("Variables")

    plt.tight_layout()

    with PdfPages(save_dir + "/compare_sigs.pdf") as pdf:
        pdf.savefig(figure, dpi=300, bbox_inches="tight")

    plt.close()



    # II. Créer le heatmap (best COSMIC signatures)
    col_ind_sort = np.sort(col_ind)
    _, ax = plt.subplots(figsize=(max(8, num_sigs * 0.6),
                                    max(4, len(noms_denovo) * 0.5)))

    im = ax.imshow(sim_matrix.iloc[:,col_ind_sort], vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Similarité cosinus")

    # Étiquettes des axes
    ax.set_xticks(range(num_sigs))
    ax.set_yticks(range(len(noms_denovo)))
    ax.set_xticklabels(ref.columns[col_ind_sort],    rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(noms_denovo, fontsize=8)
    ax.set_xlabel("Signatures de référence")
    ax.set_ylabel("Signatures de-novo")
    ax.set_title("de novo signatures against closest COSMIC signatures")

    # Ajouter les valeurs dans chaque cellule
    for i in range(sim_matrix.shape[0]):
        for j in range(num_sigs):
            val     = sim_matrix.iloc[i, col_ind_sort[j]]
            couleur = "white" if val > 0.8 else "black"
            # couleur = "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=couleur)

    plt.tight_layout()

    plt.savefig(save_dir + "/compare_heatmap-short.png", dpi=200, bbox_inches="tight")

    plt.close()



    # III. Créer le heatmap all 
    _, ax = plt.subplots(figsize=(max(8, len(noms_ref) * 0.6),
                                    max(4, len(noms_denovo) * 0.5)))

    im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Similarité cosinus")

    # Étiquettes des axes
    ax.set_xticks(range(len(noms_ref)))
    ax.set_yticks(range(len(noms_denovo)))
    ax.set_xticklabels(noms_ref,    rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(noms_denovo, fontsize=8)
    ax.set_xlabel("Signatures de référence")
    ax.set_ylabel("Signatures de-novo")
    ax.set_title("de novo signatures against all COSMIC signatures")

    # Ajouter les valeurs dans chaque cellule
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            val     = sim_matrix.iloc[i, j]
            couleur = "white" if val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=couleur)

    plt.tight_layout()

    plt.savefig(save_dir + "/compare_heatmap-all.png", dpi=200, bbox_inches="tight")

    plt.close()

