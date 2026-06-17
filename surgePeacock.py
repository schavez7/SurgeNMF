

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import os
import string
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



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
    :param which_sigs: [array of positive intergers] that indicate which signatures to plot 
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

        # # Keep count at which signatures we're at through the windows 
        # iter = 0 
        # for win in range(num_windows):
        #     if show_plot:
        #         print("\tWindow:",win+1," out of",num_windows,"open...")
        #     figure, ax = plt.subplots(nrows=4, ncols=1)
        #     if title is not None:
        #         figure.suptitle(title)
        #     figure.set_size_inches(8,8)
        #     for p in range(4):
        #         if (iter < num_plots):
        #             ax[p].bar(range(1,num_vars+1), W[:,iter], color=colors, alpha=0.9)
        #             if (iter < num_plots-1):
        #                 ax[p].set_xticks([])
        #                 ax[p].set_xticklabels([])
        #             else:
        #                 ax[p].set_xlabel("Variables")                        
        #         else:
        #             ax[p].axis('off') 
        #         ax[p].set_title(which_sigs[iter])
        #         iter += 1
        

            # if save_loc is not None: 
            #     if (num_windows > 1):
            #         filename = save_loc + f"_{win+1}Of{num_windows}.png"
            #     else: 
            #         filename = save_loc + ".png"
            #     plt.savefig(filename, dpi=300, bbox_inches="tight")

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


        # # Keep count at which signatures we're at through the windows 
        # iter = 0 
        # for win in range(num_windows):
        #     # print("--Window:",win+1," out of",num_windows)
        #     figure, ax = plt.subplots(nrows=4, ncols=2)
        #     figure.suptitle("Signatures compared")
        #     figure.set_size_inches(12,8)
        #     for p in range(4):
        #         if (iter < num_plots):
        #             # print("----Subplot num:",p," which is signature",which_sigs[iter])
        #             ax[p,0].bar(range(1,num_vars+1), W[:,row_ind[iter]], color=colors, alpha=0.9)
        #             ax[p,1].bar(range(1,num_vars+1), Walt_np[:,col_ind[iter]], color=colors, alpha=0.9)
        #             if (iter < num_plots-1):
        #                 ax[p,0].set_xticks([])
        #                 ax[p,1].set_xticks([])
        #                 ax[p,0].set_xticklabels([])
        #                 ax[p,1].set_xticklabels([])
        #             else:
        #                 ax[p,0].set_xlabel("Variables")
        #                 ax[p,1].set_xlabel("Variables")
        #             ax[p,0].set_title(f"Denovo {iter+1}")
        #             ax[p,1].set_title(W_alt.columns[col_ind[iter]])
        #         else:
        #             ax[p,0].axis('off')
        #             ax[p,1].axis('off')
        #         iter += 1
        #     # ax[0,0].set_title("True signatures")
        #     # ax[0,1].set_title("Estimated signatures")
        # # if set1_name is not None:
        # #     ax[0,0].set_title(set1_name)
        # #     ax[0,1].set_title(set2_name)

        # if save_loc is not None: 
        #     if (num_windows > 1):
        #         filename = save_loc + f"_{win+1}Of{num_windows}.png"
        #     else: 
        #         filename = save_loc + ".png"
        #     plt.savefig(filename, dpi=300, bbox_inches="tight")
    

def plot_against_cosmic(
        W,
        save_loc = None
):
    """
    Docstring for plot_signatures
    
    :param W: [array of arrays or ndarray, no pandas.DataFrame] the signatures (as columns) within a matrix 
    :param save_plot: [string] address of file to save
    """

    # Hauteur fixe par barplot (en pouces)
    hauteur_par_plot = 2.5
    largeur          = 10

    # Dimension parameters
    num_vars, num_sigs = np.shape(W)
    # Les noms des denovo signatures
    alphabeta   = string.ascii_uppercase
    which_sigs = [f"Denovo Sig {i}" for i in alphabeta[:num_sigs]]
    num_plots = len(which_sigs)

    # For color purposes
    if ( (num_vars % 6) == 0):
        n = int(num_vars / 6)        
        colors = (["#D81B60"] * n) + (["#BFB3C3"] * n) + (["#1E88E5"] * n) + (["#004D40"] * n) + (["#FFC107"] * n) + (["#723838"] * n)

    # Hauteur totale de la figure selon le nombre de plots
    hauteur_totale = hauteur_par_plot * num_plots


    # ----------------------------------------------------------------- #
    # Télécharger les signatures cosmic 
    ref = pd.read_csv("/data/chavezs2/surgeNMF/Reference_SBS96_COSMIC_Catalogue_Ordered.txt", sep="\t", index_col=0)
    
    # Déterminer les distances entre chaque colonne
    dist_mat = cdist(W.T, ref.T, "cosine")
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    # Trouver les signatures de ref
    Walt    = ref.iloc[:, col_ind]
    Walt_np = Walt.to_numpy()


    # ----------------------------------------------------------------- #
    figure, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(largeur * 2, hauteur_totale))
    if num_plots == 1:
        axes = [axes]  # s'assurer que axes est toujours une liste
    figure.suptitle("Signatures de novo vs COSMIC", fontsize=14)

    for i in range(num_plots):
        axes[i][0].bar(range(1, num_vars+1), W[:,row_ind[i]], color=colors, alpha=0.9)
        axes[i][1].bar(range(1, num_vars+1), Walt_np[:,i], color=colors, alpha=0.9)
        axes[i][0].set_title(which_sigs[row_ind[i]])
        axes[i][1].set_title(Walt.columns[i])
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

    plt.close()

