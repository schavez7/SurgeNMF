# ----------------------------------------------------------------- #
# surgeHuatzin.py
# 
#   Créer un heatmap. 
#  CE FICHIER N'EST PAS NECESSAIRE MAINTENAT. JE LE METTRE DANS QUETZAL
# 
# Last modified: Le 29 Juin 2026
# ----------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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


def plot_heatmap(
        denovo,
        save_loc = None
):
    """
    Calcule la similarité cosinus entre denovo et la référence COSMIC, puis crée et sauvegarde le heatmap.

    :param denovo:   [np.array ou pd.DataFrame] signatures de-novo  (num_vars x num_sigs_denovo)
    :param titre:    [string] titre du heatmap
    :param save_loc: [string] chemin pour sauvegarder le heatmap
    """
    # Télécharger la référence COSMIC
    ref = pd.read_csv(f"/data/{os.getenv('USER')}/SurgeNMF/surgeNMF/Reference_SBS96_COSMIC_Catalogue_Ordered.txt", sep="\t", index_col=0)

    # Accepter pandas ou numpy pour denovo
    if isinstance(denovo, pd.DataFrame):
        noms_denovo = list(denovo.columns)
        denovo      = denovo.values
    else:
        noms_denovo = [f"Denovo {i+1}" for i in range(denovo.shape[1])]

    noms_ref = list(ref.columns)
    ref      = ref.values

    sim_matrix = cosine_similarity_matrix(denovo, ref)

    # ----------------------------------------------------------------- #
    # Créer le heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(noms_ref) * 0.6),
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
            val     = sim_matrix[i, j]
            couleur = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=couleur)

    plt.tight_layout()

    if save_loc is not None:
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        plt.savefig(save_loc, dpi=200, bbox_inches="tight")
        print(f"Heatmap sauvegardé : {save_loc}")

    plt.close()

