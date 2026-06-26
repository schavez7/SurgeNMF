# surgeHeatmap.py
#
#   Crée un heatmap de la similarité cosinus entre les signatures de-novo et la référence
#
# Utilisation: python surgeHeatmap.py <dir_resultats> <fichier_denovo> <fichier_reference>
# ----------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


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


def plot_heatmap(sim_matrix, noms_denovo, noms_ref, titre, filepath):
    """
    Crée et sauvegarde le heatmap.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(noms_ref) * 0.6),
                                    max(4, len(noms_denovo) * 0.5)))

    im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Similarité cosinus")

    # Étiquettes des axes
    ax.set_xticks(range(len(noms_ref)))
    ax.set_yticks(range(len(noms_denovo)))
    ax.set_xticklabels(noms_ref, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(noms_denovo, fontsize=8)
    ax.set_xlabel("Signatures de référence")
    ax.set_ylabel("Signatures de-novo")
    ax.set_title(titre)

    # Ajouter les valeurs dans chaque cellule
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            val = sim_matrix[i, j]
            couleur = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=couleur)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap sauvegardé : {filepath}")


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Utilisation: python surgeHeatmap.py <dir_resultats> <fichier_denovo> <fichier_reference>")
        exit(1)

    dir_resultats   = sys.argv[1]
    fichier_denovo  = sys.argv[2]
    fichier_ref     = sys.argv[3]

    # Charger les signatures de-novo et la référence
    denovo = pd.read_csv(fichier_denovo, sep="\t", index_col=0)
    ref    = pd.read_csv(fichier_ref,    sep="\t", index_col=0)

    # Garder seulement les variables communes (lignes)
    vars_communes = denovo.index.intersection(ref.index)
    denovo = denovo.loc[vars_communes]
    ref    = ref.loc[vars_communes]

    noms_denovo = list(denovo.columns)
    noms_ref    = list(ref.columns)

    # Calculer la similarité cosinus
    sim_matrix = cosine_similarity_matrix(denovo.values, ref.values)

    # Nom du fichier de-novo pour le titre (ex: "Std-NMF-Frob_denovo-sigs")
    nom_methode = os.path.splitext(os.path.basename(fichier_denovo))[0]
    titre       = f"Similarité cosinus — {nom_methode}"

    # Sauvegarder le heatmap dans le même dossier que les signatures
    dir_heatmaps = os.path.join(os.path.dirname(fichier_denovo), "..", "Heatmaps")
    os.makedirs(dir_heatmaps, exist_ok=True)
    filepath = os.path.join(dir_heatmaps, nom_methode + "_heatmap.png")

    plot_heatmap(sim_matrix, noms_denovo, noms_ref, titre, filepath)