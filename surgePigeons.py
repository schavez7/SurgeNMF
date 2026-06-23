# surgePigeons.py
# 
#   Créer deux ensembles de data: une de Gausse et l'autre de Poisson
# 
# Last modified: 17 Juin 2026
# ----------------------------------------------------------------- #

import numpy as np 
import pandas as pd
import sys
import os

from surgePeacock import plot_signatures


# ----------------------------------------------------------------- #
# Il y a un répertoire pour les résultats?
if len(sys.argv) < 3:
    print("   Ça veux 3 arguments: <fichier info> <répertoire de data>\n")
    exit(1) 
else: 
    # Passer le chemin du répertoire
    fichier_info = sys.argv[1]
    dir_data     = sys.argv[2] 

# Télécharger les informations 
info = pd.read_csv(fichier_info, sep="\t", header=None, index_col=0)
seed     = int(info.loc["seed"].values[0])
num_obs  = int(info.loc["num_obs"].values[0])
num_sigs = int(info.loc["num_sigs"].values[0])
if num_sigs < 2: 
    print(f"   {num_sigs} n'est pas un nombre intègre positif plus grand que 1")
    exit(1)

# Random generator
rng = np.random.default_rng(seed)

# Télécharger la réference
ref = pd.read_csv(f"/data/{os.getenv('USER')}/surgeNMF/Reference_SBS96_COSMIC_Catalogue_Ordered.txt", sep="\t", header=0, index_col=0)
ref_colonnes = ref.columns

# Signatures aléatoires de COSMIC 
which_sigs_nombres = rng.permutation(ref.shape[1])[:num_sigs]
which_sigs_noms = ['SBS1', 'SBS4', 'SBS7c', 'SBS13']
# which_sigs_noms    = list(ref_colonnes[which_sigs_nombres])
print(f"\n   α) Vous avez choisi {num_sigs} signatures et il sont:")
print("\t",which_sigs_noms,"\n")

# Sauvegarder les noms des signatures 
with open(fichier_info, "a") as f:
    f.write(f"sigs\t{','.join(which_sigs_noms)}\n")

# Paramètre du système
num_vars = ref.shape[0]

# Obtenir les signatures (pandas.DataFrame)
W_cosmic = ref[which_sigs_noms]
noms_lignes = W_cosmic.index
plot_signatures(W_cosmic.to_numpy(), which_sigs=W_cosmic.columns , save_loc=os.path.join(dir_data, f"vrai_W"))

# Create H based on uniform distribution (but make sure some of its values are 0 [sufficiently scattered])
H_max = 500
H_std = 0.01 * H_max 
H = rng.uniform(low=0, high=H_max, size=(num_sigs,num_obs))
# - This part ensures that some of those samples have zeros entries to ensure SSC
for n in range(num_obs):
    lignes_nulle = rng.choice(num_sigs, size=rng.integers(1, num_sigs-1), replace=False)
    H[lignes_nulle,n] = 0.0

noms_obs = [f"Samp {i}" for i in range(1,num_obs+1)]
H = pd.DataFrame(H, index=which_sigs_noms, columns=noms_obs)

# Construire la vérité 
genome_data = pd.DataFrame(np.rint(W_cosmic @ H)).clip(lower=0)
genome_data_np = genome_data.to_numpy()

# Construire les observations 
genome_data_Gaussian = pd.DataFrame(
    np.clip(np.rint(rng.normal(loc=genome_data_np, scale=H_std)).astype(np.int64), 0, None),
    index=noms_lignes,
    columns=noms_obs
)
genome_data_Poisson = pd.DataFrame(
    rng.poisson(lam=genome_data_np + 1e-10),
    index=noms_lignes,
    columns=noms_obs
)

# Sauvegarder les matrices 
fichier_W    = os.path.join(dir_data, "vrai_W.txt")
fichier_H    = os.path.join(dir_data, "vrai_H.txt")
fichier_X    = os.path.join(dir_data, "vrai_genome.txt")
fichier_Gaus = os.path.join(dir_data, "data_Gaussian_distr.txt")
fichier_Pois = os.path.join(dir_data, "data_Poisson_distr.txt")
W_cosmic.to_csv(fichier_W, sep="\t")
H.to_csv(fichier_H, sep="\t")
genome_data.to_csv(fichier_X, sep="\t")
genome_data_Gaussian.to_csv(fichier_Gaus, sep="\t")
genome_data_Poisson.to_csv(fichier_Pois, sep="\t")
print(f"   β) Vos fichiers de data sont construits:\n\t",fichier_Gaus,"\n\t",fichier_Pois,"\n\n")

