# SurgeNMF: MV-NMF-4-MSA
Minimum-volume non-negative matrix factorisation for mutational signature analysis

Last updated README: 23 June 2026

## 📁 Les dossiers 🐍 🐦 🐧 🦅 🦜 🦢 🦉 🐓 🦩 🦆 🐤

### Scripts dans surgeNMF
| Fichier | Short Description |
|---|---|
| `surgeNMF.py` | Classe NMF avec toutes les méthodes |
| `surgePigeons.py` | 🕊️ Génère les données simulées |
| `surgeCrows.py` | 🐦‍⬛ Clustering et sélection optimale |
| `surgePeacock.py` | 🦚 Visualisation des signatures |
| `Reference_SBS96_COSMIC_Catalogue_Ordered.txt` | Le catalogue des signatures de COSMIC SBS96 ordonnées |


### Scripts dans VRNMF
| Fichier | Short Description |
|---|---|
| `update_estimate.R` | |
| `update_volume.R` | | 
| `update_simplex.R` | |
| `update_unitary.R` | |
| `vpreprocess.R` | |

## 📁 Fichiers pour tester 🐍 🐦 🐧 🦅 🦜 🦢 🦉 🐓 🦩 🦆 🐤

### 🔧 Scripts Bash 
| Fichier | Short Description |
|---|---|
| `job_test_one_nmf.sh` | Lance une seule instance de NMF |
| `job_test_full_analysis.sh` | Lance l'analyse complète avec bootstrap et grid search |
| `job_test_vrNMF.sh` | Lance une seule instance de vr-NMF (version R) |

### Scripts Python
| Fichier | Short Description |
|---|---|
| `test_one_nmf.py` | Tests just one nmf instance to ensure method works |
| `test_full_analysis.py` | Tests the grid search for num of sigs and λ value on bootstrapped data |
| `test_vrNMF.R` | Runs default vrNMF as in [Seplyarskiy, 2021] |


## Comment les éxecuter 
### `job_test_one_nmf.sh`

```bash
bash job_test_one_nmf.sh --dossier Results_S --num_sigs 4 --num_obs 250 --seed 1
```

### `job_test_full_analysis.sh`
```bash
bash job_test_full_analysis.sh --dossier Results_F --num_sigs 4 --num_obs 250 --seed 1
```

### `job_test_vrNMF.sh` 
```bash
bash job_test_vrNMF.sh --dossier Results_R --num_sigs 4 --num_obs 250 --seed 1
```

## Les méthodes de NMF dans `surgeNMF.py`

🐦 surgeNMF/ 
*  Standard NMF (Frobenius norm)
*  Standard NMF (Kullback-Leibler divergence)
*  MV-NMF (assuming Gaussian data)
*  MV-NMF (assuming Poisson data)

## Travail pour Serge 
* Ἄσκησις α´ - Verifier que VR-NMF fonctionne comme il doit
* Ἄσκησις β´ - Créer des fichiers d'exemple pour data réelle 

## Reférences
* Lee et Seung. NMF papers (1999, 2001).
* Seplyarskiy, Vladimir B. _et alia_. Population sequencing data reveal a compendium of mutational processes
  in the human germ line. _Science_ **373**, pages 1030-1035 (2021).
