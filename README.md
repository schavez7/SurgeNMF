# SurgeNMF: MV-NMF-4-MSA
Minimum-volume non-negative matrix factorisation for mutational signature analysis

Last updated README: 17 June 2026

## 📁 Les fichiers 🐍 🐦 🐧 🦅 🦜 🦢 🦉 🐓 🦩 🦆 🐤

### 🔧 Scripts Bash 

| Fichier | Short Description |
|---|---|
| `job_starling.sh` | Lance une seule instance de NMF |
| `job_crows.sh` | Lance l'analyse complète avec bootstrap et grid search |

### Scripts Python
| Fichier | Short Description |
|---|---|
| `surgeNMF.py` | Classe NMF avec toutes les méthodes |
| `surgePigeons.py` | 🕊️ Génère les données simulées |
| `surgeStarling.py` | Exécute les quatre versions de NMF |
| `surgeCrows.py` | 🐦‍⬛ Clustering et sélection optimale |
| `surgePeacock.py` | 🦚 Visualisation des signatures |

### Fichier additionnel 
| Fichier | Short Description |
|---|---|
| `Reference_SBS96_COSMIC_Catalogue_Ordered.txt` | Le catalogue des signatures de COSMIC SBS96 ordonnées |
​

## Comment les éxecuter 
### `job_starling.sh`

```bash
bash job_starling.sh --dossier Results --num_sigs 4 --num_obs 250 (--seed 1)
```

### `job_crows.sh`
```bash
bash job_crows.sh --dossier Results --num_sigs 4 --num_obs 250 (--seed 42)
```

## Les méthodes de NMF dans `surgeNMF.py`

🐦 surgeNMF/ 
* Ἄσκησις α´ — Standard NMF (Frobenius norm)
* Ἄσκησις β´ — Standard NMF (Kullback-Leibler divergence)
* Ἄσκησις γ´ — MV-NMF (assuming Gaussian data)
* Ἄσκησις δ´ — MV-NMF (assuming Poisson data)

## Reférences

* 
