#!/bin/bash 

# Générer les instances des génomes et l'exécuter depuis R
#
# Last updated: 22 Juin 2026


# ------------------------------------------------------------
# BON USAGE: bash job_test_vrNMF.sh --dossier <dossier> --num_sigs <num_sigs> --num_obs <num_obs> [--seed <seed>]
# EXEMPLE:   bash job_test_vrNMF.sh --dossier Results_R   --num_sigs 4          --num_obs 250        --seed 1
# ------------------------------------------------------------


set -euo pipefail 

echo
echo 
echo " ________________________________________________________________________ "
echo " Ξ                              σεργιος                                 Ξ "
echo " ------------------------------------------------------------------------ "
echo
echo



# ----------------------------------------------------------------------------- # 
# PASSER LES ARGUMENTS ET VERIFIER QU'ILS SONT DE BONS FORMES

# Initialiser toutes les variables à vide
NOM_RESULTATS=""
NUM_SIGS=""
NUM_OBS=""
RANDOM_SEED=$(date +%s) 

# Fonction qui détermine si les nomrbes sont positives et intègres 
est_entier_positif() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

# Lecture des options longues
ARGS=$(getopt -o "" --long dossier:,num_sigs:,num_obs:,seed: -- "$@")
eval set -- "$ARGS"

while true; do
    case "$1" in
        --dossier)  NOM_RESULTATS=$2 ; shift 2 ;;
        --num_sigs) NUM_SIGS=$2      ; shift 2 ;;
        --num_obs)  NUM_OBS=$2       ; shift 2 ;;
        --seed)     RANDOM_SEED=$2   ; shift 2 ;;
        --) shift ; break ;;
    esac
done

# Vérifier chaque argument nécessaire individuellement
ERROR=0
if [ -z "${NOM_RESULTATS}" ]; then
    echo "Argument manquant : --dossier"
    ERROR=1
fi
if [ -z "${NUM_SIGS}" ]; then
    echo "Argument manquant : --num_sigs"
    ERROR=1
fi
if [ -z "${NUM_OBS}" ]; then
    echo "Argument manquant : --num_obs"
    ERROR=1
fi

# Vérifier que les nombres sont des entiers positifs
if [ -n "${NUM_SIGS}" ] && ! est_entier_positif "${NUM_SIGS}"; then
    echo "--num_sigs doit être un entier positif (reçu : ${NUM_SIGS})"
    ERROR=1
elif [ -n "${NUM_SIGS}" ] && [ "${NUM_SIGS}" -lt 2 ]; then 
    echo "--num_sigs doit être supérieur ou égal à 2 (reçu : ${NUM_SIGS})" 
    ERROR=1
fi
if [ -n "${NUM_OBS}" ] && ! est_entier_positif "${NUM_OBS}"; then
    echo "--num_obs doit être un entier positif (reçu : ${NUM_OBS})"
    ERROR=1
fi
if ! est_entier_positif "${RANDOM_SEED}"; then
    echo "--seed doit être un entier positif (reçu : ${RANDOM_SEED})"
    ERROR=1
fi

# Quitter seulement après avoir affiché TOUS les arguments manquants
if [ "$ERROR" -eq 1 ]; then
    echo ""
    echo "Usage: bash job_test_one_nmf.sh --dossier <dossier> --num_sigs <num_sigs> --num_obs <num_obs> [--seed <seed>]"
    exit 1
fi


# ----------------------------------------------------------------------------- # 
# CONSTRUIRE LES RÉPERTOIRES NÉCESSAIRES

# --- Répertoires définis
DIR_MAIN=$(pwd)
DIR_RESULTATS=${DIR_MAIN}/"${NOM_RESULTATS%/}"
DIR_SUB_RESULTATS="${DIR_RESULTATS}/Denovo_Sigs"
DIR_DATA="${DIR_RESULTATS}/Input_Data"
FICHIER_INFO="${DIR_RESULTATS}/system_info.txt"

mkdir -p "$DIR_RESULTATS" "$DIR_SUB_RESULTATS" "$DIR_DATA"


# ----------------------------------------------------------------------------- # 
# SAUVEGARDER LES INFORMATIONS DU SYSTÈME

printf "seed\t${RANDOM_SEED}\n" > "$FICHIER_INFO"
printf "num_sigs\t${NUM_SIGS}\n" >> "$FICHIER_INFO"
printf "num_obs\t${NUM_OBS}\n" >> "$FICHIER_INFO"


# ----------------------------------------------------------------------------- # 
# DÉFINIR LE Python UTILISÉ
PYTHON=/data/$USER/conda/envs/surgeNMFenv/bin/python


# ----------------------------------------------------------------------------- # 
# TÉLÉCHARGER LE MODULE R
module load R 

# ----------------------------------------------------------------------------- # 
# LES CODES PRINCIPAUX

# --- Πρῶτος: Générer l'ensemble de data
$PYTHON $DIR_MAIN/surgePigeons.py $FICHIER_INFO $DIR_DATA

# --- Δεύτερος: l'Analyse 
Rscript test_vrNMF.R $FICHIER_INFO $DIR_DATA $DIR_SUB_RESULTATS 

