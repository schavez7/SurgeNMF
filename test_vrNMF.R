# Apply VR-NMF for Gaussian data on same data 
# 
# Last update: 22 June 2026 


graphics.off()
rm(list = ls())


# ------------------------- Libraries and functions ---------------------------#
library(readxl)    # Used to read in excel files 
library(lattice)   # Used for creating the heatmaps 
library(ggplot2)   # For plotting using sigminer
suppressPackageStartupMessages(library(sigminer))  # Used to obtain the COSMIC signatures and for plotting 
suppressPackageStartupMessages(library(gridExtra)) # For plotting using lattice

source("VRNMF/vpreprocess.R")
source("VRNMF/volnmf_estimate.R")
source("VRNMF/update_volume.R")
source("VRNMF/update_simplex.R")
source("VRNMF/update_unitary.R")


# -------------------------- Les arguments dehors ---------------------------- # 
# Récupérer l'arguments
args <- commandArgs(trailingOnly = TRUE) 
fichier.info <- args[1] 
dir.data     <- args[2] 
dir.resultat <- args[3] 

# Lire le fichier txt 
info.table <- read.table(fichier.info, sep="\t", header=FALSE, row.names=1)
seed     <- as.integer(info.table["seed",1])
num.sigs <- as.integer(info.table["num_sigs",1])
num.obs  <- as.integer(info.table["num_obs",1])


# ----------------------------- Preliminaires ------------------------------- # 
set.seed(seed) 

X.data.mat <- read.table(
                paste0(dir.data,"/data_Gaussian_distr.txt"), 
                sep="\t", header=TRUE, row.names=1
)

# Sauvegarder les noms des lignes
noms.lignes <- rownames(X.data.mat)

# Extraire seulement la matrice 
rownames(X.data.mat) <- NULL 
colnames(X.data.mat) <- NULL 
X.data.mat <- as.matrix(X.data.mat)
num.vars <- dim(X.data.mat)[1] 


# Nettoyer la data 
# Clean the data by removing those variables (rows) that 
num.counts.per.mut   <- rowSums(X.data.mat)
total.mut.counts     <- sum(num.counts.per.mut)
counts.sorted.idx    <- order(num.counts.per.mut) 
counts.sorted        <- num.counts.per.mut[counts.sorted.idx]
counts.sorted.cumsum <- cumsum(counts.sorted)

# Which mutation types make up less than 1% of total counts
num.of.low.mut.types   <- sum(counts.sorted.cumsum < (0.01 * total.mut.counts))
mut.types.w.low.counts <- counts.sorted.idx[1:num.of.low.mut.types]

# Creates the adjusted data matrix 
which.mut.to.keep <- rep(TRUE, num.vars)
which.mut.to.keep[mut.types.w.low.counts] <- FALSE 
X.data.mat.adj <- X.data.mat[which.mut.to.keep, , drop=FALSE]
X.rows.removed <- X.data.mat[!which.mut.to.keep, , drop=FALSE]
num.vars.adj <- num.vars - num.of.low.mut.types 

# Leur analyse 
vol <- vol_preprocess(t(X.data.mat.adj))
volres <- volnmf_main(vol, n.comp = 4, wvol = 0, 
                      n.iter = 3e+3, vol.iter = 1e+1, c.iter = 1e+1, 
                      extrapolate = TRUE, accelerate = TRUE, verbose = FALSE
)

# Assurer que la matrice soit normalisée 
S.filtered <- volres$C * vol$col.factors / rowMeans(X.data.mat.adj)

# Reconstruire  S complet avex des lignes de zéros pour les colonnes retirée 
S.full <- matrix(0, nrow=nrow(X.data.mat), ncol=ncol(S.filtered))
S.full[which.mut.to.keep,] <- S.filtered
rownames(S.full) <- noms.lignes 
colnames(S.full) <- paste("Sig", 1:ncol(S.full))

# Plot 
fichier.pdf.path <- file.path(dir.resultat, "denovo_sigs_vrnmf.pdf")
pdf(fichier.pdf.path, width=10, height=6)
show_sig_profile( 
    S.full,
    mode = "SBS", 
    style = "cosmic",
    y_lab = paste("Vol-reg NMF devono signatures")
)
dev.off()
# p
# ggsave(fichier.pdf.path, plot=p, width=10, height=6)

print("Ça marche!")