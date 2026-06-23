#!/data/$USER/conda/envs/surgeNMFenv/bin/python
# coding: utf-8 
"""
Last updated: 28 May 2026

@author Sergio Chávez
"""
"""

Class NMF with four methods: 
    Standard NMF algorithm: Frobenius norm, Kullback-Leibler divergence
    Volume-regularisation NMF: Covariance form Frobenius norm assuming Poisson data

Tasks for Serge: 
    - For today (may 21)
        Add constraints to input data, exempli gratia, number of rows = variables and number of columns = samples 
        I've fixed the issue by removing top those past Q3+1.5*IQR and got the code to work (it removed like 20ish
        data samples for being too big). I should consider normalising? Need to clean up the preprocessing part.
"""

import numpy as np

import time
# use: 
    # t0 = time.perf_counter()
    # stuff
    # t1 = time.perf_counter()
    # print("np.delete:",t1-t0)


class NMF: 
    """
    Docstring for NMF

    This class will allow the computation of multiple different types of NMF schemes:
    - Standard NMF [Frobenius norm]
    - Standard NMF [Kullback-Leibler divergence]
    - Volume-regularisation NMF Covariance-Poisson version
    - Volume-regularisation NMF Covariance-Gaussian version

    """
    def __init__(
        self, 
        data_matrix, 
        num_sigs: int,
        seed=None
    ):
        """
        Docstring for __init__
        
        :param data_matrix: [array of arrays or np.array] N column observations of M variables each
        :param num_sigs: [natural number] number of signatures wishing to seek
        :param seed: [natural number] Default is None: will use current time to obtain one
        """
        # Preliminary tests and assignments
        try: 
            self.num_sigs = int(num_sigs)
        except (TypeError, ValueError):
            # if not isinstance(num_sigs, int) or (num_sigs < 1):
            raise TypeError("Input num_sigs must be convertible to an integer")
        if self.num_sigs < 1:
            raise ValueError("Input num_sigs must be a postive integer")
        if seed is None:
            self.seed = int(time.time() * 1000) 
            self.rng = np.random.default_rng(self.seed)
        else:
            try:  
                self.seed = int(seed)
                self.rng = np.random.default_rng(self.seed)
            except (TypeError, ValueError):
                raise TypeError("Input seed must be convertible to an integer")
        
        # Stability parameter (This is the delta within the determinant mean to avoid log (zero) )
        self.epsilon = 1.

        # Processes data by filtering out low count mutation variables and determines dimensions
        self.num_vars, self.num_obs = np.shape(data_matrix)
        self.__preprocessing(data_matrix)

    def __preprocessing(
            self,
            data_matrix
    ):
        """
        Docstring for __preprocessing

        Its goal is to remove low count mutational variables from data, determine \
          the new size of the data matrix, put in appropriate form and initialise \
          W and H.
        
        :param data_matrix: [array of arrays or np.array] Matrix where each row is a different variable and each column is an observation
        """
        # --------------------------------------------------------------------------- #
        # --- Part I ---
        # (a) Remove variables (rows) with empty or low mutation counts 
        data_matrix             = np.array(data_matrix)
        num_counts_per_var      = data_matrix.sum(axis=1)
        total_mut_counts        = num_counts_per_var.sum()
        counts_sorted_idx       = np.argsort(num_counts_per_var,kind="stable")
        counts_sorted           = num_counts_per_var[counts_sorted_idx]
        counts_sorted_cumsum    = np.cumsum(counts_sorted)

        # Which mutation types make up less than 1% of total counts
        num_of_low_mut_var      = sum(counts_sorted_cumsum < (0.01 * total_mut_counts))
        self.vars_w_low_mut_counts = counts_sorted_idx[:num_of_low_mut_var]

        # Creates the adjusted data matrix self.V
        self.which_vars_to_keep = np.ones(self.num_vars, dtype=bool)
        self.which_vars_to_keep[self.vars_w_low_mut_counts] = False 
        self.X_rows_removed     = data_matrix[~self.which_vars_to_keep, :]
        self.X_data_mat_adj     = data_matrix[self.which_vars_to_keep, :]
        self.num_vars_adj       = self.num_vars - num_of_low_mut_var

        # (b) Remove any samples (columns) with empty or low mutation counts
        # num_counts_per_obs      = self.X_data_mat_adj.sum(axis=0)
        # total_mut_counts        = num_counts_per_obs.sum()
        # counts_sorted_idx       = np.argsort(num_counts_per_obs, kind="stable")
        # counts_sorted           = num_counts_per_obs[counts_sorted_idx]
        # counts_sorted_cumsum    = np.cumsum(counts_sorted)

        # # Which mutation types make up less than 1% of total counts
        # num_of_low_mut_obs      = sum(counts_sorted_cumsum < (0.01 * total_mut_counts))
        # self.obs_w_low_mut_counts = counts_sorted_idx[:num_of_low_mut_obs]
        
        # # Creates the adjusted data matrix self.V
        # self.which_obs_to_keep = np.ones(self.num_obs, dtype=bool)
        # self.which_obs_to_keep[self.obs_w_low_mut_counts] = False 
        # self.X_cols_removed    = self.X_data_mat_adj[:, ~self.which_obs_to_keep]
        # self.X_data_mat_adj    = self.X_data_mat_adj[:, self.which_obs_to_keep]
        # self.num_obs_adj       = self.num_obs - num_of_low_mut_obs

        # (b) Remove empty samples but also those that are too larges (> 3 stds)
        # print("Shape of X", self.X_data_mat_adj.shape) 
        # Remove those empty samples first 
        sums_cols  = self.X_data_mat_adj.sum(axis=0) 
        # plt.hist(sums_cols) 
        # plt.savefig("histogram_before.png") 
        zeros_cols = (sums_cols == 0) 
        if (zeros_cols.sum() > 0): 
            self.X_data_mat_adj = self.X_data_mat_adj[:, ~zeros_cols]
            # print("   ",sum(zeros_cols),"were eliminated")
        
        # Next determine if there are samples with much more mutations than others 
        # print("New shape of X", self.X_data_mat_adj.shape)
        sums_cols = sums_cols[~zeros_cols]
        Q1 = np.percentile(sums_cols, 25)
        Q3 = np.percentile(sums_cols, 75)
        IQR = Q3 - Q1
        # borne_inf = Q1 - 1.5 * IQR 
        borne_sup = Q3 + 1.5 * IQR 
        # print("1st quartile:",Q1)
        # print("3rd quartile:",Q3)
        # print("IQR:",IQR)
        # print("Upper limite:",borne_sup)
        indices = (sums_cols <= borne_sup)
        # print("Length of good", sum(indices))
        self.X_data_mat_adj = self.X_data_mat_adj[:, indices]
        # print("New new shape of X",self.X_data_mat_adj.shape)

        # # Garder seulement les valeurs dans les bornes
        # data_propre = data[(data >= borne_inf) & (data <= borne_sup)]

        # mean_num_mut_per_obs = np.mean(self.X_data_mat_adj) 
        # if ((self.X_data_mat_adj > 10 * mean_num_mut_per_obs).sum() > 0):
        #     std_num_mut_per_obs  = np.std(self.X_data_mat_adj) 
        #     which_samps_to_keep = (self.X_data_mat_adj < (mean_num_mut_per_obs + 3 * std_num_mut_per_obs))
        #     self.X_data_mat_adj = self.X_data_mat_adj[:, which_samps_to_keep] 

        # Check if 
        sums_rows  = self.X_data_mat_adj.sum(axis=1)
        zeros_rows = sums_rows == 0
        if (zeros_rows.sum() > 0): 
            print("Problem: have rows sum to 0")
            exit(1)

        # New number of observations 
        self.num_obs_adj = self.X_data_mat_adj.shape[1]

        # plt.close()
        # plt.hist(sums_cols[indices])
        # plt.savefig("histogram_after.png")

        # --------------------------------------------------------------------------- #
        # Part III: Initialise the matrices
        # 1. Create and normalise signature matrix W
        self.W_signatures_mat_init = self.rng.random((self.num_vars_adj, self.num_sigs)) 
        self.W_signatures_mat_init = self.W_signatures_mat_init / self.W_signatures_mat_init.sum(axis=0)

        # 2. Create initial exposure matrix H 
        # self.H_exposures_mat_init  = self.rng.uniform(0, self.X_data_mat_adj.max(), size=(self.num_sigs, self.num_obs_adj)) 
        self.H_exposures_mat_init  = self.rng.uniform(0, 1, size=(self.num_sigs, self.num_obs_adj)) 

        # 3. Create the sqrt matrix based on exposure matrix and the orthogonal matrix 
        # self.D_exposures_sqrt_mat_init = self.rng.uniform(0, 1, size=(self.num_sigs, self.num_sigs))
        self.D_exposures_sqrt_mat_init = self.H_exposures_mat_init @ self.H_exposures_mat_init.T
        R_u, R_d, _ = np.linalg.svd(self.D_exposures_sqrt_mat_init)
        self.D_exposures_sqrt_mat_init = -R_u * np.sqrt(R_d)
        self.D_exposures_sqrt_mat_init[self.D_exposures_sqrt_mat_init < 0] = 0.0
        self.Q_Procrustes_orthogonal_mat_init = np.eye(self.num_vars_adj,M=self.num_sigs,dtype=float)
        # self.Q_Procrustes_orthogonal_mat_init = np.zeros((self.num_vars_adj, self.num_sigs))
        # self.Q_Procrustes_orthogonal_mat_init[:self.num_sigs,:self.num_sigs] = np.eys(num_sigs)
        
    def standard_NMF_Frobenius(
        self, 
        max_iterations: int=int(2e2),
        tolerance=1e-8
    ):
        """
        Docstring for standard_NMF_Frobenius
        
        :param self: 
        :param max_iterations [positive integer] default 200 
        :param tolerance [positive real number] default 1e-8
        """
        # Check and save inputs 
        try: 
            max_iterations = int(max_iterations)
        except (TypeError, ValueError):
            raise TypeError("Input max_iterations_general must be convertible to an integer")
        if max_iterations < 1:
            raise ValueError("Input max_iterations_general must be a postive integer")
        if tolerance <= 0:
            raise ValueError("Input tolerance must be a small positive float value")
        else: 
            tolerance = tolerance

        # Update W and H alternatively
        W_iter = self.W_signatures_mat_init.copy()
        H_iter = self.H_exposures_mat_init.copy()

        # print("-------------------------------------------------------------------------")
        # print("\n             Initiating standard NMF with Frobenius norm             \n")
        # print("-------------------------------------------------------------------------")
        # print("\nSystem parameters:\n")
        # print("\tNumber of samples:                ",self.num_obs)
        # print("\tNumber of variables used:         ",self.num_vars_adj,"out of",self.num_vars)
        # print("\tNumber of signatures seeking:     ",self.num_sigs)
        # print("\n\tNumber of iterations: (general)   ",max_iterations)
        # print("\n\tError tolerance:                  ",tolerance)
        # print("\tSeed used:                        ",self.seed)
        # print("\n\n")
        # print("Starting analysis:\n")
        # print("...\n")
        iter = 0
        err  = 1.
        # print("\tStarting analysis:\n\t...")
        while (iter < max_iterations) and (err > tolerance):
            W_prev = W_iter.copy()

            # W numerator and denominator 
            W_num = self.X_data_mat_adj @ H_iter.T
            W_den = W_iter @ H_iter @ H_iter.T
            for p in range(self.num_sigs):
                for m in range(self.num_vars_adj):                    
                    W_iter[m,p] = W_iter[m,p] * (W_num[m,p] / W_den[m,p])
                    if np.isnan(W_iter[m,p]):
                        print(W_iter[m,p],W_num[m,p],W_den[m,p])
                        print(f"\nError in W at (iter,p,i)=({iter},{m},{p}). Witer[m,p]",W_iter[m,p],"\n")
                        exit(1)
            
            # Note that WH=WD^{-1}DH where diagonal entries of D are sums of columns of W 
            # D_diag = W_iter.sum(axis=0)
            # W_iter = W_iter / D_diag**(-1)
            # H_iter = np.diag(D_diag) @ H_iter 

            # W numerator and denominator 
            H_num = W_iter.T @ self.X_data_mat_adj
            H_den = W_iter.T @ W_iter @ H_iter
            
            for p in range(self.num_sigs):       
                for n in range(self.num_obs_adj):
                    # print(p,n,"out of",self.num_sigs,self.num_obs)
                    # if (iter == 1) and (p == 0) and (n == 310):
                        # print(p,n,"out of",self.num_sigs,self.num_obs)
                        # print(H_iter[p,n], H_num[p,n], H_den[p,n])
                        # print(" ")
                        # print(W_iter,"\n")
                        # print(self.X_data_mat_adj,"\n")
                        # print(W_iter.T @ self.X_data_mat_adj,"\n")
                    H_iter[p,n] = H_iter[p,n] * (H_num[p,n] / H_den[p,n])
                    if np.isnan(H_iter[p,n]):
                        print(H_iter[p,n],H_num[p,n],H_den[p,n])
                        print(f"\nError in H at (iter,p,n)=({iter},{p},{n}). Htemp[p,n]",H_iter[p,n],"\n")   
                        exit(1)

            iter += 1
            err   = ((W_prev - W_iter)**2).sum() / (W_iter**2).sum()
            # err   = np.linalg.norm(self.X_data_mat_adj - W_temp @ H_temp, ord="fro") / np.linalg.norm(self.X_data_mat_adj, ord="fro")

        # print("\tfinished at iteration",iter,"out of",max_iterations,"with W relative error",err)

        # Normalise W's columns and revert back to original number of rows while assuring H is proper size
        W_final = np.zeros((self.num_vars, self.num_sigs))
        W_final_col_sums = W_iter.sum(axis=0)
        W_final[self.which_vars_to_keep,:] = W_iter / W_final_col_sums 
        H_final = np.diag(W_final_col_sums) @ H_iter

        # Details about final solution
        H_exposure_mat_vol    = self.__volume_D_mat(H_iter @ H_iter.T)
        optimisation_func_val = ((self.X_data_mat_adj - W_iter @ H_iter)**2).sum() + (self.X_rows_removed**2).sum()

        self.St_NMF_Frobenius = {
            "max_iterations":        max_iterations,
            "tolerance":             tolerance, 
            "err":                   err,
            "iter":                  iter,
            "H_exposure_mat_vol":    H_exposure_mat_vol,
            "optimisation_func_val": optimisation_func_val,
            "W_final":               W_final,
            "H_final":               H_final
        }

    def standard_NMF_Kullback_Leibler(
        self, 
        max_iterations=2e2,
        tolerance=1e-8
    ):
        """
        Docstring for standard_NMF_Kullback_Leibler
        
        :param self: Description
        :param max_iterations [positive integer] default 200 
        :param tolerance [positive real number] default 1e-8
        """

        # Check and save inputs 
        try: 
            max_iterations = int(max_iterations)
        except (TypeError, ValueError):
            raise TypeError("Input max_iterations_general must be convertible to an integer")
        if max_iterations < 1:
            raise ValueError("Input max_iterations_general must be a postive integer")
        if tolerance <= 0:
            raise ValueError("Input tolerance must be a small positive float value")
        else: 
            tolerance = tolerance
        # Update W and H alternatively
        W_temp = self.W_signatures_mat_init.copy()
        H_temp = self.X_data_mat_adj.max() * self.H_exposures_mat_init.copy()
        
        stability_eps = 0.01 * H_temp.max()
        
        # print("-------------------------------------------------------------------------")
        # print("\n        Initiating standard NMF with Kullback-Leibler divergence     \n")
        # print("-------------------------------------------------------------------------")
        # print("\nSystem parameters:\n")
        # print("\tNumber of samples:                ",self.num_obs)
        # print("\tNumber of variables used:         ",self.num_vars_adj,"out of",self.num_vars)
        # print("\tNumber of signatures seeking:     ",self.num_sigs)
        # print("\n\tNumber of iterations: (general)   ",max_iterations)
        # print("\n\tError tolerance:                  ",tolerance)
        # print("\tSeed used:                        ",self.seed)
        # print("\n\n")
        # print("Starting analysis:\n")
        # print("...\n")
        iter = 0
        err  = 1.
        while (iter < max_iterations) and (err > tolerance):
            # Compute first for reduced calculations      
            X_over_WH  = self.X_data_mat_adj / np.maximum(W_temp @ H_temp, stability_eps)

            # Update each
            W_prev = W_temp.copy()
            W_temp = W_temp * (X_over_WH @ H_temp.T) / np.maximum(H_temp.sum(axis=1)[None, :], stability_eps)
            H_temp = H_temp * (W_temp.T @ X_over_WH) / np.maximum(W_temp.sum(axis=0)[:, None], stability_eps)

            iter += 1
            err   = ((W_prev - W_temp)**2).sum() / (W_temp**2).sum()

        # print("finished at iteration",iter,"out of",max_iterations,"with W relative error",err)

        # Normalise W's columns and revert back to original number of rows while assuring H is proper size
        W_final = np.zeros((self.num_vars, self.num_sigs))
        W_final_col_sums = W_temp.sum(axis=0)
        W_final[self.which_vars_to_keep, :] = W_temp / W_final_col_sums
        H_final = np.diag(W_final_col_sums) @ H_temp 

        # Details about final solution 
        H_exposure_mat_vol = self.__volume_D_mat(H_temp @ H_temp.T)
        WH = W_temp @ H_temp
        KLdiv = np.zeros_like(WH)
        mask = self.X_data_mat_adj > 0
        KLdiv[mask] = self.X_data_mat_adj[mask] * np.log(self.X_data_mat_adj[mask] / np.maximum(WH[mask], 1e-12)) - self.X_data_mat_adj[mask] + WH[mask]
        optimisation_func_val = KLdiv.sum()

        self.St_NMF_KullbackLeibler = {
            "max_iterations":        max_iterations,
            "tolerance":             tolerance,
            "err":                   err,
            "iter":                  iter, 
            "H_exposure_mat_vol":    H_exposure_mat_vol,
            "optimisation_func_val": optimisation_func_val,
            "W_final":               W_final,
            "H_final":               H_final
        }
        
    def volume_regularisation_NMF_Gaussian(
            self,
            vol_reg_val=None,
            num_singular_vals=None, 
            max_iterations_general=2e2,
            max_iterations_D=2e1,
            max_iterations_S=2e1,
            tolerance=1e-8,
            W=None,
            H=None
    ):
        """
        Docstring for volume_regularisation_NMF_Gaussian

        Volume-Regularisation NMF assuming Gaussian data whose goal is to minimise the frobenius norm of: 

            ||BQ - SD||^2_2 + lambda*log(det(DD^T + eps*I))

        :param vol_reg_val: [positive real number] Hyperparameter of the volume regularisation term. Default None.
        :param num_singular_vals: [positive integer] SVD reduced to this dimension
        :param max_iterations_general: [positive integer] Max iterations for main loop
        :param max_iterations_D: [positive integer] Max iterations for finding D 
        :param max_iterations_S: [positive integer] Max iterations for finding S 
        :param tolerance: [positive real number] Acceptable error 
        :param W: Input matrix in case a standard NMF method is performed before this one
        :param H: Input matrix in case a standard NMF method is performed before this one
        """
        # print("-------------------------------------------------------------------------")
        # print("\nInitiating volume-regularisation NMF w/ Frobenius norm for Gaussian data\n")
        # print("-------------------------------------------------------------------------")
        # print("\nSystem parameters:\n")
        # print("\tNumber of samples:                ",self.num_obs)
        # print("\tNumber of variables used:         ",self.num_vars_adj,"out of",self.num_vars)
        # print("\tNumber of signatures seeking:     ",self.num_sigs)
        # print("\tNumber of nonzero singular values:",int(num_singular_vals))
        # print("\n\tVolume-regularisation term value: ",vol_reg_val)
        # print("\n\tNumber of iterations: (general)   ",max_iterations_general)
        # print("\t                      (signatures)",max_iterations_S)
        # print("\t                      (exposures) ",max_iterations_D)
        # print("\n\tError tolerance:                  ",tolerance)
        # print("\tSeed used:                        ",self.seed)
        # print("\n\n")
        self.VR_NMF_Gaussian = self.__volume_regularisation_NMF("Gaussian",
                                                                vol_reg_val=vol_reg_val,
                                                                num_singular_vals=num_singular_vals,
                                                                max_iterations_general=max_iterations_general,
                                                                max_iterations_D=max_iterations_D,
                                                                max_iterations_S=max_iterations_S, 
                                                                tolerance=tolerance,
                                                                W_input=W,
                                                                H_input=H
                                                                )

    def volume_regularisation_NMF_Poisson(
            self,
            vol_reg_val=None,
            num_singular_vals=None,
            max_iterations_general=2e2,
            max_iterations_D=2e1,
            max_iterations_S=2e1,
            tolerance=1e-8,
            W=None,
            H=None
    ):
        """
        Docstring for volume_regularisation_NMF_Poisson

        Volume-Regularisation NMF assuming Poisson data whose goal is to minimise the frobenius norm of: 

            ||BQ - SD||^2_2 + lambda*log(det(DD^T + eps*I))

        :param vol_reg_val: [positive real number] Hyperparameter of the volume regularisation term. Default None.
        :param num_singular_vals: [positive integer] SVD reduced to this dimension
        :param max_iterations_general: [positive integer] Max iterations for main loop
        :param max_iterations_D: [positive integer] Max iterations for finding D 
        :param max_iterations_S: [positive integer] Max iterations for finding S 
        :param tolerance: [positive real number] Acceptable error 
        :param W: Input matrix in case a standard NMF method is performed before this one
        :param H: Input matrix in case a standard NMF method is performed before this one
        """
        # print("-------------------------------------------------------------------------")
        # print("\n Initiating volume-regularisation NMF w/ Frobenius norm for Poisson data\n")
        # print("-------------------------------------------------------------------------")
        # print("\nSystem parameters:\n")
        # print("\tNumber of samples:                ",self.num_obs)
        # print("\tNumber of variables used:         ",self.num_vars_adj,"out of",self.num_vars)
        # print("\tNumber of signatures seeking:     ",self.num_sigs)
        # print("\tNumber of nonzero singular values:",int(num_singular_vals))
        # print("\n\tVolume-regularisation term value: ",vol_reg_val)
        # print("\n\tNumber of iterations: (general)   ",max_iterations_general)
        # print("\t                      (signatures)",max_iterations_S)
        # print("\t                      (exposures) ",max_iterations_D)
        # print("\n\tError tolerance:                  ",tolerance)
        # print("\tSeed used:                        ",self.seed)
        # print("\n\n")
        self.VR_NMF_Poisson = self.__volume_regularisation_NMF("Poisson",
                                                               vol_reg_val=vol_reg_val,
                                                               num_singular_vals=num_singular_vals,
                                                               max_iterations_general=max_iterations_general,
                                                               max_iterations_D=max_iterations_D,
                                                               max_iterations_S=max_iterations_S, 
                                                               tolerance=tolerance,
                                                               W_input=W,
                                                               H_input=H
                                                              )

    def __volume_regularisation_NMF(
            self,
            data_type,
            vol_reg_val=None,
            num_singular_vals=None,
            max_iterations_general=2e2,
            max_iterations_D=2e1,
            max_iterations_S=2e1,
            tolerance=1e-8,
            W_input=None,
            H_input=None
    ):
        """
        Docstring for volume_regularisation_NMF
        """

        # --------------------------------------------------------------------------- #
        # Checks and balances
        if vol_reg_val is None:
            raise TypeError("Missing vol_reg_val: input positive real number")
        else:
            try: 
                vol_reg_val = float(vol_reg_val)
            except (TypeError, ValueError):
                raise TypeError("Input vol_reg_val is neither 'None' or a float convertible entry")
            if vol_reg_val <= 0:
                ValueError("Input vol_reg_val is not a small positive float value")
        if num_singular_vals is None:
            num_singular_vals = self.num_vars_adj
        else:
            try:
                num_singular_vals = int(num_singular_vals)
            except (TypeError, ValueError):
                raise TypeError("Input num_singular_vals must be convertible to an integer")
            if num_singular_vals < 1:
                raise ValueError("Input num_singular_vals must be a positive integer")
        try: 
            max_iterations_general = int(max_iterations_general)
        except (TypeError, ValueError):
            raise TypeError("Input max_iterations_general must be convertible to an integer")
        if max_iterations_general < 1:
            raise ValueError("Input max_iterations_general must be a postive integer")
        try: 
            max_iterations_D = int(max_iterations_D)
        except (TypeError, ValueError):
            raise TypeError("Input max_iterations_D must be convertible to an integer")
        if max_iterations_D < 1:
            raise ValueError("Input max_iterations_D must be a postive integer")
        try: 
            max_iterations_S = int(max_iterations_S)
        except (TypeError, ValueError):
            raise TypeError("Input max_iterations_S must be convertible to an integer")
        if max_iterations_S < 1:
            raise ValueError("Input max_iterations_S must be a postive integer")
        if tolerance <= 0:
            raise ValueError("Input tolerance must be a small positive float value")
        else: 
            tolerance = tolerance

        # --------------------------------------------------------------------------- #
        # Convert data and problem into the covariance form based on respective data type
        # Standard deviation of each variable
        X_std_dev_vec = np.std(self.X_data_mat_adj, axis=1) 
        if (data_type == "Gaussian"): 
            # Divide each variable by that std deviation
            X_scaled_mat  = self.X_data_mat_adj / X_std_dev_vec[:, np.newaxis]
            # This is the second-moment form for Gaussian random variables
            P_cov_data_mat = X_scaled_mat @ X_scaled_mat.T
            P_coeff = (1 / P_cov_data_mat.max())
            # Divide the matrix by the largest value
            P_cov_data_mat = P_coeff * P_cov_data_mat
        if (data_type == "Poisson"):
            # This is the second-moment form for Poisson random variables
            # Version 1 (no dividing by the standard deviation)
            P_cov_data_mat = self.X_data_mat_adj @ self.X_data_mat_adj.T - np.diag( self.X_data_mat_adj.sum(axis=1) )
            # Version 2 (divide variables by the standard deviation) [thus far unsuccessful]
            # X_scaled_mat = self.X_data_mat_adj / X_std_dev_vec[:, np.newaxis] 
            # P_cov_data_mat = X_scaled_mat @ X_scaled_mat.T - np.diag( self.X_data_mat_adj.sum(axis=1) / X_std_dev_vec[:, np.newaxis]**2 )
        P_u, P_d, _ = np.linalg.svd(P_cov_data_mat)

        # --------------------------------------------------------------------------- #
        # Initialise all parameters needed for NMF implementation
        B_sqrt_Poisson_data_mat = -P_u[:,:num_singular_vals] @ np.diag(np.sqrt(P_d[:num_singular_vals]))
        # Combine X=BQ and record all initial matrices
        if (H_input is None) or (W_input is None):
            if (data_type == "Gaussian"):
                D_iter = self.D_exposures_sqrt_mat_init
            if (data_type == "Poisson"):
                D_iter = self.X_data_mat_adj.max() * self.D_exposures_sqrt_mat_init
            S_iter = self.W_signatures_mat_init
            Q_iter = self.Q_Procrustes_orthogonal_mat_init[:num_singular_vals,:]
        else:
            # If W and H is provided, use those to construct the S and D matrices, which will also require finding a Q
            S_iter = W_input[self.which_vars_to_keep,:]
            D_iter = H_input @ H_input.T
            D_u, D_d, _ = np.linalg.svd(D_iter)
            D_iter = -D_u @ np.diag(np.sqrt(D_d))
            Q_iter = self.__solve_for_Q(B_sqrt_Poisson_data_mat, S_iter, D_iter)

        epsilon  = 0.01 * D_iter.max()**2
        X_iter   = B_sqrt_Poisson_data_mat @ Q_iter
        eps_x_ID = epsilon * np.eye(self.num_sigs) 
        DxD_T    = D_iter @ D_iter.T + eps_x_ID

        # Calculate the error at each iteration and register the initial error
        err_general = np.zeros(max_iterations_general)
        err_general[0] = 1.

        # Keep track of the value of optimisation function and volume of exposure matrix 
        H_exposure_mat_vol       = np.zeros(max_iterations_general)
        H_exposure_mat_vol[0]    = self.__volume_D_mat(DxD_T) 
        optimisation_func_val    = np.zeros(max_iterations_general)
        optimisation_func_val[0] = ( (X_iter - S_iter @ D_iter)**2 ).sum() + vol_reg_val * H_exposure_mat_vol[0]

        # print("Starting analysis:\n...")
        iter_general = 0
        while (iter_general < (max_iterations_general-1)) and (err_general[iter_general] > tolerance):
            iter_general += 1

            # Solve for D 
            D_iter = self.__solve_for_D(D_iter, DxD_T, X_iter, S_iter, eps_x_ID, vol_reg_val, max_iterations_D, tolerance)

            # Solve for S
            # S_prev = S_iter.copy()
            S_iter = self.__solve_for_S(S_iter, D_iter, X_iter, max_iterations_S, tolerance)

            # Solve for Q
            Q_iter = self.__solve_for_Q(B_sqrt_Poisson_data_mat, S_iter, D_iter)

            # Update X = BQ
            X_iter = B_sqrt_Poisson_data_mat @ Q_iter

            # Calculate the exposure matrix volume and optimisation value
            DxD_T = D_iter @ D_iter.T + eps_x_ID
            H_exposure_mat_vol[iter_general]    = self.__volume_D_mat(DxD_T)
            optimisation_func_val[iter_general] = ( (X_iter - S_iter @ D_iter)**2 ).sum() + vol_reg_val * H_exposure_mat_vol[iter_general] 

            # Calculate optimisation function value and error 
            err_general[iter_general] = abs(optimisation_func_val[iter_general-1] - optimisation_func_val[iter_general]) / max(1., abs(optimisation_func_val[iter_general]))
            # err_general[iter_general] = ((S_prev - S_iter)**2).sum() / (S_iter**2).sum()

        err_general           = err_general[:iter_general]
        H_exposure_mat_vol    = H_exposure_mat_vol[:iter_general]
        optimisation_func_val = optimisation_func_val[:iter_general]

        # Re-adjust the variable size back the original
        W_final = np.zeros((self.num_vars, self.num_sigs))
        if (data_type == "Poisson"):
            W_final[self.which_vars_to_keep, :] = S_iter
        if (data_type == "Gaussian"):
            # W_final[self.which_vars_to_keep, :] = X_std_dev_vec[:, None] * S_iter # This was causing problems (not summing up to 1)
            W_final[self.which_vars_to_keep, :] = S_iter

        # Dictionary that holds all the information about VR-NMF
        return {"vol_reg_val":            vol_reg_val,
                "num_singular_vals":      num_singular_vals,
                "max_iterations_general": max_iterations_general,
                "max_iterations_D":       max_iterations_D,
                "max_iterations_S":       max_iterations_S,
                "tolerance":              tolerance, 
                "err_general":            err_general,
                "iter_general":           iter_general,
                "X_std_dev_vec":          X_std_dev_vec,
                "H_exposure_mat_vol":     H_exposure_mat_vol,
                "optimisation_func_val":  optimisation_func_val,
                "W_final":                W_final    
               }

    def __volume_D_mat( 
            self, 
            DxD_T,
            vol_type="logdet"
    ):
        """
        Docstring for __volume_D_mat

        Find the volume based on the log det of D

        :param self: 
        :param DxD_T [numeric matrix] Exposure matrix (DD^T + eps*I) with dims <num_sigs> x <num_sigs>
        """
        if (vol_type == "logdet"):
            vol = np.log( np.linalg.det( DxD_T ) )
        else:
            TypeError("vol_type input is bad")
        
        return vol
    
    def __solve_for_D(
            self,
            D,
            DxD_T,
            X,
            S,
            eps_x_ID,
            vol_reg_val,
            max_iterations,
            tolerance
    ):
        """
        Docstring for __solve_for_D
        
        :param self:
        :param D:
        :param DxD_T:
        :param X:
        :param S:
        :param eps_x_ID: 
        :param vol_reg_val:
        :param max_iterations:
        :param tolerance:
        """
        # Through some quick numerical simulations, both non-majorated and majorated versions converge about the rate.
        # The two resulting matrices are practically the same with negligeable differences. 
        # wo_majorate = True
        # w_majorate  = False
        wo_majorate = False
        w_majorate  = True

        # Will get rid of this later. Majorate better theoretically 
        if wo_majorate: 
            # ------ Without Majorate ------ # 
            # Terms that are used many times
            S_TxS = S.T @ S 
            X_TxS = X.T @ S

            # Require an inverse
            FM = np.linalg.inv(DxD_T)

            # This is used several times so keep a copy 
            Lip_pre = S_TxS + vol_reg_val * FM
            Lip   = np.sqrt( (Lip_pre**2).sum() )

            iter_wo_majorate = 0
            err_wo_majorate  = 1.
            D_wo_majorate = D.copy()
            while (iter_wo_majorate < max_iterations) and (err_wo_majorate > tolerance):
                D_wo_majorate_prev = D_wo_majorate.copy()
                
                # Update D and ensure that are no non-negative numbers 
                gradient = D_wo_majorate @ Lip_pre - X_TxS
                D_wo_majorate = D_wo_majorate - (1/Lip) * gradient
                D_wo_majorate[D_wo_majorate < 0] = 0.
                
                err_wo_majorate   = ((D_wo_majorate_prev - D_wo_majorate)**2).sum() / (D_wo_majorate**2).sum()
                iter_wo_majorate += 1

            D = D_wo_majorate

        if w_majorate:
            # ------ With Majorate ------ # 
            S_TxS = S.T @ S 
            X_TxS = X.T @ S

            iter_D = 0
            err_D  = 1.
            while (iter_D < max_iterations) and (err_D > tolerance):
                D_prev = D.copy()

                # Require an inverse matrix, I've not looked into finding a work around
                FM = np.linalg.inv( D @ D.T + eps_x_ID )

                # Need this twice so let's not compute twice
                Lip_pre = S_TxS + vol_reg_val * FM
                Lip = np.sqrt( (Lip_pre**2).sum() )
                gradient = D @ Lip_pre - X_TxS

                # Updata D and ensure there are no non-negative numbers
                D = D - (1/Lip) * gradient
                D[D < 0] = 0.

                err_D = ((D_prev - D)**2).sum() / (D**2).sum()
                iter_D += 1

            D = D

        if wo_majorate and w_majorate:
            print("\nIter:",iter_wo_majorate,", error:", err_wo_majorate)
            print("Iter:",iter_D,", error:",err_D,"\n")
            print("Err in difference",((D_wo_majorate - D)**2).sum() / (D**2).sum())
            print("\n",D_wo_majorate,"\n\n",D)

        return D
    
    def __solve_for_S(
            self,
            S,
            D,
            X,
            max_iterations,
            tolerance
    ):
        """
        Docstring for __solve_for_S

        # Solve for S similarly to solving D but without the volume term, then project each vector 
        # onto the probability simplex. 
        
        :param self:
        :param S:
        :param D:
        :param X:
        :param max_iterations:
        :param tolerance:
        """
        # Function needed for Bisection method
        def central_diff(x, nu, h=1e-6):
            def g(x, nu):
                temp = x - nu 
                temp[temp > 0.] = 0.
                return np.linalg.norm(temp)**2 + nu*(x.sum() - 1) - len(x)*nu**2
            h = 1e-6
            return (g(x, nu + h) - g(x, nu - h)) / (2. * h)
        
        # If want to check how the two versions differ
        check = False

        # Precalculate matrices 
        R    = D @ D.T
        XD_T = X @ D.T
        Lip  = np.sqrt((R**2).sum())

        iter_S = 0
        err_S  = 1.
        while (iter_S < max_iterations) and (err_S > tolerance):
            S_prev = S.copy()

            # Compute the gradient and update S similar to find D but without volume term
            gradient = S @ R - XD_T 
            S_temp = S - (1/Lip) * gradient
            S_temp[S_temp < 0] = 0.

            # Project the answer into the probability simplex            
            S_theirs = np.zeros(S_prev.shape)
            S_mines  = np.zeros(S_prev.shape)
            for col in range(S_temp.shape[1]):
                # 1: Sort (ascending) each vector's values
                S_vec_sorted = np.sort(S_temp[:,col]) # ascending order
                # 2: Do a cumulative sum
                S_vec_cum_sum = np.cumsum(S_vec_sorted[::-1])
                # 3: subtract the bound (here 1) and divide by array(1:length(cumulative sum))
                averages = (S_vec_cum_sum - 1.) / np.arange(1,len(S_vec_cum_sum)+1)
                # 4: When does the average become positive
                condition = (averages[:-1] - averages[1:]) > 0.
                if (condition.sum() == 0):
                    ind = len(averages) - 1
                    # print("Has NOT True(s) in condition", ind)
                else:
                    ind = np.where(condition)[0][0]
                    # print("Has True(s) in condition", ind)
                
                # Save the found projection
                S_theirs[:,col] = S_temp[:,col] - averages[ind]
                S_theirs[S_theirs[:,col] < 0,col] = 0.

                # # Prepare bisection method
                # fa = 1. 
                # fb = 1. 
                # # Need to ensure values have opposite signs
                # count = 0
                # while (fa * fb) > 0 and (count < 100):
                #     a  = -S_temp.max() * self.rng.random(1)
                #     b  =  S_temp.max() * self.rng.random(1)
                #     fa = central_diff(S_temp[:,col], a) 
                #     fb = central_diff(S_temp[:,col], b) 
                #     count += 1

                # if (count == 100):
                #     raise ValueError("Have trouble finding a two oppisite signs values")

                # # Begin bisection method    
                # fm = 1.
                # count = 0
                # while (count < 100) and (abs(fm) > self.tolerance_vol_reg_NMF_cov_Poisson):
                #     mid = (a + b) / 2.
                #     fm  = central_diff(S_temp[:,col], mid)

                #     if fa * fm < 0:
                #         b  = mid 
                #         fb = fm
                #     else:
                #         a  = mid 
                #         fa = fm 

                #     count += 1
                #     if check:
                #         print(count, mid, fm)

                # S_mines[:,col] = S_temp[:,col] - mid 
                # S_mines[S_mines[:,col] < 0, col] = 0.
                
                # if check:
                #     print("\n",averages[ind],"\n",S_theirs[:,col])
                #     print("\n",mid,"\n",S_mines[:,col])
                #     print("\nDiff:", ((S_mines[:,col] - S_theirs[:,col])**2).sum() )

                #     exit(1)
            
            # S = S_mines 
            S = S_theirs

            iter_S += 1
            err_S  = ((S_prev - S)**2).sum() / (S**2).sum()

        return S
    
    def __solve_for_Q(
            self, 
            B, 
            S, 
            D
    ):
        """
        Docstring self.__solve_for_Q

        Perform Procrustes method

        :param self
        :param B
        :param S 
        :param D
        """
        Q_u, _, Q_v_T = np.linalg.svd( (D.T @ S.T) @ B)
        return Q_v_T[:self.num_sigs,:].T @ Q_u.T


if __name__ == "__main__":
    
    print("Fait autre chose! - Pigeon")