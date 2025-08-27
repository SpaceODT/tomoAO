"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""


import tomoAO.tools.tomography_tools as tools


import numpy as np
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False


import matplotlib.pyplot as plt
from scipy.sparse import block_diag

import scipy

from time import time



class tomoReconstructor:
    """
    Instantiates a LinearMMSE reconstructor.

    Parameters
    ----------
    tel : Telescope
        Telescope object.
    atmModel : Atmosphere
        Atmosphere object.
    guideStar : list of Source
        List of guide star objects.
    mmseStar : list of Source
        List of target star objects.
    dm : DeformableMirror
        Deformable mirror object.
    outputRecGrid : ndarray
        A mask where the phase is to be reconstructed.
    validSubapMask : bool
        A multi-dimensional mask with the valid subapertures per WFS channel.
    model : str, optional
        Whether 'zonal' (default) or modal.
    noiseCovariance : float or ndarray, optional
        The noise covariance matrix as a scalar or a matrix.
    lag : float, optional
        The AO system lag that can be compensated through tomography.
    weightOptimDir : float, optional
        A vector with the relative weights for each optimization direction.
    os : int, optional
        The over-sampling factor [1, 2, 4] (default=2) to apply to the reconstructed phase w.r.t the input slopes-maps.
    zernikeMode : ndarray, optional
        Zernike modes used for modal removal sampled with the required `os` factor.

    Returns
    -------
    None
        This class instantiates the direct matrices and generates an MMSE reconstructor.
    """

    def __init__(self, aoSys, weight_vector=None, alpha=None,
                 model='zonal', noise_covariance=None, lag=0,
                 weightOptimDir=-1, os=2, zernikeMode=None, indexation="xxyy", order="C"):



        self.tel = aoSys.tel
        self.atmModel = aoSys.atm


        if isinstance(aoSys.lgsAst, list):
            # It's already a list, no need to convert
            self.guideStar = aoSys.lgsAst
        else:
            # Convert to a list
            self.guideStar = [aoSys.lgsAst]
        self.nGuideStar = len(self.guideStar)


        if isinstance(aoSys.mmseStar, list):
            # It's already a list, no need to convert
            self.mmseStar = aoSys.mmseStar
        else:
            # Convert to a list
            self.mmseStar = [aoSys.mmseStar]

        self.nMmseStar = len(self.mmseStar) if self.mmseStar is not None else 0


        self.dm = aoSys.dm

        self._alpha = alpha
        self._lag = lag
        self.weightOptimDir = weightOptimDir
        self.os = os
        self.model = model
        self.zernikeMode = zernikeMode
        self.indexation = indexation
        self.order = order


        self.outputRecGrid = aoSys.outputReconstructiongrid

        #All valid subap mask
        self.unfiltered_subap_mask = aoSys.unfiltered_subap_mask

        #Valid subap mask
        self._filtered_subap_mask = aoSys.filtered_subap_mask

        self.unfiltered_act_mask = aoSys.unfiltered_act_mask

        if weight_vector is None:
            self.weight_vector = self.computeDefaultWeightVector()
        else:
            self.weight_vector = weight_vector

        if noise_covariance is None:
            self.noise_covariance = self.computeDefaultNoiseCovariance()
        else:
            self.noise_covariance = noise_covariance






        if isinstance(self.weightOptimDir, str):
            if self.weightOptimDir.lower() == 'avr' or self.weightOptimDir.lower() == 'average':
                self.weightOptimDir = 1 / self.nMmseStar * np.ones(self.nMmseStar)
            else:
                raise ValueError('Keyword for optimization weights not recognized')
        elif self.weightOptimDir != -1:  # Default optimization in individual directions
            if len(self.weightOptimDir) != self.nMmseStar:
                raise ValueError('The weights are not the same size as the number of optimization directions')
            else:
                self.weightOptimDir = self.weightOptimDir / np.sum(self.weightOptimDir)

        # %% scale r0 in case of different wavelengths
        wvlScale = self.guideStar[0].wavelength / self.atmModel.wavelength
        self.atmModel.r0 = self.atmModel.r0 * wvlScale ** 1.2



        # %% FITTING MATRIX
        if self.dm is not None:
            iFittingMatrix = 2*self.dm.modes[self.outputRecGrid.flatten("F"),]
            # iFittingMatrix = 2*self.dm.modes[self.outputRecGrid.flatten("F"),]


            if cuda_available:
                self.fittingMatrix = cp.linalg.pinv(cp.asarray(iFittingMatrix), rcond=1e-3).get()
            else:
                self.fittingMatrix = np.linalg.pinv(iFittingMatrix, rcond=1e-3)
        else:
            self.fittingMatrix = None

        # %% RECONSTRUCTOR
        self.buildReconstructor()




    @property
    def noise_covariance(self):
        return self._noise_covariance

    @noise_covariance.setter
    def noise_covariance(self, val):
        if val is not None:
            if self.model == 'modal' and isinstance(val, (int, float)):
                    n_mode = len(self.zernikeMode)
                    val = [[val] * n_mode] * n_mode
                    val = [[val[i][j] if i == j else 0 for j in range(n_mode)] for i in range(n_mode)]
            elif isinstance(val, (int, float)):
                val = val * np.eye(self.Gamma.shape[0])
        else:
            #Default noise covariance matrix TODO: Evaluate if this should be the default setting
            val = 1e-3 * self.alpha * np.diag(1 / (self.weight_vector.flatten("F") + 1e-8))

        self._noise_covariance = val


    def computeDefaultNoiseCovariance(self):
        self.noise_covariance = None


    @property
    def weight_vector(self):
        return self._weight_vector

    @weight_vector.setter
    def weight_vector(self, val):

        if val is not None:
            self._weight_vector = val

        else:
            # #Default weight vector TODO: Evaluate if this should be the default setting
            # weight_vector = np.ones([2 * np.count_nonzero(self.unfiltered_subap_mask), self.nGuideStar])
            # filteredAllValidMask = self.filtered_subap_mask.T[self.unfiltered_subap_mask.T]

            weight_vector = np.ones([2 * np.count_nonzero(self.filtered_subap_mask), self.nGuideStar])
            filteredAllValidMask = self.filtered_subap_mask.T[self.filtered_subap_mask.T]

            filteredAllValidMask = np.tile(filteredAllValidMask, 2)
            if np.any(filteredAllValidMask == 0):
                weight_vector[~filteredAllValidMask] = 0
            self._weight_vector = weight_vector


    def computeDefaultWeightVector(self):
        self.weight_vector = None


    @property
    def filtered_subap_mask(self):
        return self._filtered_subap_mask

    @filtered_subap_mask.setter
    def filtered_subap_mask(self, val):
        self._filtered_subap_mask = val


    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val


    @property
    def R_unfiltered(self):
        return self._R_unfiltered

    def buildCovarianceMatrices(self):
        print("Building the covariance matrices")
        self.Cox = []
        for i in range(len(self.mmseStar)):
            if cuda_available:
                res = tools.spatioAngularCovarianceMatrix_gpu(self.tel, self.atmModel, [self.mmseStar[i]], self.guideStar,
                                                          self.unfiltered_subap_mask, self.os)
            else:
                res = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, [self.mmseStar[i]], self.guideStar,
                                    self.unfiltered_subap_mask, self.os)
            self.Cox.append(res)


        if cuda_available:
            self.Cxx = tools.spatioAngularCovarianceMatrix_gpu(self.tel, self.atmModel, self.guideStar, self.guideStar,
                                                           self.unfiltered_subap_mask, self.os)

        else:
            self.Cxx = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, self.guideStar, self.guideStar,
                                                        self.unfiltered_subap_mask, self.os)


    def buildGamma(self):
        print("Building Gamma")
        Gamma, gridMask = tools.sparseGradientMatrixAmplitudeWeighted(self.filtered_subap_mask, amplMask=None,
                                                                      os=self.os)
        Gamma = [Gamma] * self.nGuideStar  # Replicate Gamma nMmseStar times
        gridMask = [gridMask] * self.nGuideStar
        Gamma = [np.asarray(mat) for mat in Gamma]
        self.Gamma = block_diag(Gamma).todense()
        self.gridMask = gridMask


    def buildReconstructor(self):
        print("Using GPU" if cuda_available else "GPU not available, using CPU")

        start = time()
        self.buildGamma()
        print(f"Took {time()-start} seconds to build Gamma")


        start = time()
        self.buildCovarianceMatrices()
        print(f"Took {time()-start} seconds to build the covariance matrices")


        print("Building the reconstructor")
        start_init = time()


        if cuda_available:
            self.Cox = cp.asarray(self.Cox)
            self.Cxx = cp.asarray(self.Cxx)
            self.noise_covariance = cp.asarray(self.noise_covariance)
            self.Gamma = cp.asarray(self.Gamma)



        if self.weightOptimDir == -1:
            self.RecStatSA = [None] * self.nMmseStar

            for k in range(self.nMmseStar):

                if cuda_available:
                    start = time()

                    self.RecStatSA[k] = (self.Cox[k] @ self.Gamma.T @ cp.linalg.pinv(self.Gamma @ self.Cxx @ self.Gamma.T + self.noise_covariance)).get()
                    print(f"Second part took {time() - start} seconds")

                else:
                    
                    self.RecStatSA[k] = self.Cox[k] @ self.Gamma.T @ np.linalg.pinv(self.Gamma @ self.Cxx @ self.Gamma.T + self.noise_covariance)

        else:  # weighted sum over all the optimisation directions
            # This code uses the ss stars to compute a weighted average tomographic
            # reconstructor over all of them
            self.RecStatSA = [None]
            CoxWAvr = np.sum([ self.Cox[k] * self.weightOptimDir[k] for k in range(self.nMmseStar)], axis=0)
            self.RecStatSA[0] = CoxWAvr @  self.Gamma.T @ np.linalg.pinv(self.Gamma @ self.Cxx @  self.Gamma.T + self.noise_covariance)


        self.reconstructor = np.array(self.fittingMatrix @ self.RecStatSA[0]) * self.guideStar[0].wavelength


        self.filtering_matrix = tools.get_filtering_matrix(self.unfiltered_subap_mask.copy(),
                                                           self.filtered_subap_mask.copy(), 
                                                           self.nGuideStar)

        self.reconstructor = np.array(self.reconstructor@self.filtering_matrix)


        if self.order == "C":
            self.signal_permutation_matrix = tools.get_signal_permutation_matrix(self.unfiltered_subap_mask, self.nGuideStar)
            self.dm_permutation_matrix = tools.get_dm_permutation_matrix(self.unfiltered_act_mask)
            self.reconstructor = np.array(self.dm_permutation_matrix@self.reconstructor@self.signal_permutation_matrix)


        if self.indexation == "xyxy":
            xyxy_permutation_matrix = tools.get_xyxy_permutation_matrix(self.reconstructor.shape[-1])
            self.reconstructor = self.reconstructor @ xyxy_permutation_matrix

        # unfiltered_mask = self.unfiltered_subap_mask.copy()
        # filtered_mask = self.filtered_subap_mask.copy()

        # M = np.zeros((np.count_nonzero(filtered_mask), np.count_nonzero(unfiltered_mask)))

        # unfiltered_mask_arr = unfiltered_mask.flatten("F") 
        # filtered_mask_arr = filtered_mask.flatten("F") 

        # count_row = 0
        # count_col = 0


        # for i in range(len(unfiltered_mask_arr)):
        #     if unfiltered_mask_arr[i] and filtered_mask_arr[i]:
        #         M[count_row, count_col] = 1
        #         count_row += 1
        #         count_col +=1

        #     elif unfiltered_mask_arr[i]:
        #         count_col +=1
        
        # M = scipy.linalg.block_diag(*[M]*2)

        # M = scipy.linalg.block_diag(*[M]*self.nGuideStar)

        # self.reconstructor = np.array(self.reconstructor@M)


        



        print(f"Took {time()-start_init} seconds to build the reconstructor")





