"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""


import tomoAO.tools.tomography_tools as tools


import numpy as np
cuda_available = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False


import matplotlib.pyplot as plt
from scipy.sparse import block_diag

from time import time






class tomoReconstructor:
    def __init__(self, aoSys, weight_vector=None, alpha=None,
                 model='zonal', noise_covariance=None, lag=0,
                 weightOptimDir=-1, os=2, zernikeMode=None):
        """Initialization function.

        Instantiate a LinearMMSE reconstructor

        :param tel: Telescope object [Telescope]
        :param atmModel: atmosphere object [Atmosphere]
        :param guideStar: a list of guide star objects [list of Source objects]
        :param mmseStar: a list of target star objects [list of Source objects]
        :param dm: deformable mirror object [Deformable Mirror]
        :param outputRecGrid: a mask where the phase is to be reconstructed
        :param validSubapMask: a multi-dim mask with the valid subapertures per WFS channel [bool]
        :param model: whether 'zonal' (default) or modal [str]
        :param noiseCovariance: the noise covariance matrix as a scalar [float] or a matrix [ndarray]
        :param lag: the AO system lag that can be compensated through tomography [float]
        :param weightOptimDir: a vector with the relative weights for each optimization direction [float]
        :param os: the over-sampling factor [1, 2, 4] (default=2) to apply to the reconstructed phase w.r.t the inpup slopes-maps [int]
        :param zernikeMode: zernike modes used for modal removal sampled with the required os factor [ndarray]

        :return: This class instantiates the direct matrices and generates an MMSE reconstructor
        """

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

        self.outputRecGrid = aoSys.outputReconstructiongrid

        #All valid subap mask
        self.unfiltered_subap_mask = aoSys.unfiltered_subap_mask

        #Valid subap mask
        self._filtered_subap_mask = aoSys.filtered_subap_mask

        if weight_vector is None:
            self.weight_vector = self.computeDefaultWeightVector()
        else:
            self.weight_vector = weight_vector

        if noise_covariance is None:
            self.noise_covariance = self.computeDefaultNoiseCovariance()
        else:
            self.noise_covariance = noise_covariance






        # self._weight_vector = weight_vector
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


        # Gamma, gridMask = tools.sparseGradientMatrixAmplitudeWeighted(self.unfiltered_subap_mask, amplMask=None,
        #                                                               os=self.os)
        #
        # Gamma = [Gamma] * self.nGuideStar  # Replicate Gamma nMmseStar times
        # gridMask = [gridMask] * self.nGuideStar
        # Gamma = [np.asarray(mat) for mat in Gamma]
        # self.Gamma = block_diag(Gamma)
        # self.gridMask = gridMask


        # %% FITTING MATRIX
        if self.dm is not None:
            iFittingMatrix = 2*self.dm.modes[self.outputRecGrid.flatten("F"),]

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
            val = 1e-3 * self.alpha * np.diag(1 / (self.weight_vector.flatten(order='F') + 1e-8))

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
            weight_vector = np.ones([2 * np.count_nonzero(self.unfiltered_subap_mask), self.nGuideStar])
            filteredAllValidMask = self.filtered_subap_mask.T[self.unfiltered_subap_mask]
            filteredAllValidMask = np.tile(filteredAllValidMask, 2)
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
            res = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, [self.mmseStar[i]], self.guideStar,
                                                self.unfiltered_subap_mask, self.os)
            self.Cox.append(res)


        self.Cxx = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, self.guideStar, self.guideStar,
                                                 self.unfiltered_subap_mask, self.os)


    def buildGamma(self):
        print("Building Gamma")
        Gamma, gridMask = tools.sparseGradientMatrixAmplitudeWeighted(self.unfiltered_subap_mask, amplMask=None,
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
        start = time()


        if cuda_available:
            self.Cox = cp.asarray(self.Cox)
            self.Cxx = cp.asarray(self.Cxx)
            self.noise_covariance = cp.asarray(self.noise_covariance)
            self.Gamma = cp.asarray(self.Gamma)

        if self.weightOptimDir == -1:
            self.Reconstructor = [None] * self.nMmseStar

            for k in range(self.nMmseStar):

                if cuda_available:
                    self.Reconstructor[k] = (
                                self.Cox[k] @ self.Gamma.T @ cp.linalg.pinv(self.Gamma @ self.Cxx @ self.Gamma.T + self.noise_covariance)).get()
                else:
                    self.Reconstructor[k] = self.Cox[k] @ self.Gamma.T @ np.linalg.pinv(self.Gamma @ self.Cxx @ self.Gamma.T + self.noise_covariance)

        else:  # weighted sum over all the optimisation directions
            # This code uses the ss stars to compute a weighted average tomographic
            # reconstructor over all of them
            self.Reconstructor = [None]
            CoxWAvr = np.sum([ self.Cox[k] * self.weightOptimDir[k] for k in range(self.nMmseStar)], axis=0)
            self.Reconstructor[0] = CoxWAvr @  self.Gamma.T @ np.linalg.pinv(self.Gamma @ self.Cxx @  self.Gamma.T + self.noise_covariance)


        self._R_unfiltered = self.fittingMatrix @ self.Reconstructor[0]
        print(f"Took {time()-start} seconds to build the reconstructor")





