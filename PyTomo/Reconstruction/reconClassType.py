"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""


import PyTomo.tools.tomography_tools as tools


import numpy as np
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False


import matplotlib.pyplot as plt
from scipy.sparse import block_diag


DEBUG = False





class LinearMMSE:
    def __init__(self,aoSys, weight_vector=None, intMat=None, static_maps_matrices=None, alpha=None,
                 model='zonal', noiseCovariance=None, lag=0, weightOptimDir=-1, os=2, zernikeMode=None, ordering='keck', minioning_flag='ON'):
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
        :param ordering: keck(default)|sim Whereas to interleave the x-y slopes (keck ordering) or to leave allX-then-allY for simulations

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

        self.outputRecGrid = aoSys.outputReconstructiongrid

        self.unfilteredSubapMask = aoSys.subap_mask

        self.runtime_valid_subap_mask = tools.check_subap_mask_dims(self.unfilteredSubapMask, len(self.guideStar))
        self.model = model
        self.zernikeMode = zernikeMode

        self._noiseCovariance = noiseCovariance


        self._lag = lag
        self.weightOptimDir = weightOptimDir
        self.os = os
        self.ordering = ordering
        self.mmseReconstructor = None

        self.weight_vector = weight_vector

        self.static_maps_matrices = static_maps_matrices

        if static_maps_matrices is not None:
            self.inv_cov_mat = static_maps_matrices.invcov

        if intMat is not None:
            self.intMat = intMat

        self.weight_vector = weight_vector
        self.alpha = alpha
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

        # %% Discrete gradient matrix
        # In case the validSubapMask is different per WFS channel iterate
        if len(self.unfilteredSubapMask) == self.nGuideStar:
            blk_Gamma = []
            for i in range(self.nGuideStar):

                validSubapMask = self.unfilteredSubapMask[i]  # Replace this with the code to get a different validSubapMask for each iteration

                # Call the function with different validSubapMask
                #TODO: Aqui tem que entrar a matriz com todas válidas
                Gamma, gridMask = tools.sparseGradientMatrixAmplitudeWeighted(validSubapMask, amplMask=None, os=self.os)

                # reorder the rows of Gamma to provide x-y slopes instead of the allX-allY used in simulations
                if self.ordering == 'keck':
                    N = np.count_nonzero(validSubapMask)
                    idx = np.arange(N * 2).reshape(2, N)
                    idx = idx.transpose().reshape(-1)
                    # Reorder the rows of the matrix
                    Gamma = Gamma.tocsr()[idx, :]

                # Append the results to the list
                blk_Gamma.append((Gamma, gridMask))

            # Create a block diagonal matrix from the Gamma results
            blk_diag_Gamma = block_diag(blk_Gamma)
            self.Gamma = blk_diag_Gamma

        else:  # otherwise it's the same for all, in which case one Gradient Matrix is computed and block_diag()

            Gamma, gridMask = tools.sparseGradientMatrixAmplitudeWeighted(self.unfilteredSubapMask, amplMask=None, os=self.os)
            #pdb.set_trace()
            # breakpoint()
            self.Gamma_single = Gamma.copy()
            self.ordering = None
            # reorder the rows of Gamma to provide x-y slopes instead of the allX-allY used in simulations
            if self.ordering == 'keck':
                # block of code extracting y-x-y-x for compatibility with Keck's IM!!!
                N = np.count_nonzero(self.unfilteredSubapMask)
                Gamma_copy = Gamma.copy()
                # Gamma_dense = Gamma_tmp.todense()
                Gamma = np.zeros(Gamma_copy.shape)
                Gamma[0::2] = Gamma_copy[N:, :]
                Gamma[1::2] = Gamma_copy[0:N, :]



            Gamma = [Gamma] * self.nGuideStar  # Replicate Gamma nMmseStar times
            gridMask = [gridMask] * self.nGuideStar

        # TODO: the scaling of the discrete gradient matrix needs to be worked out

        Gamma = [np.asarray(mat) for mat in Gamma]
        self.Gamma = block_diag(Gamma)
        self.gridMask = gridMask

        # %% FITTING MATRIX
        if self.dm is not None:
            iFittingMatrix = 2*self.dm.modes[self.outputRecGrid.flatten("F"),]

            if cuda_available:
                self.fittingMatrix = cp.linalg.pinv(cp.asarray(iFittingMatrix), rcond=1e-3).get()
            else:
                self.fittingMatrix = np.linalg.pinv(iFittingMatrix, rcond=1e-3)
        else:
            self.fittingMatrix = None




        # %% LGS-to-science cross-covariance matrix
        self.Cox = []
        for i in range(len(self.mmseStar)):
            res = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, [self.mmseStar[i]], self.guideStar,
                                                self.unfilteredSubapMask, self.os)
            self.Cox.append(res)


        # %% LGS-to-LGS cross-covariance matrix
        self.Cxx = tools.spatioAngularCovarianceMatrix(self.tel, self.atmModel, self.guideStar, self.guideStar,
                                                 self.unfilteredSubapMask, self.os)

        # %%
        self.solveLmmse()

        # %%

        self._R_unfiltered = self.fittingMatrix @ self.mmseReconstructor[0]


    @property
    def noiseCovariance(self):
        return self._noiseCovariance

    @noiseCovariance.setter
    def noiseCovariance(self, val):
        if val is not None:
            if self.model == 'modal' and isinstance(val, (int, float)):
                n_mode = len(self.zernikeMode)
                val = [[val] * n_mode] * n_mode
                val = [[val[i][j] if i == j else 0 for j in range(n_mode)] for i in range(n_mode)]
            elif isinstance(val, (int, float)):
                val = val * np.eye(self.Gamma.shape[0])

            self._noiseCovariance = val
            self.solveLmmse()


    def getNoiseCovariance(self):
        if self._noiseCovariance is None:
            return 1e-3 * self.alpha * np.diag(1 / (self.weight_vector.flatten(order='F') + 1e-8))
        else:
            return self._noiseCovariance


    # %%
    def solveLmmse(self):
        print(" -->> mmse solver!")

        m_Cox = self.Cox
        m_Cxx = self.Cxx
        m_Cn = self.getNoiseCovariance()
        m_Gamma = self.Gamma

        if cuda_available:
            m_Cox = cp.asarray(m_Cox)
            m_Cxx = cp.asarray(m_Cxx)
            m_Cn = cp.asarray(m_Cn)
            m_Gamma = cp.asarray(m_Gamma.todense())

        if self.weightOptimDir == -1:
            m_mmseReconstructor = [None] * self.nMmseStar
            for k in range(self.nMmseStar):

                if cuda_available:
                    m_mmseReconstructor[k] = (m_Cox[k] @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)).get()
                else:
                    m_mmseReconstructor[k] = m_Cox[k] @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)


        else:  # weighted sum over all the optimisation directions
            # This code uses the ss stars to compute a weighted average tomographic
            # reconstructor over all of them
            m_mmseReconstructor = [None]
            m_CoxWAvr = np.sum([m_Cox[k] * self.weightOptimDir[k] for k in range(self.nMmseStar)], axis=0)
            m_mmseReconstructor[0] = m_CoxWAvr @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)
        self.mmseReconstructor = m_mmseReconstructor

    def mtimes(self, data):
        return self.mmseReconstructor @ data

    def update(self, runtime_valid_subap_mask, runtime_valid_act_mask=None, update_matrices=None):

        # if not set by user, then the runtime_valid_act_mask is the global one for this DM
        if runtime_valid_act_mask is None:
            runtime_valid_act_mask = np.array([True] * len(self.dm.validAct.flatten())).astype(bool)

        runtime_valid_subap_mask = tools.check_subap_mask_dims(runtime_valid_subap_mask, len(self.guideStar))

        runtime_valid_subap_mask = tools.toggle(runtime_valid_subap_mask)

        nSubap = self.unfilteredSubapMask.shape[0]

        ids = []
        idx = []
        maxValidSubap = np.count_nonzero(self.unfilteredSubapMask)
        refRecGrid = tools.reconstructionGrid(self.unfilteredSubapMask, self.os)
        for k in range(len(self.guideStar)):
            maskS = runtime_valid_subap_mask[:, k][self.unfilteredSubapMask.flatten()]
            res = np.where(maskS)[0]
            ids.append(np.concatenate((res, res + maxValidSubap)) + k * 2 * maxValidSubap)

            # Compute the discrete reconstruction mask from the subap_mask passed on as input
            maskX = tools.reconstructionGrid(np.reshape(runtime_valid_subap_mask[:, k], (nSubap, nSubap)), self.os)
            res = np.where(maskX[refRecGrid])[0]
            idx.append(res + k * np.count_nonzero(refRecGrid))

        # convert lists to arrays
        ids = np.concatenate(ids)
        idx = np.concatenate(idx)

        # valid DM actuators

        maskA = runtime_valid_act_mask[self.dm.validAct.reshape(-1, 1)]
        ida = np.where(maskA)[0]

        print(" -->> mmse updator!")

        m_Cxx = self.Cxx[idx][:, idx]
        m_Cn = self._noiseCovariance

        # Check if m_Cn is a scalar (int, float, etc.)
        if np.isscalar(m_Cn):
            # Handle the case where it's a scalar, by creating a diagonal matrix
            m_Cn = m_Cn * np.eye(len(ids))
        else:
            # Assuming m_Cn is a NumPy array or similar
            m_Cn = m_Cn[ids, ids]
        m_Gamma = self.Gamma.toarray()
        m_Gamma = m_Gamma[ids][:, idx]

        if self.weightOptimDir == -1:
            m_mmseReconstructor = [None] * self.nMmseStar
            for k in range(self.nMmseStar):
                m_Cox = self.Cox[k][:, idx]
                m_mmseReconstructor[k] = m_Cox @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)
        else:  # weighted sum over all the optimisation directions
            # This code uses the ss stars to compute a weighted average tomographic
            # reconstructor over all of them
            m_mmseReconstructor = [None]
            m_Cox = self.Cox
            m_CoxWAvr = np.sum([m_Cox[k][:, idx] * self.weightOptimDir[k] for k in range(self.nMmseStar)], axis=0)
            m_mmseReconstructor[0] = m_CoxWAvr @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)

        self.mmseReconstructor = m_mmseReconstructor

        self.fittingMatrix = np.linalg.pinv(self.dm.modes[self.outputRecGrid.flatten()][:, ida], rcond=1e-3)

        # TODO the reshape from 2D to 3D ought to be made with the toggle_frame function yet it is not working
        runtime_valid_subap_mask = tools.toggle(runtime_valid_subap_mask)
        # sz = np.sqrt(validSubapMask.shape[0]).astype(int)
        # validSubapMask = np.reshape(validSubapMask, (sz, sz, 4), order='F')

        self._R_unfiltered = self.fittingMatrix @ self.mmseReconstructor[0]


    @property
    def R_unfiltered(self):
        return self._R_unfiltered


