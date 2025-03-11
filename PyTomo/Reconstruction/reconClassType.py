"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""

# import os
# os.chdir('/Users/ccorreia/git/rtcu-kapa/')
# print(os.getcwd())
#from aoreconstructor.lib.reconSupportFunctions import *
#from aoreconstructor.lib.tools import *

# from PyAO.lib.shwfs_cmds import *
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

# import ktl

# Libraries needed for logging
# import coloredlogs
import logging

# Library needed to interact with configuration files.

# import ktl
#from aoscripts.mock import read_write as ktl  # replacement for a fake read/write function

log = logging.getLogger('')

# import coloredlogs

# Section: Set up the log.

# Set this flag to always see debug logs
DEBUG = False
# Set up the base logger all threads will use, once we know the debug flag

# coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
# coloredlogs.DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
# # Adjust color of the log depending on the DEBUG variable.
# if DEBUG:
#     coloredlogs.install(level='DEBUG')
# else:
#     coloredlogs.install(level='INFO')
#

log = logging.getLogger('')

#
# def safeReadKeyword(service, keyword, default=None):
#     try:
#         r = ktl.read(service, keyword)
#         return r
#     except Exception as e:
#         # Prepare error message
#         errmsg = f"Exception raised while reading '{service} {keyword}' keyword.\n"
#         # Log error
#         log.error(errmsg + str(e))
#         # Assigned default value
#         log.info(f"Assigned the default value= '{default}' passed as input.\n")
#         # since default=None, a user can check outside the function whether any value has been read out
#         return default
#
#
# def safeWriteKeyword(service, keyword, value=None):
#     try:
#         status = ktl.write(service, keyword, value)
#         return status
#     except Exception as e:
#         # Prepare error message
#         errmsg = f"Exception raised while writing '{service} {keyword}' keyword.\n"
#         # Log error
#         log.error(errmsg + str(e))
#         # Assigned default value
#         log.info('Value not written')
#         return value





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

        # # Define the input arguments using argparse-like approach
        # parser = argparse.ArgumentParser(description='LinearMMSE constructor')
        # parser.add_argument('sampling', type=float, help='Sampling')
        # parser.add_argument('diameter', type=float, help='Diameter')
        # parser.add_argument('atmModel', type=Atmosphere, help='Atmosphere Model')
        # parser.add_argument('guideStar', type=Source, help='Guide Star')
        # parser.add_argument('--mmseStar', default=None, type=aoSystem.Source, help='MMSE Star')
        # parser.add_argument('--telescope', default=None, type=aoSystem.telescope, help='Telescope')
        # parser.add_argument('--pupil', default=True, type=bool, help='Pupil')
        # parser.add_argument('--unit', default=None, type=float, help='Unit')
        # parser.add_argument('--model', default='zonal', type=str, help='Model')
        # parser.add_argument('--zernikeMode', default=None, type=int, help='Zernike Mode')
        # parser.add_argument('--noiseCovariance', default=0, type=float, help='Noise Covariance')
        # parser.add_argument('--tilts', default='Z', type=str, help='Tilts')
        # parser.add_argument('--lag', default=0, type=int, help='Lag')
        # parser.add_argument('--xyOutput', default=None, type=int, help='xyOutput')
        # parser.add_argument('--G', default=0, type=float, help='G')
        # parser.add_argument('--P', default=1, type=float, help='P')
        # parser.add_argument('--weightOptimDir', default=-1, type=float, help='Weight Optim Dir')

        # Parse the input arguments
        # args = parser.parse_args()

        # if mode = "compressed":
        #     self.all_valid_subap_mask = Y
        #     self.actual_valid_subap_mask = Y
        # elif mode = "constant":
        #     self.all_valid_subap_mask = X
        #     self.actual_valid_subap_mask = Y



        #self.tel = tel
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

        #self.atmModel = atmModel
        #if isinstance(guideStar, list):
        # It's already a list, no need to convert
        #    self.guideStar = guideStar
        #else:
        # Convert to a list
        #    self.guideStar = [guideStar]

        #if isinstance(mmseStar, list):
        # It's already a list, no need to convert
        #    self.mmseStar = mmseStar
        #else:
        # Convert to a list
        #    self.mmseStar = [mmseStar]

        #self.dm = dm
        self.dm = aoSys.dm

        #self.outputRecGrid = outputRecGrid
        self.outputRecGrid = aoSys.outputReconstructiongrid

        #self.unfilteredSubapMask = validSubapMask
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
                Gamma, gridMask = sparseGradientMatrixAmplitudeWeighted(validSubapMask, amplMask=None, os=self.os)

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

                # block of code extracting x-y-x-y
                # N = np.count_nonzero(self.unfilteredSubapMask)
                # idx = np.arange(N * 2).reshape(2, N)
                # idx = idx.transpose().reshape(-1)
                # # Reorder the rows of the matrix
                # Gamma = Gamma.tocsr()[idx, :]

                # GammaReorder = Gamma.toarray().copy()
                # GammaReorder[::2,:] = Gamma.toarray()[1::2,:]
                # GammaReorder[1::2, :] = Gamma.toarray()[0::2, :]
                # Gamma = GammaReorder
            # Gamma = Gamma_tmp.todense()
            # breakpoint()

            Gamma = [Gamma] * self.nGuideStar  # Replicate Gamma nMmseStar times
            gridMask = [gridMask] * self.nGuideStar
            # Use np.block() to create a block diagonal matrix
            # blk_diag_Gamma = np.block(
            #    [[matrix if i == j else np.zeros_like(matrix) for j in range(4)] for i, matrix in enumerate(Gamma)])
            # blk_diag_Gamma = block_diag(Gamma)
        # TODO: the scaling of the discrete gradient matrix needs to be worked out

        # IMmodel = blk_diag_Gamma@self.dm.modes[self.outputRecGrid.flatten(),]
        # IMmodel = Gamma[0]@self.dm.modes[self.outputRecGrid.flatten(),]
        # breakpoint()
        # scaling = np.max(self.intMat[0], axis=1) / np.max(IMmodel, axis=1)
        # self.scaling = scaling
        # self.Gamma = np.diag(scaling) @ blk_diag_Gamma
        # self.Gamma_one = np.diag(scaling) @ Gamma[0]
        # self.Gamma_one = Gamma[0]
        # breakpoint()
        # self.Gamma = block_diag([self.Gamma_one]*self.nGuideStar)
        Gamma = [np.asarray(mat) for mat in Gamma]
        self.Gamma = block_diag(Gamma)
        # breakpoint()
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

        # %%
        #self.R = assembleRecon(self, self.runtime_valid_subap_mask, self.dm.validAct, minioning_flag=minioning_flag)

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
        # breakpoint()
        if self._noiseCovariance is None:
            return 1e-3 * self.alpha * np.diag(1 / (self.weight_vector.flatten(order='F') + 1e-8))

        #TODO: Como seria esta função?
        else:
            return self._noiseCovariance


    # %%
    def solveLmmse(self):
        print(" -->> mmse solver!")
        # RecStatSA_py = Cox@GammaBlkDiag.T@np.linalg.pinv(GammaBlkDiag@Cxx@GammaBlkDiag.T+Cn);

        m_Cox = self.Cox
        m_Cxx = self.Cxx

        m_Cn = self.getNoiseCovariance()

        # m_Cn = self._noiseCovariance

        # m_Cn = self.getNoiseCovariance()

        #import pdb
        #pdb.set_trace()
        # breakpoint()

        #TODO:
        # m_Gamma = np.diag(np.array(self.weight_vector).flatten())@self.Gamma

        # Cn = 1e-3 * alpha * np.diag(1 / (weight.flatten(order='F') + 1e-8))
        #
        # IM = block_diag(*IM)
        # # return np.linalg.pinv(IM.T @ IM + 1e-3*alpha*np.linalg.pinv(Cxx) + 1) @ IM.T
        # rec = Cox @ IM.T @ np.linalg.pinv(IM @ Cxx @ IM.T + Cn)
        #
        m_Gamma = self.Gamma
        # breakpoint()

        if cuda_available:
            m_Cox = cp.asarray(m_Cox)
            m_Cxx = cp.asarray(m_Cxx)
            m_Cn = cp.asarray(m_Cn)
            m_Gamma = cp.asarray(m_Gamma.todense())

        if self.weightOptimDir == -1:
            m_mmseReconstructor = [None] * self.nMmseStar
            for k in range(self.nMmseStar):
                # breakpoint()

                if cuda_available:
                    m_mmseReconstructor[k] = (m_Cox[k] @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)).get()
                else:
                    m_mmseReconstructor[k] = m_Cox[k] @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)

                # m_mmseReconstructor[k] = m_Cox[k] @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)

        else:  # weighted sum over all the optimisation directions
            # This code uses the ss stars to compute a weighted average tomographic
            # reconstructor over all of them
            m_mmseReconstructor = [None]
            m_CoxWAvr = np.sum([m_Cox[k] * self.weightOptimDir[k] for k in range(self.nMmseStar)], axis=0)
            m_mmseReconstructor[0] = m_CoxWAvr @ m_Gamma.T @ np.linalg.pinv(m_Gamma @ m_Cxx @ m_Gamma.T + m_Cn)
        self.mmseReconstructor = m_mmseReconstructor

    # def rec_assembler(self, aoSys):
    #     actuator_ttp_removal, act_minion_matrix, slope_tt_removal, pinv_slope_rem_tt, pinv_slope_rem_focus = \
    #         tools.modal_removal(aoSys.subap_mask, self.unfilteredSubapMask, aoSys.dm.modes, noise_weight_scaling=1)
    #
    #     R = self.R_unfiltered() # computes the unfiltered reconstructor
    #     R_assembly = tools.assemble_reconstructor(R, actuator_ttp_removal, act_minion_matrix,
    #                                               slope_tt_removal, pinv_slope_rem_tt,
    #                                               pinv_slope_rem_focus, len(aoSys.lgsAst))
    #     return R_assembly

    def mtimes(self, data):
        return self.mmseReconstructor @ data

    def update(self, runtime_valid_subap_mask, runtime_valid_act_mask=None, update_matrices=None):



        # self.weight_vector = weight_vector



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

        # TODO the modal_removal is not yet compatible with the index vectors passed on as input to the update function

        #self.R = assembleRecon(self, runtime_valid_subap_mask, runtime_valid_act_mask[ida])

    @property
    def R_unfiltered(self):
        return self._R_unfiltered


# TODO This class shall be renamed to WLSR for weighted least-squares reconstructor
# class WLSR:
#     def __init__(self, aoSys, intMat, static_maps_matrices, runtime_valid_subap_mask=None, runtime_valid_act_mask=None, noiseCovariance=1, nTruncModes=0, minioning_flag='ON', alpha=None, weight_vector=None, algorithm='bayesian'):
#         """Initialization function.
#
#         Instantiate a glao reconstructor
#
#         :param:
#         :param ordering: keck(default)|sim Whereas to interleave the x-y slopes (keck ordering) or to leave allX-then-allY for simulations
#
#         :return: This class reads the interaction matrix and generates a MMSE-like reconstructor
#         """
#         self.lsReconstructor = None
#
#         self.dm = aoSys.dm
#         self.guideStar = aoSys.lgsAst
#         self.unfilteredSubapMask = aoSys.subap_mask
#
#         # if not set by user, then the runtime_valid_subap_mask is the global one for this WFS
#         if runtime_valid_subap_mask is None:
#             self.runtime_valid_subap_mask = np.expand_dims(aoSys.subap_mask, axis=-1)  # at first use unfiltered mask
#         else:
#             self.runtime_valid_subap_mask = runtime_valid_subap_mask
#
#         # if not set by user, then the runtime_valid_act_mask is the global one for this DM
#         if runtime_valid_act_mask is None:
#             self.runtime_valid_act_mask = np.array([True] * len(self.dm.validAct.flatten())).astype(bool)
#         else:
#             self.runtime_valid_act_mask = runtime_valid_act_mask
#
#         self.nTruncModes = nTruncModes
#         self.alpha = alpha
#         self.weight_vector = weight_vector
#         self.static_maps_matrices = static_maps_matrices
#         self.inv_cov_mat = static_maps_matrices.invcov
#         self.algorithm = algorithm
#
#         if intMat is not None:
#             self.intMat = intMat
#
#         # the original Imat remains unfiltered as a property of the class. A runtime copy is created for processing
#         self.runtime_intMat = self.intMat.copy()
#         #import pdb
#         #pdb.set_trace()
#         if np.isscalar(noiseCovariance):
#             self.noiseCovariance = np.eye(self.intMat.shape[0])
#         else:
#             self.noiseCovariance = noiseCovariance
#
#         # The fitting matrix is an identity with dim=all_valid_actuators for a given DM
#         self.fittingMatrix = np.eye(len(self.dm.validAct))
#
#         # call the noise-weighted least-squares reconstructor
#         # self.solveWeightedLeastSquares(noiseCovariance)
#
#         # reconstructor matrix assembly
#         # self.R_filtered = getRecFiltered(self, self.unfilteredSubapMask, self.dm.validAct)
#
#         #self.update(self.runtime_valid_subap_mask, self.runtime_valid_act_mask)
#
#         # call the noise-weighted least-squares reconstructor
#         self.solveWeightedLeastSquares()
#
#         self.R = assembleRecon(self, self.runtime_valid_subap_mask, self.dm.validAct, minioning_flag=minioning_flag)
#     def solveWeightedLeastSquares(self):
#
#         if self.algorithm is None or self.algorithm.lower() == 'bayesian':
#             # R = (H' W  H + \alpha \Sigma_phi^-1)^\dag H'W
#             #self.R_weightedLeastSquaresReconstructor = \
#             #    reconstructBayesian(self.runtime_intMat, self.weight_vector, self.inv_cov_mat, self.alpha)
#
#             self.R_weightedLeastSquaresReconstructor = \
#                 averaging_bayesian_reconstructor_dm_space(self.intMat, self.weight_vector, self.inv_cov_mat, self.alpha)
#             #self.R_weightedLeastSquaresReconstructor = np.linalg.inv((self.runtime_intMat.T * self.weight_matrix.T) @ self.runtime_intMat
#             #                                                         + self.alpha * 1e-3 * self.inv_cov_mat + np.eye(   self.runtime_intMat.shape[1])) @ (self.runtime_intMat.T * self.weight_matrix.T)  # the one is there to penalize piston
#
#         elif self.algorithm.lower() == 'svd':
#             # R = tsvd(H' \Sigma_noise  H ) H' \Sigma_noise
#             # self.R_weightedLeastSquaresReconstructor = tsvd(
#             #     self.runtime_intMat.T @ self.runtime_noiseCovarianceMatrix @ self.runtime_intMat,
#             #     self.nTruncModes) @ self.runtime_intMat.T @ self.runtime_noiseCovarianceMatrix
#             self.R_weightedLeastSquaresReconstructor = tsvd(
#                 self.intMat.T @ self.noiseCovariance @ self.intMat,
#                 self.nTruncModes) @ self.intMat.T @ self.noiseCovariance
#         else:
#             # R = (H' \Sigma_noise  H)^\dag H' \Sigma_noise
#             self.R_weightedLeastSquaresReconstructor = np.linalg.pinv(
#                 self.runtime_intMat.T @ self.runtime_noiseCovarianceMatrix @ self.runtime_intMat) @ self.runtime_intMat.T @ self.runtime_noiseCovarianceMatrix
#
#
#     #@property
#     #def R_unfiltered(self):
#     #    return self._R_unfiltered
#     @property
#     def R_unfiltered(self):
#         return self.fittingMatrix @ self.R_weightedLeastSquaresReconstructor
#
#     def __mul__(self, data):
#         return self.R @ data
#
#     def update(self, runtime_valid_subap_mask, runtime_valid_act_mask, minioning_flag='ON'):
#
#         self.runtime_valid_subap_mask = check_subap_mask_dims(runtime_valid_subap_mask, len(self.guideStar))
#
#         runtime_valid_subap_mask = toggle(self.runtime_valid_subap_mask)
#         nSubap = self.unfilteredSubapMask.shape[0]
#
#         ids = []
#         maxValidSubap = np.count_nonzero(self.unfilteredSubapMask)
#         for k in range(len(self.guideStar)):
#             maskS = runtime_valid_subap_mask[:, k][self.unfilteredSubapMask.flatten()]
#             res = np.where(maskS)[0]
#             ids.append(np.concatenate((res, res + maxValidSubap)) + k * 2 * maxValidSubap)
#
#         # convert lists to arrays
#         ids = np.concatenate(ids)
#
#         # valid DM actuators
#         maskA = runtime_valid_act_mask[self.dm.validAct.reshape(-1, 1)]
#         ida = np.where(maskA)[0]
#
#         print(" -->> WLSR updator!")
#
#         self.runtime_noiseCovarianceMatrix = self.noiseCovariance.copy()
#
#         self.runtime_noiseCovarianceMatrix = self.runtime_noiseCovarianceMatrix[ids][:, ids]
#         self.runtime_intMat = self.intMat[ids][:, :]
#
#         # call the noise-weighted least-squares reconstructor
#         self.solveWeightedLeastSquares()
#
#         # reconstructor matrix assembly
#         runtime_valid_subap_mask = tools.toggle(runtime_valid_subap_mask)
#
#         self.R = assembleRecon(self, runtime_valid_subap_mask, self.dm.validAct, minioning_flag=minioning_flag)


if __name__ == '__main__':
    pass