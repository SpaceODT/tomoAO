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



import logging
logger = logging.getLogger(__name__)

class tomoReconstructor:
    """
    Linear MMSE tomographic reconstructor.

    Parameters
    ----------
    ao_sys : AOSystem
        AO system object containing telescope, atmosphere, guide stars, DM, and masks.
    weight_vector : ndarray, optional
        Slope weights per subaperture. Defaults to binary weights derived from valid
        subaperture masks.
    alpha : float, optional
        Regularisation scalar.
    model : str, optional
        Reconstruction model: ``'zonal'`` (default) or ``'modal'``.
    noise_covariance : float or ndarray, optional
        Measurement noise covariance — scalar (expanded to a diagonal matrix) or
        full matrix.
    optim_dir_weights : int, str, or array-like, optional
        Weights across optimisation directions. ``-1`` (default) builds one
        reconstructor per direction; ``'average'`` averages uniformly; a 1-D
        array sets explicit normalised weights.
    os : int, optional
        Oversampling factor {1, 2, 4} for the reconstructed phase grid (default 2).
    zernike_modes : ndarray, optional
        Zernike modes used for modal removal, sampled at the ``os`` resolution.
    coupling_matrix : ndarray, optional
        DM coupling matrix applied to actuator commands (identity by default).
    remove_tt_focus : bool, optional
        Project out tip/tilt/focus from the reconstructor (default ``False``).
    indexation : str, optional
        Slope vector ordering: ``'xxyy'`` (default) or ``'xyxy'``.
    order : str, optional
        Array ravelling order for permutation matrices: ``'C'`` (default) or ``'F'``.
    filter_subapertures : bool, optional
        Use the filtered subaperture mask as the working mask (default ``False``).
    """

    def __init__(
        self,
        ao_sys,
        weight_vector=None,
        alpha=None,
        model='zonal',
        noise_covariance=None,
        optim_dir_weights=-1,
        os=2,
        zernike_modes=None,
        coupling_matrix=None,
        remove_tt_focus=False,
        indexation='xxyy',
        order='C',
        filter_subapertures=False,
    ):
        self.tel = ao_sys.tel
        self.atm_model = ao_sys.atm

        # Guide stars
        lgs = ao_sys.lgsAst
        self.guide_stars = lgs if isinstance(lgs, list) else [lgs]
        self.n_guide_stars = len(self.guide_stars)

        # Optimisation (MMSE) stars
        mmse = ao_sys.mmseStar
        self.mmse_stars = mmse if isinstance(mmse, list) else [mmse]
        self.n_mmse_stars = len(self.mmse_stars) if self.mmse_stars is not None else 0

        self.dm = ao_sys.dm
        self._alpha = alpha
        self.os = os
        self.model = model
        self.zernike_modes = zernike_modes
        self.indexation = indexation
        self.order = order
        self.filter_subapertures = filter_subapertures
        self.remove_tt_focus = remove_tt_focus

        self.output_rec_grid = ao_sys.outputReconstructiongrid

        if self.filter_subapertures:
            self.unfiltered_subap_mask = ao_sys.filtered_subap_mask
            self._filtered_subap_mask = ao_sys.filtered_subap_mask
        else:
            self.unfiltered_subap_mask = ao_sys.unfiltered_subap_mask
            self._filtered_subap_mask = ao_sys.filtered_subap_mask

        self.list_filtered_subap_mask = ao_sys.list_filtered_subap_mask
        self.unfiltered_act_mask = ao_sys.unfiltered_act_mask

        # --- weights -------------------------------------------------------
        if weight_vector is None:
            self._init_default_weight_vector()
        else:
            self._weight_vector = np.asarray(weight_vector)

        # --- noise covariance ----------------------------------------------
        if noise_covariance is None:
            self._init_default_noise_covariance()
        else:
            self.noise_covariance = noise_covariance  # goes through setter

        # --- coupling matrix -----------------------------------------------
        n_valid_act = np.count_nonzero(self.unfiltered_act_mask)
        self.coupling_matrix = (
            np.eye(n_valid_act) if coupling_matrix is None else coupling_matrix
        )

        # --- permutation / filtering matrices ------------------------------
        n_slopes = np.count_nonzero(self.unfiltered_subap_mask) * 2 * self.n_guide_stars

        if self.order == 'C':
            self.signal_permutation_matrix = tools.get_signal_permutation_matrix(
                self.unfiltered_subap_mask, self.n_guide_stars
            )
            self.dm_permutation_matrix = tools.get_dm_permutation_matrix(
                self.unfiltered_act_mask
            )
        else:
            self.signal_permutation_matrix = np.eye(n_slopes)
            self.dm_permutation_matrix = np.eye(n_valid_act)

        if self.filter_subapertures:
            self.subap_filtering_matrix = (
                tools.get_different_filtered_subap_masks_filtering_matrix(
                    self.list_filtered_subap_mask, self.filtered_subap_mask
                )
            )
        else:
            self.subap_filtering_matrix = np.eye(n_slopes)

        if self.indexation == 'xyxy':
            self.xyxy_permutation_matrix = tools.get_xyxy_permutation_matrix(n_slopes)
        else:
            self.xyxy_permutation_matrix = np.eye(n_slopes)

        # --- optimisation direction weights --------------------------------
        self.optim_dir_weights = self._parse_optim_dir_weights(optim_dir_weights)

        # Scale r0 to guide-star wavelength
        wvl_scale = self.guide_stars[0].wavelength / self.atm_model.wavelength
        self.atm_model.r0 *= wvl_scale ** 1.2

        # --- fitting matrix ------------------------------------------------
        if self.dm is not None:
            i_fitting = 2 * self.dm.modes[self.output_rec_grid.flatten('F'), ]
            if cuda_available:
                self.fitting_matrix = cp.linalg.pinv(
                    cp.asarray(i_fitting), rcond=1e-3
                ).get()
            else:
                self.fitting_matrix = np.linalg.pinv(i_fitting, rcond=1e-3)
        else:
            self.fitting_matrix = None

        self._build_reconstructor()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val

    @property
    def filtered_subap_mask(self):
        return self._filtered_subap_mask

    @filtered_subap_mask.setter
    def filtered_subap_mask(self, val):
        self._filtered_subap_mask = val

    @property
    def noise_covariance(self):
        return self._noise_covariance

    @noise_covariance.setter
    def noise_covariance(self, val):
        """Accept a scalar, list, or ndarray. Scalars are expanded to a diagonal matrix."""
        if isinstance(val, (int, float)):
            if self.model == 'modal':
                n_mode = len(self.zernike_modes)
                val = val * np.eye(n_mode)
            else:
                val = val * np.eye(self.Gamma.shape[0])
        self._noise_covariance = val

    @property
    def weight_vector(self):
        return self._weight_vector

    @weight_vector.setter
    def weight_vector(self, val):
        self._weight_vector = np.asarray(val)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_default_weight_vector(self):
        weights = []
        for i in range(self.n_guide_stars):
            mask_i = self.list_filtered_subap_mask[i]
            valid = mask_i.T[self.filtered_subap_mask.T]
            valid = np.tile(valid, 2)
            weights.append(valid.astype(float))
        self.list_weight_vector = weights
        self._weight_vector = np.concatenate(weights)

    def _init_default_noise_covariance(self):
        self._noise_covariance = (
            1e-3 * self.alpha
            * np.diag(1.0 / (self.weight_vector.flatten('F') + 1e-8))
        )

    def _parse_optim_dir_weights(self, val):
        if isinstance(val, str):
            if val.lower() in ('avr', 'average'):
                return np.ones(self.n_mmse_stars) / self.n_mmse_stars
            raise ValueError(f"Unrecognised optimisation weight keyword: '{val}'")
        if val == -1:
            return -1
        weights = np.asarray(val, dtype=float)
        if len(weights) != self.n_mmse_stars:
            raise ValueError(
                f"Length of optim_dir_weights ({len(weights)}) does not match "
                f"n_mmse_stars ({self.n_mmse_stars})"
            )
        return weights / weights.sum()

    def _build_per_lgs_validity_matrix(self):
        """
        Block-diagonal matrix in common-mask slope space that zeros out slopes
        not valid for each individual LGS.

        Shape: ``(2 * n_lgs * n_common, 2 * n_lgs * n_common)``.
        Reduces to identity when every per-LGS mask equals the common mask.
        """
        from scipy.linalg import block_diag as scipy_block_diag

        common_flat = self.filtered_subap_mask.flatten('F')
        blocks = []
        for mask_i in self.list_filtered_subap_mask:
            # which positions in the common mask are also valid for this LGS
            valid_in_common = mask_i.flatten('F')[common_flat]
            validity = np.tile(valid_in_common.astype(float), 2)  # x and y slopes
            blocks.append(np.diag(validity))
        return scipy_block_diag(*blocks)

    def _build_covariance_matrices(self):
        """Compute slope-space auto- and cross-covariance matrices."""
        fn = (
            tools.spatioAngularCovarianceMatrix_gpu
            if cuda_available
            else tools.spatioAngularCovarianceMatrix
        )


        self.Cox = [
            fn(self.tel, self.atm_model, [star], self.guide_stars,
               self.filtered_subap_mask, self.os)
            for star in self.mmse_stars
        ]
        self.Cxx = fn(
            self.tel, self.atm_model, self.guide_stars, self.guide_stars,
            self.filtered_subap_mask, self.os
        )

    def _build_gamma(self):
        """Build block-diagonal slope-to-phase gradient matrix."""
        gamma_blocks = []
        self.grid_masks = []
        for _ in range(self.n_guide_stars):
            G, grid_mask = tools.sparseGradientMatrixAmplitudeWeighted(
                self.filtered_subap_mask, amplMask=None, os=self.os
            )
            gamma_blocks.append(np.asarray(G))
            self.grid_masks.append(grid_mask)
        self.Gamma = block_diag(gamma_blocks).todense()

    def _build_reconstructor(self):
        logger.info(
            "Building reconstructor using %s",
            "GPU (CUDA)" if cuda_available else "CPU",
        )

        self._build_gamma()
        self._build_covariance_matrices()

        xp = cp if cuda_available else np
        pinv = cp.linalg.pinv if cuda_available else np.linalg.pinv

        Cox = xp.asarray(self.Cox)   # (n_mmse, n_rec, n_slopes)
        Cxx = xp.asarray(self.Cxx)
        N   = xp.asarray(self.noise_covariance)
        G   = xp.asarray(self.Gamma)

        GCG_N = G @ Cxx @ G.T + N

        if self.optim_dir_weights == -1:
            GT_inv = G.T @ pinv(GCG_N)
            self.rec_per_dir = [
                _to_numpy(Cox[k] @ GT_inv) for k in range(self.n_mmse_stars)
            ]
        else:
            w = xp.asarray(self.optim_dir_weights)
            Cox_avr = xp.stack(
                [Cox[k] * w[k] for k in range(self.n_mmse_stars)]
            ).sum(axis=0)
            self.rec_per_dir = [_to_numpy(Cox_avr @ G.T @ pinv(GCG_N))]

        # Keep numpy copies of covariance matrices for inspection
        self.Cox  = _to_numpy(Cox)
        self.Cxx  = _to_numpy(Cxx)
        self.noise_covariance = _to_numpy(N)
        self.Gamma = np.asarray(_to_numpy(G))

        # Filtering matrix (unfiltered → filtered subapertures)
        self.filtering_matrix = tools.get_filtering_matrix(
            self.unfiltered_subap_mask.copy(),
            self.filtered_subap_mask.copy(),
            self.n_guide_stars,
        )

        # Per-LGS validity in common-mask slope space.
        # For filter_subapertures=False this zeros out slopes that are outside
        # each LGS's individual mask after the unfiltered→common mapping.
        # For filter_subapertures=True the subap_filtering_matrix already
        # handles per-LGS masking, so per_lgs_validity is not inserted.
        if not self.filter_subapertures:
            per_lgs_validity = self._build_per_lgs_validity_matrix()

        # Optional tip/tilt/focus removal
        if self.remove_tt_focus:
            self.act_ptt_rem_mat, self.slopes_tt_rem_mat, self.slope_ttf_proj = (
                tools.modalRemovalMatrices(
                    self.weight_vector,
                    self.unfiltered_act_mask,
                    self.filtered_subap_mask,
                    self.dm.nValidAct,
                    self.n_guide_stars,
                )
            )
            if self.filter_subapertures:
                chain = (
                    self.dm_permutation_matrix
                    @ self.act_ptt_rem_mat
                    @ self.coupling_matrix
                    @ self.fitting_matrix
                    @ self.rec_per_dir[0]
                    @ self.slopes_tt_rem_mat
                    @ self.filtering_matrix
                    @ self.signal_permutation_matrix
                    @ self.xyxy_permutation_matrix
                    @ self.subap_filtering_matrix
                )
            else:
                chain = (
                    self.dm_permutation_matrix
                    @ self.act_ptt_rem_mat
                    @ self.coupling_matrix
                    @ self.fitting_matrix
                    @ self.rec_per_dir[0]
                    @ self.slopes_tt_rem_mat
                    @ per_lgs_validity
                    @ self.filtering_matrix
                    @ self.signal_permutation_matrix
                    @ self.xyxy_permutation_matrix
                )
        else:
            if self.filter_subapertures:
                chain = (
                    self.dm_permutation_matrix
                    @ self.coupling_matrix
                    @ self.fitting_matrix
                    @ self.rec_per_dir[0]
                    @ self.filtering_matrix
                    @ self.signal_permutation_matrix
                    @ self.xyxy_permutation_matrix
                    @ self.subap_filtering_matrix
                )
            else:
                chain = (
                    self.dm_permutation_matrix
                    @ self.coupling_matrix
                    @ self.fitting_matrix
                    @ self.rec_per_dir[0]
                    @ per_lgs_validity
                    @ self.filtering_matrix
                    @ self.signal_permutation_matrix
                    @ self.xyxy_permutation_matrix
                )

        self.reconstructor = np.array(chain) * self.guide_stars[0].wavelength


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """Return a NumPy array regardless of whether *arr* is NumPy or CuPy."""
    return arr.get() if cuda_available and isinstance(arr, cp.ndarray) else np.asarray(arr)



