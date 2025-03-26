import tomoAO.tools.tomography_tools as tools
from tomoAO.Reconstruction.reconClassType import tomoReconstructor
from tomoAO.IO import load_from_ini
from tomoAO.Simulation import AOSystem

from scipy.linalg import block_diag

import numpy as np
cuda_available = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False


def tomographic_reconstructor_phase_space(config_file, config_dir, alpha, weight_vector=None, noise_covariance=None, debug=False):

    config_vars = load_from_ini(config_file, ao_mode="MLAO", config_dir=config_dir)

    aoSys = AOSystem(config_vars)

    rec = tomoReconstructor(aoSys, alpha=alpha, weight_vector=weight_vector, noise_covariance=noise_covariance, os=2)

    if debug:
        return rec

    return rec._R_unfiltered



def tomographic_reconstructor_dm_space(aoSys, IM, weight, alpha, Cn=None):

    # %% LGS-to-LGS cross-covariance matrix
    Cxx = tools.spatioAngularCovarianceMatrix(aoSys.tel, aoSys.atm, aoSys.lgsAst, aoSys.lgsAst,
                                              aoSys.act_mask, os=1, dm_space=True)
    # %% LGS-to-science cross-covariance matrix
    Cox = []
    for i in range(len([aoSys.mmseStar])):
        res = tools.spatioAngularCovarianceMatrix(aoSys.tel, aoSys.atm, [[aoSys.mmseStar][i]],
                                                  aoSys.lgsAst,
                                                  aoSys.act_mask, os=1, dm_space=True)
        Cox.append(res)

    Cox = Cox[0]

    # TODO this scaling was chosen roughly to match the original invcov matrix read from file
    Cxx = Cxx / 36
    Cox = Cox / 36

    if Cn is None:
        Cn = 1e-3 * alpha * np.diag(1 / (weight.flatten(order='F') + 1e-8))

    IM = block_diag(*IM)

    if cuda_available:
        Cox = cp.asarray(Cox)
        IM = cp.asarray(IM)
        Cxx = cp.asarray(Cxx)
        Cn = cp.asarray(Cn)

        return (Cox @ IM.T @ np.linalg.pinv(IM @ Cxx @ IM.T + Cn)).get()

    return Cox @ IM.T @ np.linalg.pinv(IM @ Cxx @ IM.T + Cn)




def averaging_bayesian_reconstructor_dm_space(IM, weight_vector, inv_cov_mat, alpha):
    """
    This function computes the Bayesian reconstructor as:

    :math:`R = {(H^{T} W^{T} + A)}^{-1} H^{T} W^{T}`

    Args:
        IM:
        weight_vector:
        inv_cov_mat:
        median_intensity:

    Returns:

    """

    n_lgs = weight_vector.shape[1]

    IMw = IM.copy()
    n_act = IM.shape[-1]

    for k in range(n_lgs):
        W = np.tile(np.reshape(weight_vector[:, k], (-1, 1)), (1, n_act))
        IMw[k, :, :] = IMw[k, :, :] * W

    IMw = block_diag(*IMw)
    IM = block_diag(*IM)

    # perform the regularized inversion
    return 1 / n_lgs * np.hstack([np.eye(n_act)] * n_lgs) @ np.linalg.inv(
        IMw.T @ IM + alpha * 1e-3 * block_diag(*[inv_cov_mat]*n_lgs) + 1.) @ (
        IMw.T)  # the one is there to penalize piston













