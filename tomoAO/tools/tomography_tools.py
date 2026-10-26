#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""

import numpy as np
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False

from scipy.sparse import csr_matrix
from scipy.special import kv, gamma
from scipy.linalg import block_diag

import math

from time import time


from math import gamma, pow, sin, pi
from numba import njit, prange

import aotools


bessel_i_kernel_code = r'''
extern "C" __global__
void bessel_i2_kernel(const double* x, double* result, const double n, const int terms, const int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    double xi = x[idx];
    double sum_result = 0.0;
    double coeff = pow(xi / 2.0, n) / tgamma(n + 1);  // Coefficient term

    double term = coeff;
    sum_result += term;

    double x_sq_half = pow(xi / 2.0, 2);  // (x/2)^2 term for recurrence

    // Loop over the terms in the series
    for (int m = 1; m < terms; m++) {
        term *= x_sq_half / (m * (m + n));  // Recurrence relation
        sum_result += term;
    }

    result[idx] = sum_result;
}
'''
if cuda_available:
    bessel_i_kernel = cp.RawKernel(bessel_i_kernel_code, 'bessel_i2_kernel')

def bessel_i(n, x, terms=10):
    x = cp.asarray(x, dtype=cp.float64)
    result = cp.zeros_like(x, dtype=cp.float64)

    threads_per_block = 256
    blocks_per_grid = (x.size + threads_per_block - 1) // threads_per_block

    bessel_i_kernel((blocks_per_grid,), (threads_per_block,), (x, result, n, terms, x.size))

    return result


def bessel_k(n, x, terms=10):
    x = cp.asarray(x, dtype=cp.float64)

    i_n = bessel_i(n, x, terms)
    i_neg_n = bessel_i(-n, x, terms)

    return (cp.pi / 2) * (i_neg_n - i_n) / cp.sin(n * cp.pi)

if cuda_available:
    first_part_cst = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) / (2 ** (5 / 6) * cp.pi ** (8 / 3)))
    first_part_out = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) * gamma(5 / 6) / (2 * cp.pi ** (8 / 3)))
else:
    first_part_cst = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) / (2 ** (5 / 6) * np.pi ** (8 / 3)))
    first_part_out = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) * gamma(5 / 6) / (2 * np.pi ** (8 / 3)))


@njit(parallel=True)
def bessel_i_cpu(x, n, terms):
    size = x.shape[0]
    result = np.zeros(size, dtype=np.float64)

    for idx in prange(size):
        xi = x[idx]
        coeff = pow(xi / 2.0, n) / gamma(n + 1.0)
        term = coeff
        sum_result = term
        x_sq_half = pow(xi / 2.0, 2)

        for m in range(1, terms):
            term *= x_sq_half / (m * (m + n))
            sum_result += term

        result[idx] = sum_result

    return result


def bessel_k_cpu(n, x, terms=20):
    x = np.asarray(x, dtype=np.float64)
    i_n = bessel_i_cpu(x, n, terms)
    i_neg_n = bessel_i_cpu(x, -n, terms)

    with np.errstate(divide='ignore', invalid='ignore'):
        k = (pi / 2.0) * (i_neg_n - i_n) / np.sin(n * pi)
        # Avoid division by zero for integer n
        k[np.isclose(np.sin(n * pi), 0.0)] = np.nan
    return k





def covariance_matrix(rho, r0, L0):

    L0r0ratio = (L0 / r0) ** (5 / 3)
    cst = first_part_cst * L0r0ratio
    index = rho != 0

    if cuda_available:
        out = cp.ones(rho.shape, dtype=rho.dtype) * first_part_out * L0r0ratio
        u = 2 * cp.pi * rho[index] / L0

    else:
        out = np.ones(rho.shape, dtype=rho.dtype) * (24 * gamma(6 / 5) / 5) ** (5 / 6) * \
              (gamma(11 / 6) * gamma(5 / 6) /
               (2 * np.pi ** (8 / 3))) * L0r0ratio
        u = 2 * np.pi * rho[index] / L0

    if cuda_available:
        u = cp.asarray(u)
        out[index] = cst * u ** (5 / 6) * bessel_k(5 / 6, u, terms=10)
        out = out
    else:
        out[index] = cst * u ** (5 / 6) * bessel_k_cpu(5/6, u, 10)


    return out


def spatioAngularCovarianceMatrix(tel, atm, src1, src2, mask, os, dm_space=0):
    # Compute the discrete reconstruction mask from the subap_mask passed on as input

    recIdx = reconstructionGrid(mask, os, dm_space)
    if src1 == src2:  # auto-covariance matrix

        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                if j >= i:
                    phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                    for l in range(0, len(atm.altitude)):
                        # CROSS COVARIANCE BETWEEN TWO STARS
                        rho1 = sourceGridCoordinates(src1[i], atm.altitude[l], nPts, tel.D)
                        rho2 = sourceGridCoordinates(src2[j], atm.altitude[l], nPts, tel.D)

                        dist = dists(rho1.T, rho2.T)

                        phaseCovElem[:, :, l] = covariance_matrix(
                            dist, atm.r0, atm.L0) * atm.fractionalR0[l]

                    crossCovCell[i, j] = np.sum(phaseCovElem, axis=2)[
                                         recIdx.flatten("F"), :][:, recIdx.flatten("F")]


        # populate the off-diagonal blocks left null in the previous step
        for i in range(0, len(src1)):
            for j in range(0, len(src1)):
                if j < i:
                    crossCovCell[i, j] = np.transpose(crossCovCell[j, i])

        crossCovMat = np.block(
            [[crossCovCell[i, j] for j in range(len(src1))] for i in range(len(src2))])
        crossCovMat = np.vstack(crossCovMat)
        return crossCovMat
    else:

        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):

                phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                for l in range(0, len(atm.altitude)):
                    # CROSS COVARIANCE BETWEEN TWO STARS
                    rho1 = sourceGridCoordinates(src1[i], atm.altitude[l], nPts, tel.D)
                    rho2 = sourceGridCoordinates(src2[j], atm.altitude[l], nPts, tel.D)

                    dist = dists(rho1.T, rho2.T)

                    phaseCovElem[:, :, l] = covariance_matrix(
                        dist, atm.r0, atm.L0) * atm.fractionalR0[l]

                crossCovCell[i, j] = np.sum(phaseCovElem, axis=2)[
                                     recIdx.flatten("F"), :][:, recIdx.flatten("F")]


        crossCovMat = np.block(
            [[crossCovCell[i, j] for j in range(len(src2))] for i in range(len(src1))])
        if len(src1) > 1:
            crossCovMat = np.vstack(crossCovMat)
        return crossCovMat




arcsec2radian = np.pi / 180 / 3600
def spatioAngularCovarianceMatrix_gpu(tel, atm, src1, src2, mask, os, dm_space=0):

    recIdx = reconstructionGrid(mask, os, dm_space)
    if src1 == src2:

        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                if j >= i:

                    phaseCovElem = cp.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                    for l in range(0, len(atm.altitude)):
                        # CROSS COVARIANCE BETWEEN TWO STARS
                        rho1 = sourceGridCoordinates(src1[i], atm.altitude[l], nPts, tel.D)
                        rho2 = sourceGridCoordinates(src2[j], atm.altitude[l], nPts, tel.D)

                        dist = dists(rho1.T, rho2.T)  # not sure why, using the transpose makes it equal to Matlab's

                        phaseCovElem[:, :, l] = covariance_matrix(
                            dist, atm.r0, atm.L0) * atm.fractionalR0[l]


                    crossCovCell[i, j] = (cp.sum(phaseCovElem, axis=2)[
                                         recIdx.flatten("F"), :][:, recIdx.flatten("F")]).get()

        # populate the off-diagonal blocks left null in the previous step
        for i in range(0, len(src1)):
            for j in range(0, len(src1)):
                if j < i:
                    crossCovCell[i, j] = np.transpose(crossCovCell[j, i])

        crossCovMat = np.block(
            [[crossCovCell[i, j] for j in range(len(src1))] for i in range(len(src2))])
        crossCovMat = np.vstack(crossCovMat)


        return crossCovMat
    else:

        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                phaseCovElem = cp.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                for l in range(0, len(atm.altitude)):
                    # CROSS COVARIANCE BETWEEN TWO STARS
                    rho1 = sourceGridCoordinates(src1[i], atm.altitude[l], nPts, tel.D)
                    rho2 = sourceGridCoordinates(src2[j], atm.altitude[l], nPts, tel.D)

                    dist = dists(rho1.T, rho2.T)  # not sure why, using the transpose makes it equal to Matlab's


                    unique_values, inverse_indices = cp.unique(dist, return_inverse=True)  # Extract unique values
                    transformed_values = covariance_matrix(unique_values, atm.r0, atm.L0) * atm.fractionalR0[l]
                    phaseCovElem[:, :, l] = transformed_values[inverse_indices].reshape(dist.shape)  # Reconstruct matrix

                crossCovCell[i, j] = (cp.sum(phaseCovElem, axis=2)[
                                      recIdx.flatten("F"), :][:, recIdx.flatten("F")]).get()

        crossCovMat = crossCovCell.squeeze(axis=0)
        crossCovMat = cp.hstack(crossCovMat)

        if len(src1) > 1:
            crossCovMat = cp.vstack(crossCovMat)

        return crossCovMat


def sourceGridCoordinates(src, altitude, nPts, D):
    """
    Coordinates at a layer of the nPts x nPts phase nodes seen in the direction of src.

    The node q of the WFS looking at src sees the pupil point p = R(-rotation) (q + offset),
    with src.offset_x, src.offset_y in m and src.rotation in rad (0 if absent; see
    fieldTransformToSourceOffset for the conversion from an OOPAO FieldTransformer).
    p is then projected along the line of sight onto the layer,
        p (1 - altitude / src.altitude) + altitude * direction,
    so for a LGS the offset is compressed by the cone like the rest of the pupil.

    Returns
    -------
    rho : ndarray (nPts, nPts), complex
        x + 1j * y in m, same layout as meshgrid.
    """
    X, Y = meshgrid(nPts, D, offset_x=getattr(src, 'offset_x', 0) or 0,
                    offset_y=getattr(src, 'offset_y', 0) or 0)
    theta = getattr(src, 'rotation', 0) or 0
    if theta != 0:
        X, Y = np.cos(theta) * X + np.sin(theta) * Y, -np.sin(theta) * X + np.cos(theta) * Y

    coneCompressionFactor = 1 - altitude / src.altitude
    x = src.coordinates[0] * altitude * arcsec2radian * np.cos(src.coordinates[1] * np.pi / 180)
    y = src.coordinates[0] * altitude * arcsec2radian * np.sin(src.coordinates[1] * np.pi / 180)
    return (X * coneCompressionFactor + x) + 1j * (Y * coneCompressionFactor + y)


def fieldTransformToSourceOffset(shift_x, shift_y, rotation_angle, pixelSize):
    """
    Offset [m] and rotation [rad] for sourceGridCoordinates of a WFS behind an OOPAO
    FieldTransformer with the given shift (pixels) and rotation (degrees).

    The FieldTransformer maps the pupil pixel p to R (p - c) + c + shift, with (x, y) =
    (column, row) and c = n/2 (half a pixel away from the pupil centre (n-1)/2), so the WFS
    node q sees the pupil at R^-1 (q - d - shift) + d = R^-1 (q + offset), d = (1/2, 1/2) px,
    i.e. offset = -shift - (I - R) d.

    Returns
    -------
    offset_x, offset_y : float, m
    rotation : float, rad
    """
    theta = np.deg2rad(rotation_angle)
    d = 0.5
    offset_x = -shift_x - (d - (np.cos(theta) * d - np.sin(theta) * d))
    offset_y = -shift_y - (d - (np.sin(theta) * d + np.cos(theta) * d))
    return offset_x * pixelSize, offset_y * pixelSize, theta


def dists(rho1, rho2):
    if cuda_available:
        return cp.abs(cp.subtract.outer(rho1.flatten(), rho2.flatten()))

    else:
        return np.abs(np.subtract.outer(rho1.flatten(), rho2.flatten()))



def meshgrid(nPts, D, offset_x=0, offset_y=0, stretch_x=1, stretch_y=1):
    x = np.linspace(-D / 2, D / 2, nPts)
    X, Y = np.meshgrid(x * stretch_x, x * stretch_y)

    X = X + offset_x
    Y = Y + offset_y
    return X, Y


def reconstructionGrid(mask, os, dm_space=False):
    # os stands for oversampling. Can be 1 or 2.
    # If os=1, reconstructionGrid pitch = subaperturePitch,
    # if os=2, reconstructionGrid pitch = subaperturePitch/2
    from scipy.signal import convolve2d
    if os == 1 and dm_space:
        val = mask
    elif os == 1 and not dm_space:
        val = convolve2d(np.ones((2, 2)), mask).astype('bool')
    elif os == 2:
        nElements = os * mask.shape[0] + 1  # Linear number of lenslet+actuator
        validLensletActuator = np.zeros((nElements, nElements), dtype=bool)
        index = np.arange(1, nElements, 2)  # Lenslet index
        validLensletActuator[np.ix_(index, index)] = mask
        kernel = np.ones((3, 3))
        output = convolve2d(validLensletActuator, kernel, mode='same')
        val = output.astype(bool)
    return val


def sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask=None, os=2):
    """
    Sparse gradient matrix weighted by the pupil illumination.

    Gamma maps the phase on the (os*nLenslet+1)^2 reconstruction nodes (subaperture corners
    plus os-1 intermediate points per side) to the x and y slopes of the valid lenslets:
    rows = [x slopes; y slopes] in validLenslet column-major order, columns = the nodes kept
    in gridMask (column-major).

    Each slope is a weighted mean of the one-step phase differences between neighbouring
    nodes of the lenslet's (os+1)x(os+1) node block:
        s_x = os * sum_ij w_ij (phi(i,j+1) - phi(i,j)) / sum_ij w_ij
        w_ij = r_i * a(i,j) * a(i,j+1)
    with a = amplMask (node illumination, see amplitudeMaskForGradient) and r_i the row
    profile of the standard stencil ([1 2 1]/4 for os = 2, [1 3 8 3 1]/16 for os = 4,
    trapezoidal otherwise); s_y likewise in the other direction. Hence:
     - with an all-ones mask (default) it is identical to the standard stencil
     - a difference that involves a dark node has zero weight, so phase values outside the
       pupil never enter the slopes
     - a pure ramp gives the same slope on every lenslet whatever the mask; the mask only
       matters for non-linear phase
    A lenslet without any illuminated node pair in one direction gets an all-zero row.

    Parameters
    ----------
    validLenslet : ndarray (nLenslet, nLenslet), bool
    amplMask : ndarray (os*nLenslet+1, os*nLenslet+1), optional
        Node illumination (default ones).
    os : int
        Oversampling: 2 (default), 4, or any integer >= 1.

    Returns
    -------
    Gamma : np.matrix (2*nValid, nNodes)
    gridMask : ndarray (os*nLenslet+1, os*nLenslet+1), bool
    """
    nLenslet = validLenslet.shape[0]
    nMap = os * nLenslet + 1
    if amplMask is None:
        amplMask = np.ones((nMap, nMap))
    amplMask = np.asarray(amplMask, dtype=float)
    if amplMask.shape != (nMap, nMap):
        raise ValueError(f"amplMask must be {nMap} x {nMap}")
    dsa = 1  # slope units: phase difference per subaperture

    if os == 2:  # row profile of the standard stencil
        r = np.array([1, 2, 1]) / 4
    elif os == 4:
        r = np.array([1, 3, 8, 3, 1]) / 16
    else:
        r = np.r_[0.5, np.ones(os - 1), 0.5] / os

    jL, iL = np.nonzero(validLenslet.T)  # column-major, as the old loops
    nValid = iL.size
    nnzPer = (os + 1) ** 2
    I = np.zeros(2 * nnzPer * nValid, dtype=int)
    J = np.zeros_like(I)
    V = np.zeros(2 * nnzPer * nValid)
    gridMask = np.zeros((nMap, nMap), dtype=bool)
    ii, jj = np.meshgrid(np.arange(os + 1), np.arange(os + 1), indexing='ij')
    emptyRows = 0
    c = 0
    for k in range(nValid):
        i0 = iL[k] * os
        j0 = jL[k] * os
        a = amplMask[i0:i0 + os + 1, j0:j0 + os + 1]
        nodes = np.ravel_multi_index((i0 + ii, j0 + jj), (nMap, nMap), order='F')
        gridMask[i0:i0 + os + 1, j0:j0 + os + 1] = True

        # x slope: differences along columns, rows weighted by r
        wx = r[:, None] * a[:, :-1] * a[:, 1:]
        Sx = np.zeros((os + 1, os + 1))
        if wx.sum() > 0:
            wx = wx / wx.sum() * os / dsa
            Sx[:, :-1] -= wx
            Sx[:, 1:] += wx
        else:
            emptyRows += 1

        # y slope: differences along rows, columns weighted by r
        wy = r[None, :] * a[:-1, :] * a[1:, :]
        Sy = np.zeros((os + 1, os + 1))
        if wy.sum() > 0:
            wy = wy / wy.sum() * os / dsa
            Sy[:-1, :] -= wy
            Sy[1:, :] += wy
        else:
            emptyRows += 1

        idx = c + np.arange(nnzPer)
        I[idx] = k
        J[idx] = nodes.ravel(order='F')
        V[idx] = Sx.ravel(order='F')
        I[idx + nnzPer] = nValid + k
        J[idx + nnzPer] = nodes.ravel(order='F')
        V[idx + nnzPer] = Sy.ravel(order='F')
        c += 2 * nnzPer

    Gamma = csr_matrix((V, (I, J)), shape=(2 * nValid, nMap ** 2))
    Gamma = Gamma[:, gridMask.flatten('F')].todense()
    if emptyRows > 0:
        import warnings
        warnings.warn(f"{emptyRows} slope rows have no illuminated node pair and are set to zero")
    return Gamma, gridMask


def transformPupil(pupil, shift_x=0, shift_y=0, rotation_angle=0):
    """
    Pupil seen by a SH whose EM field is shifted (pixels) and rotated (degrees) as by OOPAO's
    FieldTransformer: the input point p goes to R (p - c) + c + shift, with (x, y) =
    (column, row) and c = n/2, so the SH pixel p sees the pupil at R^-1 (p - c - shift) + c.
    Bilinear interpolation, as in FieldTransformer.
    """
    from scipy.ndimage import map_coordinates
    pupil = np.asarray(pupil, dtype=float)
    if shift_x == 0 and shift_y == 0 and rotation_angle == 0:
        return pupil
    n = pupil.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    theta = np.deg2rad(rotation_angle)
    c = n / 2
    dx, dy = X - c - shift_x, Y - c - shift_y
    x_in = np.cos(theta) * dx + np.sin(theta) * dy + c
    y_in = -np.sin(theta) * dx + np.cos(theta) * dy + c
    return map_coordinates(pupil, [y_in, x_in], order=1, mode='grid-constant', cval=0)


def amplitudeMaskForGradient(pupil, validLenslet, os=2, method='hybrid',
                             shift_x=0, shift_y=0, rotation_angle=0):
    """
    Node illumination for sparseGradientMatrixAmplitudeWeighted.

    Returns the (os*nLenslet+1)^2 illumination of the reconstruction nodes (node k at
    x_k = k*d/os, d the subaperture size, i.e. on pixel edges of the pupil). The pupil
    resolution must be a multiple of nLenslet.

    method
     'hybrid' (default) area fraction for the nodes inside the pupil (pupil interpolated at
              the node > 0.5), 0 for the nodes outside it, which carry no valid phase
     'area'   illuminated fraction of the (d/os)x(d/os) cell centred on the node (exact
              fractional pixel overlaps)
     'point'  pupil bilinearly interpolated at the node
    Nodes not used by any valid lenslet are set to 0.

    shift_x, shift_y, rotation_angle : SH mis-registration, in pixels and degrees, with the
    FieldTransformer convention (FieldTransformer.shift_x[k], ...). The pupil is first
    transformed as seen by that SH. Give validLenslet of that SH.
    """
    validLenslet = np.asarray(validLenslet, dtype=bool)
    nLenslet = validLenslet.shape[0]
    P = transformPupil(pupil, shift_x, shift_y, rotation_angle)
    nRes = P.shape[0]
    if nRes % nLenslet:
        raise ValueError(f"pupil resolution ({nRes}) must be a multiple of nLenslet ({nLenslet})")
    nPx = nRes // nLenslet
    nMap = os * nLenslet + 1
    xk = np.arange(nMap) * nPx / os  # node positions in pixel-edge units

    method = method.lower()
    if method == 'hybrid':
        return (amplitudeMaskForGradient(P, validLenslet, os, 'area')
                * (amplitudeMaskForGradient(P, validLenslet, os, 'point') > 0.5))
    elif method == 'area':
        h = nPx / os
        lo, hi = xk[:, None] - h / 2, xk[:, None] + h / 2
        p = np.arange(1, nRes + 1)[None, :]
        W = np.maximum(0, np.minimum(hi, p) - np.maximum(lo, p - 1))  # pixel/cell overlaps
        area = W.sum(axis=1)
        amplMask = (W @ P @ W.T) / np.outer(area, area)
    elif method == 'point':
        from scipy.interpolate import RegularGridInterpolator
        x = np.arange(nRes) + 0.5
        interp = RegularGridInterpolator((x, x), P, method='linear', bounds_error=False, fill_value=0)
        Yo, Xo = np.meshgrid(xk, xk, indexing='ij')
        amplMask = interp(np.stack([Yo, Xo], axis=-1))
    else:
        raise ValueError("method must be 'hybrid', 'area' or 'point'")

    used = np.zeros((nMap, nMap), dtype=bool)  # keep the nodes used by valid lenslets
    for i, j in zip(*np.nonzero(validLenslet)):
        used[i * os:i * os + os + 1, j * os:j * os + os + 1] = True
    amplMask[~used] = 0
    amplMask[np.abs(amplMask - 1) < 1e-12] = 1
    return amplMask


def get_xyxy_permutation_matrix(N):
    xyxy_permutation_matrix = np.zeros((N, N))
    for n in range(N//2):
        xyxy_permutation_matrix[n, 2*n] = 1  
        xyxy_permutation_matrix[n + N//2, 2*n+1] = 1 

    return xyxy_permutation_matrix


def get_signal_permutation_matrix(mask, n_lgs):

    N = mask.shape[0]

    c_ids = np.array([f"[{i}, {j}]" for i in range(N) for j in range(N) if mask[i, j]])
    f_ids = np.array([f"[{i}, {j}]" for j in range(N) for i in range(N) if mask[i, j]])


    permutation_Signal_C2F = np.zeros((len(c_ids), len(c_ids)))

    for row_id, c_id in enumerate(c_ids):
        permutation_Signal_C2F[np.where(f_ids==c_id)[0][0], row_id] = 1

    permutation_Signal_C2F = block_diag(*[permutation_Signal_C2F]*2*n_lgs)

    return permutation_Signal_C2F

def get_dm_permutation_matrix(mask):

    N = mask.shape[0]

    c_ids = np.array([f"[{i}, {j}]" for i in range(N) for j in range(N) if mask[i, j]])
    f_ids = np.array([f"[{i}, {j}]" for j in range(N) for i in range(N) if mask[i, j]])


    permutation_DM_F2C = np.zeros((len(c_ids), len(c_ids)))

    for row_id, c_id in enumerate(c_ids):
        permutation_DM_F2C[np.where(f_ids==c_id)[0][0], row_id] = 1

    return permutation_DM_F2C



def get_filtering_matrix(unfiltered_mask, filtered_mask, n_lgs):
    filtering_matrix = np.zeros((np.count_nonzero(filtered_mask), np.count_nonzero(unfiltered_mask)))

    unfiltered_mask_arr = unfiltered_mask.flatten("F") 
    filtered_mask_arr = filtered_mask.flatten("F")    
    

    count_row = 0
    count_col = 0


    for i in range(len(unfiltered_mask_arr)):
        if unfiltered_mask_arr[i] and filtered_mask_arr[i]:
            filtering_matrix[count_row, count_col] = 1
            count_row += 1
            count_col +=1

        elif unfiltered_mask_arr[i]:
            count_col +=1
    
    filtering_matrix = block_diag(*[filtering_matrix]*2*n_lgs)


    return filtering_matrix


def get_different_filtered_subap_masks_filtering_matrix(list_filtered_subap_mask, filtered_subap_mask):
    list_filtering_matrices = []
    for valid_mask in list_filtered_subap_mask:
        filtering_matrix = np.zeros((np.count_nonzero(valid_mask), np.count_nonzero(filtered_subap_mask)))

        filtered_subap_mask_arr = filtered_subap_mask.flatten()
        valid_mask_arr = valid_mask.flatten()

        count_row = 0
        count_col = 0

        for i in range(len(filtered_subap_mask_arr)):
            if filtered_subap_mask_arr[i] and valid_mask_arr[i]:
                filtering_matrix[count_row, count_col] = 1
                count_row += 1
                count_col += 1

            elif filtered_subap_mask_arr[i]:
                count_col += 1
        list_filtering_matrices.append(block_diag(*[filtering_matrix] * 2))

    filtering_matrix = block_diag(*list_filtering_matrices)
    filtering_matrix = filtering_matrix.T

    return filtering_matrix


def modalRemovalMatrices(weight, act_mask, subap_mask, n_valid_act, n_lgs):
    if len(weight) != np.count_nonzero(subap_mask):
        weight = np.ones([np.count_nonzero(subap_mask)*2, n_lgs])

    # 1 %% Piston-Tip-Tilt removal in DM space
    zern = aotools.zernikeArray(3, act_mask.shape[0]).reshape(3, -1).T
    rem = zern[act_mask.flatten(), :]
    actPTTremMat = np.eye(n_valid_act) - rem @ np.linalg.pinv(rem)

    
    # 2 %% TT removal in slope space    
    nValidLenslet = np.count_nonzero(subap_mask)
    nLinLenslet = subap_mask.shape[0]
    rem = np.tile(np.array((1, 0)), (nValidLenslet, 1)).flatten()  # x-y slope ordering
    rem = np.array((rem, np.flipud(rem))).T

    slopesTTremMat_ = []
    pinvRemTT_ = []
    for k in range(n_lgs):
        w_ = weight[:, k]
        weightMatrix = np.tile(w_, (2, 1))
        pinvRemTT_.append(np.linalg.inv((rem.T * weightMatrix) @ rem) @ (rem.T * weightMatrix))
        slopesTTremMat_.append(np.eye(2 * nValidLenslet) - rem @ pinvRemTT_[k])
        # slopesTTremMat_.append((np.eye(2 * nValidLenslet) - rem @ pinvRemTT_[k]).T)
    slopesTTremMat = block_diag(*slopesTTremMat_)

    # 3 %% Focus removal in slope space
    zern = aotools.zernikeArray(3, nLinLenslet).reshape(3, -1).T[:, -2:]
    rem = zern[subap_mask.flatten(), :]
    # TODO replace hard-coded value by somehing agnostic, general and intelligent
    # this factor is applied to the focus projection matrix to comply with existing foccents projected read from file in the old reconstructor
    rem = rem.flatten() / 1585.8872704595105
    pinvRemFocus_ = []
    for k in range(n_lgs):
        w_ = weight[:, k]
        pinvRemFocus_.append(1 / ((rem.T * w_) @ rem) * (rem.T * w_))
    pinvRemFocus = np.hstack(pinvRemFocus_)

    if n_lgs == 1:
        return actPTTremMat, slopesTTremMat, np.vstack((block_diag(*pinvRemTT_), pinvRemFocus))
    else:
        # compute common (i.e. average) tip/tilt over the nLgs channels
        common_tilt = 1 / n_lgs * np.hstack(pinvRemTT_)
        # compute matrix-based common-tilt removal from each channel
        A = np.tile(np.array([[1, 0], [0, 1]]), (4, 1))
        M = A @ np.linalg.pinv(A)
        # the vstack() applies I-A@A**-1 to accomplish the common-tilt removal from each channel
        return actPTTremMat, slopesTTremMat, -np.vstack(
            [(np.eye(2 * n_lgs) - M) @ (block_diag(*pinvRemTT_)), -common_tilt, pinvRemFocus])

