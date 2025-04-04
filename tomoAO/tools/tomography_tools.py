#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""

import numpy as np
cuda_available = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        cuda_available = True
except:
    cuda_available = False

from scipy.sparse import csr_matrix
from scipy.special import gamma, kv

import math

from time import time

#
# def bessel_i2(n, x, terms=10):
#     x = cp.asarray(x, dtype=cp.float64)
#     sum_result = cp.zeros_like(x, dtype=cp.float64)
#
#     for m in range(terms):
#         term = (1 / (math.factorial(m) * math.gamma(m + n + 1))) * (x / 2) ** (2 * m + n)
#         sum_result += term
#
#     return sum_result
#
# def bessel_k2(n, x, terms=10):
#     print("old")
#     start = time()
#     x = cp.asarray(x, dtype=cp.float64)
#
#     if cp.any(x <= 0):
#         raise ValueError("x must be positive for K_n(x)")
#
#     i_n = bessel_i(n, x, terms)
#     i_neg_n = bessel_i(-n, x, terms)
#
#     print(f"Bessel part took {time() - start} seconds\n")
#
#     result = (np.pi / 2) * (i_neg_n - i_n) / np.sin(n * cp.pi)
#
#     return result


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


first_part_cst = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) / (2 ** (5 / 6) * np.pi ** (8 / 3)))
first_part_out = (24 * gamma(6 / 5) / 5) ** (5 / 6) * (gamma(11 / 6) * gamma(5 / 6) / (2 * cp.pi ** (8 / 3)))

def covariance_matrix(rho, r0, L0):
    # print(f"covaraiance matrix\n")
    # start_init = time()

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
        out[index] = cst * u ** (5 / 6) * kv(5 / 6, u)

    return out


def spatioAngularCovarianceMatrix(tel, atm, src1, src2, mask, os, dm_space=0):
    # Compute the discrete reconstruction mask from the subap_mask passed on as input

    recIdx = reconstructionGrid(mask, os, dm_space)
    if src1 == src2:  # auto-covariance matrix

        arcsec2radian = np.pi / 180 / 3600
        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                if j >= i:

                    phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                    for l in range(0, len(atm.altitude)):
                        # CROSS COVARIANCE BETWEEN TWO STARS
                        # STAR 1
                        coneCompressionFactor = 1 - \
                                                atm.altitude[l] / src1[i].altitude
                        x = src1[i].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            np.cos(src1[i].coordinates[1] * np.pi / 180)
                        y = src1[i].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            np.sin(src1[i].coordinates[1] * np.pi / 180)
                        X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
                                        stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)

                        rho1 = X + 1j * Y


                        # STAR 2
                        coneCompressionFactor = 1 - \
                                                atm.altitude[l] / src2[j].altitude
                        x = src2[j].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            np.cos(src2[j].coordinates[1] * np.pi / 180)
                        y = src2[j].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            np.sin(src2[j].coordinates[1] * np.pi / 180)
                        X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
                                        stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                        rho2 = X + 1j * Y

                        dist = dists(rho1.T, rho2.T)  # not sure why, using the transpose makes it equal to Matlab's

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
        # breakpoint()
        arcsec2radian = np.pi / 180 / 3600
        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                for l in range(0, len(atm.altitude)):
                    # CROSS COVARIANCE BETWEEN TWO STARS
                    # STAR 1
                    coneCompressionFactor = 1 - \
                                            atm.altitude[l] / src1[i].altitude
                    x = src1[i].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        np.cos(src1[i].coordinates[1] * np.pi / 180)
                    y = src1[i].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        np.sin(src1[i].coordinates[1] * np.pi / 180)
                    X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
                                    stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                    rho1 = X + 1j * Y
                    # STAR 2
                    coneCompressionFactor = 1 - \
                                            atm.altitude[l] / src2[j].altitude
                    x = src2[j].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        np.cos(src2[j].coordinates[1] * np.pi / 180)
                    y = src2[j].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        np.sin(src2[j].coordinates[1] * np.pi / 180)
                    X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
                                    stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                    rho2 = X + 1j * Y

                    dist = dists(rho1.T, rho2.T)  # not sure why, using the transpose makes it equal to Matlab's
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
    start_init = time()

    recIdx = reconstructionGrid(mask, os, dm_space)
    if src1 == src2:
        print("\n------------------------------")
        print("Starting Cxx")
        print("------------------------------\n")


        arcsec2radian = cp.pi / 180 / 3600
        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                if j >= i:

                    phaseCovElem = cp.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                    for l in range(0, len(atm.altitude)):
                        # CROSS COVARIANCE BETWEEN TWO STARS
                        # STAR 1

                        coneCompressionFactor = 1 - \
                                                atm.altitude[l] / src1[i].altitude
                        x = src1[i].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            cp.cos(src1[i].coordinates[1] * cp.pi / 180)
                        y = src1[i].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            cp.sin(src1[i].coordinates[1] * cp.pi / 180)
                        X, Y = meshgrid(nPts, tel.D, offset_x=x.get(), offset_y=y.get(),
                                        stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)

                        rho1 = X + 1j * Y

                        # STAR 2
                        coneCompressionFactor = 1 - \
                                                atm.altitude[l] / src2[j].altitude
                        x = src2[j].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            cp.cos(src2[j].coordinates[1] * cp.pi / 180)
                        y = src2[j].coordinates[0] * atm.altitude[l] * \
                            arcsec2radian * \
                            cp.sin(src2[j].coordinates[1] * cp.pi / 180)
                        X, Y = meshgrid(nPts, tel.D, offset_x=x.get(), offset_y=y.get(),
                                        stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                        rho2 = X + 1j * Y

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

        print(f"Cxx took {time() - start_init} seconds")

        return crossCovMat
    else:
        print("\n------------------------------")
        print("Starting Cox")
        print("------------------------------\n")

        arcsec2radian = cp.pi / 180 / 3600
        nPts = recIdx.shape[0]
        crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
        for i in range(0, len(src1)):
            for j in range(0, len(src2)):
                phaseCovElem = cp.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
                for l in range(0, len(atm.altitude)):
                    # CROSS COVARIANCE BETWEEN TWO STARS
                    # STAR 1

                    coneCompressionFactor = 1 - \
                                            atm.altitude[l] / src1[i].altitude
                    x = src1[i].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        cp.cos(src1[i].coordinates[1] * cp.pi / 180)
                    y = src1[i].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        cp.sin(src1[i].coordinates[1] * cp.pi / 180)
                    X, Y = meshgrid(nPts, tel.D, offset_x=x.get(), offset_y=y.get(),
                                    stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                    rho1 = X + 1j * Y

                    # STAR 2
                    coneCompressionFactor = 1 - \
                                            atm.altitude[l] / src2[j].altitude
                    x = src2[j].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        cp.cos(src2[j].coordinates[1] * cp.pi / 180)
                    y = src2[j].coordinates[0] * atm.altitude[l] * \
                        arcsec2radian * \
                        cp.sin(src2[j].coordinates[1] * cp.pi / 180)
                    X, Y = meshgrid(nPts, tel.D, offset_x=x.get(), offset_y=y.get(),
                                    stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
                    rho2 = X + 1j * Y

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

        print(f"Cox took {time() - start_init} seconds\n")
        return crossCovMat


def dists(rho1, rho2):
    if cuda_available:
        return cp.abs(cp.subtract.outer(rho1.flatten(), rho2.flatten()))

    else:
        return np.abs(np.subtract.outer(rho1.flatten(), rho2.flatten()))

    # dist = cp.abs(cp.subtract.outer(rho1.flatten(), rho2.flatten()))
    # return dist


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


# arcsec2radian = np.pi / 180 / 3600
#
# def compute_phase_cov(i, j, nPts, tel, atm, src1, src2, recIdx):
#     phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
#
#     src1_cos = np.cos([s.coordinates[1] * np.pi / 180 for s in src1])
#     src1_sin = np.sin([s.coordinates[1] * np.pi / 180 for s in src1])
#     src2_cos = np.cos([s.coordinates[1] * np.pi / 180 for s in src2])
#     src2_sin = np.sin([s.coordinates[1] * np.pi / 180 for s in src2])
#
#     for l in range(len(atm.altitude)):
#         alt = atm.altitude[l]
#
#         # Precompute compression factor
#         compFactor1 = 1 - alt / src1[i].altitude
#         compFactor2 = 1 - alt / src2[j].altitude
#
#         # Compute source coordinates
#         x1 = src1[i].coordinates[0] * alt * arcsec2radian * src1_cos[i]
#         y1 = src1[i].coordinates[0] * alt * arcsec2radian * src1_sin[i]
#         x2 = src2[j].coordinates[0] * alt * arcsec2radian * src2_cos[j]
#         y2 = src2[j].coordinates[0] * alt * arcsec2radian * src2_sin[j]
#
#         # Compute meshgrid
#         X1, Y1 = meshgrid(nPts, tel.D, offset_x=x1, offset_y=y1, stretch_x=compFactor1, stretch_y=compFactor1)
#         X2, Y2 = meshgrid(nPts, tel.D, offset_x=x2, offset_y=y2, stretch_x=compFactor2, stretch_y=compFactor2)
#
#         # Compute complex rho values
#         rho1 = X1 + 1j * Y1
#         rho2 = X2 + 1j * Y2
#
#         # Compute distances and covariance
#         dist = dists(rho1, rho2)
#         phaseCovElem[:, :, l] = covariance_matrix(dist, atm.r0, atm.L0) * atm.fractionalR0[l]
#
#     return np.sum(phaseCovElem, axis=2)[recIdx.flatten("F"), :][:, recIdx.flatten("F")]
#

# from joblib import Parallel, delayed
#
# def spatioAngularCovarianceMatrix2(tel, atm, src1, src2, mask, os, dm_space=0):
#     print("new")
#     recIdx = reconstructionGrid(mask, os, dm_space)
#     if src1 == src2:  # auto-covariance matrix
#         start = time()
#         arcsec2radian = np.pi / 180 / 3600
#         nPts = recIdx.shape[0]
#         crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
#         for i in range(0, len(src1)):
#             for j in range(0, len(src2)):
#                 if j >= i:
#
#                     phaseCovElem = np.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)))
#                     for l in range(0, len(atm.altitude)):
#                         # CROSS COVARIANCE BETWEEN TWO STARS
#                         # STAR 1
#                         coneCompressionFactor = 1 - \
#                                                 atm.altitude[l] / src1[i].altitude
#                         x = src1[i].coordinates[0] * atm.altitude[l] * \
#                             arcsec2radian * \
#                             np.cos(src1[i].coordinates[1] * np.pi / 180)
#                         y = src1[i].coordinates[0] * atm.altitude[l] * \
#                             arcsec2radian * \
#                             np.sin(src1[i].coordinates[1] * np.pi / 180)
#                         X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
#                                         stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
#
#                         rho1 = X + 1j * Y
#
#                         # STAR 2
#                         coneCompressionFactor = 1 - \
#                                                 atm.altitude[l] / src2[j].altitude
#                         x = src2[j].coordinates[0] * atm.altitude[l] * \
#                             arcsec2radian * \
#                             np.cos(src2[j].coordinates[1] * np.pi / 180)
#                         y = src2[j].coordinates[0] * atm.altitude[l] * \
#                             arcsec2radian * \
#                             np.sin(src2[j].coordinates[1] * np.pi / 180)
#                         X, Y = meshgrid(nPts, tel.D, offset_x=x, offset_y=y,
#                                         stretch_x=coneCompressionFactor, stretch_y=coneCompressionFactor)
#                         rho2 = X + 1j * Y
#
#                         dist = dists(rho1.T, rho2.T)  # not sure why, using the transpose makes it equal to Matlab's
#
#                         phaseCovElem[:, :, l] = covariance_matrix(
#                             dist, atm.r0, atm.L0) * atm.fractionalR0[l]
#
#                     crossCovCell[i, j] = np.sum(phaseCovElem, axis=2)[
#                                          recIdx.flatten("F"), :][:, recIdx.flatten("F")]
#
#         # populate the off-diagonal blocks left null in the previous step
#         for i in range(0, len(src1)):
#             for j in range(0, len(src1)):
#                 if j < i:
#                     crossCovCell[i, j] = np.transpose(crossCovCell[j, i])
#
#         crossCovMat = np.block(
#             [[crossCovCell[i, j] for j in range(len(src1))] for i in range(len(src2))])
#         crossCovMat = np.vstack(crossCovMat)
#         print(f"\nTook {time() - start} seconds")
#         return crossCovMat
#
#     else:
#         # arcsec2radian = np.pi / 180 / 3600
#         start=time()
#         nPts = recIdx.shape[0]
#         crossCovCell = np.empty((len(src1), len(src2)), dtype=object)
#
#         results = Parallel(n_jobs=-1)(
#             delayed(compute_phase_cov)(i, j, nPts, tel, atm, src1, src2, recIdx) for i in range(len(src1)) for j in range(len(src2))
#         )
#
#         print(results)
#
#         for idx, (i, j) in enumerate([(i, j) for i in range(len(src1)) for j in range(len(src2))]):
#             crossCovCell[i, j] = results[idx]
#
#
#         print(f"crossCovCell shape: {crossCovCell.shape}")
#         crossCovMat = np.block(
#             [[crossCovCell[i, j] for j in range(len(src1))] for i in range(len(src2))])
#
#         crossCovMat = np.vstack(crossCovMat)
#         print(f"\nTook {time() - start} seconds")
#         return crossCovMat
#
#
# import numba as nb
#
#
#
#
# from numba import cuda
#
# arcsec2radian = np.pi / 180 / 3600
#
#
# @cuda.jit
# def compute_covariance(
#         rho1_real, rho1_imag, rho2_real, rho2_imag,
#         r0, L0, fracR0, phaseCovElem, nPts
# ):
#     """CUDA kernel to compute covariance matrix in parallel."""
#     i, j, l = cuda.grid(3)
#     if i < nPts ** 2 and j < nPts ** 2:
#         dist = cp.sqrt((rho1_real[i] - rho2_real[j]) ** 2 + (rho1_imag[i] - rho2_imag[j]) ** 2)
#         phaseCovElem[i, j, l] = covariance_matrix(dist, r0, L0) * fracR0[l]
#
#
# def spatioAngularCovarianceMatrix1(tel, atm, src1, src2, mask, os, dm_space=0):
#     """Optimized GPU version of spatioAngularCovarianceMatrix."""
#     recIdx = reconstructionGrid(mask, os, dm_space)
#     nPts = recIdx.shape[0]
#
#     crossCovCell = [[None for _ in range(len(src2))] for _ in range(len(src1))]
#
#     for i in range(len(src1)):
#         for j in range(len(src2)):
#             if src1 == src2 and j < i:
#                 continue
#
#             phaseCovElem = cp.zeros((nPts ** 2, nPts ** 2, len(atm.altitude)), dtype=cp.float64)
#
#             for l in range(len(atm.altitude)):
#                 # STAR 1 (Precompute values for GPU)
#                 coneFactor1 = 1 - atm.altitude[l] / src1[i].altitude
#                 x1 = src1[i].coordinates[0] * atm.altitude[l] * arcsec2radian * cp.cos(
#                     cp.radians(src1[i].coordinates[1]))
#                 y1 = src1[i].coordinates[0] * atm.altitude[l] * arcsec2radian * cp.sin(
#                     cp.radians(src1[i].coordinates[1]))
#                 X1, Y1 = meshgrid(nPts, tel.D, offset_x=x1.get(), offset_y=y1.get(), stretch_x=coneFactor1, stretch_y=coneFactor1)
#                 X1 = cp.asarray(X1)
#                 Y1 = cp.asarray(Y1)
#                 rho1_real, rho1_imag = X1.ravel(), Y1.ravel()
#
#                 # STAR 2 (Precompute values for GPU)
#                 coneFactor2 = 1 - atm.altitude[l] / src2[j].altitude
#                 x2 = src2[j].coordinates[0] * atm.altitude[l] * arcsec2radian * cp.cos(
#                     cp.radians(src2[j].coordinates[1]))
#                 y2 = src2[j].coordinates[0] * atm.altitude[l] * arcsec2radian * cp.sin(
#                     cp.radians(src2[j].coordinates[1]))
#                 X2, Y2 = meshgrid(nPts, tel.D, offset_x=x2.get(), offset_y=y2.get(), stretch_x=coneFactor2, stretch_y=coneFactor2)
#                 X2 = cp.asarray(X2)
#                 Y2 = cp.asarray(Y2)
#
#                 rho2_real, rho2_imag = X2.ravel(), Y2.ravel()
#
#                 # Define CUDA grid
#                 threads_per_block = (16, 16, 1)
#                 blocks_per_grid = (
#                     (nPts ** 2 + threads_per_block[0] - 1) // threads_per_block[0],
#                     (nPts ** 2 + threads_per_block[1] - 1) // threads_per_block[1],
#                     (len(atm.altitude) + threads_per_block[2] - 1) // threads_per_block[2]
#                 )
#
#                 # Call CUDA kernel
#                 compute_covariance[blocks_per_grid, threads_per_block](
#                     rho1_real, rho1_imag, rho2_real, rho2_imag,
#                     atm.r0, atm.L0, atm.fractionalR0,
#                     phaseCovElem, nPts
#                 )
#
#             # Store results
#             crossCovCell[i][j] = cp.sum(phaseCovElem, axis=2)[recIdx.flatten("F"), :][:, recIdx.flatten("F")]
#
#     # Fill in symmetric entries for auto-covariance
#     if src1 == src2:
#         for i in range(len(src1)):
#             for j in range(i):
#                 crossCovCell[i][j] = cp.transpose(crossCovCell[j][i])
#
#     # Construct final covariance matrix
#     crossCovMat = cp.block([[crossCovCell[i][j] for j in range(len(src2))] for i in range(len(src1))])
#
#     return cp.vstack(crossCovMat) if len(src1) > 1 else crossCovMat
#


def sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask, os=2):

    nLenslet = validLenslet.shape[0]

    nMap = os * nLenslet + 1

    if amplMask is None:
        amplMask = np.ones((nMap, nMap))

    nValidLenslet_ = np.count_nonzero(validLenslet)
    dsa = 1

    i0x = np.tile([0, 1, 2], 3)

    j0x = np.repeat([0, 1, 2], 3)

    i0y = np.tile([0, 1, 2], 3)

    j0y = np.repeat([0, 1, 2], 3)

    s0x = np.array([-1 / 4, -1 / 2, -1 / 4, 0, 0, 0, 1 / 4, 1 / 2, 1 / 4]) * (1 / dsa)

    s0y = -np.array([1 / 4, 0, -1 / 4, 1 / 2, 0, -1 / 2, 1 / 4, 0, -1 / 4]) * (1 / dsa)

    Gv = np.array([[-2, 2, -1, 1], [-2, 2, -1, 1], [-1, 1, -2, 2], [-1, 1, -2, 2]])

    i_x = np.zeros(9 * nValidLenslet_)
    j_x = np.zeros(9 * nValidLenslet_)
    s_x = np.zeros(9 * nValidLenslet_)
    i_y = np.zeros(9 * nValidLenslet_)
    j_y = np.zeros(9 * nValidLenslet_)
    s_y = np.zeros(9 * nValidLenslet_)

    jMap0, iMap0 = np.meshgrid(np.arange(0, 3), np.arange(0, 3))

    gridMask = np.zeros((nMap, nMap), dtype=bool)

    u = np.arange(0, 9)

    for jLenslet in range(0, nLenslet):
        jOffset = os * (jLenslet)
        for iLenslet in range(0, nLenslet):
            if validLenslet[iLenslet, jLenslet]:
                I = (iLenslet) * os + 1
                J = (jLenslet) * os + 1

                a = amplMask[I - 1:I + os, J - 1:J + os]

                numIllum = np.sum(a)

                if numIllum == (os + 1) ** 2:
                    iOffset = os * (iLenslet)
                    i_x[u] = i0x + iOffset
                    j_x[u] = j0x + jOffset
                    s_x[u] = s0x
                    i_y[u] = i0y + iOffset
                    j_y[u] = j0y + jOffset
                    s_y[u] = s0y
                    u = u + (os + 1) ** 2

                    gridMask[iMap0 + iOffset, jMap0 + jOffset] = True

    indx = np.ravel_multi_index((i_x.astype(int), j_x.astype(int)), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int), j_y.astype(int)), (nMap, nMap), order='F')

    v = np.tile(np.arange(0, 2 * nValidLenslet_), (u.size, 1)).T

    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten(), np.concatenate((indx, indy)))),
                       shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma.todense()
    Gamma = Gamma[:, gridMask.flatten("F")]

    return Gamma, gridMask










