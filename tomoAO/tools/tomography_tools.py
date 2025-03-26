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



def bessel_i(n, x, terms=10):
    x = cp.asarray(x, dtype=cp.float64)
    sum_result = cp.zeros_like(x, dtype=cp.float64)

    for m in range(terms):
        term = (1 / (math.factorial(m) * math.gamma(m + n + 1))) * (x / 2) ** (2 * m + n)
        sum_result += term

    return sum_result

def bessel_k(n, x, terms=10):
    x = cp.asarray(x, dtype=cp.float64)

    if cp.any(x <= 0):
        raise ValueError("x must be positive for K_n(x)")

    i_n = bessel_i(n, x, terms)
    i_neg_n = bessel_i(-n, x, terms)

    return (np.pi / 2) * (i_neg_n - i_n) / np.sin(n * cp.pi)


def covariance_matrix(rho, r0, L0):
    L0r0ratio = (L0 / r0) ** (5 / 3)
    cst = (24 * gamma(6 / 5) / 5) ** (5 / 6) * \
          (gamma(11 / 6) / (2 ** (5 / 6) * np.pi ** (8 / 3))) * L0r0ratio
    out = np.ones(rho.shape, dtype=rho.dtype) * (24 * gamma(6 / 5) / 5) ** (5 / 6) * \
          (gamma(11 / 6) * gamma(5 / 6) /
           (2 * np.pi ** (8 / 3))) * L0r0ratio
    index = rho != 0
    u = 2 * np.pi * rho[index] / L0
    if cuda_available:
        out[index] = cst * u ** (5 / 6) * bessel_k(5 / 6, cp.asarray(u), terms=10).get()

    else:
        out[index] = cst * u ** (5 / 6) * kv(5 / 6, u)

    return out



def dists(rho1, rho2):
    dist = np.abs(np.subtract.outer(rho1.flatten(), rho2.flatten()))
    return dist


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










