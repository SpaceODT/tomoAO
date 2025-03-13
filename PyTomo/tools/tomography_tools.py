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

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.special import gamma, kv
from scipy.linalg import inv
import aotools
from scipy.signal import convolve2d
import os
# Library needed to interact with configuration files.
import configparser
import math

import dataclasses

import struct
# -------------------------------------------------

@dataclasses.dataclass
class ReconMatrices:
    weight0: np.ndarray
    KL_matrix: float
    invcov: float
    ttp: float
    sub_actuator_map: float
    actuator_actuator_map: float
    foccents: float
    sub_aperture_mask: bool
    act_mask: bool

    @classmethod
    def load_matrices(cls, config_vars: configparser.SectionProxy):
        # load the number of subapertures
        nb_sub_apertures = int(config_vars['nb_sub_apertures'])

        # load the number of actuators
        nb_actuators = int(config_vars['nb_actuators'])

        # load the weight0 matrix to regularize the reconstructor
        weight0 = np.fromfile(config_vars['path2parms'] + config_vars['weight0'], dtype=np.int8)

        # load the KL_matrix
        KL_matrix = np.fromfile(config_vars['path2parms'] + config_vars['KL_matrix'], dtype=np.float32)
        KL_matrix = KL_matrix.reshape((nb_actuators, nb_actuators))

        # load the invcov matrix
        invcov = readFromBinary(config_vars['path2parms'] + config_vars['invcov'], (nb_actuators, nb_actuators),">f")

        # load the ttp (tip-tilt-piston-removal) matrix
        ttp = readFromBinary(config_vars['path2parms'] + config_vars['ttp'], (nb_actuators, nb_actuators),">f")

        # load the sub_actuator_map matrix
        sub_actuator_map = np.fromfile(config_vars['path2parms'] + config_vars['sub_actuator_map'], dtype=np.int8)
        sub_actuator_map = sub_actuator_map.reshape((nb_sub_apertures, nb_actuators)).T

        # load the actuatorActuatorMap matrix
        actuator_actuator_map = np.fromfile(config_vars['path2parms'] + config_vars['actuator_actuator_map'],
                                            dtype=np.int8)
        actuator_actuator_map = actuator_actuator_map.reshape((nb_actuators, nb_actuators))

        # load the zern_to_cent matrix
        zern_to_cent = readFromBinary(config_vars['path2parms'] + config_vars['zernToCent'], (2*nb_sub_apertures,10),">f")

        # extract the focus-to-centroids
        foccents = zern_to_cent[:, 3]

        # load the sub-aperture mask
        sub_aperture_mask = np.loadtxt(config_vars['path2parms'] + config_vars['sub_aperture_mask'], dtype=str,
                                       delimiter=",").astype(bool)

        # load the actuator mask
        act_mask = np.loadtxt(config_vars['path2parms'] + config_vars['actuator_mask'], dtype=str,
                              delimiter=",").astype(bool)


        return cls(weight0=weight0, KL_matrix=KL_matrix,
                   invcov=invcov, ttp=ttp, sub_actuator_map=sub_actuator_map,
                   actuator_actuator_map=actuator_actuator_map,
                   foccents=foccents, sub_aperture_mask=sub_aperture_mask, act_mask=act_mask)


def readFromBinary(filename, expected_shape,file_format):
    binary_file = open(filename, "rb")
    # Prepare a vector to receive the file.
    data_vector = np.zeros(expected_shape).ravel()
    if file_format == ">f":
        buffer_size = 4
    elif file_format == ">b":
        buffer_size = 1
    # For each binary elements contained in the file opened.
    for k in np.arange(np.size(data_vector)):
        # Unpack the binary element and add it to the vector.
        data_vector[k] = struct.unpack(">f", binary_file.read(buffer_size))[0]

    # put back in correct shape
    return data_vector.reshape(expected_shape)



def idx2idx(idx_valid, mask):
    # pre-allocate the output mask with valid subapertures
    out = np.zeros_like(mask)
    # compute the index of subapertures expressed as a linear index on the 2D array
    idx = np.where(mask)
    # from all possible, select those indicated by the sequential index idx_valid
    out[idx[0][idx_valid], idx[1][idx_valid]] = 1
    return out



def oneDim2twoDim(mask, vals):
    out = np.zeros(mask.shape)
    inc = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]:
                out[i][j] = vals[inc]
                inc += 1
    return out



def toindex(mask):
    return np.where(mask.flatten())[0]


def tomask(index):
    return np.array([True] * len(index.flatten())).astype(bool)


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



def cellCovMatrix(tel, atm, src1, src2, recIdx):
    return -1


def sparseGradientMatrixAmplitudeWeighted_old(validLenslet, amplMask, os=2):
    import numpy as np
    from scipy.sparse import csr_matrix
    validLenslet = np.ones(validLenslet.shape, dtype=bool)
    nLenslet = validLenslet.shape[0]

    osFactor = os

    if amplMask is None:
        amplMask = np.ones((osFactor * nLenslet + 1, osFactor * nLenslet + 1))

    nMap = osFactor * nLenslet + 1
    nValidLenslet_ = np.count_nonzero(validLenslet)
    dsa = 1

    if osFactor == 2:
        i0x = np.tile([1, 2, 3], 3)
        j0x = np.repeat([1, 2, 3], 3)
        i0y = np.tile([1, 2, 3], 3)
        j0y = np.repeat([1, 2, 3], 3)
        s0x = np.array([-1 / 4, -1 / 2, -1 / 4, 0, 0, 0, 1 / 4, 1 / 2, 1 / 4]) * (1 / dsa)
        s0y = -np.array([1 / 4, 0, -1 / 4, 1 / 2, 0, -1 / 2, 1 / 4, 0, -1 / 4]) * (1 / dsa)
        Gv = np.array([[-2, 2, -1, 1], [-2, 2, -1, 1], [-1, 1, -2, 2], [-1, 1, -2, 2]])
        i_x = np.zeros(9 * nValidLenslet_)
        j_x = np.zeros(9 * nValidLenslet_)
        s_x = np.zeros(9 * nValidLenslet_)
        i_y = np.zeros(9 * nValidLenslet_)
        j_y = np.zeros(9 * nValidLenslet_)
        s_y = np.zeros(9 * nValidLenslet_)
        iMap0, jMap0 = np.meshgrid(np.arange(1, 4), np.arange(1, 4))
        gridMask = np.zeros((nMap, nMap), dtype=bool)
        u = np.arange(1, 10)
    elif osFactor == 4:
        # Define variables and calculations for osFactor = 4
        # Implement the calculations for osFactor = 4
        # ...
        pass

    # Perform accumulation of x and y stencil row and col subscript and weight
    for jLenslet in range(1, nLenslet + 1):
        jOffset = osFactor * (jLenslet - 1)
        for iLenslet in range(1, nLenslet + 1):
            if validLenslet[iLenslet - 1, jLenslet - 1]:
                I = (iLenslet - 1) * osFactor + 1
                J = (jLenslet - 1) * osFactor + 1

                a = amplMask[I - 1:I + osFactor, J - 1:J + osFactor]
                numIllum = np.sum(a)

                if numIllum == (osFactor + 1) ** 2:
                    iOffset = osFactor * (iLenslet - 1)
                    i_x[u - 1] = i0x + iOffset
                    j_x[u - 1] = j0x + jOffset
                    s_x[u - 1] = s0x
                    i_y[u - 1] = i0y + iOffset
                    j_y[u - 1] = j0y + jOffset
                    s_y[u - 1] = s0y
                    u = u + (osFactor + 1) ** 2
                    gridMask[iMap0 + iOffset - 1, jMap0 + jOffset - 1] = True
                elif numIllum != (osFactor + 1) ** 2:
                    # Perform calculations for numIllum != (osFactor+1)**2
                    # ...
                    pass

    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order='F')
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T
    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
                       shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma[:, gridMask.ravel()]

    return Gamma, gridMask



def sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask, os=2):

    nLenslet = validLenslet.shape[0]

    if amplMask is None:
        amplMask = np.ones((os * nLenslet + 1, os * nLenslet + 1))

    nMap = os * nLenslet + 1
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


def toggle(mat):
    sz = np.sqrt(len(mat)).astype(int)
    if mat.ndim == 2:
        new_mat = np.reshape(mat, (sz, sz, mat.shape[-1]), order="F")
    elif mat.ndim == 3:
        new_mat = np.reshape(mat, (len(mat) ** 2, mat.shape[-1]), order="F")

    return new_mat

def check_subap_mask_dims(validSubapMask, n_channels):
    if validSubapMask.ndim == 3:
        if n_channels != validSubapMask.shape[-1] and validSubapMask.shape[-1] == 1:
            validSubapMask_ = np.repeat(validSubapMask, n_channels, axis=2)
        elif n_channels == validSubapMask.shape[-1]:
            validSubapMask_ = validSubapMask
        else:
            # TODO raise exception
            # Prepare error message
            errmsg = f"Number of channels different from number of subaperture masks passed as input.\n"
            errmsg += f"Will not compute the reconstructor."
            # Log error
            log.critical(errmsg)
            raise Exception("Make sure the number of guide-stars == number of subaperture masks passed as input")
    else:
        # add 3rd dimension
        validSubapMask_ = np.expand_dims(validSubapMask, axis=-1)
        validSubapMask_ = np.repeat(validSubapMask_, n_channels, axis=2)
    return validSubapMask_


def svd_pro(G):

    from scipy.linalg import svd
    if issparse(G):
        U, S, V = svds(G)
    else:
        U, S, V = svd(G)

    return U, np.pad(np.diag(S), ((0, U.shape[1] - S.shape[0]), (0, 0))), V.T

def tsvd(G, no_svd_to_truncate=0, U=None, S=None, V=None):
    if U is None or S is None or V is None:
        U, S, V = svd_pro(G)

    # Transform S to the normal shape
    if no_svd_to_truncate == 0:
        keep = len(np.diag(S))
    else:
        keep = len(np.diag(S)) - no_svd_to_truncate

    Ured = U[:, :keep]
    Vred = V[:, :keep]
    S = S[:keep, :keep]

    invS = np.diag(1.0 / np.diag(S))
    invS = invS[:keep, :keep]
    return Vred @ invS @ Ured.T









