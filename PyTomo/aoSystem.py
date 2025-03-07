"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""

import os
import pdb

#os.chdir('//')

from .tomography_tools import *

# %% USE OOPAO, define a geometry and compute the cross-covariance matrix for all the layers
# from aotools.astronomy import FLUX_DICTIONARY

# import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.ShackHartmann import ShackHartmann

# %% -----------------------     read parameter file   ----------------------------------
# from aoscripts.ltaoReconstructor.parameterFile_Keck_LTAO import initializeParameterFile
# param = initializeParameterFile()


class AOSystem:
    def __init__(self, param):
        # %% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object (not Keck for now)
        tel = Telescope(resolution=param['resolution'],
                        diameter=param['diameter'],
                        samplingTime=param['samplingTime'],
                        centralObstruction=param['centralObstruction'])

        thickness_spider = 0.05  # size in m
        angle = [45, 135, 225, 315]  # in degrees
        offset_X = [-0.4, 0.4, 0.4, -0.4]  # shift offset of the spider
        offset_Y = None

        tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

        from scipy.io import loadmat
        # data = loadmat('./aodata/tel_pupil.mat')  # Load the .mat file
        data = loadmat(f'{param["path2parms"]}tel_pupil.mat')
        pupil = data['pup']  # Extract the array

        # print(pupil)
        tel.pupil = pupil
        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        ngs = Source(optBand=param['opticalBand'],
                     magnitude=param['magnitude'],
                     altitude=param['srcAltitude'])
        # combine the NGS to the telescope using '*' operator:
        ngs * tel
        # %% LGS objects
        # TODO replace this static code with a for loop to generate as many lgsGs as instructed by the user
        lgsAst = []
        ntemp = param['n_lgs']
        for kLgs in range(param["n_lgs"]):
            #print(f"{kLgs} {ntemp} KOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOK")
            lgs = Source(optBand=param['opticalBand'],
                      magnitude=param['lgs_magnitude'],
                      altitude=param['lgs_altitude'],
                      coordinates=[param['lgs_zenith'][kLgs], param['lgs_azimuth'][kLgs]])
            lgsAst.append(lgs)

        # lgs1 = Source(optBand=param['opticalBand'],
        #               magnitude=param['magnitude'],
        #               altitude=param['srcAltitude'],
        #               coordinates=[param['lgs_zenith'][0], param['lgs_azimuth'][0]])
        #
        # lgs2 = Source(optBand=param['opticalBand'],
        #               magnitude=param['magnitude'],
        #               altitude=param['srcAltitude'],
        #               coordinates=[param['lgs_zenith'][1], param['lgs_azimuth'][1]])
        #
        # lgs3 = Source(optBand=param['opticalBand'],
        #               magnitude=param['magnitude'],
        #               altitude=param['srcAltitude'],
        #               coordinates=[param['lgs_zenith'][2], param['lgs_azimuth'][2]])
        #
        # lgs4 = Source(optBand=param['opticalBand'],
        #               magnitude=param['magnitude'],
        #               altitude=param['srcAltitude'],
        #               coordinates=[param['lgs_zenith'][3], param['lgs_azimuth'][3]])
        #
        # lgsAst = [lgs1, lgs2, lgs3, lgs4]
        # lgsAst = lgsAst[:param["n_lgs"]] # retain only the n_lgs first sources

        # %% science targets
        sciSrc = Source(optBand='K',
                        magnitude=0,
                        altitude=np.Inf,
                        coordinates=[0, 0])

        # %% science targets

        recCalSrc = Source(optBand='Na',
                           magnitude=0,
                           altitude=np.Inf,
                           coordinates=[0, 0])
        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        # atm = Atmosphere(telescope=tel,
        #                  r0=param['r0']*np.cos(param['zenithAngleInDeg']*np.pi/180) ** (3 / 5),
        #                  L0=param['L0'],
        #                  windSpeed=param['windSpeed'],
        #                  fractionalR0=param['fractionnalR0'],
        #                  windDirection=param['windDirection'],
        #                  altitude=np.array(param['altitude'])*np.cos(param['zenithAngleInDeg']*np.pi/180),
        #                  param=param)

        atm = Atmosphere(telescope=tel,
                         r0=param['r0'],
                         L0=param['L0'],
                         windSpeed=param['windSpeed'],
                         fractionalR0=param['fractionnalR0'],
                         windDirection=param['windDirection'],
                         altitude=np.array(param['altitude']),
                         param=param)
        # initialize atmosphere
        atm.initializeAtmosphere(tel)

        atm.update()


        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration(param)


        # set coordinate vector to match the Keck actuator location
        # TODO make path relative
        # act_mask = np.loadtxt("./aodata/act_mask_keck.txt", dtype=str, delimiter=",")
        # breakpoint()
        act_mask = np.loadtxt(f'{param["path2parms"]}act_mask_keck.txt', dtype=str, delimiter=",")
        act_mask = act_mask.astype(bool)
        # breakpoint()

        X, Y = meshgrid(param['nActuator'], tel.D, offset_x=0.0, offset_y=0.0, stretch_x=1, stretch_y=1)
        act_mask = np.pad(act_mask, pad_width=2, mode='constant', constant_values=0)
        coordinates = np.array([X[act_mask], Y[act_mask]]).T

        # if no coordinates specified, create a cartesian dm
        resolution = tel.resolution

        # TODO this cannot be set by default, since the wavefront resolution is set only during the MMSE reconstructor. THere is a loophole here!
        #tel.resolution = 2 * param['nSubaperture'] + 1
        tel.resolution = param['dm_resolution']# this is to compute a low-resolution DM IF, where low-resolution is the wavefront reconstruction resolution
        dm = DeformableMirror(telescope=tel,
                              nSubap=param['nSubaperture'],
                              mechCoupling=param['mechanicalCoupling'],
                              misReg=misReg,
                              coordinates=coordinates,
                              pitch=tel.D / (param['nActuator'] - 1))


        dm.act_mask = act_mask
        tel.resolution = resolution
        # breakpoint()
        # %% -----------------------     Wave Front Sensor   ----------------------------------
        #pdb.set_trace()
        wfs = ShackHartmann(telescope=tel,
                            nSubap=param['nSubaperture'],
                            lightRatio=0.5)

        # Load the subap_mask
        # TODO make path relative
        # subap_mask = np.loadtxt("./aodata/subap_mask_keck.txt", dtype=str, delimiter=",")
        subap_mask = np.loadtxt(f'{param["path2parms"]}{param["sub_aperture_mask"]}', dtype=str, delimiter=",")

        subap_mask = subap_mask.astype(bool)
        wfs.valid_subapertures = subap_mask
        wfs.subap_mask = subap_mask #Force existence of subap_mask variable for backwards compatibility

        # %% scale r0 in case of different wavelengths
        # wvlScale = lgs1.wavelength / atm.wavelength
        # atm.r0 = atm.r0 * wvlScale ** (1.2)
        # %% LGS-to-LGS cross-covariance matrix


        # %% -----------------------     Wave Front Reconstruction   ----------------------------------

        outputReconstructiongrid = reconstructionGrid(subap_mask, param['os'], dm_space=False)


        # %% -----------------------     Self Allocation   ----------------------------------

        self.param = param
        self.atm = atm
        self.tel = tel
        self.dm = dm
        self.wfs = wfs
        self.lgsAst = lgsAst
        self.mmseStar = ngs
        self.misreg = misReg
        self.outputReconstructiongrid = outputReconstructiongrid
        self.sciSrc = sciSrc
        self.act_mask = act_mask
        self.subap_mask = subap_mask

