"""
Created on Tue Apr 18 15:25:00 2023

@author: ccorreia@spaceodt.net
"""

import tomoAO.tools.tomography_tools as tools

# %% USE OOPAO, define a geometry and compute the cross-covariance matrix for all the layers
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.ShackHartmann import ShackHartmann




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


        # data = loadmat(f'{param["path2matrices"]}tel_pupil.mat')
        # pupil = data['pup']  # Extract the array
        #
        # tel.pupil = pupil
        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        ngs = Source(optBand=param['opticalBand'],
                     magnitude=param['magnitude'],
                     altitude=param['srcAltitude'])

        # combine the NGS to the telescope using '*' operator:
        ngs * tel
        # %% LGS objects


        lgsAst = [Source(optBand=param['opticalBand'],
                      magnitude=param['lgs_magnitude'],
                      altitude=param['lgs_altitude'],
                      coordinates=[param['lgs_zenith'][kLgs], param['lgs_azimuth'][kLgs]])
                  for kLgs in range(param["n_lgs"])]

        # %% science targets
        sciSrc = Source(optBand='K',
                        magnitude=0,
                        altitude=np.inf,
                        coordinates=[0, 0])

        # %% science targets

        recCalSrc = Source(optBand='Na',
                           magnitude=0,
                           altitude=np.inf,
                           coordinates=[0, 0])
        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object

        atm = Atmosphere(telescope=tel,
                         r0=param['r0'],
                         L0=param['L0'],
                         windSpeed=param['windSpeed'],
                         fractionalR0=param['fractionnalR0'],
                         windDirection=param['windDirection'],
                         altitude=np.array(param['altitude']),
                         param=param)


        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration(param)


        # set coordinate vector to match the Keck actuator location
        act_mask = np.loadtxt(param["actuator_mask"], dtype=bool, delimiter=",")
        if act_mask.shape[0] != param['nActuator']:
            act_mask = np.pad(act_mask, pad_width=int(param['nSubapExtra']/2), mode='constant', constant_values=0)

        X, Y = tools.meshgrid(param['nActuator'], tel.D, offset_x=0.0, offset_y=0.0, stretch_x=1, stretch_y=1)


        coordinates = np.array([X[act_mask], Y[act_mask]]).T

        self.dm_coordinates = coordinates
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
        # %% -----------------------     Wave Front Sensor   ----------------------------------
        wfs = ShackHartmann(telescope=tel,
                            nSubap=param['nSubaperture'],
                            lightRatio=0.5)


        unfiltered_subap_mask = np.loadtxt(param["unfiltered_subap_mask"],
                                           dtype=bool, delimiter=",")

        if unfiltered_subap_mask.shape[0] != param['nSubaperture']:
            unfiltered_subap_mask = np.pad(unfiltered_subap_mask,
                                           pad_width=int(param['nSubapExtra']/2),
                                           mode='constant',
                                           constant_values=0)


        filtered_subap_mask = np.loadtxt(param["filtered_subap_mask"],
                                         dtype=bool, delimiter=",")

        if filtered_subap_mask.shape[0] != param['nSubaperture']:
            filtered_subap_mask = np.pad(filtered_subap_mask,
                                         pad_width=int(param['nSubapExtra']/2),
                                         mode='constant',
                                         constant_values=0)


        wfs.valid_subapertures = unfiltered_subap_mask
        wfs.subap_mask = unfiltered_subap_mask #Force existence of subap_mask variable for backwards compatibility



        # %% -----------------------     Wave Front Reconstruction   ----------------------------------

        outputReconstructiongrid = tools.reconstructionGrid(unfiltered_subap_mask, param['os'], dm_space=False)


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
        self.unfiltered_subap_mask = unfiltered_subap_mask
        self.filtered_subap_mask = filtered_subap_mask

