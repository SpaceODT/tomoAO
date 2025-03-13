import numpy as np

from .mockAttributes import mockTelescope, mockSource, mockAtmosphere, mockMisregistration, mockDeformableMirror

import PyTomo.tools.tomography_tools as tools


class mockAOSystem():
    def __init__(self, param):
        self.param = param

        self.tel = mockTelescope(resolution = self.param["resolution"],
                                 diameter = self.param["diameter"])

        self.lgsAst = []
        for kLgs in range(param["n_lgs"]):
            self.lgsAst.append(mockSource(optBand=param["opticalBand"],
                                          coordinates=[param['lgs_zenith'][kLgs],param['lgs_azimuth'][kLgs]],
                                          altitude=param['lgs_altitude']))

        self.mmseStar = mockSource(optBand=param["opticalBand"],altitude=param['srcAltitude'])


        self.atm = mockAtmosphere( r0=param['r0'],
                         L0=param['L0'],
                         fractionalR0=param['fractionnalR0'],
                         altitude=np.array(param['altitude']))

        self.misReg = mockMisregistration(param)

        self.subap_mask = np.loadtxt(f'{param["path2parms"]}{param["sub_aperture_mask"]}', dtype=str, delimiter=",")
        self.subap_mask = self.subap_mask.astype(bool)

        self.act_mask = np.loadtxt(f'{param["path2parms"]}act_mask_keck.txt', dtype=str, delimiter=",")
        self.act_mask = self.act_mask.astype(bool)

        X, Y = tools.meshgrid(param['nActuator'], self.tel.D, offset_x=0.0, offset_y=0.0, stretch_x=1, stretch_y=1)
        self.act_mask = np.pad(self.act_mask, pad_width=2, mode='constant', constant_values=0)

        coordinates = np.array([X[self.act_mask], Y[self.act_mask]]).T

        self.dm = mockDeformableMirror(telescope=self.tel,
                              nSubap=param['nSubaperture'],
                              dm_resolution    = param['dm_resolution'],
                              mechCoupling=param['mechanicalCoupling'],
                              misReg=self.misReg,
                              coordinates=coordinates,
                              pitch=self.tel.D / (param['nActuator'] - 1),
                              param=param)

        self.outputReconstructiongrid = tools.reconstructionGrid(self.subap_mask, param['os'], dm_space=False)

