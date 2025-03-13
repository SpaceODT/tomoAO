import numpy as np
from joblib import Parallel, delayed




class mockAttribute:
    def __init__(self, param):
        self.param = param



class mockTelescope:
    def __init__(self, resolution, diameter):
        self.D = diameter
        self.resolution = resolution


class mockSource:
    def __init__(self, optBand:str, coordinates: list=[0,0], altitude: float=np.inf):

        self.coordinates = coordinates

        self.altitude = altitude

        self.photometry = self.getPhotometry()[optBand]

        self.wavelength = self.photometry[0]


    def getPhotometry(self):
        phot = {
            "U": [0.360e-6, 0.070e-6, 1.96e12],
            "B": [0.440e-6, 0.100e-6, 5.38e12],
            "V0": [0.500e-6, 0.090e-6, 3.64e12],
            "V": [0.550e-6, 0.090e-6, 3.31e12],
            "R": [0.640e-6, 0.150e-6, 4.01e12],
            "R2": [0.650e-6, 0.300e-6, 7.9e12],
            "R3": [0.600e-6, 0.300e-6, 8.56e12],
            "R4": [0.670e-6, 0.300e-6, 7.66e12],
            "I": [0.790e-6, 0.150e-6, 2.69e12],
            "I1": [0.700e-6, 0.033e-6, 0.67e12],
            "I2": [0.750e-6, 0.033e-6, 0.62e12],
            "I3": [0.800e-6, 0.033e-6, 0.58e12],
            "I4": [0.700e-6, 0.100e-6, 2.02e12],
            "I5": [0.850e-6, 0.100e-6, 1.67e12],
            "I6": [1.000e-6, 0.100e-6, 1.42e12],
            "I7": [0.850e-6, 0.300e-6, 5.00e12],
            "I8": [0.750e-6, 0.100e-6, 1.89e12],
            "I9": [0.850e-6, 0.300e-6, 5.00e12],
            "I10": [0.900e-6, 0.300e-6, 4.72e12],
            "J": [1.215e-6, 0.260e-6, 1.90e12],
            "J2": [1.550e-6, 0.260e-6, 1.49e12],
            "H": [1.654e-6, 0.290e-6, 1.05e12],
            "Kp": [2.1245e-6, 0.351e-6, 0.62e12],
            "Ks": [2.157e-6, 0.320e-6, 0.55e12],
            "K": [2.179e-6, 0.410e-6, 0.70e12],
            "K0": [2.000e-6, 0.410e-6, 0.76e12],
            "K1": [2.400e-6, 0.410e-6, 0.64e12],
            "L": [3.547e-6, 0.570e-6, 2.5e11],
            "M": [4.769e-6, 0.450e-6, 8.4e10],
            "Na": [0.589e-6, 0, 3.3e12],  # bandwidth is zero?
            "EOS": [1.064e-6, 0, 3.3e12],  # bandwidth is zero?
            "IR1310": [1.310e-6, 0, 2e12]  # bandwidth is zero?
        }
        return phot


class mockAtmosphere:
    def __init__(self, r0: float, L0: float, fractionalR0: list, altitude: list):

        self.r0 = r0
        self.L0 = L0
        self.altitude = altitude
        self.fractionalR0 = fractionalR0
        self.wavelength = 500 * 1e-9

class mockMisregistration(mockAttribute):
    def __init__(self, param):
        super().__init__(param)

        self.rotationAngle = param['rotationAngle']
        self.shiftX = param['shiftX']
        self.shiftY = param['shiftY']
        self.anamorphosisAngle = param['anamorphosisAngle']
        self.tangentialScaling = param['tangentialScaling']
        self.radialScaling = param['radialScaling']


class mockDeformableMirror(mockAttribute):
    def __init__(self, param, telescope, nSubap, dm_resolution, mechCoupling,
                 misReg, coordinates=None, pitch=None, modes=None, sign=1, flip_lr=False, flip=False, floating_precision: int = 64):
        super().__init__(param)

        self.tel = telescope

        self.misReg = misReg

        if pitch is not None:
            self.pitch = pitch
        else:
            self.pitch = self.tel.D / nSubap

        self.resolution = dm_resolution

        self.mechCoupling = mechCoupling

        self.sign = sign

        self.flip_lr = flip_lr

        self.flip = flip

        self.floating_precision = floating_precision

        self.xIF0 = coordinates[:, 0]
        self.yIF0 = coordinates[:, 1]

        self.nAct = len(self.xIF0)
        self.nActAlongDiameter = self.tel.D / self.pitch

        validAct = (np.arange(0, self.nAct))

        self.validAct = validAct.astype(int)
        self.nValidAct = self.nAct

        xIF0 = self.xIF0[self.validAct]
        yIF0 = self.yIF0[self.validAct]

        xIF3, yIF3 = self.anamorphosis(xIF0, yIF0, self.misReg.anamorphosisAngle *
                                       np.pi/180, self.misReg.tangentialScaling, self.misReg.radialScaling)

        xIF4, yIF4 = self.rotateDM(
            xIF3, yIF3, self.misReg.rotationAngle*np.pi/180)


        xIF = xIF4-self.misReg.shiftX
        yIF = yIF4-self.misReg.shiftY

        self.xIF = xIF
        self.yIF = yIF


        u0x = self.resolution/2+xIF*self.resolution/self.tel.D
        u0y = self.resolution/2+yIF*self.resolution/self.tel.D

        self.nIF = len(xIF)
        # store the coordinates
        self.coordinates = np.zeros([self.nIF, 2])
        self.coordinates[:, 0] = xIF
        self.coordinates[:, 1] = yIF


        if modes is not None:
            self.modes = modes
        else:
            def joblib_construction():
                Q = Parallel(n_jobs=8, prefer='threads')(
                    delayed(self.modesComputation)(i, j) for i, j in zip(u0x, u0y))
                return Q

            self.modes = np.squeeze(np.moveaxis(
                np.asarray(joblib_construction()), 0, -1))



    def anamorphosis(self, x, y, angle, mRad, mNorm):

        mRad += 1
        mNorm += 1
        xOut = x * (mRad*np.cos(angle)**2 + mNorm * np.sin(angle)**2) + \
            y * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)
        yOut = y * (mRad*np.sin(angle)**2 + mNorm * np.cos(angle)**2) + \
            x * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)

        return xOut, yOut

    def rotateDM(self, x, y, angle):
        xOut = x*np.cos(angle)-y*np.sin(angle)
        yOut = y*np.cos(angle)+x*np.sin(angle)
        return xOut, yOut

    def modesComputation(self, i, j):
        x0 = i
        y0 = j
        cx = (1 + self.misReg.radialScaling) * (self.resolution /
                                                self.nActAlongDiameter) / np.sqrt(
            2 * np.log(1. / self.mechCoupling))
        cy = (1 + self.misReg.tangentialScaling) * (self.resolution /
                                                    self.nActAlongDiameter) / np.sqrt(
            2 * np.log(1. / self.mechCoupling))

        #                    Radial direction of the anamorphosis
        theta = self.misReg.anamorphosisAngle * np.pi / 180
        x = np.linspace(0, 1, self.resolution) * self.resolution
        X, Y = np.meshgrid(x, x)

        #                Compute the 2D Gaussian coefficients
        a = np.cos(theta) ** 2 / (2 * cx ** 2) + np.sin(theta) ** 2 / (2 * cy ** 2)
        b = -np.sin(2 * theta) / (4 * cx ** 2) + np.sin(2 * theta) / (4 * cy ** 2)
        c = np.sin(theta) ** 2 / (2 * cx ** 2) + np.cos(theta) ** 2 / (2 * cy ** 2)

        G = self.sign * \
            np.exp(-(a * (X - x0) ** 2 + 2 * b * (X - x0) * (Y - y0) + c * (Y - y0) ** 2))

        if self.flip_lr:
            G = np.fliplr(G)

        if self.flip:
            G = np.flip(G)

        output = np.reshape(G, [1, self.resolution ** 2])
        if self.floating_precision == 32:
            output = np.float32(output)

        return output