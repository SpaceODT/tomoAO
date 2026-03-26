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
from OOPAO.Asterism import Asterism



import logging
logger = logging.getLogger(__name__)

class AOSystem:
    """
    Container for the objects that make up an AO system.

    Parameters
    ----------
    tel : Telescope
        OOPAO Telescope object.
    atm : Atmosphere
        OOPAO Atmosphere object.
    dm : DeformableMirror
        OOPAO DeformableMirror object.
    lgsAst : list[Source] or Asterism
        Laser guide star sources.
    mmse_star : Source or list[Source]
        Optimisation (MMSE) target(s).
    filtered_subap_mask : ndarray (2D) or list[ndarray] or ndarray (3D)
        Valid subaperture mask(s) after flux filtering.

        - **2D array**: the same mask is used for every LGS.
        - **list of 2D arrays** or **3D array** (first axis = LGS index):
          one mask per LGS. The common mask used for covariance computation
          is derived via ``mask_operation``.
    act_mask : ndarray, optional
        Boolean actuator mask. Defaults to ``dm.act_mask`` when not provided.
    wfs : object, optional
        WFS object stored for reference. Not used internally.
    sci_src : Source, optional
        Science target source stored for reference.
    unfiltered_subap_mask : ndarray, optional
        Full (unfiltered) subaperture mask representing every physically
        illuminated subaperture. Defaults to the common filtered mask when
        not provided.
    mask_operation : {'union', 'intersection'}, optional
        How to derive the single common mask from per-LGS masks.
        Only relevant when ``filtered_subap_mask`` is a list or 3D array.
        Default is ``'union'``.
    os : int, optional
        Oversampling factor {1, 2, 4} for the reconstruction grid. Default 2.
    """

    def __init__(
        self,
        param,
        tel,
        atm,
        dm,
        lgsAst,
        filtered_subap_mask,
        unfiltered_act_mask=None,
        mmse_star = None,
        wfs=None,
        sci_src=None,
        unfiltered_subap_mask=None,
        filtered_subap_mask_operation='union',
        os=2,
    ):



        # --- LGS sources ---------------------------------------------------
        if isinstance(lgsAst, Asterism):
            lgs_list = lgsAst.src
        elif isinstance(lgsAst, list):
            lgs_list = lgsAst
        else:
            raise TypeError(
                f"lgsAst must be a list of Source objects or an Asterism, "
                f"got {type(lgsAst).__name__}."
            )

        n_lgs = len(lgs_list)


        # --- subaperture masks ---------------------------------------------
        list_filtered_subap_mask, common_filtered_mask = _parse_filtered_subap_mask(
            filtered_subap_mask, n_lgs, filtered_subap_mask_operation
        )

        if unfiltered_subap_mask is None:
            unfiltered_subap_mask = common_filtered_mask
            logger.debug("unfiltered_subap_mask not provided — using common filtered mask.")

        # --- reconstruction grid -------------------------------------------
        output_rec_grid = tools.reconstructionGrid(
            common_filtered_mask, os, dm_space=False
        )




        # --- actuator mask -------------------------------------------------
        tel.resolution = param["dm_resolution"]
        new_dm = DeformableMirror(telescope=tel,
                              nSubap=param['nSubaperture'],
                              mechCoupling=param['mechanicalCoupling'],
                              misReg=dm.misReg,
                              coordinates=dm.coordinates,
                              pitch=dm.pitch)

        if unfiltered_act_mask is None:
            new_dm.unfiltered_act_mask = dm.unfiltered_act_mask
        else:
            new_dm.unfiltered_act_mask = unfiltered_act_mask

        # TODO: non fried geometry not supported yet

        # Re-instate original telescope resolution
        tel.resolution = param["resolution"]

        # if unfiltered_act_mask is None:
        #     if not hasattr(dm, 'unfiltered_act_mask'):
        #         raise ValueError(
        #             "unfiltered_act_mask must be provided explicitly when dm.unfiltered_act_mask is not set."
        #         )
        #     unfiltered_act_mask = dm.unfiltered_act_mask


        if mmse_star is None:
            mmse_star = Source(optBand=param['opticalBand'],
                         magnitude=param['magnitude'],
                         altitude=param['srcAltitude'])

        # --- store ---------------------------------------------------------
        self.tel = tel
        self.atm = atm
        self.dm = new_dm
        self.wfs = wfs
        self.lgsAst = lgs_list
        self.mmseStar = mmse_star
        self.sciSrc = sci_src
        self.filtered_subap_mask = common_filtered_mask
        self.list_filtered_subap_mask = list_filtered_subap_mask
        self.unfiltered_subap_mask = unfiltered_subap_mask
        self.unfiltered_act_mask = new_dm.unfiltered_act_mask
        self.outputReconstructiongrid = output_rec_grid


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _parse_filtered_subap_mask(filtered_subap_mask, n_lgs, mask_operation):
    """
    Parse ``filtered_subap_mask`` into ``(list_filtered_subap_mask, common_mask)``.

    Accepts:
    - a single 2D boolean ndarray (same mask for all LGS),
    - a list of 2D boolean ndarrays (one per LGS),
    - a 3D boolean ndarray with shape ``(n_lgs, N, N)``.
    """
    # --- single mask -------------------------------------------------------
    if isinstance(filtered_subap_mask, np.ndarray) and filtered_subap_mask.ndim == 2:
        return [filtered_subap_mask] * n_lgs, filtered_subap_mask

    # --- per-LGS masks -----------------------------------------------------
    if isinstance(filtered_subap_mask, list):
        mask_list = [np.asarray(m, dtype=bool) for m in filtered_subap_mask]
    elif isinstance(filtered_subap_mask, np.ndarray) and filtered_subap_mask.ndim == 3:
        mask_list = [filtered_subap_mask[i] for i in range(filtered_subap_mask.shape[0])]
    else:
        raise TypeError(
            "filtered_subap_mask must be a 2D ndarray, a list of 2D ndarrays, "
            "or a 3D ndarray (shape n_lgs × N × N). "
            f"Got {type(filtered_subap_mask).__name__}"
            + (f" with ndim={filtered_subap_mask.ndim}" if hasattr(filtered_subap_mask, 'ndim') else "")
            + "."
        )

    if len(mask_list) != n_lgs:
        raise ValueError(
            f"Number of filtered masks ({len(mask_list)}) must match "
            f"number of LGS sources ({n_lgs})."
        )

    if mask_operation == 'union':
        common_mask = np.logical_or.reduce(mask_list)
    elif mask_operation == 'intersection':
        common_mask = np.logical_and.reduce(mask_list)
    else:
        raise ValueError(
            f"mask_operation must be 'union' or 'intersection', got '{mask_operation!r}'."
        )

    return mask_list, common_mask




class old_AOSystem:
    def __init__(self, param, **kwargs):
        # %% -----------------------     TELESCOPE   ----------------------------------
        if "tel" not in kwargs:

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
        else:
            tel = kwargs["tel"]

        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        if "ngs" not in kwargs:
            ngs = Source(optBand=param['opticalBand'],
                         magnitude=param['magnitude'],
                         altitude=param['srcAltitude'])
        else:
            ngs = kwargs["ngs"]
        # combine the NGS to the telescope using '*' operator:

        # %% LGS objects
        if "lgsAst" not in kwargs:
            lgsAst = [Source(optBand=param['opticalBand'],
                          magnitude=param['lgs_magnitude'],
                          altitude=param['lgs_altitude'],
                          coordinates=[param['lgs_zenith'][kLgs], param['lgs_azimuth'][kLgs]])
                      for kLgs in range(param["n_lgs"])]

        else:
            lgsAst = kwargs["lgsAst"].src


        # %% science targets
        if "sciSrc" not in kwargs:
            sciSrc = Source(optBand='K',
                            magnitude=0,
                            altitude=np.inf,
                            coordinates=[0, 0])
        else:
            sciSrc = kwargs["sciSrc"]



        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        if "atm" not in kwargs:
            atm = Atmosphere(telescope=tel,
                             r0=param['r0'],
                             L0=param['L0'],
                             windSpeed=param['windSpeed'],
                             fractionalR0=param['fractionnalR0'],
                             windDirection=param['windDirection'],
                             altitude=np.array(param['altitude']),
                             param=param)
        else:
            atm = kwargs["atm"]


        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        if "dm" not in kwargs:
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
            tel.resolution = param['dm_resolution']# this is to compute a low-resolution DM IF, where low-resolution is the wavefront reconstruction resolution
            dm = DeformableMirror(telescope=tel,
                                  nSubap=param['nSubaperture'],
                                  mechCoupling=param['mechanicalCoupling'],
                                  misReg=misReg,
                                  coordinates=coordinates,
                                  pitch=tel.D / (param['nActuator'] - 1))


            dm.act_mask = act_mask
            dm.unfiltered_act_mask = act_mask
            tel.resolution = resolution

        else:

            tel.resolution = param["dm_resolution"]
            dm = DeformableMirror(telescope=tel,
                                  nSubap=param['nSubaperture'],
                                  mechCoupling=param['mechanicalCoupling'],
                                  misReg=kwargs["dm"].misReg,
                                  coordinates=kwargs["dm"].coordinates,
                                  pitch=kwargs["dm"].pitch)

            dm.unfiltered_act_mask = kwargs["dm"].unfiltered_act_mask

            # TODO: non fried geometry not supported yet

            # Re-instate original telescope resolution
            tel.resolution = param["resolution"]


        # %% -----------------------     Wave Front Sensor   ----------------------------------
        if "wfs" not in kwargs:
            wfs = ShackHartmann(telescope=tel,
                                src=lgsAst[0],
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

        else:
            wfs = kwargs["wfs"]


        
        if "unfiltered_subap_mask" not in kwargs:
            unfiltered_subap_mask = wfs.valid_subapertures.copy()
        else:
            unfiltered_subap_mask = kwargs["unfiltered_subap_mask"]
        

        if "filtered_subap_mask" not in kwargs:
            filtered_subap_mask = wfs.valid_subapertures.copy()
            list_filtered_subap_mask = [filtered_subap_mask] * len(lgsAst)

        else:
            if len(kwargs["filtered_subap_mask"].shape) > 2:
                list_filtered_subap_mask = kwargs["filtered_subap_mask"]

                if "filtered_subap_mask_operation" not in kwargs:
                    mask_operation = "union"
                else:
                    mask_operation = kwargs["filtered_subap_mask_operation"]

                if mask_operation == "union":
                    filtered_subap_mask = np.logical_or.reduce(list_filtered_subap_mask)
                elif mask_operation == "intersection":
                    filtered_subap_mask = np.logical_and.reduce(list_filtered_subap_mask)
                else:
                    raise ValueError(f"'filtered_subap_mask_operation' must be 'union' or 'intersection', got '{mask_operation}'")

            else:
                filtered_subap_mask = kwargs["filtered_subap_mask"]
                list_filtered_subap_mask = [filtered_subap_mask] * len(lgsAst)

        # %% -----------------------     Wave Front Reconstruction   ----------------------------------

        outputReconstructiongrid = tools.reconstructionGrid(filtered_subap_mask, param['os'], dm_space=False)



        # %% -----------------------     Self Allocation   ----------------------------------

        self.param = param
        self.atm = atm
        self.tel = tel
        self.dm = dm
        self.wfs = wfs
        self.lgsAst = lgsAst
        self.mmseStar = ngs
        self.outputReconstructiongrid = outputReconstructiongrid
        self.sciSrc = sciSrc

        self.unfiltered_subap_mask = unfiltered_subap_mask
        self.filtered_subap_mask = filtered_subap_mask
        self.list_filtered_subap_mask = list_filtered_subap_mask
        self.unfiltered_act_mask = dm.unfiltered_act_mask
