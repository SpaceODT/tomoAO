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
    param : dict
        Configuration dictionary. Requires the keys ``'dm_resolution'``,
        ``'nSubaperture'`` and ``'mechanicalCoupling'`` to rebuild the
        deformable mirror, ``'resolution'`` to restore the telescope
        resolution, and ``'opticalBand'``, ``'magnitude'`` and
        ``'srcAltitude'`` to build the default ``mmse_star``.
    tel : Telescope
        OOPAO Telescope object.
    atm : Atmosphere
        OOPAO Atmosphere object.
    dm : DeformableMirror
        OOPAO DeformableMirror object.
    lgsAst : list[Source] or Asterism
        Laser guide star sources.
    filtered_subap_mask : ndarray (2D) or list[ndarray] or ndarray (3D)
        Valid subaperture mask(s) after flux filtering.

        - **2D array**: the same mask is used for every LGS.
        - **list of 2D arrays** or **3D array** (first axis = LGS index):
          one mask per LGS. The common mask used for covariance computation
          is derived via ``filtered_subap_mask_operation``.
    unfiltered_act_mask : ndarray, optional
        Boolean actuator mask. Defaults to ``dm.unfiltered_act_mask`` when
        not provided.
    mmse_star : Source or list[Source], optional
        Optimisation (MMSE) target(s). Defaults to a single ``Source`` built
        from ``param``.
    wfs : object, optional
        WFS object stored for reference. Not used internally.
    sci_src : Source, optional
        Science target source stored for reference.
    unfiltered_subap_mask : ndarray, optional
        Full (unfiltered) subaperture mask representing every physically
        illuminated subaperture. Defaults to the common filtered mask when
        not provided.
    filtered_subap_mask_operation : {'union', 'intersection'}, optional
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
        # TODO: Change lgsAst to generic Ast
        if isinstance(lgsAst, Asterism):
            lgs_list = lgsAst.src
        elif isinstance(lgsAst, list):
            lgs_list = lgsAst
        else:
            raise TypeError(
                f"lgsAst must be a list of Source objects or an Asterism, "
                f"got {type(lgsAst).__name__}."
            )

        for i, src in enumerate(lgs_list):
            if wfs is not None and wfs.em_field_transform is not None:
                src.offset_x = wfs.em_field_transform.shift_x[i]
                src.offset_y = wfs.em_field_transform.shift_y[i]
            else:
                src.offset_x = 0
                src.offset_y = 0


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



        if mmse_star is None:
            mmse_star = Source(optBand=param['opticalBand'],
                         magnitude=param['magnitude'],
                         altitude=param['srcAltitude'])

            mmse_star.offset_x = 0
            mmse_star.offset_y = 0
        else:
            if isinstance(mmse_star, list):
                for i in range(len(mmse_star)):
                    mmse_star[i].offset_x = 0
                    mmse_star[i].offset_y = 0
            else:
                if mmse_star.offset_x is None:
                    mmse_star.offset_x = 0
                if mmse_star.offset_y is None:
                    mmse_star.offset_y = 0


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




