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
    wfs : ShackHartmann, optional
        OOPAO SH (one SH per LGS). Its ``em_field_transform`` (FieldTransformer)
        gives the shift and rotation of each SH. Without it the SHs are assumed aligned.
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

    Attributes
    ----------
    wfs_shift_x, wfs_shift_y, wfs_rotation_angle : list[float]
        Shift [px] and rotation [deg] of each SH (FieldTransformer convention).
        Each LGS also gets ``offset_x``, ``offset_y`` [m] and ``rotation`` [rad],
        the same mis-registration as used by the covariance matrices.
    wfs_pupils : list[ndarray]
        Pupil seen by each SH: ``wfs.sh_data['src_k'].pupil_mask`` (binary) for a
        geometric SH, ``tools.transformPupil(tel.pupil, ...)`` (fractional edges)
        otherwise.
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

        # --- WFS mis-registrations (OOPAO FieldTransformer) ----------------
        # each SH is described by its FieldTransformer shift [px] and rotation [deg] and by the
        # pupil it sees; the sources carry the equivalent offset [m] and rotation [rad] used by
        # the covariance matrices (see tools.sourceGridCoordinates)
        n_lgs = len(lgs_list)
        field_transform = None if wfs is None else getattr(wfs, 'em_field_transform', None)
        if field_transform is not None:
            for name in ('anamorphosisAngle', 'tangentialScaling', 'radialScaling'):
                if np.any(np.asarray(getattr(field_transform, name, 0)) != 0):
                    logger.warning("FieldTransformer.%s is not modelled by tomoAO and is ignored.", name)
            wfs_shift_x = [float(v) for v in field_transform.shift_x]
            wfs_shift_y = [float(v) for v in field_transform.shift_y]
            wfs_rotation_angle = [float(v) for v in field_transform.rotation_angle]
        else:
            wfs_shift_x = [0.] * n_lgs
            wfs_shift_y = [0.] * n_lgs
            wfs_rotation_angle = [0.] * n_lgs

        pixel_size = tel.D / tel.resolution
        for i, src in enumerate(lgs_list):
            src.offset_x, src.offset_y, src.rotation = tools.fieldTransformToSourceOffset(
                wfs_shift_x[i], wfs_shift_y[i], wfs_rotation_angle[i], pixel_size
            )

        # pupil seen by each SH, for the gradient operators. The geometric SH of OOPAO computes
        # the slopes only inside its binary pupil_mask (fluxMap == max), so that is the pupil it
        # sees; a diffractive SH sees the transformed pupil with its fractional edge pixels
        if getattr(wfs, 'is_geometric', False) and hasattr(wfs, 'sh_data'):
            wfs_pupils = [np.asarray(wfs.sh_data[f'src_{i}'].pupil_mask, dtype=float)
                          for i in range(n_lgs)]
        else:
            wfs_pupils = [tools.transformPupil(tel.pupil, wfs_shift_x[i], wfs_shift_y[i],
                                               wfs_rotation_angle[i])
                          for i in range(n_lgs)]


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
                if getattr(mmse_star, 'offset_x', None) is None:
                    mmse_star.offset_x = 0
                if getattr(mmse_star, 'offset_y', None) is None:
                    mmse_star.offset_y = 0


        # --- store ---------------------------------------------------------
        self.tel = tel
        self.atm = atm
        self.dm = new_dm
        self.wfs = wfs
        self.wfs_shift_x = wfs_shift_x
        self.wfs_shift_y = wfs_shift_y
        self.wfs_rotation_angle = wfs_rotation_angle
        self.wfs_pupils = wfs_pupils
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




