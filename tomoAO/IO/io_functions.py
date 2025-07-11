import configparser
import os
import numpy as np
from astropy.io import fits


def load_from_ini(config_file, ao_mode=None, config_dir=os.path.dirname(__file__)):
    # Instantiate configparser
    config = configparser.ConfigParser()
    config.optionxform = str
    # Path of the configuration file is given as the "configdir" argument
    # Read the configuration file
    config.read(config_dir + config_file)

    # Get the section of the config file containing the reconstructor
    # parameters for the chosen AO_MODE
    if ao_mode is None:
        ao_mode = 'default'

    parm = dict(config[ao_mode].items())

    return {key: eval(value) for key, value in
            parm.items()}  # returns a dictionary with entries in specified data types


def open_influence_matrix(config, baseline_folder=None, filename=None):
    if baseline_folder is None:
        baseline_folder = config["path2imx"]

    if filename is None:
        filename = config["influence_matrix"]

    #with fits.open(baseline_folder + "influence_matrix_5Mar.fits") as hdul:
    with fits.open(baseline_folder+filename) as hdul:
        hdul.verify('fix')
        IM = hdul[0].data

    # For now, concatenating n_channels times the same IM
    n_channels = config["n_lgs"]

    return np.array([IM] * n_channels)