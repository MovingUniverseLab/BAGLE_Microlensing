import numpy as np
import pylab as plt
from astropy.table import Table
from astropy.time import Time
from astropy import units 
from astropy.coordinates import SkyCoord
from astropy import time as atime, coordinates as coord, units as u
import time


class EventDataDict(dict):
    """
    A dictionary object to hold the photometric and astrometric data for 
    a specified target. 

    The object must be initialized with
        target: event name
        raL: R.A. of the target
        decL: Dec. of the target
    and other entries are expected by the Multinest fitter and 
    can be set at or after initialization. If they are not 
    set at initialization, they will be initiliazed. 

    The other eexpected entries include:

    For each photometric data set, the dictionary should contain:
        
        t_phot1: Numpy array of times in MJD.
        mag1: Numpy array of magnitudes.
        mag_err1: Numpy array of magnitude errors. 
        t_phot2:  ... additional data sets...
        mag2:     ... additional data sets...
        mag_err2: ... additional data sets...


    where the '1' at the end is incremented for additional photometric data sets. 

    For each astrometric data set, the dictionary contains:
        
        t_ast1: Numpy array of times in MJD.
        xpos1: Numpy array of positions on the sky in the East direction in arcsec.
        ypos1: Numpy array of positions on the sky in the North direction in arcsec.
        xpos_err1: Numpy array of positional errors on the sky in the East direction in arcsec.
        ypos_err1: Numpy array of positional errors on the sky in the North direction in arcsec.
        t_ast2:    ... additional data sets...
        xpos2:     ... additional data sets...
        ypos2:     ... additional data sets...
        xpos_err2: ... additional data sets...
        ypos_err2: ... additional data sets...

    where the index is incremented for additional astrometric data sets.

    There should also be entries containing a list of the names and file locations
    of all the photometric and astrometric data sets loaded. This is 
    used during fitting and for reporting (and convenient reloading).
    These entries MUST match the order and length of the corresponding
    data sets. The entries are:

        phot_data: list of names for photometric data sets (e.g. I_OGLE, Kp_Keck, Ch1_Spitzer, MOA)
        phot_files: list of filename strings for photometric data sets
        ast_data: list of names for astrometric data sets (e.g. Kp_Keck)
        ast_files: list of filename strings for astrometric data sets

    One last note, if a data set is used to provide both photometry and astrometry, it
    should have the same "phot_data" and "ast_data" key. This will be used to tie parameters
    together between the two sets during fitting. 

    """
    
    def __init__(self, val=None):
        """
        EventDataDict must be instantiated with a dictionary containing

            target
            raL
            decL
        
        See class notes for a full description of what should be defined on 
        the dictionary for normal use. 
        """
        if ((val is None) or ('target' not in val) or
                ('raL' not in val) or ('decL' not in val)):
            msg = "EventDataDict must be instantiated with a dictionary containing target, raL, and decL."
            raise Exception(msg)

        super().__init__(val)

        # Check that all the required fields are made
        # and set to none if not used.
        self['phot_data'] = []
        self['ast_data'] = []
        self['phot_files'] = []
        self['ast_files'] = []

        return
        

    
def getdata(target, phot_data=['I_OGLE'], ast_data=['Kp_Keck'],
            time_format='mjd', verbose=False):
    """
    Helper function to illustrate how to load photometric and astrometric data.
    You will likely want to write your own version of this. 

    Get the photometric and astrometric data for the specified target. 
    Specify the types of data through the phot_data and ast_data lists. 

    Inputs
    ----------
    target : str
        Target name (lower case)

    Optional Inputs
    --------------------
    phot_data : list
        List of strings specifying the data sets. Options include:
        I_OGLE, Kp_Keck, Ch1_Spitzer, MOA

    ast_data : list
        List of strings specifying the data sets. Options include:
        Kp_Keck

    time_format : string
        The time format (default = 'mjd') such as mjd, year, jd.

    verbose : bool
        Print out extra information. 

    Returns
    ----------
    data_dict : EventDataDict
        A dictionary containing the data. See documentation on data.EventDataDict for details. 

        For each photometric data set, the dictionary contains:
        
            t_phot1
            mag1
            mag_err1

        where the '1' at the end is incremented for additional data sets. 

        For each astrometric data set, the dictionary contains:
        
            t_ast1
            xpos1
            ypos1
            xpos_err1
            ypos_err1

        where the index is incremented for additional astrometric data sets (useful for the future).

        There are two additional entries in the dictionary which contain the R.A. and Dec.
        of the lensing target... this is the photocenter of the joint lens/source system and 
        is hard-coded in module tables. Note that if only a single
        astrometry data set is requested, the returned keys are t_ast, xpos, ypos, xpos_err, ypos_err
        with no index on the end.

        data['raL']
        data['decL']

    """
    # Setup some internal dictionaries to hold many objects.
    # These are just for convenience.
    ra = {'mb09260' :  '17:58:28.561',
          'mb10364' :  '17:57:05.401',
          'ob110037' : '17:55:55.83',
          'ob110310' : '17:51:25.39',
          'ob110462' : '17:51:40.19',
          'ob110462_corr' : '17:51:40.19',
          'ob110462_new' : '17:51:40.19',
          'ob110462_op_bc' : '17:51:40.19',
          'ob110462_new2' : '17:51:40.19',
          'ob110462_new3' : '17:51:40.19',
          'ob110462_mroz22' : '17:51:40.19',
          'ob110462_mroz22_raw' : '17:51:40.19',
          'ob110462_trunc' : '17:51:40.19',
          'ob110462_corr_trunc' : '17:51:40.19',
          'ob110462_new_trunc' : '17:51:40.19',
          'ob110462_new_corr_trunc' : '17:51:40.19',
          'ob110462_22feb' : '17:51:40.19',
          'ob110462_23feb' : '17:51:40.19',
          'ob110462_23apr' : '17:51:40.19',
          'ob120169' : '17:49:51.38',
          'ob140613' : '17:53:57.68', 
          'ob150029' : '17:59:46.60', 
          'ob150211' : '17:29:26.18',
          'ob170302' : '17:41:35.93',
          'ob170328' : '17:54:09.56',
          'ob170019' : '17:52:18.74',
          'ob170095' : '17:51:27.94',
          'ob190017' : '17:59:03.52',
          'ob191000' : '17:47:01.67',
          'ob191080' : '18:10:04.47',
          'ob190241' : '17:54:10.76',
          'kb200101' : '17:45:11.03',
          'kb200122' : '17:40:27.69',
          'kb200122_short' : '17:40:27.69',
          'ob040361' : '17:46:35.41',
          'ob060095' : '17:57:23.14',
          'ob020061' : '17:35:55.97',
          'mb19284'  : '18:05:55.084',}
    
    dec = {'mb09260' :  '-26:50:20.88',
           'mb10364' :  '-34:27:05.01',
           'ob110037' : '-30:33:39.7',
           'ob110310' : '-30:24:35.0',
           'ob110462' : '-29:53:26.3',
           'ob110462_corr' : '-29:53:26.3',
           'ob110462_new' : '-29:53:26.3',
           'ob110462_op_bc' : '-29:53:26.3',
           'ob110462_new2' : '-29:53:26.3',
           'ob110462_new3' : '-29:53:26.3',
           'ob110462_mroz22' : '-29:53:26.3',
           'ob110462_mroz22_raw' : '-29:53:26.3',
           'ob110462_trunc' : '-29:53:26.3',
           'ob110462_corr_trunc' : '-29:53:26.3',
           'ob110462_new_trunc' : '-29:53:26.3',
           'ob110462_new_corr_trunc' : '-29:53:26.3',
           'ob110462_22feb' : '-29:53:26.3',
           'ob110462_23feb' : '-29:53:26.3',
           'ob110462_23apr' : '-29:53:26.3',
           'ob120169' : '-35:22:28.0',
           'ob140613' : '-28:34:21.6', 
           'ob150029' : '-28:38:41.8', 
           'ob150211' : '-30:58:54.3',
           'ob170019' : '-33:00:04.0', 
           'ob170095' : '-33:08:06.6',
           'ob190017' : '-27:32:49.2',
           'ob170302' : '-34:33:19.3',
           'ob170328' : '-28:44:52.6',
           'ob191000' : '-26:30:15.9',
           'ob191080' : '-27:52:01.4',
           'ob190241' : '-29:39:21.9',
           'kb200101' : '-25:24:28.98',
           'kb200122' : '-34:58:32.27',
           'kb200122_short' : '-34:58:32.27',
           'ob040361' : '-33:46:19.7',
           'ob060095' : '-28:46:32.0',
           'ob020061' : '-27:16:01.8',
           'mb19284'  : '-30:20:12.95',}
    
    # The values in astrom_file are from the latest analysis directories
    astrom_file = {'ob120169' : '/u/jlu/work/microlens/OB120169/a_2020_08_18/ob120169_astrom_p5_2020_08_18.fits',
                   'ob140613' : '/u/jlu/work/microlens/OB140613/a_2020_08_18/ob140613_astrom_p5_2020_08_18_os.fits',
                   'ob150029' : '/u/jlu/work/microlens/OB150029/a_2020_08_18/ob150029_astrom_p4_2020_08_18.fits',
                   'ob150211' : '/u/jlu/work/microlens/OB150211/a_2020_08_18/ob150211_astrom_p5_2020_08_18.fits',
                   'ob170095' : '/u/jlu/work/microlens/OB170095/a_2021_09_18/notes/ob170095_astrom_p4_2021_09_19.fits'} # TEMP
    
    # The values in astrom_file are from the latest analysis directories
    astrom_hst = {'mb09260_f606w'  : '/u/jlu/work/microlens/MB09260/a_2021_07_08/mb09260_f606w_astrom_p4_2021_07_08.fits',
                  'mb09260_f814w'  : '/u/jlu/work/microlens/MB09260/a_2021_07_08/mb09260_f814w_astrom_p4_2021_07_08.fits',
                  'mb10364_f606w' : '/u/jlu/work/microlens/MB10364/a_2021_07_08/mb10364_f606w_astrom_p5_2021_07_08.fits',
                  'mb10364_f814w' : '/u/jlu/work/microlens/MB10364/a_2021_07_08/mb10364_f814w_astrom_p5_2021_07_08.fits',
                  'ob110037_f606w' : '/u/jlu/work/microlens/OB110037/a_2021_07_08/ob110037_f606w_astrom_p5_2021_07_08.fits',
                  'ob110037_f814w' : '/u/jlu/work/microlens/OB110037/a_2021_07_08/ob110037_f814w_astrom_p5_2021_07_08.fits',
                  'ob110310_f606w' : '/u/jlu/work/microlens/OB110310/a_2021_07_08/ob110310_f606w_astrom_p4_2021_07_08.fits',
                  'ob110310_f814w' : '/u/jlu/work/microlens/OB110310/a_2021_07_08/ob110310_f814w_astrom_p4_2021_07_08.fits',
                  'ob110462_f606w' : '/u/jlu/work/microlens/OB110462/a_2021_07_08/ob110462_f606w_astrom_p5_nomay_2021_07_08.fits',
                  'ob110462_f814w' : '/u/jlu/work/microlens/OB110462/a_2021_07_08/ob110462_f814w_astrom_p5_nomay_2021_07_08.fits',
                  'ob110462_corr_f606w' : '/u/jlu/work/microlens/OB110462/a_2021_12_20/ob110462_f606w_astrom_p5_nomay_2021_12_20.fits',
                  'ob110462_corr_f814w' : '/u/jlu/work/microlens/OB110462/a_2021_12_20/ob110462_f814w_astrom_p5_nomay_2021_12_20.fits',
                  'ob110462_new_f606w' : '/u/jlu/work/microlens/OB110462/a_2021_12_28/ob110462_f606w_astrom_p5_nomay_bias_color_corr_2021_12_28.fits',
                  'ob110462_new_f814w' : '/u/jlu/work/microlens/OB110462/a_2021_12_28/ob110462_f814w_astrom_p5_nomay_bias_color_corr_2021_12_28.fits',
                  'ob110462_op_bc_f606w' : '/u/jlu/work/microlens/OB110462/a_2022_03_01/ob110462_f606w_astrom_p5_nomay_bias_corr_2021_12_28.fits',
                  'ob110462_op_bc_f814w' : '/u/jlu/work/microlens/OB110462/a_2022_03_01/ob110462_f814w_astrom_p5_nomay_bias_corr_2021_12_28.fits',
                  'ob110462_new2_f606w' : '/u/jlu/work/microlens/OB110462/a_2022_03_01/ob110462_f606w_astrom_p5_nomay_bias_corr_color_offset_2021_12_28.fits',
                  'ob110462_new2_f814w' : '/u/jlu/work/microlens/OB110462/a_2022_03_01/ob110462_f814w_astrom_p5_nomay_bias_corr_color_offset_2021_12_28.fits',
                  'ob110462_22feb_f606w' : '/u/jlu/work/microlens/OB110462/a_2022_02_02/ob110462_f606w_astrom_p5_nomay_bias_color_corr_gaia_2022_02_02.fits',
                  'ob110462_22feb_f814w' : '/u/jlu/work/microlens/OB110462/a_2022_02_02/ob110462_f814w_astrom_p5_nomay_bias_color_corr_gaia_2022_02_02.fits',
                  'ob110462_23feb_f606w' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/ob110462finn_f606w_astrom_p6.fits',
                  'ob110462_23feb_f814w' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/ob110462finn_f814w_astrom_p6.fits',
                  'ob110462_23apr_f606w' : '/u/jlu/work/microlens/OB110462/a_2023_04_23/all_epochs/ob110462finn_f606w_astrom_p6.fits',
                  'ob110462_23apr_f814w' : '/u/jlu/work/microlens/OB110462/a_2023_04_23/all_epochs/ob110462finn_f814w_astrom_p6.fits',
                  'ob110462_mroz22_f606w' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/OB110462astrom_f606w_p5_bias.fits',
                  'ob110462_mroz22_f814w' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/OB110462astrom_f814w_p5_bias.fits',
                  'ob110462_mroz22_f606w_raw' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/ob110462_f606w_astrom_p5.fits',
                  'ob110462_mroz22_f814w_raw' : '/u/jlu/work/microlens/OB110462/a_2022_11_15/ob110462_f814w_astrom_p5.fits',
                  'mb19284_f814w'  : '/u/jlu/work/microlens/MB19284/mb19284_astrom_p4_2021_10_1_hst.fits'}
    
    photom_file = {'ob110037' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0037.dat',
                   'ob110310' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0310.dat',
                   'ob110462' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462.dat',
                   'ob110462_corr' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_corr.dat',
                   'ob110462_new' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_corr.dat',
                   'ob110462_op_bc' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_corr.dat',
                   'ob110462_new2' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_corr.dat',
                   'ob110462_new3' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022.dat',
                   'ob110462_mroz22': '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_10_corr.dat',
                   'ob110462_trunc' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_trunc.dat',
                   'ob110462_corr_trunc' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_corr_trunc.dat',
                   'ob110462_new_trunc' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_trunc.dat',
                   'ob110462_new_corr_trunc' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_corr_trunc.dat',
                   'ob110462_22feb' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_corr.dat',
                   'ob110462_23feb' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_10_corr.dat',
                   'ob110462_23apr' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462_2022_10_corr.dat',
                   'ob120169' : '/g/lu/data/microlens/ogle/v2019_06/OGLE-2012-BLG-0169.dat',
                   'ob140613' : '/g/lu/data/microlens/ogle/v2019_06/OGLE-2014-BLG-0613.dat', 
                   'ob150029' : '/g/lu/data/microlens/ogle/v2019_06/OGLE-2015-BLG-0029.dat', 
                   'ob150211' : '/g/lu/data/microlens/ogle/v2019_06/OGLE-2015-BLG-0211.dat',
                   'ob170302' : '/g/lu/data/microlens/ogle/v2020_01_ews/OGLE-2017-BLG-0302.dat',
                   'ob170328' : '/g/lu/data/microlens/ogle/v2020_01_ews/OGLE-2017-BLG-0328.dat',
                   'ob170019' : '/g/lu/data/microlens/ogle/ews/OB170019.dat',
                   'ob170095' : '/g/lu/data/microlens/ogle/ews/OB170095.dat',
                   'ob190017' : '/g/lu/data/microlens/ogle/ews/OB190017.dat',
                   'ob191000' : '/g/lu/data/microlens/ogle/v2020_01_ews/OGLE-2019-BLG-1000.dat',
                   'ob191080' : '/g/lu/data/microlens/ogle/v2020_01_ews/OGLE-2019-BLG-1080.dat',
                   'ob190241' : '/g/lu/data/microlens/ogle/v2020_01_ews/OGLE-2019-BLG-0241.dat',
                   'ob040361' : '/g/lu/data/microlens/ogle/OGLE-2004-BLG-0361.dat',
                   'ob020061' : '/g/lu/data/microlens/ogle/OGLE-2002-BLG-0061.dat',
                   'ob060095' : '/g/lu/data/microlens/ogle/OGLE-2006-BLG-0095.dat',}
    
    photom_spitzer = {'ob120169': None,
                      'ob140613': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob140613_phot_2.txt',
                      'ob150029': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150029_phot_2.txt',
                      'ob150211': '/g/lu/data/microlens/spitzer/calchi_novati_2015/ob150211_phot_3.txt'}
    
    photom_moa = {'mb09260' : '/g/lu/data/microlens/moa/MB09260/mb09260-MOA2R-10000.phot.dat',
                  'mb10364' : '/g/lu/data/microlens/moa/MB10364/mb10364-MOA2R-10000.phot.dat',
                  'mb11039' : '/g/lu/data/microlens/moa/MB11039/mb11039-MOA2R-10000.phot.dat', # OB110037
                  'mb11332' : '/g/lu/data/microlens/moa/MB11332/mb11332-MOA2R-10000.phot.dat', # OB110310
                  'mb11191' : '/g/lu/data/microlens/moa/MB11191/mb11191-MOA2R-10000.phot.dat', # OB110462
                  'mb19284' : '/g/lu/data/microlens/moa/MB19284/MOA-2019-BLG-284_arb_calib.dat'}
    
    photom_kmt = {'kb200101' : '/g/lu/data/microlens/kmtnet/alerts_2020/kb200101/KMTA19_I.pysis.txt',
                  'kb200122' : '/g/lu/data/microlens/kmtnet/alerts_2020/kb200122/KMTA37_I.pysis'}
    
    # THIS ONE IS A TEMPORARY HACK... 
    photom_kmt_dia = {'kb200122' : '/Users/casey/scratch/proposals/hst28_ddt/kmt_dia.dat',
                      'kb200122_short' : '/Users/casey/scratch/proposals/hst28_ddt/kmt_dia_short.dat'}
    
    data_sets = {'mb09260' : {'MOA'   :      photom_moa['mb09260'],
                              'HST_f606w' :  astrom_hst['mb09260_f606w'],
                              'HST_f814w' :  astrom_hst['mb09260_f814w']},
                 'mb10364' : {'MOA'   :      photom_moa['mb10364'],
                              'MOA_TEST'  :  photom_moa['mb10364'],
                              'HST_f606w' :  astrom_hst['mb10364_f606w'],
                              'HST_f814w' :  astrom_hst['mb10364_f814w']},
                 'ob110037': {'I_OGLE':      photom_file['ob110037'],
                              'MOA'   :      photom_moa['mb11039'],
                              'HST_f606w' :  astrom_hst['ob110037_f606w'],
                              'HST_f814w' :  astrom_hst['ob110037_f814w']},
                 'ob110310': {'I_OGLE':      photom_file['ob110310'],
                              'MOA'   :      photom_moa['mb11332'],
                              'HST_f606w' :  astrom_hst['ob110310_f606w'],
                              'HST_f814w' :  astrom_hst['ob110310_f814w']},
                 'ob110462': {'I_OGLE':      photom_file['ob110462'],
                              'MOA'   :      photom_moa['mb11191'],
                              'HST_f606w' :  astrom_hst['ob110462_f606w'],
                              'HST_f814w' :  astrom_hst['ob110462_f814w']},
                 'ob110462_corr': {'I_OGLE': photom_file['ob110462_corr'],
                              'HST_f606w' :  astrom_hst['ob110462_corr_f606w'],
                              'HST_f814w' :  astrom_hst['ob110462_corr_f814w']},
                 'ob110462_new': {'I_OGLE':  photom_file['ob110462_new'],
                              'MOA'   :      photom_moa['mb11191'],
                              'HST_f606w' :  astrom_hst['ob110462_new_f606w'],
                              'HST_f814w' :  astrom_hst['ob110462_new_f814w']},
                 'ob110462_op_bc': {'I_OGLE':  photom_file['ob110462_op_bc'],
                                 'MOA'   :      photom_moa['mb11191'],
                                 'HST_f606w' :  astrom_hst['ob110462_op_bc_f606w'],
                                 'HST_f814w' :  astrom_hst['ob110462_op_bc_f814w']},
                 'ob110462_new2': {'I_OGLE':  photom_file['ob110462_new2'],
                              'MOA'   :      photom_moa['mb11191'],
                              'HST_f606w' :  astrom_hst['ob110462_new2_f606w'],
                              'HST_f814w' :  astrom_hst['ob110462_new2_f814w']},
                 'ob110462_new3': {'I_OGLE':  photom_file['ob110462_new3'],
                                   'MOA'   :      photom_moa['mb11191']},
                 'ob110462_mroz22': {'I_OGLE':  photom_file['ob110462_mroz22'],
                                     'MOA'   :      photom_moa['mb11191'],
                                     'HST_f606w' :  astrom_hst['ob110462_mroz22_f606w'],
                                     'HST_f814w' :  astrom_hst['ob110462_mroz22_f814w']},
                 'ob110462_mroz22_raw': {'I_OGLE':  photom_file['ob110462_mroz22'],
                                         'MOA'   :      photom_moa['mb11191'],
                                         'HST_f606w' :  astrom_hst['ob110462_mroz22_f606w_raw'],
                                         'HST_f814w' :  astrom_hst['ob110462_mroz22_f814w_raw']},
                 'ob110462_trunc': {'I_OGLE':      photom_file['ob110462_trunc'],
                                    'MOA'   :      photom_moa['mb11191']},
                 'ob110462_corr_trunc': {'I_OGLE':      photom_file['ob110462_corr_trunc'],
                                         'MOA'   :      photom_moa['mb11191']},
                 'ob110462_new_trunc': {'I_OGLE':      photom_file['ob110462_new_trunc'],
                                        'MOA'   :      photom_moa['mb11191']},
                 'ob110462_new_corr_trunc': {'I_OGLE':      photom_file['ob110462_new_corr_trunc'],
                                             'MOA'   :      photom_moa['mb11191']},
                 'ob110462_22feb': {'I_OGLE':  photom_file['ob110462_22feb'],
                                    'MOA'   :      photom_moa['mb11191'],
                                    'HST_f606w' :  astrom_hst['ob110462_22feb_f606w'],
                                    'HST_f814w' :  astrom_hst['ob110462_22feb_f814w']},
                 'ob110462_23feb': {'I_OGLE':  photom_file['ob110462_23feb'],
                                    'HST_f606w' :  astrom_hst['ob110462_23feb_f606w'],
                                    'HST_f814w' :  astrom_hst['ob110462_23feb_f814w']},
                 'ob110462_23apr': {'I_OGLE':  photom_file['ob110462_23apr'],
                                    'HST_f606w' :  astrom_hst['ob110462_23apr_f606w'],
                                    'HST_f814w' :  astrom_hst['ob110462_23apr_f814w']},
                 'ob120169': {'I_OGLE':      photom_file['ob120169'],
                              'Kp_Keck':     astrom_file['ob120169']},
                 'ob140613': {'I_OGLE':      photom_file['ob140613'],
                              'Kp_Keck':     astrom_file['ob140613'],
                              'Ch1_Spitzer': photom_spitzer['ob140613']},
                 'ob150029': {'I_OGLE':      photom_file['ob150029'],
                              'Kp_Keck':     astrom_file['ob150029'],
                              'Ch1_Spitzer': photom_spitzer['ob150029']},
                 'ob150211': {'I_OGLE':      photom_file['ob150211'],
                              'Kp_Keck':     astrom_file['ob150211'],
                              'Ch1_Spitzer': photom_spitzer['ob150211']},
                 'ob170302': {'I_OGLE':      photom_file['ob170302']},
                 'ob170328': {'I_OGLE':      photom_file['ob170328']},
                 'ob170019': {'I_OGLE':      photom_file['ob170019']},
                 'ob170095': {'I_OGLE':      photom_file['ob170095'],
                              'Kp_Keck':     astrom_file['ob170095']},
                 'ob190017': {'I_OGLE':      photom_file['ob190017']},
                 'ob191000': {'I_OGLE':      photom_file['ob191000']},
                 'ob191080': {'I_OGLE':      photom_file['ob191080']},
                 'ob190241': {'I_OGLE':      photom_file['ob190241']},
                 'kb200101': {'KMT'   :      photom_kmt['kb200101']},
                 'kb200122': {'KMT'   :      photom_kmt['kb200122'],
                              'KMT_DIA'    : photom_kmt_dia['kb200122']},
                 'kb200122_short': {'KMT_DIA'    : photom_kmt_dia['kb200122_short']},
                 'ob040361': {'I_OGLE':      photom_file['ob040361']},
                 'ob020061': {'I_OGLE':      photom_file['ob020061']},
                 'ob060095': {'I_OGLE':      photom_file['ob060095']},
                 'mb19284' : {'MOA'   :      photom_moa['mb19284'],
                              'HST_f814w' :  astrom_hst['mb19284_f814w']},
                 }
    
#    ra = {'mb09260' :  '17:58:28.561',
#          'mb10364' :  '17:57:05.401',
#          'ob110037' : '17:55:55.83',
#          'ob110310' : '17:51:25.39',
#          'ob110462' : '17:51:40.19',
#          'ob120169' : '17:49:51.38'
#    }
#    
#    dec = {'mb09260' :  '-26:50:20.88',
#           'mb10364' :  '-34:27:05.01',
#           'ob110037' : '-30:33:39.7',
#           'ob110310' : '-30:24:35.0',
#           'ob110462' : '-29:53:26.3',
#           'ob120169' : '-35:22:28.0'
#    }
#    
#    # The values in astrom_file are from the latest analysis directories
#    astrom_file = {'ob120169' : '/u/jlu/work/microlens/OB120169/a_2020_08_18/ob120169_astrom_p5_2020_08_18.fits'} # TEMP
#    
#    # The values in astrom_file are from the latest analysis directories
#    astrom_hst = {'mb09260_f606w'  : '/u/jlu/work/microlens/MB09260/a_2021_07_08/mb09260_f606w_astrom_p4_2021_07_08.fits',
#                  'mb09260_f814w'  : '/u/jlu/work/microlens/MB09260/a_2021_07_08/mb09260_f814w_astrom_p4_2021_07_08.fits',
#                  'mb10364_f606w' : '/u/jlu/work/microlens/MB10364/a_2021_07_08/mb10364_f606w_astrom_p5_2021_07_08.fits',
#                  'mb10364_f814w' : '/u/jlu/work/microlens/MB10364/a_2021_07_08/mb10364_f814w_astrom_p5_2021_07_08.fits',
#                  'ob110037_f606w' : '/u/jlu/work/microlens/OB110037/a_2021_07_08/ob110037_f606w_astrom_p5_2021_07_08.fits',
#                  'ob110037_f814w' : '/u/jlu/work/microlens/OB110037/a_2021_07_08/ob110037_f814w_astrom_p5_2021_07_08.fits',
#                  'ob110310_f606w' : '/u/jlu/work/microlens/OB110310/a_2021_07_08/ob110310_f606w_astrom_p4_2021_07_08.fits',
#                  'ob110310_f814w' : '/u/jlu/work/microlens/OB110310/a_2021_07_08/ob110310_f814w_astrom_p4_2021_07_08.fits',
#                  'ob110462_f606w' : '/u/jlu/work/microlens/OB110462/a_2021_07_08/ob110462_f606w_astrom_p5_nomay_2021_07_08.fits',
#                  'ob110462_f814w' : '/u/jlu/work/microlens/OB110462/a_2021_07_08/ob110462_f814w_astrom_p5_nomay_2021_07_08.fits'
#    }
#    
#    photom_file = {'ob110037' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0037.dat',
#                   'ob110310' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0310.dat',
#                   'ob110462' : '/g/lu/data/microlens/ogle/OGLE-2011-BLG-0462.dat',
#                   'ob120169' : '/g/lu/data/microlens/ogle/v2019_06/OGLE-2012-BLG-0169.dat'
#    }
#    
#    photom_spitzer = {'ob120169': None
#                      }
#    
#    photom_moa = {'mb09260' : '/g/lu/data/microlens/moa/MB09260/mb09260-MOA2R-10000.phot.dat',
#                  'mb10364' : '/g/lu/data/microlens/moa/MB10364/mb10364-MOA2R-10000.phot.dat',
#                  'mb11039' : '/g/lu/data/microlens/moa/MB11039/mb11039-MOA2R-10000.phot.dat', # OB110037
#                  'mb11332' : '/g/lu/data/microlens/moa/MB11332/mb11332-MOA2R-10000.phot.dat', # OB110310
#                  'mb11191' : '/g/lu/data/microlens/moa/MB11191/mb11191-MOA2R-10000.phot.dat' # OB110462
#                 }
#    
#    photom_kmt = {'kb200101' : '/g/lu/data/microlens/kmtnet/alerts_2020/kb200101/KMTA19_I.pysis.txt',
#                  'kb200122' : '/g/lu/data/microlens/kmtnet/alerts_2020/kb200122/KMTA37_I.pysis'}
#    
#    data_sets = {'mb09260' : {'MOA'   :      photom_moa['mb09260'],
#                              'HST_f606w' :  astrom_hst['mb09260_f606w'],
#                              'HST_f814w' :  astrom_hst['mb09260_f814w']},
#                 'mb10364' : {'MOA'   :      photom_moa['mb10364'],
#                              'MOA_TEST'  :  photom_moa['mb10364'],
#                              'HST_f606w' :  astrom_hst['mb10364_f606w'],
#                              'HST_f814w' :  astrom_hst['mb10364_f814w']},
#                 'ob110037': {'I_OGLE':      photom_file['ob110037'],
#                              'MOA'   :      photom_moa['mb11039'],
#                              'HST_f606w' :  astrom_hst['ob110037_f606w'],
#                              'HST_f814w' :  astrom_hst['ob110037_f814w']},
#                 'ob110310': {'I_OGLE':      photom_file['ob110310'],
#                              'MOA'   :      photom_moa['mb11332'],
#                              'HST_f606w' :  astrom_hst['ob110310_f606w'],
#                              'HST_f814w' :  astrom_hst['ob110310_f814w']},
#                 'ob110462': {'I_OGLE':      photom_file['ob110462'],
#                              'MOA'   :      photom_moa['mb11191'],
#                              'HST_f606w' :  astrom_hst['ob110462_f606w'],
#                              'HST_f814w' :  astrom_hst['ob110462_f814w']},
#                 'ob120169': {'I_OGLE':      photom_file['ob120169'],
#                              'Kp_Keck':     astrom_file['ob120169']}
#    }
    

    # Load up the data for the object we care about.
    data_in = {}

    # Load the RA and Dec
    target_coords = SkyCoord(ra[target], dec[target], 
                             unit = (units.hourangle, units.deg), frame = 'icrs')
    data_in['target'] = target
    data_in['raL'] = target_coords.ra.degree
    data_in['decL'] = target_coords.dec.degree

    # Keep track of the data files we used.
    phot_files = []
    ast_files = []
    
    # Load up the photometric data.
    for pp in range(len(phot_data)):
        filt = phot_data[pp]

        if filt not in data_sets[target].keys():
            raise RuntimeError('Failed to find photometric data set {0:s} for {1:s}'.format(filt, target))
        
        phot_files.append(data_sets[target][filt])
                              
        if filt == 'I_OGLE':
            # Read in photometry table.
            t, m, me = read_ogle_lightcurve(data_sets[target][filt])

        if filt == 'Kp_Keck':
            t, m, me = read_keck_lightcurve(data_sets[target][filt], target)

        if filt == 'Ch1_Spitzer':
            t, m, me = read_spitzer_lightcurve(data_sets[target][filt])

        if filt == 'MOA':
            t, m, me = read_moa_lightcurve(data_sets[target][filt])

        if filt[0:3] == 'HST':
            t, m, me = read_hst_lightcurve(data_sets[target][filt], target)

        if filt == 'KMT':
            t, m, me = read_kmt_lightcurve(data_sets[target][filt])

        if filt == 'KMT_DIA':
            t, m, me = read_kmt_dia_lightcurve(data_sets[target][filt])

        # Set time to proper format
        if time_format == 'mjd':
            t = t.mjd
        if time_format == 'jyear':
            t = t.j_year
        if time_format == 'jd':
            t = t.jd

        # Insert the data into the dictionary.
        suffix = '{0:d}'.format(pp + 1)
        if len(phot_data) == 1:
            suffix = '1'

        data_in['t_phot' + suffix] = t
        data_in['mag' + suffix] = m
        data_in['mag_err' + suffix] = me

    for aa in range(len(ast_data)):
        filt = ast_data[aa]

        if filt not in data_sets[target].keys():
            raise RuntimeError('Failed to find astrometric data set {0:s} for {1:s}'.format(filt, target))

        ast_files.append(data_sets[target][filt])
        
        if filt == 'Kp_Keck':
            t, x, y, xe, ye = read_keck_astrometry(data_sets[target][filt], target)

        if filt[0:3] == 'HST':
            t, x, y, xe, ye = read_hst_astrometry(data_sets[target][filt], target)
            
            
        # Set time to proper format
        if time_format == 'mjd':
            t = t.mjd
        if time_format == 'jyear':
            t = t.j_year
        if time_format == 'jd':
            t = t.jd

        # Insert the data into the dictionary.
        suffix = '{0:d}'.format(aa + 1)
            
        data_in['t_ast' + suffix] = t
        data_in['xpos' + suffix] = x
        data_in['ypos' + suffix] = y
        data_in['xpos_err' + suffix] = xe
        data_in['ypos_err' + suffix] = ye

    # Keep a record of the types of data.
    data_in['phot_data'] = phot_data
    data_in['ast_data'] = ast_data

    data_in['phot_files'] = phot_files
    data_in['ast_files'] = ast_files

    data_obj = EventDataDict(data_in)

    return data_obj
    

# Convenience functions for reading in different types of data files.
def read_ogle_lightcurve(filename):
    pho = Table.read(filename, format = 'ascii')
    t = Time(pho['col1'], format='jd', scale='utc')
    m = pho['col2']
    me = pho['col3']
    
    return (t, m, me)

def read_moa_lightcurve(filename):
    pho = Table.read(filename, format='ascii')
    # Convert HJD provided by MOA into JD.
    # https://geohack.toolforge.org/geohack.php?pagename=Mount_John_University_Observatory&params=43_59.2_S_170_27.9_E_region:NZ-CAN_type:landmark
    moa = coord.EarthLocation(lat=-43.986667 * u.deg,lon=170.465*u.deg, height=1029*u.meter)
    t_hjd = atime.Time(pho['col1'], format='jd', scale = 'utc')
    ltt = t_hjd.light_travel_time(target_coords, 'heliocentric', location=moa)
    
    t = t_hjd - ltt
    m = pho['col5']
    me = pho['col6']

    return (t, m, me)
    

def read_kmt_lightcurve(filename):
    pho = Table.read(filename, format='ascii')
    t = Time(pho['HJD'] + 2450000.0, format='jd', scale='utc')
    m = pho['mag']
    me = pho['mag_err']

    return (t, m, me)

def read_kmt_dia_lightcurve(filename):
    pho = Table.read(filename, format='ascii')
    t = Time(pho['col1'] + 2450000.0, format='jd', scale='utc')
    m = 27.68-2.5*np.log10(pho['col2']+27300)
    me = -1.08 * pho['col3']/(pho['col2'] + 27300)

    return (t, m, me)


def read_keck_lightcurve(filename, target, verbose=False):
    """
    File should be a FITS table with all of the stars (not just the target).
    """
    pho = Table.read(filename)
    tdx = np.where(pho['name'] == target)[0][0]
    t = Time(pho['t'][tdx, :], format='jyear', scale='utc')
    m = pho['m'][tdx, :]
    
    # Add empirical photometric errors based on nearest neighbors. 
    pho['m0e'] = np.nanstd(pho['m'], axis=1)
    pho['dr'] = np.hypot(pho['x0'] - pho['x0'][tdx],
                             pho['y0'] - pho['y0'][tdx])
    pho['dm'] = pho['m0'] - pho['m0'][tdx]
    
    # We can't use the normal me because it captures the source variability.
    # Calculate from surrounding stars. Don't use the target itself.
    dr = pho['dr']
    dm = np.abs(pho['dm'])
    
    # Iterate until we get some stars.
    n_neigh = 0
    nn = 1
    dr_start = 2.5
    dm_start = 1.5
    
    while n_neigh < 3:
        r_factor = 1.0 + (nn / 10.)  # Grow search radius by 10% each round.
        m_factor = 1.0 + (nn / 5.)   # Grow mag search by 20% each round.
        rdx = np.where((dr < dr_start*r_factor) & (dm < dm_start*m_factor) & (pho['m0e'] != 0))[0]
        rdx = rdx[rdx != tdx]  # Drop the target.
        n_neigh = len(rdx)
        nn += 1
        
    # For all the magniudes of the surrounding stars (in individual epochs),
    # mask out the invalid values.
    me_neigh = np.nanmean( pho['m0e'][rdx] )
    if verbose:
        print('Found {0:d} neighbors within:'.format(n_neigh))
        print('  dr = {0:0.2f} arcsec'.format(dr_start * r_factor))
        print('  dm = {0:0.2f} mag'.format(dm_start * m_factor))
        print(pho['name','m0','m0e','dr','dm'][rdx])

    if np.isnan(me_neigh):
        me_neigh = 0.025
        if verbose:
            print('Using hard-coded me_neigh')

    if verbose: 
        print('me_neigh = {0:.3f} mag'.format(me_neigh))
                
        me = np.ones(len(t), dtype=float) * me_neigh

    return (t, m, me)

        
def read_spitzer_lightcurve(filename):
    pho = Table.read(filename, format='ascii')
    t = Time(pho['col1']  + 2450000.0, format='jd', scale='utc')
    f = pho['col2']
    fe = pho['col3']
    m = 25.0 - 2.5 * np.log10(f)
    me = 1.086 * fe / f

    return (t, m, me)

def read_hst_lightcurve(filename, target):
    """
    File is a FITS table containing all stars. 
    Expected format is flystar.StarTable with columns
    for 'name', and 2D columns for 't', 'm', and 'me'.
    """
    pho = Table.read(filename)
    if 'ob110462' in target:
        tdx = np.where(pho['name'] == 'OB110462')[0][0]
    elif target == 'ob110462_op_bc':
        tdx = np.where(pho['name'] == 'OB110462')[0][0]
    elif target == 'ob110462_new2':
        tdx = np.where(pho['name'] == 'OB110462')[0][0]
    else:
        tdx = np.where(pho['name'] == target.upper())[0][0]
#    tdx = np.where(pho['name'] == target.upper())[0][0]
    good_idx = np.where(~np.isnan(pho[tdx]['t']))[0] # get rid of nans
    
    t = Time(pho['t'][tdx, good_idx], format='jyear', scale='utc')
    m = pho['m'][tdx, good_idx]
    me = pho['me'][tdx, good_idx]
    # Make sure t is increasing
    if t[0] > t[-1]:
        t = t[::-1]
        m = m[::-1]
        me = me[::-1]

    return (t, m, me)
    
def read_keck_astrometry(filename, target):
    ast = Table.read(filename)
    tdx = np.where(ast['name'] == target)[0][0]
    t = Time(ast['t'][tdx, :], format='jyear', scale='utc')
    x = ast['x'][tdx, :] * -1.0   # East in +x direction
    y = ast['y'][tdx, :]
    xe = ast['xe'][tdx, :]
    ye = ast['ye'][tdx, :]

    return (t, x, y, xe, ye)

def read_hst_astrometry(filename, target):
    ast = Table.read(filename)
    if 'ob110462' in target:
        tdx = np.where(ast['name'] == 'OB110462')[0][0]
    elif target == 'ob110462_op_bc':
        tdx = np.where(ast['name'] == 'OB110462')[0][0]
    elif target == 'ob110462_new2':
        tdx = np.where(ast['name'] == 'OB110462')[0][0]
    else:
        tdx = np.where(ast['name'] == target.upper())[0][0]
#    tdx = np.where(ast['name'] == target.upper())[0][0]
    good_idx = np.where(~np.isnan(ast[tdx]['t']))[0] # get rid of nans 
    t = Time(ast['t'][tdx, good_idx], format='jyear', scale='utc')
    x = ast['x'][tdx, good_idx] * -1.0   # East in +x direction
    y = ast['y'][tdx, good_idx]
    xe = ast['xe'][tdx, good_idx]
    ye = ast['ye'][tdx, good_idx]
    
    # Make sure t is increasing
    if t[0] > t[-1]:
        t = t[::-1]
        x = x[::-1]
        y = y[::-1]
        xe = xe[::-1]
        ye = ye[::-1]

    return (t, x, y, xe, ye)

    
