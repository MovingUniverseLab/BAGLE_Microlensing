from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
import numpy as np 
from astropy.time import Time
from astropy.coordinates.builtin_frames.utils import get_jd12
import erfa
import matplotlib.pyplot as plt
from bagle import model
from matplotlib.ticker import MaxNLocator

def convert_bagle_mulens_psbl_phot(ra, dec, 
                                   t0_in, u0_in, tE_in, 
                                   piEE_in, piEN_in, t0par,
                                   q_in, alpha_in, sep,
                                   mod_in='bagle', plot=True):
    """
    alpha_in : degrees.

    mod_in = 'bagle' or 'mulens'
    """
    ##########
    # Convert between helio and geo projected.
    ##########
    if mod_in == 'bagle':
        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='helio',
                                        murel_in='SL', murel_out='LS',
                                        coord_in='EN', coord_out='tb',
                                        plot=plot)

    elif mod_in == 'mulens':
        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='geo',
                                        murel_in='LS', murel_out='SL',
                                        coord_in='tb', coord_out='EN',
                                        plot=plot)
        
    else:
        raise Exception("mod_in must be 'bagle' or 'mulens'")

    t0_out, u0_out, tE_out, piEE_out, piEN_out = output

    ##########
    # Calculate the trajectory made by the source-lens relative the 
    # binary axis (this is different, because the source-lens relative motion
    # changes across the two frames).
    ##########
    murel_in = np.rad2deg(np.arctan2(piEN_in, piEE_in))
    murel_out = np.rad2deg(np.arctan2(piEN_out, piEE_out))
    # The 180 is because one is source-lens and the other is lens-source.
    if mod_in == 'bagle':
        delta_alpha = murel_out - murel_in + 180
        alpha_out = alpha_in - delta_alpha
        
        q_out = 1.0/q_in
        q_prime = 0.5 * (1 - q_in) / (1 + q_in)
        t0_out += q_prime * sep * tE_out * np.cos(np.deg2rad(alpha_out))
        u0_out += q_prime * sep * np.sin(np.deg2rad(alpha_out))
    elif mod_in == 'mulens': 
        delta_alpha = murel_out - murel_in + 180
        alpha_out = alpha_in + delta_alpha
        
        q_out = 1.0/q_in
        q_prime = 0.5 * (1 - q_out) / (1 + q_out)
        t0_out -= q_prime * sep * tE_out * np.cos(np.deg2rad(alpha_out))
        u0_out += q_prime * sep * np.sin(np.deg2rad(alpha_out))
        

    return t0_out, u0_out, tE_out, piEE_out, piEN_out, q_out, alpha_out


def convert_helio_geo_ast(ra, dec,
                          piS, xS0E_in, xS0N_in,
                          muSE_in, muSN_in, 
                          t0_in, u0_in, tE_in, 
                          piEE_in, piEN_in, t0par,
                          in_frame='helio',
                          murel_in='SL', murel_out='LS', 
                          coord_in='EN', coord_out='tb', plot=True):
    """
    NOTE: THIS IS NOT YET TESTED
    """
    day_to_yr = 365.25

    # UNITS: I think this is 1/days, NOT 1/years.
    # Need to convert, depending on input units.
    if type(ra) == str:
        ra = str(str(Angle(ra, unit = u.hourangle)))
    
    par_t0par = model.parallax_in_direction(ra, dec, t0par)
    dp_dt_t0par = model.dparallax_dt_in_direction(ra, dec, t0par)

    t0_out, _, _, _, _ = convert_helio_geo_phot(ra, dec, 
                                                t0_in, u0_in, tE_in, 
                                                piEE_in, piEN_in, t0par,
                                                in_frame,
                                                murel_in, murel_out, 
                                                coord_in, coord_out,
                                                plot=plot)

    if in_frame=='helio':
        muSE_out = muSE_in + piS * dp_dt_t0par[0] * day_to_yr
        muSN_out = muSN_in + piS * dp_dt_t0par[1] * day_to_yr
        xS0E_out = xS0E_in + muSE_in * (t0_out - t0_in) + piS * par_t0par[0]
        xS0N_out = xS0N_in + muSN_in * (t0_out - t0_in) + piS * par_t0par[1]
    elif in_frame=='geo':
        muSE_out = muSE_in - piS * dp_dt_t0par[0] * day_to_yr
        muSN_out = muSN_in - piS * dp_dt_t0par[1] * day_to_yr
        xS0E_out = xS0E_in + muSE_in * (t0_out - t0_in) - piS * par_t0par[0] - piS * (t0_out - t0_in) * dp_dt_t0par[0]
        xS0N_out = xS0N_in + muSN_in * (t0_out - t0_in) - piS * par_t0par[1] - piS * (t0_out - t0_in) * dp_dt_t0par[1]
    else:
        raise Exception('in_frame can only be "helio" or "geo"!')

    return xS0E_out, xS0N_out, muSE_out, muSN_out


def _check_input_convert_helio_geo_phot(ra, dec, 
                                        t0_in, u0_in, tE_in, 
                                        piEE_in, piEN_in, t0par,
                                        in_frame,
                                        murel_in, murel_out, 
                                        coord_in, coord_out,
                                        plot):

    var_str = ['ra', 'dec']
    for vv, var in enumerate([ra, dec]):
        if not isinstance(var, (int, float, str)):
            raise Exception('{0} ({1}) must be an integer, float, or string.'.format(var_str[vv], var))

    var_str = ['t0_in', 'u0_in', 'tE_in', 'piEE_in', 'piEN_in']
    array_like = 5 * [True]
    for vv, var in enumerate([t0_in, u0_in, tE_in, piEE_in, piEN_in]):
        if not hasattr(var, '__len__'):
            array_like[vv] = False
            if not isinstance(var, (int, float)):
                raise Exception('{0} ({1}) must be an integer, float, or array-like.'.format(var_str[vv], var))

    if np.sum(array_like) > 1:
        lens = [len([t0_in, u0_in, tE_in, piEE_in, piEN_in][i]) for i, x in enumerate(array_like) if x]
        if len(np.unique(lens)) > 1:
            raise Exception('t0_in, u0_in, tE_in, piEE_in, piEN_in must all be the same length, floats, or integers.')

    if not isinstance(t0par, (int, float)):
        raise Exception('t0par ({0}) must be either an integer or float.'.format(in_frame))
    
    if in_frame != 'helio' and in_frame != 'geo':
        raise Exception('in_frame ({0}) must be either "helio" or "geo".'.format(in_frame))

    var_str = ['murel_in', 'murel_out']
    for vv, var in enumerate([murel_in, murel_out]):
        if var != 'SL' and var != 'LS':
            raise Exception('{0} ({1}) must be "SL" or "LS".'.format(var_str[vv], in_frame))

    var_str = ['coord_in', 'coord_out']
    for vv, var in enumerate([coord_in, coord_out]):
        if var != 'EN' and var != 'tb':
            raise Exception('{0} ({1}) must be "EN" or "tb".'.format(var_str[vv], in_frame))

    if not isinstance(plot, bool):
        raise Exception('plot ({0}) must be a boolean.'.format(plot))


def convert_helio_geo_phot(ra, dec, 
                           t0_in, u0_in, tE_in, 
                           piEE_in, piEN_in, t0par,
                           in_frame='helio',
                           murel_in='SL', murel_out='LS', 
                           coord_in='EN', coord_out='tb',
                           plot=True):
    """
    Convert between heliocentric and geocentric-projected parameters.
    This converts only the subset of parameters in photometry fits
    (t0, u0, tE, piEE, piEN).

    The core conversion is assuming that source-lens, using the EN
    coordinate convention.

    ra, dec : str, float, or int
        Equatorial coordinates.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0_in : float (MJD)

    tE_in : float (days)

    piEE_in, piEN_in : float

    t0par : float (MJD)

    in_frame : 'helio' or 'geo'
        'helio' if we're converting from helio to geo.
        'geo' if we're converting from geo to helio.

    murel_in : 'SL' or 'LS'
        source-lens or lens-source for relative frame.

    murel_out : 'SL' or 'LS'
        source-lens or lens-source for relative frame.

    coord_in : 'EN' or 'tb'
        Use fixed on-sky coordinate system (Lu) or right-handed
        system based on murel and minimum separation (Gould)

    coord_out : 'EN' or 'tb'
        Use fixed on-sky coordinate system (Lu) or right-handed
        system based on murel and minimum separation (Gould)
    """
    # Check inputs.
    _check_input_convert_helio_geo_phot(ra, dec, 
                                        t0_in, u0_in, tE_in, 
                                        piEE_in, piEN_in, t0par,
                                        in_frame,
                                        murel_in, murel_out, 
                                        coord_in, coord_out,
                                        plot)
    
    if type(ra) == str:
        ra = str(str(Angle(ra, unit = u.hourangle)))

    # Flip from LS to SL as needed (conversion equations assume SL)
    if murel_in=='LS':
        piEE_in *= -1
        piEN_in *= -1

    #####
    # Calculate piEE, piEN, and tE. 
    #####
    piEE_out, piEN_out, tE_out = convert_piEvec_tE(ra, dec, t0par, 
                                                   piEE_in, piEN_in, tE_in, 
                                                   in_frame=in_frame)

    piE = np.hypot(piEE_in, piEN_in)

    #####
    # Calculate u0 and t0.
    #####
    # Define tauhat vector (same direction as piE vector).
    tauhatE_in = piEE_in/piE
    tauhatN_in = piEN_in/piE
    tauhatE_out = piEE_out/piE
    tauhatN_out = piEN_out/piE

    # Define the u0hat vector, which is orthogonal to the tauhat vector.
    # It is the direction of source-lens separation. Note that this is NOT 
    # always the same direction as Gould's beta vector (they are sometimes
    # antiparallel).
    if coord_in=='EN':
        try:
            # Handle conversion of single values.
            if np.sign(u0_in * piEN_in) < 0:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
            else:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
        except:
            # Handle conversions with array-like inputs.
            _u0hatE_in = -tauhatN_in
            _u0hatN_in = tauhatE_in
            u0hatE_in = np.where(np.sign(u0_in * piEN_in) < 0, _u0hatE_in, -_u0hatE_in)
            u0hatN_in = np.where(np.sign(u0_in * piEN_in) < 0, _u0hatN_in, -_u0hatN_in)

    elif coord_in=='tb':
        try:
            # Handle conversion of single values.
            if np.sign(u0_in) > 0:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
            else:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
        except:
            # Handle conversions with array-like inputs.
            _u0hatE_in = tauhatN_in
            _u0hatN_in = -tauhatE_in
            u0hatE_in = np.where(np.sign(u0_in) > 0, _u0hatE_in, -_u0hatE_in)
            u0hatN_in = np.where(np.sign(u0_in) > 0, _u0hatN_in, -_u0hatN_in)

    else:
        raise Exception('coord_in can only be "EN" or "tb"!')

    # Calculate t0 and u0 vector.
    t0_out, u0vec_out = convert_u0vec_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                         tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                         tauhatE_out, tauhatN_out,
                                         in_frame=in_frame)

    # Now get u0 (the scalar) and its associated sign from u0 vector.
    try:
        # Handle conversion of single values.
        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    except:
        # Handle conversions with array-like inputs.
        u0_out = np.zeros(len(t0_in))
        _u0_out = np.hypot(u0vec_out[:,0], u0vec_out[:,1])
        u0_out = np.where(u0vec_out[:,0] > 0, _u0_out, -_u0_out)

    if plot:
        #####
        # Plot conversion diagrams. 
        #####
        # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
        par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)
        
        tau_in = (t0par - t0_in)/tE_in
        tau_out = (t0par - t0_out)/tE_out
        
        vec_u0_in = [np.abs(u0_in)*u0hatE_in, 
                     np.abs(u0_in)*u0hatN_in]
        vec_tau_in = [tau_in*tauhatE_in,
                      tau_in*tauhatN_in]
        vec_u0_out = u0vec_out
        vec_tau_out = [tau_out*tauhatE_out,
                       tau_out*tauhatN_out]
        vec_par = [par_t0par[0],
                   par_t0par[1]]
        
        plot_conversion_diagram(vec_u0_in, vec_tau_in, vec_u0_out, vec_tau_out, piE, vec_par, in_frame,
                                t0par, t0_in, u0_in, tE_in, piEE_in, piEN_in,
                                t0_out, u0_out, tE_out, piEE_out, piEN_out)

    # Transform from tb to EN as needed (so user gets back what they expect).
    if coord_out=='tb':
        try:
            # Handle conversion of single values.
            x = np.sign(np.cross(np.array([tauhatE_out, tauhatN_out]), u0vec_out))
            if x > 0:
                u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
            else:
                u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        except:
            # Handle conversions with array-like inputs.
            x = np.sign(tauhatE_out * u0vec_out[1] - tauhatN_out * u0vec_out[0])
            u0_out = np.zeros(len(t0_in))
            _u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
            u0_out = np.where(x < 0, _u0_out, -_u0_out)

    # Flip from LS to SL as needed (so user gets back what they expect).
    if murel_out=='LS':
        piEE_out *= -1
        piEN_out *= -1

    return t0_out, u0_out, tE_out, piEE_out, piEN_out


def convert_u0vec_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE,
                     tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                     tauhatE_out, tauhatN_out,
                     in_frame='helio'):
    # FIXME: Fix the broadcasting stuff using the check above for the lenght
    # don't hardcode as t0_in.
    """
    *** PROPER MOTIONS ARE DEFINED AS SOURCE - LENS ***
    *** COORDINATE SYSTEM IS ON-SKY (NOT TAU-BETA) ***

    tauhat_in, tauhat_out. 
    u0hat_in, u0hat_out.

    VECTORS ARE ARRAYS DEFINED AS [E, N].

    """
    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    # Get dp_dt_t0par in 1/days.
    par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)

    # NOTE: dp_dt_t0par doesn't seem quite as good as calculating it from tauhat/tE/piE...
    # not sure why this is....
    # dp_dt_t0par = model.dparallax_dt_in_direction(ra, dec, np.array([t0par])).reshape(2,)/365.25
        
    # Calculate a bunch of values we need to get u0 and t0.
    tau_in = (t0par - t0_in)/tE_in
    u0vec_in = np.abs(u0_in) * np.array([u0hatE_in, u0hatN_in])
    try:
        tauvec_in = tau_in * np.array([tauhatE_in, tauhatN_in])
    except:
        tauvec_in = tau_in[:, np.newaxis] * np.array([tauhatE_in, tauhatN_in])
    uvec_in = u0vec_in + tauvec_in

    # Get direction of tauhat_out and u0hat_out.
    tauhat_out = np.array([tauhatE_out, tauhatN_out])
    tauhat_in = np.array([tauhatE_in, tauhatN_in])
    
    # Actually get u0 and t0.
    if in_frame=='helio':
        try:
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par)
#            u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out - piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

        except:
            par_t0par = np.tile(par_t0par,(len(t0_in),1)).T
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            _vec = u0vec_in - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par
            t0_out = t0_in - tE_out * np.sum(tauhat_out* _vec, axis=0)
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out - piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])


    elif in_frame=='geo':
        try:
            dp_dt_t0par = -1*((tauhat_in/tE_in) - (tauhat_out/tE_out))/piE
            t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par)
        #        u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out + piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        except:
            par_t0par = np.tile(par_t0par,(len(t0_in),1)).T
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            _vec = u0vec_in + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par
            t0_out = t0_in - tE_out * np.sum(tauhat_out* _vec, axis=0)
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out + piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

    else:
        raise Exception('in_frame can only be "helio" or "geo"!')

    try:
        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    except:
        u0_out = np.zeros(len(t0_out))
        _u0_out = np.hypot(u0vec_out[:,0], u0vec_out[:,1])
        u0_out = np.where(u0vec_out[:,0] > 0, _u0_out, -_u0_out)

    return t0_out, u0vec_out


def convert_piEvec_tE(ra, dec, t0par, 
                      piEE_in, piEN_in, tE_in, 
                      in_frame='helio'):     
    """
    Convert the values of piE vector and tE between the
    heliocentric and geoprojected frame.
    
    !!! NOTE: INPUT AND OUTPUT RELATIVE PROPER MOTION
    (AND HENCE piE VECTOR) ARE DEFINED TO BE SOURCE - LENS !!!

    If you want lens-source output, flip piEE and piEN by -1.

    Parameters
    ----------
    in_frame : str
        'helio' or 'geo'
        If 'helio', convert to helio to geo.
        If 'geo', convert to geo to helio.

    ra, dec : str, float, or int
        Equatorial coordinates.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0par : float
        Reference time for the geocentric-projected frame value in MJD.

    piEE_in, piEN_in, tE_in : float
        piEE, piEN, and tE of the event in the frame passed 
        in for in_frame (i.e. either helio or geo)

    Return
    ------
    piEE_out, piEN_out, tE_out : float
        piEE, piEN, and tE converted to the frame that was 
        NOT passed in for in_frame (i.e. in_frame='helio' 
        will return these values in 'geo', and vice versa)
    """
    if in_frame not in ['helio', 'geo']:
        raise Exception('in_frame must be either helio or geo.')

    # Convert from AU/day to km/s.
    au_day_to_km_s = 1731.45683

    # Magnitude of the microlensing parallax piE
    # and magnitude squared piE2.
    piE = np.hypot(piEE_in, piEN_in)
    piE2 = piEE_in**2 + piEN_in**2

    # vtilde (projected velocity) in units of AU/day.
    vtildeN_in = piEN_in/(tE_in * piE2)
    vtildeE_in = piEE_in/(tE_in * piE2) 
    
    # Convert vtilde_in from AU/day to km/s.
    vtildeN_in *= au_day_to_km_s
    vtildeE_in *= au_day_to_km_s

    # Get Earth's instantaneous velocity at t0par 
    # to get heliocentric velocity as projected on 
    # sky (in km/s)
    v_Earth_perp_E, v_Earth_perp_N = v_Earth_proj(ra, dec, t0par)

    # Convert vtilde.
    if in_frame=='helio':
        vtildeN_out = -vtildeN_in - v_Earth_perp_N
        vtildeE_out = -vtildeE_in - v_Earth_perp_E
    elif in_frame=='geo':
        vtildeN_out = -vtildeN_in + v_Earth_perp_N
        vtildeE_out = -vtildeE_in + v_Earth_perp_E
    else:
        raise Exception('in_frame can only be "helio" or "geo"!')

    # Convert piEE, piEN, and tE
    vtilde_in = np.hypot(vtildeE_in, vtildeN_in)
    vtilde_out = np.hypot(vtildeE_out, vtildeN_out)
    e_out = -vtildeE_out/vtilde_out
    n_out = -vtildeN_out/vtilde_out

    piEE_out = piE*e_out 
    piEN_out = piE*n_out 
    tE_out = (vtilde_in/vtilde_out) * tE_in
    
    return piEE_out, piEN_out, tE_out


def v_Earth_proj(ra, dec, mjd):
    """
    !!! This is the Earth-Sun vector direction. 
    (This is the opposite of how we define the parallax vector.)
    Returns in units of km/s.

    Based on MulensModel code calculation.
    (I just took out all the object oriented stuff.)
    
    ra, dec : str, float, or int
        Equatorial coordinates.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

        If float or int, needs to be in degrees.
        
    mjd : float
        Time in MJD.
    """
    # Convert from AU/day to km/s.
    au_day_to_km_s = 1731.45683

    # Make this check dec too.
    if type(ra) == str:
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    elif ((type(ra) == float) or (type(ra) == int)):
        coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    else:
        raise Exception('ra and dec must be either strings or int/floats.')

    direction = coord.cartesian.xyz.value    
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction)/np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected)/np.linalg.norm(np.cross(direction, _east_projected))

    jd = mjd+2400000.5
    time = Time(jd, format='jd', scale='tdb')
    (jd1, jd2) = get_jd12(time, 'tdb')
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2) # this is earth-sun
    # erfa returns things in a weird format... 
    # recast results into numpy array so we can use broadcasting.
    try:
        nn = earth_pv_bary.shape[0]
        earth_pv_bary_arr = np.empty((nn, 3))
        for ii in np.arange(nn):
            earth_pv_bary_arr[ii] = earth_pv_bary[ii][1]
        velocity = earth_pv_bary_arr * au_day_to_km_s
    except:
        velocity = np.asarray(earth_pv_bary[1]) * au_day_to_km_s

    v_Earth_perp_N = np.dot(velocity, _north_projected)
    v_Earth_perp_E = np.dot(velocity, _east_projected)

    return v_Earth_perp_E, v_Earth_perp_N


def plot_conversion_diagram(vec_u0_in, vec_tau_in, vec_u0_out, vec_tau_out, 
                            piE, vec_par, in_frame,
                            t0par, t0_in, u0_in, tE_in, piEE_in, piEN_in,
                            t0_out, u0_out, tE_out, piEE_out, piEN_out):
    """
    The parallax vector direction is always Earth to Sun (blue to red).
    All the input values are in the heliocentric frame, following the
    source-lens and East-North coordinate conventions.
    """
    #####
    # Figure in Lu convention (S-L frame, E-N coord)
    ####
    fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
    plt.clf()
    fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
    plt.subplots_adjust(left=0.15, right=0.7)

    if in_frame == 'helio':
        label_in = 'Src (Hel)'
        label_out = 'Src (Geo $t_r$)'
        color_in = 'red'
        color_out = 'blue'
    if in_frame == 'geo':
        label_in = 'Src (Geo $t_r$)'
        label_out = 'Src (Hel)'
        color_in = 'blue'
        color_out = 'red'

    # u0 vector
    ax.annotate('', xy=(0,0), xycoords='data',
                xytext=(vec_u0_out[0], vec_u0_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15))

    ax.annotate('', xy=(0,0), xycoords='data',
                xytext=(vec_u0_in[0], vec_u0_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15))

    # tau vector (starting at u0 vector)
    ax.annotate('', xy=(vec_u0_out[0], vec_u0_out[1]), xycoords='data',
                xytext=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15, shrinkA=0))

    ax.annotate('', xy=(vec_u0_in[0], vec_u0_in[1]), xycoords='data',
                xytext=(vec_u0_in[0] + vec_tau_in[0], vec_u0_in[1] + vec_tau_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15, shrinkA=0))

    # Plot the lens (i.e. origin)
    ax.plot(0, 0, 'o', ms=8, mec='k', color='k', label='Lens')

    # Plot the source (both helio and geo proj positions at t0)
    ax.plot(vec_u0_in[0], vec_u0_in[1], 'o', ms=8, mec=color_in, color='yellow', mew=2, label=label_in)
    ax.plot(vec_u0_out[0], vec_u0_out[1], 'o', ms=8, mec=color_out, color='yellow', mew=2, label=label_out)

    # Parallax vector
    if in_frame=='helio':
        ax.annotate('', xy=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), xycoords='data',
                    xytext=(vec_u0_out[0] + vec_tau_out[0] + vec_par[0]*piE, vec_u0_out[1] + vec_tau_out[1] + vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))
    else:
        ax.annotate('', xy=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), xycoords='data',
                    xytext=(vec_u0_out[0] + vec_tau_out[0] - vec_par[0]*piE, vec_u0_out[1] + vec_tau_out[1] - vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))

#    # Not sure why these next two lines are totally necessary...
    ax.plot(vec_u0_in[0] + vec_tau_in[0], vec_u0_in[1] + vec_tau_in[1], 'o', ms=0.001, color=color_in)
    ax.plot(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1], 'o', ms=0.001, color=color_out)

    ax.legend()
    ax.axhline(y=0, ls=':', color='gray')
    ax.axvline(x=0, ls=':', color='gray')
    ax.set_xlabel('$u_E$')
    ax.set_ylabel('$u_N$')
    ax.axis('equal')
    ax.invert_xaxis()
    ax.set_title('Lu convention (S-L frame, E-N coord)')
    
    # Text panel listing parameters.
    tleft = 0.75
    ttop = 0.8
    ttstep = 0.05

    fig.text(tleft, ttop - -1*ttstep, '$t_r$ = {0:.1f} MJD'.format(t0par), fontsize=12)

    # Input and output are helio, Lu convention. So parameters are as reported.
    if in_frame == 'helio':
        fig.text(tleft, ttop - 1*ttstep, 'Helio', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(u0_in), fontsize=12)
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        
        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(piEN_out), fontsize=12)
        fig.text(tleft, ttop - 15*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)

    if in_frame == 'geo':
        fig.text(tleft, ttop - 1*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Helio', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        # Input is Gould geo, so need to fix those to play nice with our vectors in Lu helio. 
        if np.sign(u0_in * piEN_in) > 0:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_in)), fontsize=12)
        else:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_in)), fontsize=12)

        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        
        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(piEN_out), fontsize=12)
        fig.text(tleft, ttop - 15*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)

    plt.show()
    plt.pause(1)

    #####
    # Figure in Gould convention (L-S frame, tau-beta coord)
    ####
    fig, ax = plt.subplots(1, 2, num=4, figsize=(12,6), gridspec_kw={'width_ratios': [2.5, 1]})
    plt.clf()
    fig, ax = plt.subplots(1, 2, num=4, figsize=(12,6), gridspec_kw={'width_ratios': [2.5, 1]})
    plt.subplots_adjust(left=0.15*3/4, right=0.78, wspace=0.3)

    # FIXME: is the parallax vector supposed to be the same direction
    # in Gould frame or antiparallel to the one in Lu???
    if in_frame == 'helio':
        label_in = 'Lens (Hel)'
        label_out = 'Lens (Geo $t_r$)'
        color_in = 'red'
        color_out = 'blue'
    if in_frame == 'geo':
        label_in = 'Lens (Geo $t_r$)'
        label_out = 'Lens (Hel)'
        color_in = 'blue'
        color_out = 'red'

    # u0 vector
    ax[0].annotate('', xy=(0,0), xycoords='data',
                xytext=(-vec_u0_out[0], -vec_u0_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15))

    ax[0].annotate('', xy=(0,0), xycoords='data',
                xytext=(-vec_u0_in[0], -vec_u0_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15))

    # tau vector (starting at u0 vector)
    ax[0].annotate('', xy=(-vec_u0_out[0], -vec_u0_out[1]), xycoords='data',
                   xytext=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), textcoords='data',
                   ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15, shrinkA=0))

    ax[0].annotate('', xy=(-vec_u0_in[0], -vec_u0_in[1]), xycoords='data',
                   xytext=(-vec_u0_in[0] - vec_tau_in[0], -vec_u0_in[1] - vec_tau_in[1]), textcoords='data',
                   ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15, shrinkA=0))

    # Plot the source (i.e. origin)
    ax[0].plot(0, 0, 'o', ms=8, mec='k', color='yellow', label='Source')

    # Plot the lens (both helio and geo proj positions at t0)
    ax[0].plot(-vec_u0_in[0], -vec_u0_in[1], 'o', ms=8, mec=color_in, mew=2, color='k', label=label_in)
    ax[0].plot(-vec_u0_out[0], -vec_u0_out[1], 'o', ms=8, mec=color_out, mew=2, color='k', label=label_out)

    # Parallax vector
    if in_frame=='helio':
        ax[0].annotate('', xy=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), xycoords='data',
                    xytext=(-vec_u0_out[0] - vec_tau_out[0] - vec_par[0]*piE, -vec_u0_out[1] - vec_tau_out[1] - vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))
    else:
        ax[0].annotate('', xy=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), xycoords='data',
                    xytext=(-vec_u0_out[0] - vec_tau_out[0] + vec_par[0]*piE, vec_u0_out[1] - vec_tau_out[1] + vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))

    # Not sure why these next two lines are totally necessary...
    ax[0].plot(-vec_u0_in[0] - vec_tau_in[0], -vec_u0_in[1] - vec_tau_in[1], 'o', ms=0.001, color=color_in)
    ax[0].plot(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1], 'o', ms=0.001, color=color_out)

    ax[0].legend()
    ax[0].axhline(y=0, ls=':', color='gray')
    ax[0].axvline(x=0, ls=':', color='gray')
    ax[0].set_xlabel('$u_E$')
    ax[0].set_ylabel('$u_N$')
    ax[0].axis('equal')
    ax[0].invert_xaxis()
    ax[0].set_title(r'Gould convention (L-S frame, $\tau$-$\beta$ coord)')

    q_tau_out = ax[1].quiver(0, 0, -piEE_out/piE, -piEN_out/piE, 
                           color=color_out, scale_units='xy', angles='xy', scale=1)
    
    q_tau_in = ax[1].quiver(0, 0, -piEE_in/piE, -piEN_in/piE, 
                          color=color_in, scale_units='xy', angles='xy', scale=1)

    q_u0_out = ax[1].quiver(0, 0, -piEN_out/piE, piEE_out/piE, 
                          color=color_out, scale_units='xy', angles='xy', scale=1, alpha=0.5, width=0.02)
    
    q_u0_in = ax[1].quiver(0, 0, -piEN_in/piE, piEE_in/piE, 
                         color=color_in, scale_units='xy', angles='xy', scale=1, alpha=0.5, width=0.02)
    
    if in_frame == 'helio':
        ax[1].quiverkey(q_tau_out, 1, -2.0, 0.3, r'$\hat{\tau}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_tau_in, 1, -2.4, 0.3, r'$\hat{\tau}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_out, 0, -2.0, 0.3, r'$\hat{\beta}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_in, 0, -2.4, 0.3, r'$\hat{\beta}_H$', coordinates='data', labelpos='E', labelsep=0.5)
    else:
        ax[1].quiverkey(q_tau_out, 1, -2.0, 0.3, r'$\hat{\tau}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_tau_in, 1, -2.4, 0.3, r'$\hat{\tau}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_out, 0, -2.0, 0.3, r'$\hat{\beta}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_in, 0, -2.4, 0.3, r'$\hat{\beta}_G$', coordinates='data', labelpos='E', labelsep=0.5)

    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('$E$')
    ax[1].set_ylabel('$N$')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].invert_xaxis()
    ax[1].set_aspect('equal', 'box')

    # Text panel listing parameters.
    tleft = 0.75 * 1.08
    ttop = 0.8
    ttstep = 0.05

    fig.text(tleft, ttop - -1*ttstep, '$t_r$ = {0:.1f} MJD'.format(t0par), fontsize=12)

    if in_frame == 'helio':
        fig.text(tleft, ttop - 1*ttstep, 'Helio', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        # Input is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_in * piEN_in) > 0:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_in)), fontsize=12)
        else:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_in)), fontsize=12)
            
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        
        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        # Output is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_out * piEN_out) > 0:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_out)), fontsize=12)
        else:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_out)), fontsize=12)

        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_out), fontsize=12)
        fig.text(tleft, ttop - 15*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)
    
    if in_frame == 'geo':
        fig.text(tleft, ttop - 1*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Helio', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(u0_in), fontsize=12)           
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        
        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f}'.format(t0_out), fontsize=12)
        # Output is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_out * piEN_out) > 0:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_out)), fontsize=12)
        else:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_out)), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f}'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_out), fontsize=12)
        fig.text(tleft, ttop - 15*ttstep, '$\pi_{{E,E}}/\pi_{{E,N}}$ = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)

    plt.show()
    plt.pause(1)


def convert_u0_t0_psbl(t0_in, u0_x_in, u0_y_in, tE, theta_E, 
                       q, phi, sep, mu_rel_x, mu_rel_y,
                       coords_in=None, coords_out=None, d_mas = None):
    """
    Converts time of closest approach (t0) and 
    vector distance of closest approach (u0) for PSBL events.
    The default coordinate system transformations supported are:
    Geometric Midpoint <--> Primary Center
    Geometric Midpoint <--> Center of Mass Center
    
    You can also specify your own transformation by inputing a distance
    along binary axis d_mas. BE AWARE OF THE SIGN OF d_mas.
    If output coord center is closer to primary than input d > 0.
    If output coord center is closer to secondary than input d < 0.
    
    Parameters
    -----------
    t0_in : float
        Time of closest approach between source and lens specified in coords_in in days.
    u0_x_in : float
        u0 in specified coordinate frame in coords_in in x direction.
        x direction is coordinate independent, but standard is East.
    u0_y_in : float
        u0 in specified coordinate frame in coords_in in y direction.
        y direction is coordinate independent, but standard is North.
    tE : float
        Characteristic timescale of microlensing event in days.
    theta_E : float
        Characterisitc lengthscale of microlensing event in mas.
    q : float
        Ratio between M_2/M_1.
    phi : float
        Angle between mu_rel and the binary axis in radians.
    sep : float
        Separation between primary and secondary of binary in mas.
    mu_rel_x : float
        Relative proper motion between lens and source in mas/yr in x direction.
        x direction is coordinate independent, but standard is East.
    mu_rel_y : float
        Relative proper motion between lens and source in mas/yr in y direction.
        y direction is coordinate independent, but standard is North.
    
    Optional Parameters
    --------------------
    coords_in : string or None
        Input coordinate system. 
        Must be 'geom_mid', 'prim_center', or 'COM'.
        Default is None.
    coords_out : string or None
        Output coordinate system. 
        Must be 'geom_mid', 'prim_center', or 'COM'.
        Default is None.
    d_mas : float or None
        Distance along binary axis in units of mas.
        BE AWARE OF SIGN! 
        If output coord center is closer to primary than input d > 0.
        If output coord center is closer to secondary than input d < 0.
    
    Outputs
    ---------
    u0_x_out : float
        u0 in specified coordinate frame in coords_out.
        x direction is coordinate independent, but standard is East.
    u0_y_out : float
        u0 in specified coordinate frame in coords_in.
        y direction is coordinate independent, but standard is North.
    t0_out : float
        Time of closest approach between source and lens specified in coords_out in days.
    """
    
    if d_mas is not None:
        d = d_mas/theta_E
        sign = 1
    else:
        d = None
    
    if coords_in is None and d is None:
        raise Exception('Must specify coord system or distance along binary axis')
    
    if coords_in is not None and d is not None:
        raise Exception('Can only specify default coordinate transform or distance along binary axis, not both')
    
    # Uses one of the default coord transforms if a distance is not specified
    if d is None:
        valid_coords = ['geom_mid', 'prim_center', 'COM']
        if coords_in not in valid_coords or coords_out not in valid_coords:
            raise Exception('coord_in and coord_out must be one of: {}'.format(valid_coords))
    
        q_prime = (1 - q)/(2*(1 + q))
        if coords_in == 'geom_mid' and coords_out == 'prim_center':
            d = sep/(2*theta_E)
            sign = -1
        elif coords_in == 'prim_center' and coords_out == 'geom_mid':
            d = -sep/(2*theta_E)
            sign = 1
        elif coords_in == 'geom_mid' and coords_out == 'COM':
            d = sep*q_prime/theta_E
            sign = -1
            # If q > 1, then COM is closer to secondary than primary
            # so the transformation flips
            if q > 1:
                sign *= -1
        elif coords_in == 'COM' and coords_out == 'geom_mid':
            d = sep*q_prime/theta_E
            sign = 1
            # If q > 1, then COM is closer to secondary than primary
            # so the transformation flips
            if q > 1:
                sign *= -1
    
    if u0_x_in == 0:
        u0_x_in_hat = 0
    else:
        u0_x_in_hat = u0_x_in/np.abs(u0_x_in)
    
    if u0_y_in == 0:
        u0_y_in_hat = 0
    else:
        u0_y_in_hat = u0_y_in/np.abs(u0_y_in)
    
    z = np.array([0, 0, 1])
    mu_rel_arr = np.array([mu_rel_x, mu_rel_y, 0])
    u0_in_arr = np.array([u0_x_in, u0_y_in, 0])
    mu_rel_arr_hat = mu_rel_arr/np.sqrt(mu_rel_arr[0]**2 + mu_rel_arr[1]**2)
    u0_in_arr_hat = u0_in_arr/np.sqrt(u0_in_arr[0]**2 + u0_in_arr[1]**2)
    
    C = np.dot(np.cross(mu_rel_arr_hat, u0_in_arr_hat), z)

    u0_x_out = u0_x_in + sign*C*d*np.sin(phi)*u0_x_in_hat
    u0_y_out = u0_y_in + sign*C*d*np.sin(phi)*u0_y_in_hat
    
    t0_out = t0_in + sign*tE*d*np.cos(phi)
    
    return u0_x_out, u0_y_out, t0_out
