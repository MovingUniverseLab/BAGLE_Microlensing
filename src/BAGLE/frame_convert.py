from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np 
from astropy.time import Time
from astropy.coordinates.builtin_frames.utils import get_jd12
import erfa
import matplotlib.pyplot as plt
from microlens.jlu import model


# Note: this hasn't been checked/tested yet.
def convert_helio_geo_ast(ra, dec,
                          piS, xS0E_in, xS0N_in,
                          muSE_in, muSN_in, 
                          t0_in, u0_in, tE_in, 
                          piEE_in, piEN_in, t0par,
                          in_frame='helio',
                          murel_in='SL', murel_out='LS', 
                          coord_in='EN', coord_out='tb'):

    day_to_yr = 365.25

    # UNITS: I think this is 1/days, NOT 1/years.
    # Need to convert, depending on input units. 
    par_t0par = model.parallax_in_direction(ra, dec, t0par)
    dp_dt_t0par = model.dparallax_dt_in_direction(ra, dec, t0par)

    t0_out, _, _, _, _ = convert_helio_geo_phot(ra, dec, 
                                                t0_in, u0_in, tE_in, 
                                                piEE_in, piEN_in, t0par,
                                                in_frame,
                                                murel_in, murel_out, 
                                                coord_in, coord_out)

    if in_frame=='helio':
        muSE_out = muSE_in + piS * dp_dt_t0par[0] * day_to_yr
        muSN_out = muSN_in + piS * dp_dt_t0par[1] * day_to_yr
        xS0E_out = xS0E_in + muSE_in * (t0_out - t0_in) + piS * par_t0par[0]
        xS0N_out = xS0N_in + muSN_in * (t0_out - t0_in) + piS * par_t0par[1]
    else:
        muSE_out = muSE_in - piS * dp_dt_t0par[0] * day_to_yr
        muSN_out = muSN_in - piS * dp_dt_t0par[1] * day_to_yr
        xS0E_out = xS0E_in + muSE_in * (t0_out - t0_in) - piS * par_t0par[0] - piS * (t0_out - t0_in) * dp_dt_t0par[0]
        xS0N_out = xS0N_in + muSN_in * (t0_out - t0_in) - piS * par_t0par[1] - piS * (t0_out - t0_in) * dp_dt_t0par[1]

    return xS0E_out, xS0N_out, muSE_out, muSN_out



def convert_helio_geo_phot(ra, dec, 
                           t0_in, u0_in, tE_in, 
                           piEE_in, piEN_in, t0par,
                           in_frame='helio',
                           murel_in='SL', murel_out='LS', 
                           coord_in='EN', coord_out='tb',
                           plot=True):
    """
    Attempt at generalized code.
    Not sure if it's 100% right, but it does BAGLE to Mulens Model, 
    and vice versa, correctly.

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
#    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)

    # Flip from LS to SL as needed 
    # (conversion equations assume SL)
    if murel_in=='LS':
        piEE_in *= -1
        piEN_in *= -1

    #####
    # Calculate piEE, piEN, and tE. 
    #####
    piEE_out, piEN_out, tE_out = convert_piE_tE(ra, dec, t0par, 
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

    try:
        nn = len(t0par)
        u0hatE_in = np.empty(nn)
        u0hatN_in = np.empty(nn)
        u0hatE_out = np.empty(nn)
        u0hatN_out = np.empty(nn)

        if coord_in=='EN':
            u0hatE_in = np.where(np.sign(u0_in * piEN_in) < 0, -tauhatN_in, tauhatN_in)
            u0hatN_in = np.where(np.sign(u0_in * piEN_in) < 0, tauhatE_in, -tauhatE_in)

        if coord_in=='tb':
            u0hatE_in = np.where(np.sign(u0_in) < 0, -tauhatN_in, tauhatN_in)
            u0hatN_in = np.where(np.sign(u0_in) < 0, tauhatE_in, -tauhatE_in)
    
    except:
    # Define the u0hat vector, which is orthogonal to the tauhat vector.
    # It is the source-lens separation vector.
        if coord_in=='EN':
            if np.sign(u0_in * piEN_in) < 0:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
            else:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in

        if coord_in=='tb':
            if np.sign(u0_in) > 0:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
            else:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in

    # out_frame == 'helio':
    # why doesn't this matter on u0 sign?
    # why is it piEN_in and not piEN_out? 
    # is this an edge case?
    if in_frame=='geo':
        if piEN_in > 0:
            u0hatE_out = tauhatN_out
            u0hatN_out = -tauhatE_out
        else:
            u0hatE_out = -tauhatN_out
            u0hatN_out = tauhatE_out
    # out_frame = 'geo':
    else:
        u0hatE_out = tauhatN_out
        u0hatN_out = -tauhatE_out

    _t0_out, u0vec_out = convert_u0_t0_old(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                   tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                   tauhatE_out, tauhatN_out, # u0hatE_out, u0hatN_out,
                                   in_frame=in_frame)

    x = np.sign(np.cross(-np.array([tauhatE_out, tauhatN_out]), -u0vec_out))
    if x > 0:
        u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    else:
        u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

    print(_t0_out)
    print(u0_out)

    t0_out, u0_out = convert_u0_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                   tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                   tauhatE_out, tauhatN_out, u0hatE_out, u0hatN_out,
                                   in_frame=in_frame)

    print(t0_out)
    print(u0_out)


    # Undo flip from LS to SL as needed
    # (so user gets back what they expect).
    if murel_out=='LS':
        piEE_out *= -1
        piEN_out *= -1

    tau_in = (t0par - t0_in)/tE_in
    tau_out = (t0par - t0_out)/tE_out

    vec_u0_in = [np.abs(u0_in)*u0hatE_in, 
                np.abs(u0_in)*u0hatN_in]
    vec_tau_in = [tau_in*tauhatE_in,
                 tau_in*tauhatN_in]
#    vec_u0_out = [np.abs(u0_out)*u0hatE_out,
#                np.abs(u0_out)*u0hatN_out]
    vec_u0_out = u0vec_out
    vec_tau_out = [tau_out*tauhatE_out,
                 tau_out*tauhatN_out]
    vec_par = [par_t0par[0],
               par_t0par[1]]

    if plot:
#        fig, ax = plt.subplots(1, 2, num=2, figsize=(15,6))
#        plt.clf()
#        fig, ax = plt.subplots(1, 2, num=2, figsize=(15,6))
        fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
        plt.clf()
        fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
        plt.subplots_adjust(left=0.15, right=0.7)

        ax.plot([0, vec_u0_in[0]], 
                   [0, vec_u0_in[1]], color='red')
        ax.plot([vec_u0_in[0], vec_u0_in[0] + vec_tau_in[0]], 
                   [vec_u0_in[1], vec_u0_in[1] + vec_tau_in[1]], color='red')
        ax.plot([0, vec_u0_out[0]], 
                   [0, vec_u0_out[1]], color='blue')
        ax.plot([vec_u0_out[0], vec_u0_out[0] + vec_tau_out[0]], 
                   [vec_u0_out[1], vec_u0_out[1] + vec_tau_out[1]], color='blue')
        if in_frame=='helio':
            ax.plot([vec_u0_out[0] + vec_tau_out[0], vec_u0_out[0] + vec_tau_out[0] + vec_par[0]*piE], 
                       [vec_u0_out[1] + vec_tau_out[1], vec_u0_out[1] + vec_tau_out[1] + vec_par[1]*piE], color='k')
            ax.plot([vec_u0_in[0] + vec_tau_in[0], vec_u0_in[0] + vec_tau_in[0] - vec_par[0]*piE], 
                       [vec_u0_in[1] + vec_tau_in[1], vec_u0_in[1] + vec_tau_in[1] - vec_par[1]*piE], color='k')
        else:
            ax.plot([vec_u0_out[0] + vec_tau_out[0], vec_u0_out[0] + vec_tau_out[0] - vec_par[0]*piE], 
                       [vec_u0_out[1] + vec_tau_out[1], vec_u0_out[1] + vec_tau_out[1] - vec_par[1]*piE], color='k')
            ax.plot([vec_u0_in[0] + vec_tau_in[0], vec_u0_in[0] + vec_tau_in[0] + vec_par[0]*piE], 
                       [vec_u0_in[1] + vec_tau_in[1], vec_u0_in[1] + vec_tau_in[1] + vec_par[1]*piE], color='k')
        ax.axhline(y=0, ls=':', color='gray')
        ax.axvline(x=0, ls=':', color='gray')
        ax.set_xlabel('$u_E$')
        ax.set_ylabel('$u_N$')
        ax.axis('equal')
        ax.invert_xaxis()
        ax.set_title('Red = in (H), Blue = out (G)')

#        ax[1].annotate(r'$\vec{P}$($t_{0,par}$)', xy=(0,0), xycoords='data',
#                       xytext=(vec_par[0]/np.hypot(vec_par[0], vec_par[1]), vec_par[1]/np.hypot(vec_par[0], vec_par[1])), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='k',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate(r'$\hat{\tau}_{rel}^{geo}$', xy=(0,0), xycoords='data',
#                       xytext=(tauhatE_out, tauhatN_out), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='blue',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate(r'$\hat{\tau}_{rel}^{hel}$', xy=(0,0), xycoords='data',
#                       xytext=(tauhatE_in, tauhatN_in), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='red',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate('$\hat{u}_0^{geo}$', xy=(0,0), xycoords='data',
#                       xytext=(u0hatE_out, u0hatN_out), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='blue',
#                                       lw=3.5,
#                                       ls='--'))
#        ax[1].annotate('$\hat{u}_0^{hel}$', xy=(0,0), xycoords='data',
#                       xytext=(u0hatE_in, u0hatN_in), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='red',
#                                       lw=3.5,
#                                       ls='--'))
#        ax[1].set_aspect('equal')
#        ax[1].set_xlabel('$u_E$')
#        ax[1].set_ylabel('$u_N$')
#        ax[1].axhline(y=0, ls=':', color='gray')
#        ax[1].axvline(x=0, ls=':', color='gray')
#        ax[1].set_xlim(-1.5, 1.5)
#        ax[1].set_ylim(-1.5, 1.5)
#        ax[1].invert_xaxis()
        
        tleft = 0.75
#        tleft = 0.8
        ttop = 0.8
        ttstep = 0.05
        fig.text(tleft, ttop - 0*ttstep, 't0_in = {0:.1f}'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 1*ttstep, 'u0_in = {0:.2f}'.format(u0_in), fontsize=12)
        fig.text(tleft, ttop - 2*ttstep, 'tE_in = {0:.1f}'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, 'piEE_in = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 4*ttstep, 'piEN_in = {0:.2f}'.format(piEN_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, 'piEE_in/piEN_in = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, 't0par = {0:.1f}'.format(t0par), fontsize=12)
        fig.text(tleft, ttop - 8*ttstep, 't0_out = {0:.1f}'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 9*ttstep, 'u0_out = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 10*ttstep, 'tE_out = {0:.1f}'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, 'piEE_out = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, 'piEN_out = {0:.2f}'.format(piEN_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, 'piEE_out/piEN_out = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)
        
        plt.show()
        plt.pause(1)
        import pdb
        pdb.set_trace()

    return t0_out, u0_out, tE_out, piEE_out, piEN_out


# Figure out how to write this conversion into something that takes arrays.
# Lu coord = 'EN', Gould coord = 'tb' (tau-beta)
def convert_helio_geo_phot_old(ra, dec, 
                           t0_in, u0_in, tE_in, 
                           piEE_in, piEN_in, t0par,
                           in_frame='helio',
                           murel_in='SL', murel_out='LS', 
                           coord_in='EN', coord_out='tb',
                           plot=True):
    """
    Attempt at generalized code.
    Not sure if it's 100% right, but it does BAGLE to Mulens Model, 
    and vice versa, correctly.

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
#    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)

    # Flip from LS to SL as needed 
    # (conversion equations assume SL)
    if murel_in=='LS':
        piEE_in *= -1
        piEN_in *= -1

    #####
    # Calculate piEE, piEN, and tE. 
    #####
    piEE_out, piEN_out, tE_out = convert_piE_tE(ra, dec, t0par, 
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

    try:
        nn = len(t0par)
        u0hatE_in = np.empty(nn)
        u0hatN_in = np.empty(nn)
        u0hatE_out = np.empty(nn)
        u0hatN_out = np.empty(nn)

        if coord_in=='EN':
            u0hatE_in = np.where(np.sign(u0_in * piEN_in) < 0, -tauhatN_in, tauhatN_in)
            u0hatN_in = np.where(np.sign(u0_in * piEN_in) < 0, tauhatE_in, -tauhatE_in)

        if coord_in=='tb':
            u0hatE_in = np.where(np.sign(u0_in) < 0, -tauhatN_in, tauhatN_in)
            u0hatN_in = np.where(np.sign(u0_in) < 0, tauhatE_in, -tauhatE_in)
    
    except:
    # Define the u0hat vector, which is orthogonal to the tauhat vector.
    # It is the source-lens separation vector.
        if coord_in=='EN':
            if np.sign(u0_in * piEN_in) < 0:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
            else:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in

        if coord_in=='tb':
            if np.sign(u0_in) > 0:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
            else:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in

    # out_frame == 'helio':
    # why doesn't this matter on u0 sign?
    # why is it piEN_in and not piEN_out? 
    # is this an edge case?
    if in_frame=='geo':
        if piEN_in > 0:
            u0hatE_out = tauhatN_out
            u0hatN_out = -tauhatE_out
        else:
            u0hatE_out = -tauhatN_out
            u0hatN_out = tauhatE_out
    # out_frame = 'geo':
    else:
        u0hatE_out = tauhatN_out
        u0hatN_out = -tauhatE_out

    _t0_out, u0vec_out = convert_u0_t0_old(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                   tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                   tauhatE_out, tauhatN_out, # u0hatE_out, u0hatN_out,
                                   in_frame=in_frame)

    x = np.sign(np.cross(-np.array([tauhatE_out, tauhatN_out]), -u0vec_out))
    if x > 0:
        u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    else:
        u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

    print(_t0_out)
    print(u0_out)

    t0_out, u0_out = convert_u0_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                   tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                   tauhatE_out, tauhatN_out, u0hatE_out, u0hatN_out,
                                   in_frame=in_frame)

    print(t0_out)
    print(u0_out)

    print('In frame: ', in_frame)
    print('Coord in: ', coord_in)
    print('u0 :', u0_in)
    print('u0_E :', u0hatE_in)
    print('tau x beta :', np.sign(np.cross([tauhatE_in, tauhatN_in], [u0hatE_in, u0hatN_in])))

    print('Coord out: ', coord_out)
    print('Out')
    print('u0 :', u0_out)
    print('u0_E :', u0hatE_out)
    print('tau x beta :', np.sign(np.cross([tauhatE_out, tauhatN_out], [u0hatE_out, u0hatN_out])))
    print('')


    # Undo flip from LS to SL as needed
    # (so user gets back what they expect).
    if murel_out=='LS':
        piEE_out *= -1
        piEN_out *= -1

    tau_in = (t0par - t0_in)/tE_in
    tau_out = (t0par - t0_out)/tE_out

    vec_u0_in = [np.abs(u0_in)*u0hatE_in, 
                np.abs(u0_in)*u0hatN_in]
    vec_tau_in = [tau_in*tauhatE_in,
                 tau_in*tauhatN_in]
#    vec_u0_out = [np.abs(u0_out)*u0hatE_out,
#                np.abs(u0_out)*u0hatN_out]
    vec_u0_out = u0vec_out
    vec_tau_out = [tau_out*tauhatE_out,
                 tau_out*tauhatN_out]
    vec_par = [par_t0par[0],
               par_t0par[1]]

    if plot:
#        fig, ax = plt.subplots(1, 2, num=2, figsize=(15,6))
#        plt.clf()
#        fig, ax = plt.subplots(1, 2, num=2, figsize=(15,6))
        fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
        plt.clf()
        fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
        plt.subplots_adjust(left=0.15, right=0.7)

        ax.plot([0, vec_u0_in[0]], 
                   [0, vec_u0_in[1]], color='red')
        ax.plot([vec_u0_in[0], vec_u0_in[0] + vec_tau_in[0]], 
                   [vec_u0_in[1], vec_u0_in[1] + vec_tau_in[1]], color='red')
        ax.plot([0, vec_u0_out[0]], 
                   [0, vec_u0_out[1]], color='blue')
        ax.plot([vec_u0_out[0], vec_u0_out[0] + vec_tau_out[0]], 
                   [vec_u0_out[1], vec_u0_out[1] + vec_tau_out[1]], color='blue')
        if in_frame=='helio':
            ax.plot([vec_u0_out[0] + vec_tau_out[0], vec_u0_out[0] + vec_tau_out[0] + vec_par[0]*piE], 
                       [vec_u0_out[1] + vec_tau_out[1], vec_u0_out[1] + vec_tau_out[1] + vec_par[1]*piE], color='k')
            ax.plot([vec_u0_in[0] + vec_tau_in[0], vec_u0_in[0] + vec_tau_in[0] - vec_par[0]*piE], 
                       [vec_u0_in[1] + vec_tau_in[1], vec_u0_in[1] + vec_tau_in[1] - vec_par[1]*piE], color='k')
        else:
            ax.plot([vec_u0_out[0] + vec_tau_out[0], vec_u0_out[0] + vec_tau_out[0] - vec_par[0]*piE], 
                       [vec_u0_out[1] + vec_tau_out[1], vec_u0_out[1] + vec_tau_out[1] - vec_par[1]*piE], color='k')
            ax.plot([vec_u0_in[0] + vec_tau_in[0], vec_u0_in[0] + vec_tau_in[0] + vec_par[0]*piE], 
                       [vec_u0_in[1] + vec_tau_in[1], vec_u0_in[1] + vec_tau_in[1] + vec_par[1]*piE], color='k')
        ax.axhline(y=0, ls=':', color='gray')
        ax.axvline(x=0, ls=':', color='gray')
        ax.set_xlabel('$u_E$')
        ax.set_ylabel('$u_N$')
        ax.axis('equal')
        ax.invert_xaxis()
        ax.set_title('Red = in (H), Blue = out (G)')

#        ax[1].annotate(r'$\vec{P}$($t_{0,par}$)', xy=(0,0), xycoords='data',
#                       xytext=(vec_par[0]/np.hypot(vec_par[0], vec_par[1]), vec_par[1]/np.hypot(vec_par[0], vec_par[1])), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='k',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate(r'$\hat{\tau}_{rel}^{geo}$', xy=(0,0), xycoords='data',
#                       xytext=(tauhatE_out, tauhatN_out), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='blue',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate(r'$\hat{\tau}_{rel}^{hel}$', xy=(0,0), xycoords='data',
#                       xytext=(tauhatE_in, tauhatN_in), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='red',
#                                       lw=3.5,
#                                       ls='-'))
#        ax[1].annotate('$\hat{u}_0^{geo}$', xy=(0,0), xycoords='data',
#                       xytext=(u0hatE_out, u0hatN_out), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='blue',
#                                       lw=3.5,
#                                       ls='--'))
#        ax[1].annotate('$\hat{u}_0^{hel}$', xy=(0,0), xycoords='data',
#                       xytext=(u0hatE_in, u0hatN_in), textcoords='data',
#                       ha='right',
#                       arrowprops=dict(arrowstyle= '<|-',
#                                       color='red',
#                                       lw=3.5,
#                                       ls='--'))
#        ax[1].set_aspect('equal')
#        ax[1].set_xlabel('$u_E$')
#        ax[1].set_ylabel('$u_N$')
#        ax[1].axhline(y=0, ls=':', color='gray')
#        ax[1].axvline(x=0, ls=':', color='gray')
#        ax[1].set_xlim(-1.5, 1.5)
#        ax[1].set_ylim(-1.5, 1.5)
#        ax[1].invert_xaxis()
        
        tleft = 0.75
#        tleft = 0.8
        ttop = 0.8
        ttstep = 0.05
        fig.text(tleft, ttop - 0*ttstep, 't0_in = {0:.1f}'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 1*ttstep, 'u0_in = {0:.2f}'.format(u0_in), fontsize=12)
        fig.text(tleft, ttop - 2*ttstep, 'tE_in = {0:.1f}'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, 'piEE_in = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 4*ttstep, 'piEN_in = {0:.2f}'.format(piEN_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, 'piEE_in/piEN_in = {0:.2f}'.format(piEE_in/piEN_in), fontsize=12)
        fig.text(tleft, ttop - 7*ttstep, 't0par = {0:.1f}'.format(t0par), fontsize=12)
        fig.text(tleft, ttop - 8*ttstep, 't0_out = {0:.1f}'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 9*ttstep, 'u0_out = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 10*ttstep, 'tE_out = {0:.1f}'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, 'piEE_out = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, 'piEN_out = {0:.2f}'.format(piEN_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, 'piEE_out/piEN_out = {0:.2f}'.format(piEE_out/piEN_out), fontsize=12)
        
        plt.show()
        plt.pause(1)
        import pdb
        pdb.set_trace()

    return t0_out, u0_out, tE_out, piEE_out, piEN_out

def convert_u0_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE,
                  tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                  tauhatE_out, tauhatN_out, u0hatE_out, u0hatN_out,
                  in_frame='helio'):
    """
    *** PROPER MOTIONS ARE DEFINED AS SOURCE - LENS ***
    *** COORDINATE SYSTEM IS ON-SKY (NOT TAU-BETA) ***
    tauhat_in, tauhat_out. 
    u0hat_in, u0hat_out.
    VECTORS ARE ARRAYS DEFINED AS [E, N].
    """
    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    try:
        par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)
    except:
        nn = len(t0par)
        par_t0par = model.parallax_in_direction(ra, dec, t0par)
        
    # Calculate a bunch of values we need to get u0 and t0.
    tau_in = (t0par - t0_in)/tE_in
    u0vec_in = np.abs(u0_in) * np.array([u0hatE_in, u0hatN_in])
    tauvec_in = tau_in * np.array([tauhatE_in, tauhatN_in])
    uvec_in = u0vec_in + tauvec_in

    # Get direction of tauhat_out and u0hat_out.
    tauhat_out = np.array([tauhatE_out, tauhatN_out])
    u0hat_out = np.array([u0hatE_out, u0hatN_out])
    
    # Actually get u0 and t0...
    if in_frame=='helio':
        t0_out = t0par - tE_out * np.dot(tauhat_out, uvec_in - piE*par_t0par)
        u0_out = np.dot(u0hat_out, uvec_in - piE*par_t0par)
    if in_frame=='geo':
        t0_out = t0par - tE_out * np.dot(tauhat_out, uvec_in + piE*par_t0par)
        u0_out = np.dot(u0hat_out, uvec_in + piE*par_t0par)

    return t0_out, u0_out

def convert_u0_t0_old(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE,
                  tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                  tauhatE_out, tauhatN_out, # u0hatE_out, u0hatN_out,
                  in_frame='helio'):
    """
    *** PROPER MOTIONS ARE DEFINED AS SOURCE - LENS ***
    *** COORDINATE SYSTEM IS ON-SKY (NOT TAU-BETA) ***

    tauhat_in, tauhat_out. 
    u0hat_in, u0hat_out.

    VECTORS ARE ARRAYS DEFINED AS [E, N].

    """
    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    # Get dp_dt_t0par in 1/days.
    try:
        par_t0par = model.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2,)
        dp_dt_t0par = model.dparallax_dt_in_direction(ra, dec, np.array([t0par])).reshape(2,)/365.25
    except:
        nn = len(t0par)
        par_t0par = model.parallax_in_direction(ra, dec, t0par)
        dp_dt_t0par = model.dparallax_dt_in_direction(ra, dec, t0par)/365.25
        
    # Calculate a bunch of values we need to get u0 and t0.
    tau_in = (t0par - t0_in)/tE_in
    u0vec_in = np.abs(u0_in) * np.array([u0hatE_in, u0hatN_in])
    tauvec_in = tau_in * np.array([tauhatE_in, tauhatN_in])
    uvec_in = u0vec_in + tauvec_in

    # Get direction of tauhat_out and u0hat_out.
    tauhat_out = np.array([tauhatE_out, tauhatN_out])
    tauhat_in = np.array([tauhatE_in, tauhatN_in])
#    u0hat_out = np.array([u0hatE_out, u0hatN_out])
    
    # Actually get u0 and t0...
    dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE

    if in_frame=='helio':
        t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par)
        u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par
        u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])

#        t0_out = t0par - tE_out * np.dot(tauhat_out, uvec_in - piE*par_t0par)
#        u0_out = np.dot(u0hat_out, uvec_in - piE*par_t0par)

    if in_frame=='geo':
        t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par)
        u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par
        u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])

#        t0_out = t0par - tE_out * np.dot(tauhat_out, uvec_in + piE*par_t0par)
#        u0_out = np.dot(u0hat_out, uvec_in + piE*par_t0par)

    print(np.sign(tauhat_out[1]))
#    print(np.sign(np.cross(tauhat_out, u0vec_out)))
#    return t0_out, u0_out
    return t0_out, u0vec_out


def convert_piE_tE(ra, dec, t0par, 
                   piEE_in, piEN_in, tE_in, 
                   in_frame='helio'):     
    """
    *** PROPER MOTIONS ARE DEFINED AS SOURCE - LENS ***

    If you want lens-source, flip piEE and piEN by -1.

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
        Reference time for the geocentric frame value in MJD.

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
    if ((type(ra) == float) or (type(ra) == int)):
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
