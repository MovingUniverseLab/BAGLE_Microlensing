from bagle import model, model_fitter, parallax
from microlens.jlu import astrometry
from astropy import constants as c
from astropy import units as u
from astropy.time import Time
import numpy as np
import pylab as plt
from astropy.table import Table
import pdb
import math
import os

paper_dir = '/u/mhuston/work/bagle_papers/'

def plot_pspl_parallax_belokurov():
    dim_ang = u.dimensionless_angles()
    
    # Scenario from Belokurov and Evans 2002 (Figure 1)
    # Parameters specified in the paper: 
    mL = 0.5 * u.M_sun
    dL = 150.0 * u.pc
    dS = 1500.0 * u.pc
    vTilde = 70 * u.km/u.s  # assuming this is \tilde{v}
    u0 = 1.5  # Einstein radii

    # Transverse velocity might be vtilde = \mu_rel / ( (1/d_L) - (1/d_S)) = \mu_rel * \pi_rel
    print(vTilde * (dS - dL) / dS)

    # Parameters we calculate from above or guess.
    raL = 17.5 * 15.0  # in decimal degrees
    decL = -30.0
    imag = 19.0
    b_sff = 1.0
    muS = np.array([-1.75, 6.0])  # mas/yr measured from Figure 1
    muL = (vTilde / dL).to(u.mas/u.yr, equivalencies=dim_ang)  # mas/yr
    muL = np.array([-muL.value, 0.0])

    # Derived values from definitively determined inputs.
    thetaE = ((4.0 * c.G * mL / c.c**2) * ((1./dL) - (1./dS)))**0.5
    thetaE = thetaE.to(u.mas, equivalencies=dim_ang) # mas
    xS0amp = u0 * thetaE   # mas
    xS0 = (muL / np.linalg.norm(muL))[::-1] * xS0amp # mas

    piRel = (u.AU / dL) - (u.AU / dS)
    piRel = piRel.to(u.mas, equivalencies=dim_ang)

    # Derived values from guessed at inputs.
    muRelAmp = vTilde * piRel / u.AU
    muRelAmp = muRelAmp.to(u.mas/u.yr)
    muRel = muL - muS

    print('mu_rel = [{0:4.2f}, {1:4.2f}] mas/yr'.format(muRel[0], muRel[1]))
    print('mu_rel_amp = {0:4.2f} mas/yr'.format(muRelAmp))
    print('mu_rel_amp = {0:4.2f} mas/yr'.format(np.linalg.norm(muRel)))
    print('mu_L =  [{0:4.2f}, {1:4.2f}] mas/yr, '.format(muL[0], muL[1]))
    print('mu_S =  [{0:4.2f}, {1:4.2f}] mas/yr, '.format(muS[0], muS[1]))
    print('thetaE = {0:4.2f} mas'.format(thetaE))
    print('piRel = {0:4.2f} mas'.format(piRel))
    print('xS0amp = {0:4.2f} mas'.format(xS0amp))
    print('xS0 =   [{0:4.2f} mas, {1:4.2f}], '.format(xS0[0], xS0[1]))
    
    beta = xS0amp # mas
    t0 = 57160.00  # MJD
    # muS = np.array([-2.0, 7.0])
    # muL = np.array([90.00, -24.71])

    # Convert out of astropy units
    mL = mL.value
    xS0 = xS0.value
    beta = beta.value
    dL = dL.value
    dS = dS.value
    muL = np.array([0, 0])

    # No parallax
    pspl_n = model.PSPL_PhotAstrom_noPar_Param1(mL, t0, beta, dL, dL/dS,
                                                xS0[0]*1e-3, xS0[1]*1e-3,    
                                                muL[0], muL[1],
                                                muS[0], muS[1],
                                                b_sff, imag,
                                                raL=raL, decL=decL)
    print('pspl_n.u0', pspl_n.u0)
    print('pspl_n.muS', pspl_n.muS)
    print('pspl_n.u0_hat', pspl_n.u0_hat)
    print('pspl_n.thetaE_hat', pspl_n.thetaE_hat)
    
    # With parallax
    pspl_p = model.PSPL_PhotAstrom_Par_Param1(mL, t0, beta, dL, dL/dS,
                                                xS0[0]*1e-3, xS0[1]*1e-3,    
                                                muL[0], muL[1],
                                                muS[0], muS[1],
                                                b_sff, imag,
                                                raL=raL, decL=decL)

    t = np.arange(t0 - 1000, t0 + 1000, 1)
    dt = t - pspl_n.t0

    A_n = pspl_n.get_amplification(t)
    A_p = pspl_p.get_amplification(t)

    xS_n = pspl_n.get_astrometry(t)
    xS_p_unlens = pspl_p.get_astrometry_unlensed(t)
    xS_p_lensed = pspl_p.get_astrometry(t)
    xL_p = pspl_p.get_lens_astrometry(t)

    # Plot the positions of everything
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xS_n[:, 0] * 1e3, xS_n[:, 1] * 1e3, 'r--',
                 mfc='none', mec='red', label='Src, No parallax model')
    plt.plot(xS_p_unlens[:, 0] * 1e3, xS_p_unlens[:, 1] * 1e3, 'b--',
                 mfc='none', mec='blue',
             label='Src, Parallax model, unlensed')
    plt.plot(xL_p[:, 0] * 1e3, xL_p[:, 1] * 1e3, 'k--',
                 mfc='none', mec='grey', label='Lens')
    plt.plot(xS_p_lensed[:, 0] * 1e3, xS_p_lensed[:, 1] * 1e3, 'b-', label='Src, Parallax model, lensed')
    plt.legend(fontsize=10)
    plt.gca().invert_xaxis()
    plt.xlabel('R.A. (mas)')
    plt.ylabel('Dec. (mas)')
    plt.axis('equal')
    lim = 20
    print('LIM = ', lim)
    # plt.xlim(lim, -lim) # arcsec
    # plt.ylim(-lim, lim)
    # plt.axis('tight')
    # plt.xlim(0.7, -0.7)
    # plt.ylim(-0.7, 0.7)
    plt.savefig(paper_dir + 'on_sky.png')

    # Check just the astrometric shift part.
    shift_n = pspl_n.get_centroid_shift(t) # mas
    shift_p = (xS_p_lensed - xS_p_unlens) * 1e3 # mas
    shift_n_amp = np.linalg.norm(shift_n, axis=1)
    shift_p_amp = np.linalg.norm(shift_p, axis=1)
    
    fig3 = plt.figure(3)
    plt.clf()
    f1_3 = fig3.add_axes((0.20, 0.3, 0.75, 0.6))
    plt.plot(dt, shift_n_amp, 'r--', label='No parallax model')
    plt.plot(dt, shift_p_amp, 'b--', label='Parallax model')
    plt.ylabel('Astrometric Shift (mas)')
    plt.legend(fontsize=10)
    f1_3.set_xticklabels([])

    f2_3 = fig3.add_axes((0.20, 0.1, 0.75, 0.2))
    plt.plot(dt, shift_p_amp - shift_n_amp, 'k-', label='Par - No par')
    plt.legend(fontsize=10)
    plt.axhline(0, linestyle='--', color='k')
    plt.ylabel('Diff (mas)')
    plt.xlabel('t - t0 (MJD)')
    
    plt.savefig(paper_dir + 'shift_amp_v_t.png')
    

    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(shift_n[:, 0], shift_n[:, 1], 'r-', label='No parallax')
    plt.plot(shift_p[:, 0], shift_p[:, 1], 'b-', label='Parallax')
    plt.axhline(0, linestyle='--')
    plt.axvline(0, linestyle='--')
    plt.gca().invert_xaxis()
    plt.legend(fontsize=10)
    plt.xlabel('Shift RA (mas)')
    plt.ylabel('Shift Dec (mas)')
    plt.axis('equal')
    plt.savefig(paper_dir + 'shift_on_sky.png')
    

    print('Einstein radius: ', pspl_n.thetaE_amp, pspl_p.thetaE_amp)
    print('Einstein crossing time: ', pspl_n.tE, pspl_n.tE)


    return

def plot_geometry():
    """
    Plot a geometry figure showing source, lens, proper motions, parallax vector, etc. 
    """
    ra = 17.50 * 15.0
    dec = -29.00
    lens_distance = 4000.0
    
    astrometry.plot_parallax_vector(ra, dec, lens_distance)

    # Fetch all the arrows so far and make them somewhat transparent.
    ax = plt.gca()

    for artist in ax.artists:
        if 'FancyArrow' in str(type(artist)):
            artist.set_alpha(0.3)

    # Set the label for the lens.
    ax.lines[0].set_label('Lens')
            
    # Now add a source going by.
    src_pos = np.array([ 0.5, -0.7])  # mas
    src_pm  = np.array([-0.7, -0.5])  # mas/yr)
    src_dist = 8000 # pc
    
    # Make a 1 year array of MJDs for plotting purposes. Sample every day.
    t0 = Time('2020-01-01', format='isot', scale='utc')
    mjd0 = t0.mjd
    times = np.arange(mjd0, mjd0 + 365, 1)

    # Project the microlensing parallax into parallel and perpendicular
    # w.r.t. the ecliptic... useful quantities.
    parallax_vec_at_t = parallax.parallax_in_direction(ra, dec, times)
    parallax_amp = 1.0e3 / src_dist
    print(f'src parallax: {parallax_amp:0.1f} mas')
    parallax_vec_at_t *= parallax_amp
    src_pos_at_t = src_pos.T + parallax_vec_at_t

    plt.plot(src_pos_at_t[:, 0], src_pos_at_t[:, 1], 'k.', ms=1, alpha=0.8, label='Source')
    plt.arrow(0, 0, src_pos[0], src_pos[1],
                  color='black', head_width=0.1, head_length=0.2,
                  length_includes_head=True)
    plt.text(0.18, -0.37, r'$\theta_E \vec{u}_0$', color='black', fontsize=14)
    plt.arrow(src_pos[0], src_pos[1], src_pm[0], src_pm[1],
                  color='black', head_width=0.1, head_length=0.2,
                  length_includes_head=True)
    plt.text(0.15, -0.9, r'$\vec{\mu}_S$', color='black', fontsize=14)
    
    plt.axis('equal')
    plt.xlim(1.5, -1.5)
    plt.ylim(-1.5, 1.5)

    plt.legend(markerscale=15, fontsize=14)

    plt.savefig(paper_dir + 'lens_geometry.png')
    
    return ax



def plot_pspl_ellipse():
    """
    Plot a geometry figure showing source, lens, proper motions, parallax vector, etc. 
    """
    raL_in = 17.50 * 15.  # Bulge R.A.
    decL_in = -29.0
    mL_in = 1.0  # msun
    t0_in = 57000.0
    xS0_in = np.array([0.0, 0.0])  # arcsec
    beta_in = 0.2  # mas  same as p=0.4
    muS_in = np.array([0.0, 0.0])
    muL_in = np.array([1.0, 1.0])
    dL_in = 3000.0  # pc
    dS_in = 8000.0  # pc
    b_sff_in = 1.0
    mag_src_in = 19.0

    pspl_par_in = model.PSPL_PhotAstrom_Par_Param1(mL_in,
                                                   t0_in,
                                                   beta_in,
                                                   dL_in,
                                                   dL_in / dS_in,
                                                   xS0_in[0],
                                                   xS0_in[1],
                                                   muL_in[0],
                                                   muL_in[1],
                                                   muS_in[0],
                                                   muS_in[1],
                                                   [b_sff_in],
                                                   [mag_src_in],
                                                   raL=raL_in,
                                                   decL=decL_in)

    pspl_in = model.PSPL_PhotAstrom_noPar_Param1(mL_in,
                                                 t0_in,
                                                 beta_in,
                                                 dL_in,
                                                 dL_in / dS_in,
                                                 xS0_in[0],
                                                 xS0_in[1],
                                                 muL_in[0],
                                                 muL_in[1],
                                                 muS_in[0],
                                                 muS_in[1],
                                                 [b_sff_in],
                                                 [mag_src_in])
    
    # Simulate    
    t = np.linspace(54000, 60000, 2000)

    imag_par = pspl_par_in.get_photometry(t)
    pos_par = pspl_par_in.get_astrometry(t)
    pos_src_par = pspl_par_in.get_astrometry_unlensed(t)
    pos_lens_par = pspl_par_in.get_lens_astrometry(t)
    
    imag = pspl_in.get_photometry(t)
    pos = pspl_in.get_astrometry(t)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=t.min(), vmax=t.max())
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    plt.figure(1, figsize=(12, 6))
    plt.clf()
    
    gs_kw = dict(width_ratios=[0.9,1.1], height_ratios=[2, 1])
    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                                   ['lower left', 'right']],
                                      gridspec_kw=gs_kw, figsize=(12, 6),
                                      constrained_layout=False, num=1)
    fig.suptitle('Parallax Effects')

    ax_m = axd['upper left']
    ax_mr = axd['lower left']
    ax_p = axd['right']

    ax_m.scatter(t, imag_par,
                 c=t, cmap=cmap, norm=norm, s=3)
    ax_m.scatter(t, imag,
                 c=t, cmap=cmap, norm=norm, s=0.5, alpha=0.3)
    ax_m.set_ylabel('Brightness (mag)')
    
    ax_mr.plot(t, imag_par - imag, 'k-')
    ax_mr.set_xlabel('Time (MJD)')
    ax_mr.set_ylabel('Diff (mag)')

    dt = 2000
    ax_m.set_xlim(t0_in - dt, t0_in + dt)
    ax_mr.set_xlim(t0_in - dt, t0_in + dt)

    ax_m.invert_yaxis()
    

    im=ax_p.scatter(pos_par[:, 0]*1e3, pos_par[:, 1]*1e3,
                    label='Parallax',
                 c=t, cmap=cmap, norm=norm, s=3)
    ax_p.scatter(pos[:, 0]*1e3, pos[:, 1]*1e3,
                    label='No Parallax',
                 c=t, cmap=cmap, norm=norm, s=0.5, alpha=0.3)
    ax_p.plot(pos_src_par[:, 0]*1e3, pos_src_par[:, 1]*1e3,
                  label='Unlensed Source',
                  ls='-', marker=None, color='darkorange')
    ax_p.plot(pos_lens_par[:, 0]*1e3, pos_lens_par[:, 1]*1e3,
                  label='Lens',
                  ls='--', marker=None, color='black')
    ax_p.set_xlabel(r'$\Delta \alpha^*$ (mas)')
    ax_p.set_ylabel(r'$\Delta \delta$ (mas)')
    ax_p.axis('equal')
    sz = 0.5
    ax_p.set_ylim(-sz, sz)
    ax_p.set_xlim(-sz, sz)
    ax_p.invert_xaxis()
    ax_p.legend(fontsize=12,scatterpoints=10,markerscale=2,scatteryoffsets=[0.5])
    cbar = fig.colorbar(im,pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label='time (mjd)',size=12)
    plt.tight_layout()
    plt.savefig(paper_dir + 'pspl_ellipse.png')
    
    return


def compare_with_pylima():
    from bagle.tests import test_model

    ra = 267.4640833333333
    dec = -34.62555555555556
    t0 = 55775.0
    u0_amp = 0.5
    tE = 20.0 # 200
    piEE = 0.05 # 0.5
    piEN = -0.1
    mag_src = 16
    b_sff = 0.5

    foo = test_model.plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp,
                                                 tE, piEE, piEN,
                                                 mag_src, b_sff, parallax=False)
    plt.savefig(paper_dir + 'compare_to_pylima_noparallax.png')

    foo = test_model.plot_compare_vs_pylima_pspl(ra, dec, t0, u0_amp,
                                                 tE, piEE, piEN,
                                                 mag_src, b_sff, parallax=True)
    plt.savefig(paper_dir + 'compare_to_pylima_parallax.png')

    return

def plot_helio_geo_t0():
    """
    Plot how the t0, tE, u0 all change when converting from helio to geo(tr) for a 
    single set of parameters (but maybe move t0 throughout the year). 
    """
    from bagle import helio_geo_conversion as hgc
    
    # Bulge event: some parameters from OB110462 DW solution.
    ra = '17:51:40.19'
    dec = '-29:53:26.3'
    tE_h = 290.12
    u0_h = -0.07
    piS = 0.11
    piEE_h = 0.003
    piEN_h = -0.13
    xS0 = 0.0
    yS0 = 0.0
    muSE = -2.25  # mas/yr
    muSN = -3.58
    b_sff = np.array([0.05, 0.90, 0.94])
    mag_base = np.array([16.41, 19.86, 22.04])

    t0_h_mid = 55759.28
    
    # M_L = 3.69
    # piL = 0.65
    # piRel = 0.54
    # ThetaE = 4.03 # mas

    t0_h_arr = np.arange(t0_h_mid - 365, t0_h_mid + 365, 10)

    t0_g = np.zeros(len(t0_h_arr), dtype=float)
    u0_g = np.zeros(len(t0_h_arr), dtype=float)
    tE_g = np.zeros(len(t0_h_arr), dtype=float)
    piEE_g = np.zeros(len(t0_h_arr), dtype=float)
    piEN_g = np.zeros(len(t0_h_arr), dtype=float)
    u0hatE_g = np.zeros(len(t0_h_arr), dtype=float)
    u0hatN_g = np.zeros(len(t0_h_arr), dtype=float)

    for tt in range(len(t0_h_arr)):
        t0ref = t0_h_arr[tt]

        foo = hgc.convert_helio_to_geo_phot(ra, dec, t0_h_arr[tt], u0_h, tE_h, piEE_h, piEN_h, t0ref,
                                        murel_in='SL', murel_out='SL', plot=False)

        t0_g[tt] = foo[0]
        u0_g[tt] = foo[1]
        tE_g[tt] = foo[2]
        piEE_g[tt] = foo[3]
        piEN_g[tt] = foo[4]
        u0hatE_g = foo[5]
        u0hatN_g = foo[6]


    # Lets convert our t0_h_arr to decimal year.
    time = Time(t0_h_arr, format='mjd', scale='tdb')
    plt.close(1)
    fig = plt.figure(1, figsize=(6, 10))

    ax_arr = fig.subplots(4, 1, squeeze=False, sharex=True)
    plt.subplots_adjust(left=0.2, wspace=0.3, hspace=0.05, bottom=0.08, top=0.98)
    
    ax_arr[0, 0].plot(time.decimalyear, t0_g - t0_h_arr, 'k-')
    ax_arr[0, 0].axhline(0, ls='--', color='grey')
    ax_arr[0, 0].set_ylabel(r'$t_{0,\oplus_r} - t_{0,\odot}$ (days)')

    ax_arr[1, 0].plot(time.decimalyear, u0_g, 'k-', label=r'$\oplus_r$')
    ax_arr[1, 0].axhline(u0_h, ls='--', color='grey', label=r'$\odot$')
    ax_arr[1, 0].set_ylabel(r'$u_{0,\oplus_r}$')
    ax_arr[1, 0].legend()

    ax_arr[2, 0].plot(time.decimalyear, piEE_g, 'r-', label=r'$\pi_{E,\oplus_r,E}$')
    ax_arr[2, 0].plot(time.decimalyear, piEN_g, 'b-', label=r'$\pi_{E,\oplus_r,N}$')
    ax_arr[2, 0].axhline(piEE_h, color='red',  ls='--', label='$\pi_{E,\odot,E}$')
    ax_arr[2, 0].axhline(piEN_h, color='blue', ls='--', label='$\pi_{E,\odot,N}$')
    ax_arr[2, 0].legend()
    ax_arr[2, 0].set_ylabel(r'$\vec{\pi}_{E}$')

    ax_arr[3, 0].plot(time.decimalyear, tE_g, 'k-', label=r'$\oplus_r$')
    ax_arr[3, 0].axhline(tE_h, ls='--', color='grey', label=r'$\odot$')
    ax_arr[3, 0].set_xlabel('Time of $t_0$ (yr)')
    ax_arr[3, 0].set_ylabel(r'$t_{E}$')
    ax_arr[3, 0].legend()

    plt.savefig(paper_dir + 'helio_to_geo_vs_t0.png')
    
    return

