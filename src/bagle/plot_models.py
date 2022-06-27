from bagle import model
import numpy as np
import pylab as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb

def get_source_pos(z, m1, m2, z1, z2):
    """
    Get the source position.
    This can be physical or angular, but you have to
    be consistent with the inputs.
    However, it must be using complex coordinates, 
    not vectors!

    Parameters
    ----------
    z : arr or float
        Image positions 

    m1 : arr or float
        Mass 1
    
    m2 : arr or float
        Mass 2

    z1 : arr or float
        Lens 1 position

    z2 : arr or float
        Lens 2 position

    Return
    ------
    w : arr or float
        Source position

    """
    w = z - m1/np.conj(z - z1) - m2/np.conj(z - z2)

    return w

def get_magnification_map(psbl, duration=0.05, time_steps=300):
    """
    For a given PSBL model, plot the source trajectory on top 
    of the magnification map.

    Parameters
    ----------
    psbl : model.PSBL object
        The PSBL model to use for plotting.
        
    duration : float
        Total time to plot, in units of tE.

    time_steps : int
        Number of time steps to plot for source trajectory
    """

    # An 8000 x 8000 grid takes a few seconds to run.

    # Get lenses info
    m1 = psbl.m1
    m2 = psbl.m
    xL1_0, xL2_0 = psbl.get_resolved_lens_astrometry(t_obs=psbl.t0)
    z1 = xL1_0[0][0] + 1j*xL1_0[0][1]
    z2 = xL2_0[0][0] + 1j*xL2_0[0][1]

    # Set up magnification map grid, centered on lens.
    # zgrid are the image positions, where the shots end.
    # We want to find where they start (source plane), i.e.
    # inverse ray shooting
    grid_center = psbl.xL0
    grid_size = 10 * psbl.sep # Probably a better way to do this...
    plot_radius = 5 * psbl.sep

    xmin = grid_center[0] - grid_size
    xmax = grid_center[0] + grid_size
    ymin = grid_center[1] - grid_size
    ymax = grid_center[1] + grid_size

    x = np.linspace(xmin, xmax, 8000)
    y = np.linspace(ymin, ymax, 8000)
    xgrid, ygrid = np.meshgrid(x,y)
    zgrid = xgrid + 1j*ygrid

    # Get the source positions 
    w_points = get_source_pos(zgrid, m1, m2, z1, z2)

    # There's a few points that get shot out far away
    # This trims them out
    dist2 = (w_points.real**2 + w_points.imag**2)

    # Separate into real and imaginary componenest for plotting
    wreal = w_points[np.where(dist2 < plot_radius)].real
    wimag = w_points[np.where(dist2 < plot_radius)].imag

    # FIXME: CHECK THIS 
    # Source - lens position 
    # (trajectory of source in the lens rest frame)
    tmin = psbl.t0 - ((duration / 2.0) * psbl.tE)
    tmax = psbl.t0 + ((duration / 2.0) * psbl.tE)
    dt_in_years = np.linspace(tmin, tmax, time_steps)
    xLS_unlensed = (psbl.xS0 - psbl.xL0) + np.outer(dt_in_years, psbl.muS - psbl.muL) * 1e-3

    # Plotting boundaries
    plot_xmin = min(np.min(xLS_unlensed[:,0]), xmin)
    plot_xmax = max(np.max(xLS_unlensed[:,0]), xmax)
    plot_ymin = min(np.min(xLS_unlensed[:,1]), ymin)
    plot_ymax = max(np.max(xLS_unlensed[:,1]), ymax)

    plt.figure(1, figsize=(6,6))
    plt.clf()
    # magnification map and lenses
    val = plt.hist2d(wreal, wimag, bins=500, norm = mpl.colors.LogNorm(), cmap = 'viridis')
    plt.scatter(z1.real, z1.imag, s = 5, color = 'black',
             marker = '.', label = 'Lens 1')
    plt.scatter(z2.real, z2.imag, s = 5, color = 'black',
             marker = 'v', label = 'Lens 2')

    # Source trajectory 
    plt.plot(xLS_unlensed[:,0], xLS_unlensed[:,1], color = 'red')
    plt.xlim(plot_xmin, plot_xmax)
    plt.ylim(plot_ymin, plot_ymax)
    plt.colorbar(val[3])
    plt.legend()
    plt.show()
    
def animate_PSBL(psbl, duration=10, time_steps=300, outfile='psbl_movie.gif'):
    """
    Make an animated GIF of a point-source binary-lens event. Animate the photometry
    and the astrometry. 

    Inputs
    ----------
    psbl : model.PSBL object
        The PSBL model to use for plotting.
    duration : float
        The total time to plot, in units of tE.
    
    """
    tmin = psbl.t0 - ((duration / 2.0) * psbl.tE)
    tmax = psbl.t0 + ((duration / 2.0) * psbl.tE)
    t = np.linspace(tmin, tmax, time_steps)

    # Fetch the array of images and amplifications.
    # Calculate only once for speed-ups.
    img, amp = psbl.get_all_arrays(t)
    
    # Position of source
    
    rL_1, rL_2 = psbl.get_resolved_lens_astrometry(t)
    rS = psbl.get_astrometry_unlensed(t)
    rS_img = psbl.get_astrometry(t, image_arr=img, amp_arr=amp)
    rS_img_all = psbl.get_resolved_astrometry(t, image_arr=img, amp_arr=amp)
    pS = psbl.get_photometry(t, amp_arr=amp)

    # Convert arcsec to milli-arcsec:
    rL_1 *= 1e3
    rL_2 *= 1e3
    rS *= 1e3
    rS_img *= 1e3
    rS_img_all *= 1e3

    plt.close(3)
    plt.figure(3, figsize=(12, 4))
    fig = plt.gcf()

    ax1 = plt.axes([0.08, 0.17, 0.3, 0.78])
    ax2 = plt.axes([0.48, 0.17, 0.3, 0.78])
    plt.subplots_adjust(wspace=0.44, left=0.1)
    
    # creates many different lines
    # Lines: unlensed source, lens1, lens2, lensed source (unresolved), photometry
    # Lines: lensed source image1, image2, image3, image4, image5)
    line1, = ax1.plot(rS[:, 0], rS[:, 1], 'b--', alpha=0.5,
                         label="Source, unlensed")
    line2, = ax1.plot([], 'k*', markersize=6, alpha=0.5,
                          label="Lens 1")
    line3, = ax1.plot([], 'k*', markersize=6, alpha=0.5, color='grey',
                          label="Lens 2")
    line4, = ax1.plot([], 'r-', label='Source, lensed, unresolved')
    line5, = ax2.plot(t - psbl.t0, pS, 'r-')

    ttext = ax2.text(0.98, 0.98, 'Time', fontsize=10,
                          horizontalalignment='right',
                          verticalalignment='top',
                          transform=ax2.transAxes)
                          

    lines = [line1, line2, line3, line4, line5, ttext]
    
    for ii in range(5):
        if ii == 0:
            label = 'Source, lensed images'
        else:
            label = ''
            
        line_tmp, = ax1.plot([], 'r.', markersize=6, alpha=0.5, color='purple',
                                 label=label)
        lines.append(line_tmp)

    ast_lim = np.max(np.abs(np.concatenate([rL_1, rL_2, rS, rS_img]).flatten()))
    pho_lim = np.max(pS)
    
    ax1.set_xlabel(r'$\Delta \alpha^*$ (mas)')
    ax1.set_ylabel(r'$\Delta \delta$ (mas)')
    ax1.set_xlim(ast_lim, -ast_lim)
    ax1.set_ylim(-ast_lim, ast_lim)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Brightness (mag)")
    ax2.invert_yaxis()

    # Print out all of the parameters.
    plt.figtext(0.802, 0.8, 'PSBL Model')

    fmt0 = r'M$_{{L1}}$ = {0:.2f} M$_\odot$'
    fmt1 = r'M$_{{L2}}$ = {0:.2f} M$_\odot$'
    fmt2 = r'sep = {0:.1e} arcsec'
    fmt3 = r'$\alpha$ = {0:.2f} deg'
    fmt4 = r'$\beta$ = {0:.1f} mas'
    fmt5 = r'x$_{{S0}}$ = [{0:.4f}, {1:.4f}] arcsec'
    fmt6 = r'$\mu_L$ = [{0:.2f}, {1:.2f}] mas/yr'
    fmt7 = r'$\mu_S$ = [{0:.2f}, {1:.2f}] mas/yr'
    fmt8 = r'd$_L$ = {0:.0f} pc'
    fmt9 = r'd$_S$ = {0:.0f} pc'
    fmt10 = r'mag$_S$ = {0:.2f} mag'
    fmt11 = r'b$_{{sff}}$ = {0:.3f}'
    dy = 0.05
    plt.figtext(0.805, 0.75-0*dy, fmt0.format(psbl.mL1), fontsize=12)
    plt.figtext(0.805, 0.75-1*dy, fmt1.format(psbl.mL2), fontsize=12)
    plt.figtext(0.805, 0.75-2*dy, fmt2.format(psbl.sep), fontsize=12)
    plt.figtext(0.805, 0.75-3*dy, fmt3.format(psbl.alpha), fontsize=12)
    plt.figtext(0.805, 0.75-4*dy, fmt4.format(psbl.beta), fontsize=12)
    plt.figtext(0.805, 0.75-5*dy, fmt5.format(psbl.xS0[0], psbl.xS0[1]), fontsize=12)
    plt.figtext(0.805, 0.75-6*dy, fmt6.format(psbl.muL[0], psbl.muL[1]), fontsize=12)
    plt.figtext(0.805, 0.75-7*dy, fmt7.format(psbl.muS[0], psbl.muS[1]), fontsize=12)
    plt.figtext(0.805, 0.75-8*dy, fmt8.format(psbl.dL), fontsize=12)
    plt.figtext(0.805, 0.75-9*dy, fmt9.format(psbl.dS), fontsize=12)
    for ff in range(len(psbl.mag_src)):
        plt.figtext(0.805, 0.75-(10+ff*2)*dy, fmt10.format(psbl.mag_src[ff]), fontsize=12)
        plt.figtext(0.805, 0.75-(11+ff*2)*dy, fmt11.format(psbl.b_sff[ff]), fontsize=12)


    # this function is called at every frame,
    # with i being the number of the frame that it's currently on
    def update(i, t, rL_1, rL_2, rS, rS_img, rS_img_all, pS, lines):
        lines[0].set_data(rS[:i+1, 0], rS[:i+1, 1])
        lines[1].set_data(rL_1[i, 0], rL_1[i, 1])
        lines[2].set_data(rL_2[i, 0], rL_2[i, 1])
        lines[3].set_data(rS_img[:i+1, 0], rS_img[:i+1, 1])
        lines[4].set_data(t[:i+1] - psbl.t0, pS[:i+1])
        lines[5].set_text('time = {0:.0f} days'.format(t[i] - psbl.t0))

        for jj in range(5):
            lines[6+jj].set_data(rS_img_all[i, jj, 0], rS_img_all[i, jj, 1])
        
        return lines
    
    """
    FuncAnimation takes in lots of arguments
    
    fig = background figure
    
    update = function that is called every frame
    
    len(tau) = the number of frames, so now the first argument
    passed into update (i) will be (0,1,2...len(tau))
    
    fargs specifies the other arguments to pass into update
    
    blit being true means that each frame, if there are elements
    of it that don't change from the last frame,
    it won't replot them, so this makes it faster
    
    interval = number of milliseconds between each frame
    alternatively you can specify fps in save after after the file name
    
    """
    ani = animation.FuncAnimation(fig, update, frames=len(t),
                                  fargs=[t, rL_1, rL_2, rS, rS_img, rS_img_all, pS, lines],
                                  blit=True, interval=10)
    # ani.save(outfile, writer="imagemagick", dpi=80)
    
    return ani

def animate_PSPL(pspl, duration=10, time_steps=300, outfile='pspl_movie.gif'):
    """
    Make an animated GIF of a point-source point-lens event. Animate the photometry
    and the astrometry. 

    Inputs
    ----------
    pspl : model.PSPL object
        The PSPL model to use for plotting.
    duration : float
        The total time to plot, in units of tE.
    
    """
    tmin = pspl.t0 - ((duration / 2.0) * pspl.tE)
    tmax = pspl.t0 + ((duration / 2.0) * pspl.tE)
    t = np.linspace(tmin, tmax, time_steps)

    # Position of source
    rL = pspl.get_lens_astrometry(t)
    rS = pspl.get_astrometry_unlensed(t)
    rS_img = pspl.get_astrometry(t)
    rS_img_all = pspl.get_resolved_astrometry(t)
    pS = pspl.get_photometry(t)

    # Convert arcsec to milli-arcsec:
    rL *= 1e3
    rS *= 1e3
    rS_img *= 1e3
    rS_img_all = [rS_img_all[0] * 1e3, rS_img_all[1] * 1e3]

    plt.close(3)
    plt.figure(3, figsize=(12, 4))
    fig = plt.gcf()

    ax1 = plt.axes([0.08, 0.17, 0.3, 0.78])
    ax2 = plt.axes([0.48, 0.17, 0.3, 0.78])
    plt.subplots_adjust(wspace=0.44, left=0.1)
    
    # creates many different lines
    # Lines: unlensed source, lens1, lens2, lensed source (unresolved), photometry
    # Lines: lensed source image1, image2, image3, image4, image5)
    line1, = ax1.plot(rS[:, 0], rS[:, 1], 'b--', alpha=0.5,
                         label="Source, unlensed")
    line2, = ax1.plot([], 'k*', markersize=6, alpha=0.5,
                          label="Lens")
    line3, = ax1.plot([], 'r-', label='Source, lensed, unresolved')
    line4, = ax2.plot(t - pspl.t0, pS, 'r-')

    ttext = ax2.text(0.98, 0.98, 'Time', fontsize=10,
                          horizontalalignment='right',
                          verticalalignment='top',
                          transform=ax2.transAxes)
                          

    lines = [line1, line2, line3, line4, ttext]
    
    for ii in range(2):
        if ii == 0:
            label = 'Source, lensed images'
        else:
            label = ''
            
        line_tmp, = ax1.plot([], 'r.', markersize=6, alpha=0.5, color='purple',
                                 label=label)
        lines.append(line_tmp)

    ast_lim = np.max(np.abs(np.concatenate([rL, rS, rS_img]).flatten()))
    pho_lim = np.max(pS)
    
    ax1.set_xlabel(r'$\Delta \alpha^*$ (mas)')
    ax1.set_ylabel(r'$\Delta \delta$ (mas)')
    ax1.set_xlim(ast_lim, -ast_lim)
    ax1.set_ylim(-ast_lim, ast_lim)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Brightness (mag)")
    ax2.invert_yaxis()

    # Print out all of the parameters.
    plt.figtext(0.802, 0.8, 'PSPL Model')

    fmt0 = r'M$_L$ = {0:.2f} M$_\odot$'
    fmt1 = r'R.A. = {0:.5f} deg'
    fmt2 = r'Dec. = {0:.5f} deg'
    fmt3 = r'$\beta$ = {0:.1f} mas'
    fmt4 = r'x$_{{S0}}$ = [{0:.4f}, {1:.4f}] arcsec'
    fmt5 = r'$\mu_L$ = [{0:.2f}, {1:.2f}] mas/yr'
    fmt6 = r'$\mu_S$ = [{0:.2f}, {1:.2f}] mas/yr'
    fmt7 = r'd$_L$ = {0:.0f} pc'
    fmt8 = r'd$_S$ = {0:.0f} pc'
    fmt9 = r'mag$_S$ = {0:.2f} mag'
    fmt10 = r'b$_{{sff}}$ = {0:.3f}'
    dy = 0.05
    plt.figtext(0.805, 0.75-0*dy, fmt0.format(pspl.mL), fontsize=12)
    # plt.figtext(0.805, 0.75-1*dy, fmt1.format(pspl.raL), fontsize=12)
    # plt.figtext(0.805, 0.75-2*dy, fmt2.format(pspl.decL), fontsize=12)
    plt.figtext(0.805, 0.75-3*dy, fmt3.format(pspl.beta), fontsize=12)
    plt.figtext(0.805, 0.75-4*dy, fmt4.format(pspl.xS0[0], pspl.xS0[1]), fontsize=12)
    plt.figtext(0.805, 0.75-5*dy, fmt5.format(pspl.muL[0], pspl.muL[1]), fontsize=12)
    plt.figtext(0.805, 0.75-6*dy, fmt6.format(pspl.muS[0], pspl.muS[1]), fontsize=12)
    plt.figtext(0.805, 0.75-7*dy, fmt7.format(pspl.dL), fontsize=12)
    plt.figtext(0.805, 0.75-8*dy, fmt8.format(pspl.dS), fontsize=12)
    for ff in range(len(pspl.mag_src)):
        plt.figtext(0.805, 0.75-(9+ff*2)*dy, fmt9.format(pspl.mag_src[ff]), fontsize=12)
        plt.figtext(0.805, 0.75-(10+ff*2)*dy, fmt10.format(pspl.b_sff[ff]), fontsize=12)

    # this function is called at every frame,
    # with i being the number of the frame that it's currently on
    def update(i, t, rL, rS, rS_img, rS_img_all, pS, lines):
        lines[0].set_data(rS[:i+1, 0], rS[:i+1, 1])
        lines[1].set_data(rL[i, 0], rL[i, 1])
        lines[2].set_data(rS_img[:i+1, 0], rS_img[:i+1, 1])
        lines[3].set_data(t[:i+1] - pspl.t0, pS[:i+1])
        lines[4].set_text('time = {0:.0f} days'.format(t[i] - pspl.t0))

        for jj in range(2):
            lines[5+jj].set_data(rS_img_all[jj][i, 0], rS_img_all[jj][i, 1])
        
        return lines
    
    """
    FuncAnimation takes in lots of arguments
    
    fig = background figure
    
    update = function that is called every frame
    
    len(tau) = the number of frames, so now the first argument
    passed into update (i) will be (0,1,2...len(tau))
    
    fargs specifies the other arguments to pass into update
    
    blit being true means that each frame, if there are elements
    of it that don't change from the last frame,
    it won't replot them, so this makes it faster
    
    interval = number of milliseconds between each frame
    alternatively you can specify fps in save after after the file name
    
    """
    ani = animation.FuncAnimation(fig, update, frames=len(t),
                                  fargs=[t, rL, rS, rS_img, rS_img_all, pS, lines],
                                  blit=True, interval=20)
    # ani.save(outfile, writer="imagemagick", dpi=80)
    
    return ani

def plot_PSBL(psbl, duration=10, time_steps=300, outfile='psbl_geometry.png'):
    """
    Make an animated GIF of a point-source binary-lens event. Animate the photometry
    and the astrometry. 

    Inputs
    ----------
    psbl : model.PSBL object
        The PSBL model to use for plotting.
    duration : float
        The total time to plot, in units of tE.
    
    """
    tmin = psbl.t0 - ((duration / 2.0) * psbl.tE)
    tmax = psbl.t0 + ((duration / 2.0) * psbl.tE)
    t = np.linspace(tmin, tmax, time_steps)

    # Fetch the array of images and amplifications.
    # Calculate only once for speed-ups.
    img, amp = psbl.get_all_arrays(t)
    
    # Position of source
    rL_1, rL_2 = psbl.get_resolved_lens_astrometry(t)
    rS = psbl.get_astrometry_unlensed(t)
    rS_img = psbl.get_astrometry(t, image_arr=img, amp_arr=amp)
    rS_img_all = psbl.get_resolved_astrometry(t, image_arr=img, amp_arr=amp)
    pS = psbl.get_photometry(t, amp_arr=amp)

    # Convert arcsec to milli-arcsec:
    if psbl.astrometryFlag:
        rL_1 *= 1e3
        rL_2 *= 1e3
        rS *= 1e3
        rS_img *= 1e3
        rS_img_all *= 1e3
        ast_unit = '(mas)'
    else:
        ast_unit = r'($\theta_E$)'

    plt.close(3)
    plt.figure(3, figsize=(12, 4))
    fig = plt.gcf()

    ax1 = plt.axes([0.08, 0.17, 0.3, 0.78])
    ax2 = plt.axes([0.48, 0.17, 0.3, 0.78])
    plt.subplots_adjust(wspace=0.44, left=0.1)
    
    # creates many different lines
    # Lines: unlensed source, lens1, lens2, lensed source (unresolved), photometry
    # Lines: lensed source image1, image2, image3, image4, image5)
    ax1.plot(rS[:, 0], rS[:, 1], 'b--', alpha=0.5,
                         label="Source, unlensed")
    ax1.plot(rL_1[:,0], rL_1[:,1], 'k*', markersize=6, alpha=0.5,
                          label="Lens 1")
    ax1.plot(rL_2[:,0], rL_2[:,1], 'k*', markersize=6, alpha=0.5, color='grey',
                          label="Lens 2")
    ax1.plot(rS_img[:,0], rS_img[:,1], 'r-', label='Source, lensed, unresolved')
    ax2.plot(t - psbl.t0, pS, 'r-')

    for ii in range(5):
        if ii == 0:
            label = 'Source, lensed images'
        else:
            label = ''
            
        ax1.plot(rS_img_all[:, ii, 0], rS_img_all[:, ii, 1], 'r.', 
                 markersize=2, alpha=0.5, color='purple',
                 label=label)

    # Plot the Einstein radius
    if psbl.astrometryFlag:
        thetaE = psbl.thetaE_amp
    else:
        thetaE = 1.0
    circ = plt.Circle((0, 0), thetaE, fill=False, color='black', linestyle='--')
    ax1.add_artist(circ)

    # Plot an arrow to indicate time.
    arr_dx = (rS_img[-1, 0] - rS_img[-2, 0]) * 1e1
    arr_dy = (rS_img[-1, 1] - rS_img[-2, 1]) * 1e1
    time_arrow = ax1.arrow(rS_img[-2, 0], rS_img[-2, 1],
                           arr_dx, arr_dy,
                           color='red', width=0.1)
        
    ast_lim = np.max(np.abs(np.concatenate([rL_1, rL_2, rS, rS_img]).flatten())) * 1.2
    pho_lim = np.max(pS)
    
    ax1.set_xlabel(r'$\Delta \alpha^*$ ' + ast_unit)
    ax1.set_ylabel(r'$\Delta \delta$ ' + ast_unit)
    ax1.set_xlim(ast_lim, -ast_lim)
    ax1.set_ylim(-ast_lim, ast_lim)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Brightness (mag)")
    ax2.invert_yaxis()

    # Print out all of the parameters.
    plt.figtext(0.802, 0.8, 'PSBL Model')

    fmt_dict = {'mL1': r'M$_{{L1}}$ = {0:.2f} M$_\odot$',
                'mL2': r'M$_{{L2}}$ = {0:.2f} M$_\odot$',
                'sep': r'sep = {0:.4f} arcsec or $\theta_E$',
                'alpha': r'$\alpha$ = {0:.2f} deg',
                'beta': r'$\beta$ = {0:.1f} mas',
                'xS0': r'x$_{{S0}}$ = [{0:.4f}, {1:.4f}] arcsec',
                'muL': r'$\mu_L$ = [{0:.2f}, {1:.2f}] mas/yr',
                'muS': r'$\mu_S$ = [{0:.2f}, {1:.2f}] mas/yr',
                'piE': r'$\pi_E$ = [{0:.2f}, {1:.2f}]',
                'dL': r'd$_L$ = {0:.0f} pc',
                'dS': r'd$_S$ = {0:.0f} pc',
                'mag_src': r'mag$_S$ = ',
                'b_sff': r'b$_{{sff}}$ = ',
                'tE': r't$_E$ = {0:.1f} days',
                'u0': r'u$_0$ = {0:.3f}',
                'q': 'q = {0:.3f}',
                'phi': r'$\alpha$ = {0:.2f} deg'
                }

    if psbl.astrometryFlag:
        print_vars = ['mL1', 'mL2', 'sep', 'alpha', 'beta',
                      'xS0', 'muL', 'muS', 'dL', 'dS',
                      'mag_src', 'b_sff']
    else:
        print_vars = ['tE', 'u0', 'q', 'sep', 'phi',
                      'piE', 'mag_src', 'b_sff']

    for pp in range(len(print_vars)):
        dy = 0.05
        par = print_vars[pp]
        fmt = fmt_dict[par]
        val = getattr(psbl, par)

        if par == 'mag_src' or par == 'b_sff':
            fmt += '[' + '{:.2f} ' * len(val) + ']'

        if isinstance(val, (list,np.ndarray)):
            txt = fmt.format(*val)
        else:
            txt = fmt.format(val)
            
        plt.figtext(0.805, 0.75-pp*dy, txt, fontsize=12)
        
    plt.show()
    plt.savefig(outfile)

    return

