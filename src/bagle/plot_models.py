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

def get_magnification_map_old(psbl, duration=0.05, time_steps=300):
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
    m2 = psbl.m2
    xL1_0, xL2_0 = psbl.get_resolved_lens_astrometry(t=psbl.t0)
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
    
def animate_PSBL(psbl, duration=10, time_steps=300, outfile='psbl_movie'):
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

    fmt0 = r'M$_{{L1}}$ = {0:.2f} M$_\\odot$'
    fmt1 = r'M$_{{L2}}$ = {0:.2f} M$_\\odot$'
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
    plt.figtext(0.805, 0.75-0*dy, fmt0.format(psbl.mLp), fontsize=12)
    plt.figtext(0.805, 0.75-1*dy, fmt1.format(psbl.mLs), fontsize=12)
    plt.figtext(0.805, 0.75-2*dy, fmt2.format(psbl.sep), fontsize=12)
    plt.figtext(0.805, 0.75-3*dy, fmt3.format(psbl.alpha), fontsize=12)
    plt.figtext(0.805, 0.75-4*dy, fmt4.format(psbl.beta), fontsize=12)
    plt.figtext(0.805, 0.75-5*dy, fmt5.format(psbl.xS0[0], psbl.xS0[1]), fontsize=12)
    plt.figtext(0.805, 0.75-6*dy, fmt6.format(psbl.muL[0], psbl.muL[1]), fontsize=12)
    plt.figtext(0.805, 0.75-7*dy, fmt7.format(psbl.muS[0], psbl.muS[1]), fontsize=12)
    plt.figtext(0.805, 0.75-8*dy, fmt8.format(psbl.dL), fontsize=12)
    plt.figtext(0.805, 0.75-9*dy, fmt9.format(psbl.dS), fontsize=12)
    for ff, mag_src_val in enumerate(psbl.mag_src):
        plt.figtext(0.805, 0.75-(10+ff*2)*dy, fmt10.format(mag_src_val), fontsize=12)
        plt.figtext(0.805, 0.75-(11+ff*2)*dy, fmt11.format(psbl.b_sff[ff]), fontsize=12)


    # this function is called at every frame,
    # with i being the number of the frame that it's currently on
    def update(i, t, rL_1, rL_2, rS, rS_img, rS_img_all, pS, lines):
        lines[0].set_data(rS[:i+1, 0], rS[:i+1, 1])
        lines[1].set_data([rL_1[i, 0]], [rL_1[i, 1]])
        lines[2].set_data([rL_2[i, 0]], [rL_2[i, 1]])
        lines[3].set_data(rS_img[:i+1, 0], rS_img[:i+1, 1])
        lines[4].set_data(t[:i+1] - psbl.t0, pS[:i+1])
        lines[5].set_text('time = {0:.0f} days'.format(t[i] - psbl.t0))

        for jj in range(5):
            lines[6+jj].set_data([rS_img_all[i, jj, 0]], [rS_img_all[i, jj, 1]])
        
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
    ani.save("%s.mp4" % outfile, writer="ffmpeg") 
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

    fmt0 = r'M$_L$ = {0:.2f} M$_\\odot$'
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
        lines[1].set_data([rL[i, 0]], [rL[i, 1]])
        lines[2].set_data(rS_img[:i+1, 0], rS_img[:i+1, 1])
        lines[3].set_data(t[:i+1] - pspl.t0, pS[:i+1])
        lines[4].set_text('time = {0:.0f} days'.format(t[i] - pspl.t0))

        for jj in range(2):
            lines[5+jj].set_data([rS_img_all[jj][i, 0]], [rS_img_all[jj][i, 1]])
        
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

    fmt_dict = {'mLp': r'M$_{{L1}}$ = {0:.3f} M$_\\odot$',
                'mLs': r'M$_{{L2}}$ = {0:.3f} M$_\\odot$',
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
        print_vars = ['mLp', 'mLs', 'sep', 'alpha', 'beta',
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

def plot_critical_curves(critical_curves, thetaE_amp = None, ax = None, show = True):
    """
    Plots critical_curves

    Parameters
    ----------
    critical_curves : np.array
        critical_curve object in units of thetaE generated with get_critical_curves method.

    thetaE_amp : float or None, optional
        thetaE amplitude. If provided, plot will be in mas, if not, plot will be in thetaE.
        Default is None.

    ax : matplotlib Axes object or None, optional
        Axes provided if you would like this to be in a subplot, if not a new figure will be generated.
        Default is None.

    show : bool
        Shows plot or not.
        Default is True.
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    # Plot in units of mas
    if thetaE_amp is not None:
        ax.scatter(np.real(critical_curves)*thetaE_amp, np.imag(critical_curves)*thetaE_amp, s = 1, label = 'Critical Curves')
    # Plot in units of thetaE
    else:
        ax.scatter(np.real(critical_curves), np.imag(critical_curves), s = 1, label = 'Critical Curves')

    if show:
        plt.show()

    return ax

def plot_caustics(caustics, thetaE_amp = None, ax = None, show = True):
    """
    Plots caustics

    Parameters
    ----------
    caustics : np.array
    caustics object in units of thetaE generated with get_caustics method.

    thetaE_amp : float or None, optional
        thetaE amplitude. If provided, plot will be in mas, if not, plot will be in thetaE.
        Default is None.

    ax : matplotlib Axes object or None, optional
        Axes provided if you would like this to be in a subplot, if not a new figure will be generated.
        Default is None.

    show : bool
        Shows plot or not.
        Default is True.

    Returns
    -------
    ax : matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    # Plot in units of mas
    if thetaE_amp is not None:
        ax.scatter(np.real(caustics)*thetaE_amp, np.imag(caustics)*thetaE_amp, s = 1, label = 'Caustics')
    # Plot in units of thetaE
    else:
        ax.scatter(np.real(caustics), np.imag(caustics), s = 1, label = 'Caustics')

    if show:
        plt.show()

    return ax


def gen_and_plot_caustics_and_critical_curves(mmodel, caustic_time = 't0', show = True, plot_caus = True, plot_cc = True, plot_source_trajectory = True, 
                                              min_t_trajectory_plot = 'default', max_t_trajectory_plot = 'default'):
    """
    Generates and plots caustics, critical curves, and unlensed source trajectory

    Parameters
    ----------
    mmodel : BAGLE model object
        PSBL or BSBL BAGLE model object

    caustic_time : float or 't0', optional
        Time to plot the caustics and critical curves.
        If 't0' it will plot it at mmodel.t0 time.
        Default is 't0'

    show : bool, optional
        Runs plt.show() if true.
        Default is True.

    plot_caus : bool, optional
        Plots caustics if true.
        Default is True.

    plot_cc : bool, optional
        Plots critical curves if true.
        Default is True.

    plot_source_trajectory : bool, optional
        Plots unlensed source trajectory if true.
        Default is True.

    min_t_trajectory_plot : float or 'default', optional
        Minimum time for unlensed source trajectory plot.
        When 'default' plots t0 - tE as minimum.
        Default is 'default'.

    max_t_trajectory_plot : float or 'default', optional
        Maximum time for unlensed source trajectory plot.
        When 'default' plots t0 + tE as maximum.
        Default is 'default'.

    Returns
    -------
    ax : matplotlib Axes object
    """
    
    if plot_caustics == False and plot_critical_curves == False:
        raise Exception('Must plot either caustics or critical curves or both with this function')
        
    if caustic_time == 't0':
        caustic_time = mmodel.t0
        
    critical_curves = mmodel.get_critical_curves(np.array([caustic_time]))
    caustics = mmodel.get_caustics(np.array([caustic_time]))

    fig, ax = plt.subplots(1,1)

    if plot_cc:
        ax_cc = plot_critical_curves(critical_curves, ax = ax, thetaE_amp = mmodel.thetaE_amp, show = False)

    if plot_caus:
        ax_caus = plot_caustics(caustics, ax = ax, thetaE_amp = mmodel.thetaE_amp, show = False)

    if plot_source_trajectory:
        if min_t_trajectory_plot == 'default':
            min_t_trajectory_plot = mmodel.t0 - mmodel.tE
        if max_t_trajectory_plot == 'default':
            max_t_trajectory_plot = mmodel.t0 + mmodel.tE
        times = np.arange(min_t_trajectory_plot, max_t_trajectory_plot)
        xLS_unlensed = mmodel.get_astrometry_unlensed(times)
        ax.plot(xLS_unlensed[:,0]*1e3, xLS_unlensed[:,1]*1e3, label = 'Unlensed Source Trajectory', color = 'green')
        
    ax.invert_xaxis()
    plt.axis('equal')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    plt.legend(fontsize=10)
    if show:
        plt.show()

    return ax

    
    
def all_plots(event, t, img, amp):
    """
    Get the photometry, centroid shift and astrometric shift plots for PSBL/BSBL events.

    Inputs
    ----------
    event : model.PSBL/model.BSBL object
        The PSBL/BSBL model to use for plotting.
    duration: float
        The total time to plot, in units of tE.
    
    """
    
    fig, ax = plt.subplots(3, 1, figsize=(25,40))
    
    centroid = event.get_centroid_shift(t) 
    if bsbl.parallaxFlag:
        ax[0].plot(t-event.t0, event.get_photometry(t, amp_arr=amp), linewidth = 4, color = 'deepskyblue')
        ax[0].set_xlabel('dt')
        ax[0].set_ylabel('Amplification')
        ax[0].set_title('Photometric Shift (Parallax)')
        ax[0].invert_yaxis()
        #ax[0].set_xlim(-500, 500)
        
        ax[1].plot(centroid[:,0], centroid[:,1], linewidth = 4, color = 'purple')
        ax[1].set_xlabel('RA (arcsec)')
        ax[1].set_ylabel('Dec (arcsec)')
        ax[1].set_title('Centroid Shift (Parallax)')
        ax[1].axis('equal')
        
        ax[2].plot(t-event.t0, np.linalg.norm(centroid, axis=1), linewidth = 4, color = 'purple')
        ax[2].set_xlabel('dt')
        ax[2].set_ylabel('Astrometric Shift (arcsec)')
        ax[2].set_title('Astrometric Shift (Parallax)')
        
        #ax[2].set_xlim(-5000, 5000)
        plt.show()


def plot_bsbl(bsbl, zoom, duration = 1000, time_steps=50000, caustic_finder = 'off', default = 0):
    """
    Get the source and lens trajectories for a BSBL event. 

    Inputs
    ----------
    event : model.BSBL object
        The BSBL model to use for plotting.
    duration: float
        The total time to plot, in units of tE.
    
    """

    fig, ax = plt.subplots(figsize=(30,25))
    lim = zoom
    
    plt.gca().set_xlim(lim, -lim)
    plt.gca().set_ylim(-lim, lim)
    
    tmin = bsbl.t0 - (duration/2 * bsbl.tE)
    tmax = bsbl.t0 + (duration/2 * bsbl.tE)
    
    t = np.linspace(tmin, tmax, time_steps)
    img, amp = bsbl.get_all_arrays(t)

    
    lens1, lens2 = bsbl.get_resolved_lens_astrometry(t)

    xS_resolved = bsbl.get_resolved_astrometry(t, image_arr = img)
    print(xS_resolved.shape)
    img_pri = xS_resolved[:, 0, :, :] 
    img_sec = xS_resolved[:, 1, :, :] 


    source_unlensed = bsbl.get_resolved_source_astrometry_unlensed(t)

    srce_pos_primary =source_unlensed[:, 0, :]
    srce_pos_secondary = source_unlensed[:, 1, :] 
    
    
    x = srce_pos_primary[:, 0]
    y = srce_pos_primary[:, 1]
    x2 = srce_pos_secondary[:, 0]
    y2 = srce_pos_secondary[:, 1]
    

    phot = bsbl.get_photometry(t, amp_arr=amp)
    
    plt.plot(x, y, label = 'Primary Source Position', linewidth = 4, color = 'green')
    plt.plot(x2, y2, label = 'Secondary Source Position', linewidth = 4, color='hotpink')

    if bsbl.muL[0] ==0 and bsbl.muL[1] ==0:
        plt.plot(lens1[:, 0], lens1[:, 1],  '.', markersize = 20, label = 'Primary Lens Position', color = 'red')
        plt.plot(lens2[:, 0], lens2[:, 1],  '.', markersize = 20, label = 'Secondary Lens Position', color = 'blue')
    else:
        plt.plot(lens1[:, 0], lens1[:, 1],  linewidth = 4, label = 'Primary Lens Position', color = 'red')
        plt.plot(lens2[:, 0], lens2[:, 1],  linewidth = 4, label = 'Secondary Lens Position', color = 'blue')



    for ii in range(5):
        if ii == 0:
            label_pri = 'Image Positions'
            #label_sec = 'Secondary Image Position'
        else:
            label_pri = ''
            label_sec = ''
        plt.plot(img_pri[:, ii, 0], img_pri[:, ii, 1], '.', linewidth = .1, alpha = 0.5, color = 'orange', label = label_pri)
        plt.plot(img_sec[:, ii, 0], img_sec[:, ii, 1], '.', linewidth = .1, alpha = 0.5, color = 'orange')

    #plt.title('Change in Primary and Secondary Lens and Source Position (Keplerian Orbit)')
    
    plt.xlabel(f'$\\Delta$ RA') 
    plt.ylabel(f'$\\Delta$ Dec')
    
    labels =['Primary Source Trajectory','Secondary Source Trajectory','Primary Lens Trajectory', 'Secondary Lens Trajectory', 'Images']
    ps = mpatches.Patch(facecolor='green', edgecolor = 'k') # This will create a red bar with black borders, you can leave out edgecolor if you do not want the borders
    ss = mpatches.Patch(facecolor='hotpink', edgecolor = 'k')
    pl = mpatches.Patch(facecolor='red', edgecolor = 'k')
    sl = mpatches.Patch(facecolor='blue', edgecolor = 'k')
    im = mpatches.Patch(facecolor='orange', edgecolor = 'k')
    fig.legend(handles = [ps, ss, pl, sl, im], labels=labels,
    loc='center left',
    bbox_to_anchor=(.9, 0.5),
    borderaxespad=0,
    frameon=False
)
    
    tright = .92
    ttop = .65
    #fig.text(tright, ttop - .1 , 't0 = {0:.2f} (MJD)'.format(bsbl.t0))
    #fig.text(tright, ttop - .15 , 'u0     = {0:.2f}'.format(bsbl.u0_amp))
    ##fig.text(tright, ttop - .2 , 'u0_pyL = {0:.3f}'.format(pylima_u0))
    #fig.text(tright, ttop - .2 , 'tE = {0:.2f} (day)'.format(bsbl.tE))
    
   # if bsbl.orbitFlag=='Keplerian':
        #fig.text(tright, ttop - .25 , 'iL = {0:.2f} (deg)'.format(bsbl.iL))
        #fig.text(tright, ttop - .3 , 'eL = {0:.2f}'.format(bsbl.eL))
        #fig.text(tright, ttop - .35 , 'pL = {0:.2f} (days)'.format(bsbl.pL))
        #fig.text(tright, ttop - .4 , 'iS = {0:.2f} (deg)'.format(bsbl.iS))
        #fig.text(tright, ttop - .45 , 'eS = {0:.2f}'.format(bsbl.eS))
        #fig.text(tright, ttop - .5 , 'pS = {0:.2f} (days)'.format(bsbl.pS))

              
    #plt.legend(loc='lower left') 
    
    return t, img, amp

def get_magnification_map(chosen_model, grid_size = 0.0312, plot_radius = 0.0156, lim = 0.01, bins=6000, cmap='seismic'):
    """
    For a given PSBL/BSBL model, plot the source trajectory on top 
    of the magnification map at time t0.

    Parameters
    ----------
    chosen_model : model object
        The  model to use for plotting.

    grid_size : float
        Window size in which the magnification map is generated. 

    plot_radius : float
        Radius within which the magnification map is generated.
    

    lim : float
        Limits on x and y axis for plotting purposes

    bins : int
        For the resolution of the magnification maps
    """
    
    if cmap != 'seismic':
        cmap = get_cmap(cmap)

    # An 8000 x 8000 grid takes a few seconds to run.

    # Get lenses info
    m1 = chosen_model.m1
    m2 = chosen_model.m2
    xL1_0, xL2_0 = chosen_model.get_resolved_lens_astrometry(t=np.array([chosen_model.t0]))

    z1 = xL1_0[0][0] + 1j*xL1_0[0][1]
    z2 = xL2_0[0][0] + 1j*xL2_0[0][1]

    # Set up magnification map grid, centered on lens.
    # zgrid are the image positions, where the shots end.
    # We want to find where they start (source plane), i.e.
    # inverse ray shooting
    grid_center = chosen_model.xL0_com *1e-3
    grid_size = grid_size # Probably a better way to do this...
    plot_radius = plot_radius

    xmin = grid_center[0] - grid_size
    xmax = grid_center[0] + grid_size
    ymin = grid_center[1] - grid_size
    ymax = grid_center[1] + grid_size

    x = np.linspace(xmin, xmax, bins)
    y = np.linspace(ymin, ymax, bins)
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

    plt.figure(1, figsize=(20,20))
    plt.clf()
    # magnification map and lenses
    H, xedges, yedges = np.histogram2d(wreal, wimag, bins=bins)
    val = plt.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=mpl.colors.CenteredNorm(), cmap=cmap)



    plt.plot(z1.real, z1.imag, markersize = 20, color = 'green', marker = '.', label = 'Primary Lens')
    plt.plot(z2.real, z2.imag, markersize = 20, color = 'darkslategrey', marker = 'H', label = 'Secondary Lens')

    
    plt.ylabel('Dec')
    plt.xlabel('RA')
    plt.title('Magnification Map')
    plt.xlim(lim, -lim)
    plt.ylim(-lim, lim)
    plt.colorbar(val)
    plt.legend(markerscale = 1)
    plt.show()

def get_magnification_map_timedep(chosen_model, time_skip = 500, time_choice = 1000, grid_size = 0.0312, plot_radius = 0.0156, time_steps=300, cmap = 'seismic', lim = 0.01, bins=6000):
    """
    Same as get_magnification_map() but for arbitrary time of your chosing. This function will generate a 2x2 grid with magnification maps for four times. 

    Parameters
    ----------
    chosen_model : model object
        The  model to use for plotting.

    time_choice : int
        Time before t0 to generate the first magnification map
    time_skip : int
        Intervals in the time array. 
        
    grid_size : float
        Window size in which the magnification map is generated. 

    plot_radius : float
        Radius within which the magnification map is generated.
    
    lim : float
        Limits on x and y axis for plotting purposes

    bins : int
        For the resolution of the magnification maps
    """

    # An 8000 x 8000 grid takes a few seconds to run.

    # Get lenses info

    def helper(t_obs, grid_size, plot_radius):
        m1 = chosen_model.m1
        m2 = chosen_model.m2
        xL1_0, xL2_0 = chosen_model.get_resolved_lens_astrometry(t_obs=np.array([t_obs]))
        z1 = xL1_0[0][0] + 1j*xL1_0[0][1]
        z2 = xL2_0[0][0] + 1j*xL2_0[0][1]
    
        # Set up magnification map grid, centered on lens.
        # zgrid are the image positions, where the shots end.
        # We want to find where they start (source plane), i.e.
        # inverse ray shooting
        grid_center = chosen_model.xL0 *1e-3
        grid_size = grid_size # Probably a better way to do this...
        plot_radius = plot_radius
        print(grid_size)
        print(plot_radius)

    
        xmin = grid_center[0] - grid_size
        xmax = grid_center[0] + grid_size
        ymin = grid_center[1] - grid_size
        ymax = grid_center[1] + grid_size
    
        x = np.linspace(xmin, xmax, 8000)
        y = np.linspace(ymin, ymax, 8000)
        xgrid, ygrid = np.meshgrid(x,y)
        zgrid = xgrid + 1j*ygrid
    
        # Get the source positions 
        w_points = plot_models.get_source_pos(zgrid, m1, m2, z1, z2)
    
        # There's a few points that get shot out far away
        # This trims them out
        dist2 = (w_points.real**2 + w_points.imag**2)
    
        # Separate into real and imaginary componenest for plotting
        wreal = w_points[np.where(dist2 < plot_radius)].real
        wimag = w_points[np.where(dist2 < plot_radius)].imag
    
        return z1, z2, wreal, wimag

    fig, ax = plt.subplots(2, 2, figsize=(25, 20))
    index = 0
    time = chosen_model.t0-time_choice
    count=0
    time_array = np.array([time, time+time_choice, time+time_choice*2, time+time_choice*3])
    phot = chosen_model.get_photometry(time_array)

    plt.title('Magnification Map')
    for i in range(0,2):
        for j in range(0,2):
            z1, z2, wreal, wimag = helper(time, grid_size, plot_radius)
            H, xedges, yedges = np.histogram2d(wreal, wimag, bins=bins)
            val = ax[i][j].imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=mpl.colors.CenteredNorm(), cmap=cmap)
            ax[i][j].plot(z1.real, z1.imag, markersize = 30, color = 'black', marker = '.', label = 'Primary Lens')
            ax[i][j].plot(z2.real, z2.imag, markersize = 20, color = 'darkslategrey', marker = 'H', label = 'Secondary Lens')
            ax[i][j].set_xlim(-lim, lim)
            ax[i][j].set_ylim(-lim, lim)
            ax[i][j].set_title(f'Time:{time}')
            ax[i][j].set_ylabel('Dec')
            ax[i][j].set_xlabel('RA')
            ax[i][j].legend(markerscale = 1)
            fig.colorbar(val)
            time = time + time_skip

def get_centroid_shift_map(model, grid_size=31.2, plot_radius=15.6, lim=8, bins=6000, cmap='seismic'):
    """
    Generate a color map of flux-weighted centroid shifts (in mas) for a uniform
    grid of sources using inverse ray shooting.

    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    arcsec2mas = 1e3
    mas2arcsec = 1.0 / arcsec2mas

    m1 = psbl.m1
    m2 = psbl.m2
    xL1_0, xL2_0 = psbl.get_resolved_lens_astrometry(t=np.array([psbl.t0]))
    z1_as = xL1_0[0][0] + 1j * xL1_0[0][1]   # arcsec
    z2_as = xL2_0[0][0] + 1j * xL2_0[0][1]   # arcsec
    z1_mas = z1_as * arcsec2mas #Lens position in mas
    z2_mas = z2_as * arcsec2mas#Lens position in mas

    
    
    grid_center_mas = (psbl.xL0_com + 1e-3) * arcsec2mas
    grid_size_mas = grid_size
    plot_radius_mas = plot_radius 
    lim_mas = lim

    xmin_mas = grid_center_mas[0] - grid_size_mas
    xmax_mas = grid_center_mas[0] + grid_size_mas
    ymin_mas = grid_center_mas[1] - grid_size_mas
    ymax_mas = grid_center_mas[1] + grid_size_mas

    x = np.linspace(xmin_mas, xmax_mas, bins)
    y = np.linspace(ymin_mas, ymax_mas, bins)   
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid_mas = xgrid + 1j * ygrid   
    
    z_flat = zgrid_mas.flatten()
    dist2_as = z_flat.real**2 + z_flat.imag**2
    valid_mask = dist2_mas < (plot_radius_mas ** 2) #Cut off faraway points
    z_flat_valid = z_flat[valid_mask]

    # arrays of lens positions in arcsec
    z1_arr = np.full_like(z_flat_valid, z1_mas)
    z2_arr = np.full_like(z_flat_valid, z2_mas)
    z_arr = psbl.get_image_pos_arr(z_flat_valid, z1_arr, z2_arr, m1, m2)
    mu_arr = psbl.get_amp_arr(z_arr, z1_arr, z2_arr)

    t_dummy = np.arange(len(z_flat_valid))
    xCentroid = psbl.get_astrometry(t_dummy, image_arr=z_arr, amp_arr=mu_arr)
    xCentroid_complex_mas = xCentroid[:, 0] + 1j * xCentroid[:, 1]
    shift_mas = np.abs(xCentroid_complex_as - z_flat_valid)


    shift_map = np.full(zgrid_mas.shape, np.nan, dtype=float)
    shift_map_flat = shift_map.flatten()
    shift_map_flat[valid_mask] = shift_mas
    shift_map = shift_map_flat.reshape(zgrid_as.shape)

    vmin = 0.0
    vmax = np.nanpercentile(shift_map, 99.99)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(shift_map, origin='lower', extent=[xmin_mas, xmax_mas, ymin_mas, ymax_mas],
                    cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)

    plt.plot(z1_mas.real, z1_mas.imag, 'black', marker='.', label='Primary Lens', markersize=10)
    plt.plot(z2_mas.real, z2_mas.imag, 'green', marker='^', label='Secondary Lens', markersize=10)

    plt.xlabel(r"$\Delta \alpha^*$ (mas)")
    plt.ylabel(rf"$\Delta \delta$ (mas)")
    plt.xlim(-lim, lim)   # lim in mas (user-facing)
    plt.ylim(-lim, lim)
    plt.legend()
    plt.tight_layout()
    plt.savefig('csmap.png')
    plt.show()
    

def compare_model_pkg_phot_amp(bagle_mod, time_arr, amp_pylima=None, amp_vbmicr=None, amp_mulens=None,
                          savefile='compare_model_pkg_phot_amp.png'):
    """
    Make plots to compare a BAGLE model against PyLIMA, VBM, and MulensModel. This function
    only produces a plot of the amplifications and residuals.
    """
    # Get the BAGLE amplification.
    amp_bagle = bagle_mod.get_amplification(time_arr)

    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make the figure
    ##########
    plt.close(2)
    fig, ax = plt.subplots(2, 1, sharex=True, num=2)

    if amp_pylima is not None:
        ax[0].plot(time_arr, amp_pylima, label='pyLIMA', color=colors[1], ls='-', lw=2, marker='None')
        ax[1].plot(time_arr, amp_pylima - amp_bagle, 'o', color=colors[1], lw=2, label='pyLIMA - BAGLE')
    if amp_vbmicr is not None:
        ax[0].plot(time_arr, amp_vbmicr, label='VBMicrolensing', color=colors[2], ls=':', lw=2, marker='None')
        ax[1].plot(time_arr, amp_vbmicr - amp_bagle, '^', color=colors[2], lw=2, label='VBMicrolensing - BAGLE')
    if amp_mulens is not None:
        ax[0].plot(time_arr, amp_mulens, label='MuLensModel', color=colors[3], ls='-.', lw=2, marker='None')
        ax[1].plot(time_arr, amp_mulens - amp_bagle, 'x', color=colors[3], lw=2, label='MuLensModel - BAGLE')

    ax[0].plot(time_arr, amp_bagle, label='BAGLE', color=colors[0], ls='-', lw=2, marker='None')

    ax[0].set_ylabel('Amplification')
    ax[0].legend(loc='upper right', fontsize=12)
    ax[1].legend(loc='upper left', fontsize=12)
    ax[1].set_ylabel('Difference')
    ax[1].set_xlabel('Time (MJD)')

    fig.savefig(savefile)

    print('compare_model_pkg_amp: BAGLE parameters')
    print(f"$t_{{E,\\odot}}={bagle_mod.tE:.0f}$ days, " +
          f"$u_{{0,\\odot}}={bagle_mod.u0_amp:.1f}$, " +
          f"$t_{{0,\\odot}}={bagle_mod.t0:.0f}$ MJD, " +
          f"$b_{{sff}}[0]={bagle_mod.b_sff[0]:.0f}$, " +
          f"$mag_S[0]={bagle_mod.mag_src[0]:.0f}$ mag"
          )

    return


def compare_model_pkg_phot_astrom_amp(time_arr, amp_bagle, amp_pylima=None, amp_vbmicr=None,
                          savefile='compare_model_pkg_phot_astrom_amp.png'):
    """
    Make plots to compare a BAGLE model against PyLIMA and VBM. This function
    only produces a plot of the amplifications and residuals.

    This function supports all BAGLE photometry + astrometry models.
    """
    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make the figure
    ##########
    plt.close(2)
    fig, ax = plt.subplots(2, 1, sharex=True, num=2)

    if amp_pylima is not None:
        ax[0].plot(time_arr, amp_pylima, label='pyLIMA', color=colors[1], ls='-', lw=2, marker='None')
        ax[1].plot(time_arr, amp_pylima - amp_bagle, 'o', color=colors[1], lw=2, label='pyLIMA - BAGLE')
    if amp_vbmicr is not None:
        ax[0].plot(time_arr, amp_vbmicr, label='VBMicrolensing', color=colors[2], ls=':', lw=2, marker='None')
        ax[1].plot(time_arr, amp_vbmicr - amp_bagle, '^', color=colors[2], lw=2, label='VBMicrolensing - BAGLE')

    ax[0].plot(time_arr, amp_bagle, label='BAGLE', color=colors[0], ls='-', lw=2, marker='None')

    ax[0].set_ylabel('Amplification')
    ax[0].legend(loc='upper right', fontsize=12)

    ax[1].legend(loc='upper left', fontsize=12)
    ax[1].set_ylabel('Difference')
    ax[1].set_xlabel('Time (MJD)')

    fig.savefig(savefile)

    return

def compare_model_pkg_phot_astrom_uamp(time_arr, uamp_bagle, uamp_pylima=None, uamp_vbmicr=None,
                                       savefile='compare_model_pkg_phot_astrom_uamp.png'):
    """
    Make plot of the amplitude of the u vector for different packages. This plots only
    photometry + astrometry models from BAGLE, pyLIMA, and VBM.
    """
    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make u-Amplitude Figure
    ##########
    fig, ax = plt.subplots(2, 1, sharex=True, num=2)

    if uamp_pylima is not None:
        ax[0].plot(time_arr, uamp_pylima, label='pyLIMA', color=colors[1], ls='-', lw=2, marker='None')
        ax[1].plot(time_arr, uamp_pylima - uamp_bagle, color=colors[1], label='pyLIMA - BAGLE', ls='-', lw=2, marker='None')
    if uamp_vbmicr is not None:
        ax[0].plot(time_arr, uamp_vbmicr, label='VBMicrolensing', color=colors[2], ls=':', lw=2, marker='None')
        ax[1].plot(time_arr, uamp_vbmicr - uamp_bagle, color=colors[2], label='VBMicrolensing - BAGLE', ls=':', lw=2, marker='None')

    ax[0].plot(time_arr, uamp_bagle, label='BAGLE', color=colors[0], ls='-', lw=2, marker='None')

    ax[0].set_ylabel('u ($\\theta_E$)')
    ax[0].legend(loc='upper right', fontsize=12)
    ax[1].legend(loc='lower right', fontsize=12)
    ax[1].set_ylabel('Diff')
    ax[1].set_xlabel('Time (MJD)')

    fig.savefig(savefile)

    return


def compare_model_pkg_phot_astrom_uvec(time_arr, uvec_bagle, uvec_pylima=None, uvec_vbmicr=None,
                                       savefile='compare_model_pkg_phot_astrom_uvec.png'):
    """
    Make plot of the amplitude of the u vector for different packages. This plots only
    photometry + astrometry models from BAGLE, pyLIMA, and VBM.
    """
    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make uvec Astrometry Figure
    ##########
    fig, ax = plt.subplots(2, 2, sharex=True, num=3, figsize=(12, 6))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, wspace=0.3)

    if uvec_pylima is not None:
        ax[0, 0].plot(time_arr, uvec_pylima[:, 0],
                      color=colors[1], label='pyLIMA',
                      ls='-', lw=2, marker='None')
        ax[0, 1].plot(time_arr, uvec_pylima[:, 1],
                      color=colors[1], label='pyLIMA',
                      ls='-', lw=2, marker='None')
        ax[1, 0].plot(time_arr, uvec_pylima[:, 0] - uvec_bagle[:, 0],
                      color=colors[1], label='pyLIMA - BAGLE',
                      ls='-', lw=2, marker='None')
        ax[1, 1].plot(time_arr, uvec_pylima[:, 1] - uvec_bagle[:, 1],
                      color=colors[1], label='pyLIMA - BAGLE',
                      ls='-', lw=2, marker='None')
        
    if uvec_vbmicr is not None:
        ax[0, 0].plot(time_arr, uvec_vbmicr[:, 0], 
                      color=colors[2], label='VBM',  
                      ls=':', lw=2, marker='None')
        ax[1, 0].plot(time_arr, uvec_vbmicr[:, 0] - uvec_bagle[:, 0], 
                      color=colors[2], label='VBM - BAGLE', 
                      ls=':', lw=2, marker='None')
        ax[1, 1].plot(time_arr, uvec_vbmicr[:, 1] - uvec_bagle[:, 1], 
                      color=colors[2], label='VBM - BAGLE', 
                      ls='-', lw=2, marker='None')
        ax[0, 1].plot(time_arr, uvec_vbmicr[:, 1], 
                      color=colors[2], label='VBM', 
                      ls='-', lw=2, marker='None')

    ax[0, 0].plot(time_arr, uvec_bagle[:, 0], label='BAGLE', color=colors[0], ls='-', lw=2, marker='None')
    ax[0, 1].plot(time_arr, uvec_bagle[:, 1], label='BAGLE', color=colors[0], ls='-', lw=2, marker='None')

    ax[0, 0].set_ylabel('u$_E$')
    ax[0, 0].legend(loc='upper right', fontsize=12)

    ax[1, 0].legend(loc='upper left', fontsize=12)
    ax[1, 0].set_ylabel('Difference')
    ax[1, 0].set_xlabel('Time (MJD)')

    ax[0, 1].set_ylabel('u$_N$')
    ax[0, 1].legend(loc='upper right', fontsize=12)

    ax[1, 1].legend(loc='upper left', fontsize=12)
    ax[1, 1].set_ylabel('Difference')
    ax[1, 1].set_xlabel('Time (MJD)')

    fig.savefig(savefile)

    return


def compare_model_pkg_phot_astrom_xSxL(time_mjd, xS_bagle, xL_bagle,
                                       xS_pylima=None, xL_pylima=None,
                                       xS_vbmicr=None, xL_vbmicr=None,
                                       savefile='compare_model_pkg_phot_astrom_xSxL.png'):
    """
    Make plot of the soure and lens position on the sky for different packages. This plots only
    photometry + astrometry models from BAGLE, pyLIMA, and VBM.
    """
    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make Unlensed Astrometry Figure
    ##########
    fig, ax = plt.subplots(2, 2, sharex=True, num=4, figsize=(12, 6))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, wspace=0.3)

    # ax[0, 0] == East vs. time
    # ax[0, 1] == North vs. time
    # ax[1, 0] == East residuals vs. time (mod - BAGLE)
    # ax[1, 1] == North residual vs. time (mod - BAGLE)

    if xS_pylima is not None:
        ax[0,0].plot(time_mjd, xS_pylima[:, 0] * 1e3,
                     label='xS$_{pyLIMA}$', color=colors[1],
                     ls='-', lw=2, marker='None')
        ax[0,0].plot(time_mjd, xL_pylima[:, 0] * 1e3,
                     label='xL$_{pyLIMA}$', color=colors[1],
                     ls=':', lw=2, marker='None')

        ax[0,1].plot(time_mjd, xS_pylima[:, 1] * 1e3,
                     label='xS$_{pyLIMA}$', color=colors[1],
                     ls='-', lw=2, marker='None')
        ax[0,1].plot(time_mjd, xL_pylima[:, 1] * 1e3,
                     label='xL$_{pyLIMA}$', color=colors[1],
                     ls=':', lw=2, marker='None')

        ax[1, 0].plot(time_mjd, (xS_pylima[:, 0] - xS_bagle[:, 0]) * 1e3,
                      label='xS$_{pyLIMA}$ - xS$_{BAGLE}$',
                      color=colors[1], ls='-', lw=2, marker='None')
        ax[1, 0].plot(time_mjd, (xL_pylima[:, 0] - xL_bagle[:, 0]) * 1e3,
                      label='xL$_{pyLIMA}$ - xL$_{BAGLE}$',
                      color=colors[1], ls=':', lw=2, marker='None')

        ax[1, 1].plot(time_mjd, (xS_pylima[:, 1] - xS_bagle[:, 1]) * 1e3,
                      label='xS$_{pyLIMA}$ - xS$_{BAGLE}$',
                      color=colors[1], ls='-', lw=2, marker='None')
        ax[1, 1].plot(time_mjd, (xL_pylima[:, 1] - xL_bagle[:, 1]) * 1e3,
                      label='xL$_{pyLIMA}$ - xL$_{BAGLE}$',
                      color=colors[1], ls=':', lw=2, marker='None')

    if xS_vbmicr is not None:
        ax[0,0].plot(time_mjd, xS_vbmicr[:, 0] * 1e3,
                     label='xS$_{VBM}$', color=colors[2],
                     ls='-', lw=2, marker='None')
        ax[0, 0].plot(time_mjd, xL_vbmicr[:, 0] * 1e3,
                      label='xL$_{VBM}$', color=colors[2],
                      ls=':', lw=2, marker='None')

        ax[0, 1].plot(time_mjd, xS_vbmicr[:, 1] * 1e3,
                      label='xS$_{VBM}$', color=colors[2],
                      ls='-', lw=2, marker='None')
        ax[0, 1].plot(time_mjd, xL_vbmicr[:, 1] * 1e3,
                      label='xL$_{VBM}$', color=colors[2],
                      ls=':', lw=2, marker='None')

        ax[1, 0].plot(time_mjd, (xS_vbmicr[:, 0] - xS_bagle[:, 0]) * 1e3,
                      label='xS$_{VBM}$ - xS$_{BAGLE}$',
                      color=colors[2], ls='-', lw=2, marker='None')
        ax[1, 0].plot(time_mjd, (xL_vbmicr[:, 0] - xL_bagle[:, 0]) * 1e3,
                      label='xL$_{VBM}$ - xL$_{BAGLE}$',
                      color=colors[2], ls=':', lw=2, marker='None')

        ax[1, 1].plot(time_mjd, (xS_vbmicr[:, 1] - xS_bagle[:, 1]) * 1e3,
                      label='xS$_{VBM}$ - xS$_{BAGLE}$',
                      color=colors[2], ls='-', lw=2, marker='None')
        ax[1, 1].plot(time_mjd, (xL_vbmicr[:, 1] - xL_bagle[:, 1]) * 1e3,
                      label='xL$_{VBM}$ - xL$_{BAGLE}$',
                      color=colors[2], ls=':', lw=2, marker='None')

    # E vs. t for BAGLE
    ax[0,0].plot(time_mjd, xS_bagle[:, 0] * 1e3,
                 label='xS$_{BAGLE}$', color=colors[0],
                 ls='-', lw=2, marker='None')

    ax[0,0].plot(time_mjd, xL_bagle[:, 0] * 1e3,
                 label='xL$_{BAGLE}$', color=colors[0],
                 ls=':', lw=2, marker='None')

    # N vs. t for BAGLE
    ax[0,1].plot(time_mjd, xS_bagle[:, 1] * 1e3,
                 label='xS$_{BAGLE}$', color=colors[0],
                 ls='-', lw=2, marker='None')

    ax[0,1].plot(time_mjd, xL_bagle[:, 1] * 1e3,
                 label='xL$_{BAGLE}$', color=colors[0],
                 ls=':', lw=2, marker='None')

    ax[0,0].legend(fontsize=12)
    ax[0,0].set_ylabel('$x_{\\odot} \\cdot \\hat{E}$ (mas)')
    ax[0,1].set_ylabel('$x_{\\odot} \\cdot \\hat{N}$ (mas)')
    ax[1,0].legend(fontsize=12)
    ax[1,0].set_ylabel('Difference')
    ax[1,0].set_xlabel("Time (MJD)")
    ax[1,1].set_ylabel('Difference')
    ax[1,1].set_xlabel("Time (MJD)")

    fig.savefig(savefile)

    return


def compare_model_pkg_phot_astrom_cent(time_mjd, ast_lensed_bagle,
                                       ast_lensed_pylima=None, ast_lensed_vbmicr=None,
                                       savefile='compare_model_pkg_phot_astrom_cent.png'):
    """
    Make plot of the lensed centroid on the sky for different packages. This plots only
    photometry + astrometry models from BAGLE, pyLIMA, and VBM.
    """
    # Use the following color table.
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))

    ##########
    # Make Lensed Astrometry Figure
    ##########
    fig, ax = plt.subplots(2, 2, sharex=True, num=5, figsize=(12, 6))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, wspace=0.3)

    # ax[0, 0] == East vs. time
    # ax[0, 1] == North vs. time
    # ax[1, 0] == East residuals vs. time (mod - BAGLE)
    # ax[1, 1] == North residual vs. time (mod - BAGLE)

    # PyLIMA
    if ast_lensed_pylima is not None:
        ax[0,0].plot(time_mjd, ast_lensed_pylima[:, 0] * 1e3,
                     label='x$_{lensed,pyLIMA}$',
                     color=colors[1], ls='-', lw=2, marker='None')
        ax[0, 1].plot(time_mjd, ast_lensed_pylima[:, 1] * 1e3,
                      label='x$_{lensed,pyLIMA}$',
                      color=colors[1], ls='-', lw=2, marker='None')
        ax[1, 0].plot(time_mjd, (ast_lensed_pylima[:, 0] - ast_lensed_bagle[:, 0]) * 1e3,
                      label='x$_{lensed,pyLIMA}$ - x$_{lensed,BAGLE}$',
                      color=colors[1], ls='-', lw=2, marker='None')
        ax[1, 1].plot(time_mjd, (ast_lensed_pylima[:, 1] - ast_lensed_bagle[:, 1]) * 1e3,
                      label='x$_{lensed,pyLIMA}$ - x$_{lensed,BAGLE}$',
                      color=colors[1], ls='-', lw=2, marker='None')

        # VBM
    if ast_lensed_vbmicr is not None:
        ax[0,0].plot(time_mjd, ast_lensed_vbmicr[:, 0] * 1e3,
                     label='x$_{lensed,VBMicrolensing}$',
                     color=colors[2], ls='-', lw=2, marker='None')
        ax[0, 1].plot(time_mjd, ast_lensed_vbmicr[:, 1] * 1e3,
                      label='x$_{lensed,VBMicrolensing}$',
                      color=colors[2], ls='-', lw=2, marker='None')
        ax[1, 0].plot(time_mjd, (ast_lensed_vbmicr[:, 0] - ast_lensed_bagle[:, 0]) * 1e3,
                      label='x$_{lensed,VBM}$ - x$_{lensed,BAGLE}$',
                      color=colors[2], ls='-', lw=2, marker='None')
        ax[1, 1].plot(time_mjd, (ast_lensed_vbmicr[:, 1] - ast_lensed_bagle[:, 1]) * 1e3,
                      label='x$_{lensed,VBM}$ - x$_{lensed,BAGLE}$',
                      color=colors[2], ls='-', lw=2, marker='None')

    # BAGLE
    ax[0,0].plot(time_mjd, ast_lensed_bagle[:, 0] * 1e3,
                 label='x$_{lensed,BAGLE}$',
                 color=colors[0], ls='-', lw=2, marker='None')
    ax[0,1].plot(time_mjd, ast_lensed_bagle[:, 1] * 1e3,
                 label='x$_{lensed,BAGLE}$',
                 color=colors[0], ls='-', lw=2, marker='None')

    ax[0,0].legend(fontsize=12)
    ax[0,0].set_ylabel('$x_{\\odot} \\cdot \\hat{E}$ (mas)')
    ax[0,1].set_ylabel('$x_{\\odot} \\cdot \\hat{N}$ (mas)')
    ax[1,0].legend(fontsize=12)
    ax[1,0].set_ylabel('Difference')
    ax[1,0].set_xlabel("Time (MJD)")
    ax[1,1].set_ylabel('Difference')
    ax[1,1].set_xlabel("Time (MJD)")

    fig.savefig(savefile)

    return
