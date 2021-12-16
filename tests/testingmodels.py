from src.micromodel import model 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


"""
Tutorial for using model.py

Starting with PSPL (point source, point lens)
First declare an instance of PSPL with model.PSPL(args)
as follows 
"""
#event1 = model.PSPL(15, 0, [20,20], -0.5, [0,0], [3,0], 150, 250, 10)

"""
The arguments to put into model.PSPL are as follows:
    1. Mass of the lens in solar masses
    2. Time of closest approach 
    3. [RA, Dec] position of source on the sky at closest approach 
    4. Angle between the source and lens on the sky at closest approach (mas)
    5. [RA, Dec] lens proper motion mas/yr
    6. [RA, Dec] source proper motion mas/yr
    7. Distance from observer to lens in parsecs
    8. Distance from observer to source in parsecs
    9. Image base flux

Once you have created the event, you can pass a list of times into multiple functions:
    self.get_amplification, self.get_photometry, self.get_centroid_shift, self.get_astrometry,
    self.get_astrometry_unlensed, self.get_lens_astrometry, self.get_resolved_shift, self.get_resolved_amplification,
    self.get_resolved_astrometry
An example is shown below:
    
Imagine I want to plot from -2, to 2 Einstein crossing times
"""
#t = np.array(range(-100,100,1))
#t = (t/50)*event1.tE
#A = event1.get_amplification(t)
#plt.plot(t,A)
"""
If you want to create an animation, use the following code: 
"""
#event1.animate(4, 150, 40, 'pspl', [10,10], 3, "yes")
"""
Animate takes in the following arguments:
    1. Number of Einstein times plotted
    2. Number of time steps, either side of the peak 
    3. Time for each frame in milliseconds
    4. Name of the created HTML file
    5. Size of figure in centimetres
    6. # of Einstein Radii plotted in vertical direction (horizontal direction set so both axes have the same scale)
    7. Whether the unresolved astrometry is plotted or not, enter "yes" or "no"
Now pspl.html, should be created in the same folder as this file, double click to open it in your defauly browser.
"""

ra = 269.9441667
dec = -28.6449444
mL = 10.0
t0 = 55150.0
xS0 = [0, 0]
beta = -2.0
muL = [0, 0]
muS = [5, 0]
dL = 4000
dS = 8000
blen_frac = 1
imag_base = 10

event2 = model.PSPL_parallax(ra, dec, mL, t0, xS0, beta, muL, muS, dL, dS, blen_frac, imag_base)
print('tE = ', event2.tE)
print('thetaE = ', event2.thetaE_amp)
event2.animate(7, 150, 30, 'psplparallax', [8,8], 3, "yes")
"""
Using PSPL_parallax is the pretty much the same, except the first 2 arguments when you call the class are
the lens positions in [RA,Dec], then all the other parameters are the same as for pspl
"""
#event3 = model.FSPL(15, 0, [20,20], -1.5, [0,0], [230,0], 150, 250, 40, 50, 10)
"""
To make an instance of a uniformly bright event, you must call FSPL, FSPL takes in all the same arguments as PSPL, except that after the source
distance it takes in:
    n = number of points used to approximate the boundary of the source
    R = source radius in solar radii
which in the above code are 40 and 150 respectively
"""
#event3.animate(4, 150, 30, 'FSPL', [10,10], 3)
"""
Creating an animation is exactly the same as above.
"""
#event4 = model.FSPL_Limb(11, 0, [0,0], -2.5, [0,0], [2,0], 4000, 8000, 40, 150, 0.5, 35, 10)
"""
In the limb darkening model, after the radius of the star, the next 2 parameters are
    u = limb darkening parameter, determines how darkened the source is. Must be a number between 0 and 1
        u = 0 is a uniformly bright source
    m = number of concentric circles in the integration grid
"""
#event4.animate(1.5, 150, 30, "FSPL_limb", [20,20], 3)
"""
t = np.array(range(-150,150,1))
t = (t/50)*event1.tE
A1 = event1.get_amplification(t)
A2 = event3.get_amplification(t)
A3 = event4.get_amplification(t)
plt.plot(t,A1, label='pspl')
plt.plot(t,A2, label='fspl')
plt.plot(t,A2, label='limb')
plt.legend()
"""

# event5 = model.FSBL(4, 6, 0, [0,0], -0.5, [0,0], [2,0], 4000, 8000, 40, 1, 0.004, 0, 0.5, 10)
# event5.animate(1., 300, 30, "FSBL", [20,20], 2.)
"""
t = np.array(range(-150,151,1))
t = (t/150)*event5.tE
t1 = t
values = event5.get_animation_points(t)
magnification = values[0]
images = values[2]
centroid = values[3]
x = centroid[:,0]
y = centroid[:,1]
caustic = event5.get_caustic(t)[0]
lens = event5.get_lens_astrometry(t)
lens1 = lens[0]
lens2 = lens[1]

t = np.array(range(-7,8,1))
t = (t/5)*event5.tE
source = event5.get_astrometry_unlensed(t)

figure = plt.figure(figsize=[20,20.25])
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)                           #sets up the figure
ax1 = figure.add_subplot(2,1,1)
ax2 = figure.add_subplot(2,1,2)    
figure.subplots_adjust(hspace=.25)

ax2.plot(t1, magnification)
ax2.set_xlabel("Time (days)", fontsize=40)
ax2.set_ylabel("Magnification", fontsize=40)


ax1.plot(x,y, 'm', markersize=5, label="Image Centroid")
ax1.scatter(source[0][0], source[0][1], c='r', label="Source")

for i in range(len(source)):
    circle=plt.Circle(source[i],event5.radius,color='r')
    ax1.add_patch(circle)
    ax1.scatter(source[i][0], source[i][1], c='r')

ax1.scatter(caustic[:,0], caustic[:,1], s=5, label="Caustic")
ax1.scatter(lens1[0][0], lens1[0][1], s=50, c='y', label="Lens")
ax1.scatter(lens2[0][0], lens2[0][1], s=50, c='y')
ax1.set_xlabel("RA (arcsec)", fontsize=40)
ax1.set_ylabel("Dec (arcsec)", fontsize=40)
lgnd = ax1.legend(fontsize=30, markerscale=8)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]

plt.savefig("Deviation.png")
"""
"""
t = np.array(range(-150,151,1))
t = (t/150)*event5.tE
tE = t/event5.tE
A = event5.get_amplification(t)


fig = plt.figure(figsize=[20,20.25])
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)                           #sets up the figure
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)    
fig.subplots_adjust(hspace=.25)
    

ax2.plot(tE,A)
ax2.set_xlabel("Time (tE)", fontsize=40)
ax2.set_ylabel("Magnification", fontsize=40)

t2 = np.array(range(-4,5,1))
t2 = 0.75*(t2/4)*event5.tE
caustic = event5.get_caustic(t2)
centre = event5.get_astrometry_unlensed(t2)
source = []
for i in range(len(centre)):
    ys = []
    for j in range(event5.n + 1):
        y1 = centre[i][0] + event5.radius*np.cos(2 * np.pi * j/event5.n)
        y2 = centre[i][1] + event5.radius*np.sin(2 * np.pi * j/event5.n)
        ys.append((y1,y2))
    source.append(np.array(ys))
lens = event5.get_lens_astrometry(t2)


ax1.scatter(lens[0][0][0], lens[0][0][1], s=30, label="Masses", c='r')
ax1.scatter(lens[1][0][0], lens[1][0][1], s=30, c='r')
ax1.scatter(caustic[0][:,0], caustic[0][:,1], label="Caustic", s=10)
ax1.scatter(source[0][:,0], source[0][:,1], c='g', label="Source Progression", s=10)    
for j in range(len(centre)-1):
    ax1.scatter(source[j+1][:,0], source[j+1][:,1], c='g', s=10)
ax1.set_xlim(-0.003, 0.003)
ax1.set_ylim(-0.0015, 0.0015)
ax1.set_xlabel("RA", fontsize=40)
ax1.set_ylabel("Dec", fontsize=40)
ax1.legend(fontsize=25, markerscale=3)

    
plt.savefig("causticcrossing.png")
"""

                








