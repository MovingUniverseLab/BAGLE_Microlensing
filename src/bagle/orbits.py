import os, sys, math
import numpy as np
from astropy.table import Table, Column, MaskedColumn
import matplotlib.pyplot as plt
import astropy.constants as c
from astropy import units as u
from copy import deepcopy
from random import choice

days_per_year = 365.25

class Orbit(object):

    def oal2xy(self, t, mass=None, dist=None, accel=False):
        """
        Projected on-sky positions of both binary components about their CoM.

        Uses the Thiele-Innes constants (Koren et al. 2016) to map the orbit
        onto the plane of the sky. The centre-of-mass motion is not included;
        add it separately.

        Parameters
        ----------
        t : array_like
            Epochs, in the same time units as ``self.p`` and ``self.tp``
            (days in the BAGLE models).
        mass, dist, accel : optional
            Unused; kept for backward compatibility.

        Returns
        -------
        x, y : numpy array, shape = [len(t)]
            East and North offsets of the primary from the centre of mass,
            in the units of ``self.aleph`` (arcsec in the BAGLE models).
        x2, y2 : numpy array, shape = [len(t)]
            East and North offsets of the secondary, in the units of
            ``self.aleph2``.

        Notes
        -----
        Required attributes:

        * ``w`` -- argument of periastron of the primary (deg).
        * ``o`` -- longitude of the ascending node, Omega (deg).
        * ``i`` -- inclination (deg).
        * ``e`` -- eccentricity; values outside [0, 1] return (-1e8, -1e8).
        * ``p``, ``tp`` -- period and time of periastron.
        * ``aleph``, ``aleph2`` -- semi-major axes of the primary and the
          secondary about the centre of mass.

        Angle conventions:

        * Omega is the position angle of the ascending node, measured from
          North through East. Both components share the same line of nodes
          and the same ascending node at PA = Omega.
        * The ascending node is where a component crosses the plane of the
          sky moving toward +Z, the line-of-sight coordinate
          ``Z = conC * X + conH * Y`` (taken as pointing away from the
          observer).
        * ``w`` is measured in the orbital plane from the ascending node to
          the primary's periastron, in the direction of orbital motion.
        * The secondary uses ``w + 180``: its periastron is ``w + 180``
          from the ascending node, or equivalently ``w`` beyond the node
          at PA = Omega + 180.
        * i = 0 is face-on with motion from North through East (position
          angle increasing); 0 < i < 90 is prograde, i = 90 edge-on, and
          i > 90 retrograde (position angle decreasing). For i = 0 the
          primary's periastron is at PA = Omega + w and the secondary's at
          PA = Omega + w + 180.

        Examples
        --------
        >>> orb = orbits.Orbit()
        >>> orb.w = omega       # deg
        >>> orb.o = big_omega   # deg
        >>> orb.i = incl        # deg
        >>> orb.e = e_mag
        >>> orb.p = p           # days
        >>> orb.tp = tp         # days
        >>> orb.aleph = aleph   # arcsec
        >>> orb.aleph2 = aleph2 # arcsec
        >>> x, y, x2, y2 = orb.oal2xy(t)
        """

        
        if self.e <0 or self.e>1:
            return(-1e8, -1e8)
        meanMotion = 2.0 * math.pi / self.p

        ecc_sqrt = np.sqrt(1.0 - self.e**2)

        # Mean anomaly
        M = meanMotion * (t - self.tp)
        
        # Eccentric anomaly
        E = self.eccen_anomaly(M, self.e)

        # True Anomaly
        eta = 2* np.arctan(np.sqrt((1+self.e)/(1-self.e)) * np.tan(E/2))

        

        # elliptical rectangular coordinates Of the Orbit
        cos_E = np.cos(E)
        sin_E = np.sin(E)

        X = cos_E - self.e
        Y = ecc_sqrt * sin_E

        #----------
        # Calculate Thiele-Innes Constants for the Primary
        #----------
        cos_bigOm = np.cos(math.radians(self.o))
        sin_bigOm = np.sin(math.radians(self.o))
        cos_i = np.cos(math.radians(self.i))
        sin_i = np.sin(math.radians(self.i))
        cos_om = np.cos(math.radians(self.w))
        sin_om = np.sin(math.radians(self.w))
        

        self.conA = self.aleph * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
        self.conB = self.aleph * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
        self.conC = self.aleph * (sin_om * sin_i)
        self.conF = self.aleph * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
        self.conG = self.aleph * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
        self.conH = self.aleph * (cos_om * sin_i)


        
        #----------
        # Calculate Thiele-Innes Constants for the Secondary
        #----------
        
        cos_om2 = np.cos(math.radians(self.w+180))
        sin_om2 = np.sin(math.radians(self.w+180))

        
        self.conA2 = self.aleph2 * (cos_om2 * cos_bigOm  - sin_om2 * sin_bigOm * cos_i)
        self.conB2 = self.aleph2 * (cos_om2 * sin_bigOm  + sin_om2 * cos_bigOm * cos_i)
        self.conC2 = self.aleph2 * (sin_om2 * sin_i)
        self.conF2 = self.aleph2 * (-sin_om2 * cos_bigOm - cos_om2 * sin_bigOm * cos_i)
        self.conG2 = self.aleph2 * (-sin_om2 * sin_bigOm + cos_om2 * cos_bigOm * cos_i)
        self.conH2 = self.aleph2 * (cos_om2 * sin_i)



        # calculate x and y
        #The x and y that come out are the delta position of the star which you would then need to add to the center of mass 
        #x = (self.conB * X) + (self.conG * Y) + self.vx * (t-t0_com)/days_per_year + self.x0 
        #y = (self.conA * X) + (self.conF * Y) + self.vy * (t-t0_com)/days_per_year + self.y0
        #x2 = (self.conB2 * X) + (self.conG2 * Y) + self.vx * (t-t0_com)/days_per_year + (self.x0)
        #y2 = (self.conA2 * X) + (self.conF2 * Y) + self.vy * (t-t0_com)/days_per_year + (self.y0)
        #self.K1 = 3
        #self.gamma = 2
        #rv = self.K1 * (np.cos(math.radians(self.w)) + eta) + self.e * np.cos(math.radians(self.w)) + self.gamma
        #a1_num = self.K1  * self.p * np.sqrt(1-self.e**2)
        #a1_den = 2*np.pi * np.sin(self.i)
        #a1 = a1_num/a1_den
        #a1 = np.abs(a1)
        #rv is radial velocity
            
        
        # Unit conversions
        # r  from AU     to arcsec
        # v  from AU/yr  to mas/yr
        
        #x /= cc.dist
        #y /= cc.dist

        # x, y, x2, y2 = x/self.dS, y/self.dS, x2/self.dS, y2/self.dS
        x = (self.conB * X) + (self.conG * Y)
        y = (self.conA * X) + (self.conF * Y) 
        x2 = (self.conB2 * X) + (self.conG2 * Y) 
        y2 = (self.conA2 * X) + (self.conF2 * Y) 
        
        return (x, y, x2, y2)
        
    """
    Exerpted from orbits.py from gcworks
    """
    def kep2xyz(self, epochs):
        """
        Generates (r, v, a) in AU, AU/yr, and AU/yr^2 respectively from keplarian parameters. 
        
        epoch: numpy array
           Array of times (in years) that you would like to compute the parameters for
        
        mass: float or array-like
           Mass of primary object
        

        An example call is:

        orb = orbits.Orbit()
        orb.w = omega # [degrees]
        orb.o = bigOm # [degrees]
        orb.i = incl # [degrees]
        orb.e = e_mag # float between 0 and 1
        orb.p = p # [years]
        orb.t0 = t0 # [years] This is initial
        orb.mass = mass # [Msun]

        (r, v, a) = orb.kep2xyz(array([refTime]))
        
        """

        GM = self.mass * c.G.to("cm3/(Msun s2)")

        epoch_num = len(epochs)

        # meanMotion in radians per year
        meanMotion = 2.0 * math.pi / self.p
        
        # Semi-major axis in AU
        axis = (self.p**2 * self.mass)**(1.0/3.0)

        ecc_sqrt = np.sqrt(1.0 - self.e**2)        
        
        #switch angular quantities to radians
        w = math.radians(self.w)
        o = math.radians(self.o)
        i = math.radians(self.i)
        
        # Mean anomaly
        mean_anomaly = meanMotion * (epochs - self.t0)

        #----------
        # Now for each epoch we compute the x and y positions
        #----------

        # Eccentric anomaly
        E = self.eccen_anomaly(mean_anomaly, self.e)
            
        cos_E = np.cos(E)
        sin_E = np.sin(E)
        
        Edot = meanMotion / (1.0 - (self.e * cos_E))

        X = cos_E - self.e
        Y = ecc_sqrt * sin_E

        #----------
        # Calculate Thiele-Innes Constants
        #----------
        cos_bigOm = np.cos(o)
        sin_bigOm = np.sin(o)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        
        cos_om = np.cos(w)
        sin_om = np.sin(w)

        self.conA = axis * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
        self.conB = axis * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
        self.conC = axis * (sin_om * sin_i)
        self.conF = axis * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
        self.conG = axis * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
        self.conH = axis * (cos_om * sin_i)
        
        # initialize zero arrays for r, v, and a
        r = np.zeros((epoch_num, 3), dtype='float64')
        v = np.zeros((epoch_num, 3), dtype='float64')
        a = np.zeros((epoch_num, 3), dtype='float64')

        r[:,0] = (self.conB * X) + (self.conG * Y)
        r[:,1] = (self.conA * X) + (self.conF * Y)
        r[:,2] = (self.conC * X) + (self.conH * Y)
        
        v[:,0] = Edot * ((-self.conB * sin_E) + (self.conG * ecc_sqrt * cos_E))
        v[:,1] = Edot * ((-self.conA * sin_E) + (self.conF * ecc_sqrt * cos_E))
        v[:,2] = Edot * ((-self.conC * sin_E) + (self.conH * ecc_sqrt * cos_E))
        
    
        # Calculate accleration
        for ii in range(epoch_num):
            rmag_cm = (np.sqrt( (r[ii,:]**2).sum() )*(u.au)).to("cm").value
            a[ii,:] = -GM * (r[ii,:]*(u.au)).to("cm").value / rmag_cm**3
        
        # from cm/s^2 to AU/yr^2
        a = (a*(u.cm/u.s**2)).to("au/yr2").value

        return (r, v, a)


    def eccen_anomaly(self, m, ecc, thresh=1e-10):
        """
        Calculates the eccentricity anomaly
        
        m: numpy array
            Mean anomalies
            
        ecc: float between 0-1
            The eccentricity of the orbit
        """
        # set default values

        if (ecc < 0. or ecc >= 1.):
            raise Exception('Eccentricity must be >= 0 and < 1')
            
        #
        # Range reduction of m to -pi < m <= pi
        #
        mx = m.copy()

        ## ... m > pi
        zz = (np.where(mx > math.pi))[0]
        mx[zz] = mx[zz] % (2.0 * math.pi)
        zz = (np.where(mx > math.pi))[0]
        mx[zz] = mx[zz] - (2.0 * math.pi)
        
        # ... m < -pi
        zz = (np.where(mx <= -math.pi))[0]
        mx[zz] = mx[zz] % (2.0 * math.pi)
        zz = (np.where(mx <= -math.pi))[0]
        mx[zz] = mx[zz] + (2.0 * math.pi)

        #
        # Bail out for circular orbits...
        #
        if (ecc == 0.0):
            return mx

        aux   = (4.0 * ecc) + 0.50
        alpha = (1.0 - ecc) / aux

        beta = mx/(2.0*aux)
        aux = np.sqrt(beta**2 + alpha**3)
   
        z=beta+aux
        zz=(np.where(z <= 0.0))[0]
        z[zz]=beta[zz]-aux[zz]

        test=abs(z)**0.3333333333333333

        z =  test.copy()
        zz = (np.where(z < 0.0))[0]
        z[zz] = -z[zz]

        s0=z-alpha/z
        s1 = s0-(0.0780 * s0**5) / (1.0 + ecc)
        e0 = mx + ecc*((3.0 * s1) - (4.0 * s1**3))

        se0=np.sin(e0)
        ce0=np.cos(e0)

        f  = e0 - (ecc*se0) - mx
        f1 = 1.0 - (ecc*ce0)
        f2 = ecc*se0
        f3 = ecc*ce0
        f4 = -1.0 * f2
        u1 = -1.0 * f/f1
        u2 = -1.0 * f/(f1 + 0.50*f2*u1)
        u3 = -1.0 * f/(f1 + 0.50*f2*u2
                 + 0.166666666666670*f3*u2*u2)
        u4 = -1.0 * f/(f1 + 0.50*f2*u3
                 + 0.166666666666670*f3*u3*u3
                 + 0.0416666666666670*f4*u3**3)

        eccanom=e0+u4

        zz = (np.where(eccanom >= 2.00*math.pi))[0]
        eccanom[zz]=eccanom[zz]-2.00*math.pi
        zz = (np.where(eccanom < 0.0))[0]
        eccanom[zz]=eccanom[zz]+2.00*math.pi

        # Now get more precise solution using Newton Raphson method
        # for those times when the Kepler equation is not yet solved
        # to better than 1e-10
        # (modification J. Wilms)

        mmm = mx.copy()
        ndx = (np.where(mmm < 0.))[0]
        mmm[ndx] += (2.0 * math.pi)
        diff = eccanom - ecc*np.sin(eccanom) - mmm

        ndx = (np.where(abs(diff) > 1e-10))[0]
        for i in ndx:
            # E-e sinE-M
            fe = eccanom[i]-ecc*np.sin(eccanom[i])-mmm[i]
            # f' = 1-e*cosE
            fs = 1.0 - ecc*np.cos(eccanom[i])
            oldval=eccanom[i]
            eccanom[i]=oldval-fe/fs

            loopCount = 0
            while (abs(oldval-eccanom[i]) > thresh):
                # E-e sinE-M
                fe = eccanom[i]-ecc*np.sin(eccanom[i])-mmm[i]
                # f' = 1-e*cosE
                fs = 1.0 - ecc*np.cos(eccanom[i])
                oldval=eccanom[i]
                eccanom[i]=oldval-fe/fs
                loopCount += 1
                
                if (loopCount > 10**6):
                    msg = 'eccen_anomaly: Could not converge for e = %f' % ecc
                    raise EccAnomalyError(msg)

            while (eccanom[i] >=  math.pi):
                eccanom[i] = eccanom[i] - (2.0 * math.pi)
                
            while (eccanom[i] < -math.pi ):
                eccanom[i] = eccanom[i] + (2.0 * math.pi)

        return eccanom
    
class EccAnomalyError(Exception):
    def __init__(self, message):
        self.message = message

def a_to_P(mass, a):
    """
    Goes from semimajor axis in AU to period in years
    
    Parameters
    ---------- 
    mass: float or array-like
        Primary object mass in Msun.
    
    a: float or array-like
        Semimajor axis in AU.
        
    Returns
    ---------- 
    period: float or array-like
        Orbital period in years.
    """
    
    G_units = c.G.to("AU3/(M_sun*year2)").value
    period = (a**3*4*(np.pi**2)/G_units/mass)**(1/2)
    return period


def relative_state_to_elements(t0, r_ENZ, v_ENZ, a):
    """
    Keplerian elements (``Orbit.oal2xy`` convention) from a relative state.

    Parameters
    ----------
    t0 : float
        Epoch of the state (MJD).
    r_ENZ : array_like, shape (3,)
        Position of the secondary relative to the primary at ``t0``: East,
        North, and line of sight (any length unit L).
    v_ENZ : array_like, shape (3,)
        Velocity of the secondary relative to the primary at ``t0`` (L / yr).
    a : float
        Semi-major axis of the relative orbit (L). Must exceed ``|r| / 2``
        for a bound orbit.

    Returns
    -------
    e : float
        Eccentricity.
    i : float
        Inclination (deg).
    big_omega : float
        Longitude of the ascending node (deg), ``Orbit.o``.
    omega_pri : float
        Argument of periastron of the primary's orbit (deg), ``Orbit.w``.
    p : float
        Period (days).
    tp : float
        Time of periastron (MJD).

    Notes
    -----
    ``Orbit.oal2xy`` places a body at ``aleph * (P X + Q Y)`` with
    ``X = cos E - e``, ``Y = sqrt(1 - e^2) sin E`` and, in (North, East,
    line of sight) components,

    * ``P = (cos w cos O - sin w sin O cos i, cos w sin O + sin w cos O cos i,
      sin w sin i)``,
    * ``Q = (-sin w cos O - cos w sin O cos i, -sin w sin O + cos w cos O cos i,
      cos w sin i)``,

    so the orbit normal is ``W = P x Q = (sin O sin i, -cos O sin i, cos i)``.
    The secondary uses ``w + 180``, so the secondary-minus-primary orbit has
    ``w_rel = w_pri + 180`` and semi-major axis ``aleph + aleph2``. The
    elements are recovered from the angular momentum (``W``), the
    eccentricity vector (``P``; the position at ``t0`` for circular orbits),
    and the true anomaly at ``t0``.
    """
    # Work in (North, East, line of sight) to match the Thiele-Innes terms.
    r = np.array([r_ENZ[1], r_ENZ[0], r_ENZ[2]], dtype=float)
    v = np.array([v_ENZ[1], v_ENZ[0], v_ENZ[2]], dtype=float)
    r_amp = np.linalg.norm(r)
    if not 0.5 * r_amp < a:
        raise ValueError(f"Unbound orbit: a = {a} must exceed "
                         f"|r|/2 = {0.5 * r_amp}")

    # Gravitational parameter from vis-viva (L^3 / yr^2) and mean motion.
    GM = np.dot(v, v) / (2.0 / r_amp - 1.0 / a)
    n = np.sqrt(GM / a ** 3)  # rad / yr

    # Orbit normal -> inclination and node.
    h = np.cross(r, v)
    W = h / np.linalg.norm(h)
    i = np.arccos(np.clip(W[2], -1.0, 1.0))
    big_omega = np.arctan2(W[0], -W[1])

    # Periastron direction of the relative orbit; for circular orbits the
    # position at t0 is used (so t0 is the periastron time).
    e_vec = np.cross(v, h) / GM - r / r_amp
    e = np.linalg.norm(e_vec)
    if e > 1e-12:
        P = e_vec / e
    else:
        e = 0.0
        P = r / r_amp

    # Argument of periastron from the node: P = cos w n + sin w (W x n).
    node = np.array([np.cos(big_omega), np.sin(big_omega), 0.0])
    omega_rel = np.arctan2(np.dot(P, np.cross(W, node)), np.dot(P, node))

    # True -> eccentric -> mean anomaly at t0.
    Q = np.cross(W, P)
    nu = np.arctan2(np.dot(r, Q), np.dot(r, P))
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * nu),
                         np.sqrt(1.0 + e) * np.cos(0.5 * nu))
    M = E - e * np.sin(E)

    # Periastron time and period in days.
    tp = t0 - M / n * days_per_year
    p = 2.0 * np.pi / n * days_per_year

    # The primary's argument of periastron is 180 deg from the relative orbit's.
    omega_pri = (np.rad2deg(omega_rel) - 180.0) % 360.0

    return e, np.rad2deg(i), np.rad2deg(big_omega) % 360.0, omega_pri, p, tp


def add_positions(ss):
    """
    Adds x and y positions randomly in a box of length and width 40000 AU for each system.
    
    Parameters
    ----------   
    ss: astropy table
        Star system table without positions
        
    Returns
    -------
    ss_temp: astropy table
        Star system table with positions added
    """
    ss_temp = deepcopy(ss)
    
    ss_temp.add_column( Column(np.zeros(len(ss_temp), dtype=float), name='x', description='AU') )
    ss_temp.add_column( Column(np.zeros(len(ss_temp), dtype=float), name='y', description='AU') )
    
    sign_x = np.array([choice([-1,1]) for i in range(len(ss_temp))])
    sign_y = np.array([choice([-1,1]) for i in range(len(ss_temp))])
    ss_temp['x'] = sign_x*20000*np.random.rand(len(ss_temp))
    ss_temp['y'] = sign_y*20000*np.random.rand(len(ss_temp))
    
    return ss_temp

def add_mult_positions(companions, ss_pos, logAge, add_velocities = False):
    """
    Adds x and y positions of multiple companions by transforming keplerian parameters to xyz in AU
    using code origially from gcworks and random initial times. Then adding them to the random posiiton of the primary object.
    
    Parameters
    ----------    
    ss_pos: astropy table
        Star system table with positions added with add_positions()
    
    companion: astropy table
        Companion table without positions
    
    logAge: float or int
        Log of age of cluster with age in years
        
    Returns
    -------
    companion_temp: astropy table
        Companion table with positions added
        
    """
    companions_temp = deepcopy(companions)
    
    companions_temp.add_column( Column(np.zeros(len(companions_temp), dtype=float), name='x', description='AU') )
    companions_temp.add_column( Column(np.zeros(len(companions_temp), dtype=float), name='y', description='AU') )
    
    if add_velocities == True:
        companions_temp.add_column( Column(np.zeros(len(companions_temp), dtype=float), name='v_x', description='AU/yr') )
        companions_temp.add_column( Column(np.zeros(len(companions_temp), dtype=float), name='v_y', description='AU/yr') )
        
    orb = Orbit()
    for i in companions_temp:
        orb.w = i['omega'] #degrees
        orb.o = i['Omega'] #degrees
        orb.i = i['i'] #degrees
        orb.e = i['e'] #between 0 and 1
        orb.p = a_to_P(ss_pos[i['system_idx']]['systemMass'],10**i['log_a']) #year
        orb.t0 = (10**logAge)*np.random.rand() #year
        orb.mass = ss_pos[i['system_idx']]['mass']
        
        (r, v, a) = orb.kep2xyz(np.array([10**logAge]))
               
        x = r[0][0]
        y = r[0][1]
        
        #putting positions relative to primary object
        i['x'] = ss_pos[i['system_idx']]['x'] + x #AU
        i['y'] = ss_pos[i['system_idx']]['y'] + y #AU
        
        if add_velocities == True:
            i['v_x'] = v[0][0] #AU/yr
            i['v_y'] = v[0][1] #AU/yr
    
    
    return companions_temp

def distance_to_center_of_mass(ss_pos, companions_pos):
    """
    Adds extra column to star system and companions table with x and y distance to the center of mass in AU.
    Assumes hierarchical triples (two closest stars orbit their center of mass and triple orbits them) and no quads+.
    
    Parameters
    ----------    
    ss_pos: astropy table
        Star system table with positions added with add_positions()
    
    companion_pos: astropy table
        Companion table with positions added with add_mult_positions()
        
    Results
    -------- 
    ss_pos_temp: astropy table
        Star system table with distance to center of mass in AU added
    
    companion_pos_temp: astropy table
        Companion table with distance to center of mass in AU added
    """
    
    ss_pos_temp = deepcopy(ss_pos)
    companions_pos_temp = deepcopy(companions_pos)
    
    ss_pos_temp.add_column( Column(np.zeros(len(ss_pos_temp), dtype=float), name='com_x') )
    ss_pos_temp.add_column( Column(np.zeros(len(ss_pos_temp), dtype=float), name='com_y') )
    companions_pos_temp.add_column( Column(np.zeros(len(companions_pos_temp), dtype=float), name='com_x') )
    companions_pos_temp.add_column( Column(np.zeros(len(companions_pos_temp), dtype=float), name='com_y') )
    
    companions_pos_temp.sort(['system_idx','log_a'])
    
    for i in range(len(ss_pos_temp)):
        if ss_pos_temp[i]['isMultiple'] == True:
            companion_indicies = np.where(companions_pos_temp['system_idx'] == i)[0]
            
            primary_mass = ss_pos_temp[i]['mass']
            companion_masses = companions_pos_temp[companion_indicies]['mass']
            
            primary_x = ss_pos_temp[i]['x']
            companion_x = companions_pos_temp[companion_indicies]['x']
            com_x = (primary_mass*primary_x + companion_masses[0]*companion_x[0])/(primary_mass + companion_masses[0])
            
            primary_y = ss_pos_temp[i]['y']
            companion_y = companions_pos_temp[companion_indicies]['y']
            com_y = (primary_mass*primary_y + companion_masses[0]*companion_y[0])/(primary_mass + companion_masses[0])
            
            ss_pos_temp[i]['com_x'] = com_x - primary_x
            companions_pos_temp[companion_indicies[0]]['com_x'] = com_x - companion_x[0]
            ss_pos_temp[i]['com_y'] = com_y - primary_y
            companions_pos_temp[companion_indicies[0]]['com_y'] = com_y - companion_y[0]
            
            # Assumes hierarchical triples 
            if len(companion_indicies) == 2:
                center_mass = primary_mass + companion_masses[0]
                com_x_out = (center_mass*com_x + companion_masses[1]*companion_x[1])/(center_mass + companion_masses[1])
                com_y_out = (center_mass*com_y + companion_masses[1]*companion_y[1])/(center_mass + companion_masses[1])
                
                companions_pos_temp[companion_indicies[1]]['com_x'] = com_x_out - companion_x[1]
                companions_pos_temp[companion_indicies[1]]['com_y'] = com_y_out - companion_y[1]
                
    return ss_pos_temp, companions_pos_temp
                
            
def plot_projected_cluster(ss_pos, companions_pos):
    """
    Plots projected cluster with lines between companions and primary stars
    
    Parameters
    ----------   
    ss_pos: astropy table
        Star system table with positions added with add_positions()
    
    companion_pos: astropy table
        Companion table with positions added with add_mult_positions()
    """
    plt.figure(figsize=(10,10))
    plt.plot(ss_pos['x'], ss_pos['y'],linestyle='none',marker='o' )
    plt.plot(companions_pos['x'], companions_pos['y'],linestyle='none',marker='.' )
    
    #makes lines between companion and primary star
    for i in companions_pos:
        plt.plot([i['x'], ss_pos[i['system_idx']]['x']],[i['y'], ss_pos[i['system_idx']]['y']],color='grey',linewidth=1)
        
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.show()
    
    return

def plot_companion_orbit(ss, companions_pos, logAge, t0 = None, system = None):
    """
    Plots the orbit of one system assuming the primary object is at (0,0). By default random companion and initial time.
    
    Parameters
    ----------       
    ss: astropy table
        Star system table (does not matter if it has positions or not)
    
    companion_pos: astropy table
        Companion table with positions added with add_mult_positions()
    
    logAge: float or int
        Log of age of cluster with age in years
        
    t0: float or int, optional
        Initial time of creation of the system in years.
        Default random.
    
    system: int, optional
        Index of desired companion in companion_pos table.
        Default random.
    """
    
    if system == None:
        system = np.random.randint(len(companions_pos))
    if t0 == None:
        t0 = (10**logAge)*np.random.rand() #year

    companion = companions_pos[system]
    
    orb = Orbit()
    orb.w = companion['omega'] #degrees
    orb.o = companion['Omega'] #degrees
    orb.i = companion['i'] #degrees
    orb.e = companion['e'] #between 0 and 1
    orb.p = a_to_P(ss[companion['system_idx']]['mass'],10**companion['log_a']) #year
    orb.t0 = t0 #year
        
    (r, v, a) = orb.kep2xyz(np.linspace(1, 10**logAge), mass=ss[companion['system_idx']]['mass'])
        
    for i in r:
        plt.plot(i[0],i[1], marker='.', color = 'blue')
        if i[0] == r[-1][0]:
            plt.plot(i[0],i[1], marker='*', color = 'gold', markersize=15)
            
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.show()
    
    return