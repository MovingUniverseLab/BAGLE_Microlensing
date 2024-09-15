class Orbit(object):
    def xyz2kep(self, r, v, re, ve, epoch, mass=None, dist=None):
        """
        Converts from cartesian (position, velocity) elements to Keplerian
        elements (inclination, eccentricity, omega, bigOmega, t0, period).

        @param r: 3D vector containing position of secondary (arcsec).
        @type r: numarray
        @param v: 3D vector containing velocity of secondary (km/s).
        @type v: numarray
        @param re: 3D vector containing position errors (arcsec).
        @type re: numarray
        @param ve: 3D vector containing velocity errors (km/s).
        @type veh: numarray
        @param epoch: The epoch at which the r and v vectors were observed.
        @type epoch: float

        @kwparam mass: Black hole mass in solar mass units
                       (default pulled from Constants).
        @kwparam dist: Black hole distance in parsec
                       (default pulled from Constants).

        @return elements: object containing orbital elements
        """
        cc = objects.Constants()

        if (mass != None):
            cc.mass = mass
        if (dist != None):
            cc.dist = dist
            cc.asy_to_kms = cc.dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)

        GM = cc.msun * cc.mass * cc.G

        # Convert position from arcsec to cm
        r *= cc.dist * cc.cm_in_au
        re *= cc.dist * cc.cm_in_au

        # Convert velocity from km/s to cm/s
        v = v * 1.0e5
        ve = ve * 1.0e5

        r_mag = sqrt((r**2).sum())
        v_mag = sqrt((v**2).sum())
        re_mag = sqrt(( (r*re)**2 ).sum()) / r_mag
        ve_mag = sqrt(( (v*ve)**2 ).sum()) / v_mag

        ### --- Now everything should be in CGS ---###

        # Check for unbound orbits... we don't handle those here
        if (v_mag > sqrt(2.0 * GM / r_mag)):
            raise ValueError('Velocity exceeds escape velocity: v_mag = %e vs. v_esc = %e, mass=%e' % (v_mag, sqrt(2.0 * GM / r_mag), cc.mass))

        # Angular momentum vector
        h = util.cross_product(r, v)
        h_mag = sqrt((h**2).sum())

        he = zeros(3, dtype=float64)
        he[0] =  (re[1]*v[2])**2 + (r[1]*ve[2])**2
        he[0] += (re[2]*v[1])**2 + (r[2]*ve[1])**2
        he[1] =  (re[0]*v[2])**2 + (r[0]*ve[2])**2
        he[1] += (re[2]*v[0])**2 + (r[2]*ve[0])**2
        he[2] =  (re[0]*v[1])**2 + (r[0]*ve[1])**2
        he[2] += (re[1]*v[0])**2 + (r[1]*ve[0])**2

        he = sqrt(he)
        he_mag = sqrt(( (h*he)**2 ).sum()) / h_mag

        # Inclination
        u = -h[2] / h_mag
        incl = math.degrees(arccos(u))

        dudx = h[2] * (v[1]*h[2] - v[2]*h[1]) / h_mag**3
        dudx = dudx - (v[1]/h_mag)
        dudy = h[2] * (v[2]*h[0] - v[0]*h[2]) / h_mag**3
        dudy = dudy + (v[0]/h_mag)
        dudz = h[2] * (v[0]*h[1] - v[1]*h[0]) / h_mag**3
        dudvx = h[2] * (r[2]*h[1] - r[1]*h[2]) / h_mag**3
        dudvx = dudvx + (r[1]/h_mag)
        dudvy = h[2] * (r[0]*h[2] - r[2]*h[0]) / h_mag**3
        dudvy = dudvy - (r[0]/h_mag)
        dudvz = h[2] * (r[1]*h[0] - r[0]*h[1]) / h_mag**3

        incl_err = (dudvx*ve[0])**2 + (dudvy*ve[1])**2 + (dudvz*ve[2])**2
        incl_err += (dudx*re[0])**2 + (dudy*re[1])**2 + (dudz*re[2])**2
        incl_err = incl_err / (1.0 - u**2) # derivative of acos
        incl_err = math.degrees(sqrt(incl_err))

        # eccentricity
        e = (util.cross_product(v, h) / GM) - (r/r_mag)
        e_mag = sqrt((e**2).sum())

        dedx = zeros(3, dtype=float64)
        dedx[0] = (v[1]**2 + v[2]**2) / GM
        dedx[0] += (r[0]**2 / r_mag**3) - (1.0 / r_mag)
        dedx[1] = (-v[0]*v[1] / GM) + (r[0]*r[1] / r_mag**3)
        dedx[2] = (-v[0]*v[2] / GM) + (r[0]*r[2] / r_mag**3)

        dedy = zeros(3, dtype=float64)
        dedy[0] = (-v[1]*v[0] / GM) + (r[1]*r[1] / r_mag**3)
        dedy[1] = ((v[0]**2 + v[2]**2) / GM)
        dedy[1] += (r[1]**2 / r_mag**3) - (1.0 / r_mag)
        dedy[2] = (-v[1]*v[2] / GM) + (r[1]*r[2] / r_mag**3)

        dedz = zeros(3, dtype=float64)
        dedz[0] = (-v[2]*v[0] / GM) + (r[2]*r[0] / r_mag**3)
        dedz[1] = (-v[2]*v[1] / GM) + (r[2]*r[1] / r_mag**3)
        dedz[2] = ((v[0]**2 + v[1]**2) / GM)
        dedz[2] += (r[2]**2 / r_mag**3) - (1.0 / r_mag)

        dedvx = zeros(3, dtype=float64)
        dedvx[0] = -(v[1]*r[1] + v[2]*r[2]) / GM
        dedvx[1] = (2.0*v[0]*r[1] - v[1]*r[0]) / GM
        dedvx[2] = (2.0*v[0]*r[2] - v[2]*r[0]) / GM

        dedvy = zeros(3, dtype=float64)
        dedvy[0] = (2.0*v[1]*r[0] - v[0]*r[1]) / GM
        dedvy[1] = -(v[0]*r[0] + v[2]*r[2]) / GM
        dedvy[2] = (2.0*v[1]*r[2] - v[2]*r[1]) / GM

        dedvz = zeros(3, dtype=float64)
        dedvz[0] = (2.0*v[2]*r[0] - v[0]*r[2]) / GM
        dedvz[1] = (2.0*v[2]*r[1] - v[1]*r[2]) / GM
        dedvz[2] = -(v[0]*r[0] + v[1]*r[1]) / GM

        ee_mag = ((dedx*e).sum() * re[0] / e_mag)**2
        ee_mag += ((dedy*e).sum() * re[1] / e_mag)**2
        ee_mag += ((dedz*e).sum() * re[2] / e_mag)**2
        ee_mag += ((dedvx*e).sum() * ve[0] / e_mag)**2
        ee_mag += ((dedvy*e).sum() * ve[1] / e_mag)**2
        ee_mag += ((dedvz*e).sum() * ve[2] / e_mag)**2

        ee = zeros(3, dtype=float64)
        ee[0] = (v[1]*he[2])**2 + (ve[1]*h[2])**2
        ee[0] += (v[2]*he[1])**2 + (ve[2]*h[1])**2

        ee[1] = (v[2]*he[0])**2 + (ve[2]*h[0])**2
        ee[1] += (v[0]*he[2])**2 + (ve[0]*h[2])**2

        ee[2] = (v[0]*he[1])**2 + (ve[0]*h[1])**2
        ee[2] += (v[1]*he[0])**2 + (ve[1]*h[0])**2

        ee /= GM**2
        ee += (r / r_mag)**2 * ((re / r)**2 + (re_mag / r_mag)**2)
        ee = sqrt(ee)
        ee_mag2 = sqrt(( (e * ee)**2 ).sum()) / e_mag

        # Line of nodes vector
        Om = util.cross_product(array([0.0, 0.0, 1.0]), h)
        Om_mag = sqrt((Om**2).sum())

        Ome = array([he[1], he[2], 0.0])
        Ome_mag = sqrt(( (Om * Ome)**2 ).sum()) / Om_mag

        # bigOmega = PA to the ascending node
        bigOm = math.degrees(arctan2(Om[0], Om[1]))

        bigOme =  v[2]**2 * ((re*h)**2).sum()
        bigOme += r[2]**2 * ((ve*h)**2).sum()
        bigOme /= Om_mag**4
        bigOme = math.degrees( sqrt(bigOme) )

        # omega = angle from bigOmega to periapse
        cos_om = ((Om / Om_mag) * (e / e_mag)).sum()
        omega = math.degrees( arccos(cos_om) )

        # dot product of Om and e
        Om_e = (Om*e).sum()

        dodx = -(Om_e * Om[0] * v[2]) / (e_mag * Om_mag**3)
        dodx += ((e[0]*v[2]) + (Om*dedx).sum()) / (e_mag * Om_mag)
        dodx -= (Om_e * (e*dedx).sum()) / (e_mag**3 * Om_mag)

        dody = -(Om_e*Om[1]*v[2]) / (e_mag * Om_mag**3)
        dody += ((e[1]*v[2]) + (Om*dedy).sum()) / (e_mag * Om_mag)
        dody -= (Om_e * (e*dedy).sum()) / (e_mag**3 * Om_mag)

        dodz = (Om_e * (Om*v).sum()) / (e_mag * Om_mag**3)
        dodz += ((-e[0]*v[0]) + (-e[1]*v[1]) + (Om*dedz).sum()) / (e_mag*Om_mag)
        dodz -= (Om_e * (e*dedz).sum()) / (e_mag**3 * Om_mag**3)

        dodvx = (Om_e*Om[0]*r[2]) / (e_mag * Om_mag**3)
        dodvx += ((-e[0]*r[2]) + (Om*dedvx).sum()) / (e_mag * Om_mag)
        dodvx -= (Om_e * (e*dedvx).sum()) / (e_mag**3 * Om_mag)

        dodvy = (Om_e*Om[1]*r[2]) / (e_mag * Om_mag**3)
        dodvy += ((-e[1]*r[2]) + (Om*dedvy).sum()) / (e_mag * Om_mag)
        dodvy -= (Om_e * (e*dedvy).sum()) / (e_mag**3 * Om_mag)

        dodvz = (-Om_e * (Om*r).sum()) / (e_mag * Om_mag**3)
        dodvz += ((e[0]*r[0]) + (e[1]*r[1]) + (Om*dedvz).sum()) /(e_mag*Om_mag)
        dodvz -= (Om_e * (e*dedvz).sum()) / (e_mag**3 * Om_mag)

        omega_err = (dodvx*ve[0])**2 + (dodvy*ve[1])**2 + (dodvz*ve[2])**2
        omega_err += (dodx*re[0])**2 + (dody*re[1])**2 + (dodz*re[2])**2
        omega_err = omega_err / (1.0 - cos_om**2)
        omega_err = math.degrees(sqrt(omega_err))

        if (omega_err > 180.0):
            omega_err = 179.999

        if (e[2] < 0):
            omega = 360.0 - omega

        # Semi major axis
        tmp = (2.0 / r_mag) - (v_mag**2 / GM)
        if (tmp == 0): tmp = 0.00001
        a = 1.0 / tmp

        ae = ((r * re / r_mag**3)**2).sum() + ((v * ve / GM)**2).sum()
        ae = sqrt(ae) * 2.0 * a**2

        # Period
        a_AU = a / cc.cm_in_au
        ae_AU = ae / cc.cm_in_au

        p = sqrt((a_AU / cc.mass) * a_AU**2)
        pe = (3.0/2.0) * p * ae_AU / a_AU

        #----------
        # Thiele-Innes Constants
        #----------
        cos_om = cos( math.radians(omega) )
        sin_om = sin( math.radians(omega) )
        cos_bigOm = cos( math.radians(bigOm) )
        sin_bigOm = sin( math.radians(bigOm) )
        cos_i = cos( math.radians(incl) )
        sin_i = sin( math.radians(incl) )

        conA = a * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
        conB = a * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
        conC = a * (sin_om * sin_i)
        conF = a * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
        conG = a * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
        conH = a * (cos_om * sin_i)

        # Eccentric Anomaly
        cos_E = (r[1]*conG - r[0]*conF) / (conA*conG - conF*conB)
        cos_E += e_mag
        sin_E = (r[0]*conA - r[1]*conB) / (conA*conG - conF*conB)
        sin_E /= sqrt(1.0 - e_mag**2)
        eccA = arctan2(sin_E, cos_E)

        # Eccentric Anomaly
        eccAe = (r_mag / (e_mag * a))**2 * ((re_mag / r_mag)**2 + (ae / a)**2)
        eccAe += (tmp * ee_mag / e_mag)**2
        eccAe /= (1.0 - tmp**2)
        eccAe = sqrt(eccAe)

        # Time of periapse passage
        tmp = p / (2.0 * math.pi)
        t0 = tmp * (eccA - e_mag*sin_E)
        t0 = epoch - t0

        deedv = array([(e*dedvx).sum(), (e*dedvy).sum(), (e*dedvz).sum()])
        deedr = array([(e*dedx).sum(), (e*dedy).sum(), (e*dedz).sum()])

        t0e3 = (tmp * (eccA - e_mag*sin_E) * pe / p)**2
        t0e3 = t0e3 + (tmp * (1 - e_mag*cos_E) * eccAe)**2
        t0e3 = t0e3 + (tmp * sin_E * ee_mag)**2
        t0e3 = sqrt(t0e3)
        t0e = t0e3

        phase = abs(epoch - t0) * 2.0 / p
        phaseErr = sqrt((2.0*t0e/p)**2 + (phase*pe/p)**2)

        eccA = math.degrees(eccA)
        eccAe = math.degrees(eccAe)

        #if (isnan(incl) == 1):
        #    pdb.set_trace()

        # Fix bigOmega
        if (bigOm < 0.0):
            bigOm = bigOm + 360.0

        # Orbital Parameters
        self.w = omega
        self.we = omega_err
        self.o = bigOm
        self.oe = bigOme
        self.i = incl
        self.ie = incl_err
        self.e = e_mag
        self.ee = ee_mag
        self.p = p
        self.pe = pe
        self.t0 = t0
        self.t0e = t0e
        self.ph = phase
        self.phe = phaseErr
        self.a = a

        # Vectors useful for later calculations
        self.hvec = h
        self.hevec = he
        self.evec = e
        self.eevec = ee
        self.ovec = Om
        self.oevec = Ome
        
        # Thiele-Innes Constants - convert from cm to AU
        self.conA = conA / cc.cm_in_au
        self.conB = conB / cc.cm_in_au
        self.conC = conC / cc.cm_in_au
        self.conF = conF / cc.cm_in_au
        self.conG = conG / cc.cm_in_au
        self.conH = conH / cc.cm_in_au

    def kep2xyz(self, epochs, mass=None, dist=None,relRedshift=False,GROrbit=False):
        """
        After setting all the orbital elements on this orbit object, call
        kep2xyz in order to return a tuple containing:
        
        Input:
        relRedshift -- if True, the relativistic redshift (special relativity and general
                       relativity) is added to the z component of the velocity. This effect
                       is NOT a dynamical effect on the orbit but an effect on the light 
                       propagation that affects the RV observations
        
        GROrbit -- if True, the secular post-Newtonian effects are taken into account 
                   (only secular, i.e. advance of argument of periapse)
        
        r -- radius vector [arcsec]
        
        v -- velocity vector [mas/yr]
        
        a -- acceleration vector [mas/yr^2]
        

        An example call is:

        orb = orbits.Orbit()
        orb.w = omega
        orb.o = bigOm
        orb.i = incl
        orb.e = e_mag
        orb.p = p
        orb.t0 = t0

        (r, v, a) = orb.kep2xyz(array([refTime]))
        
        """
        cc = objects.Constants()

        if (mass != None):
            cc.mass = mass
        if (dist != None):
            cc.dist = dist
            cc.asy_to_kms = cc.dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)

        GM = cc.mass * cc.msun * cc.G #cm^3/s^2



        ecnt = len(epochs)

        # meanMotion in radians per year
        meanMotion = 2.0 * math.pi / self.p
        
        # Semi-major axis in AU
        axis = (self.p**2 * cc.mass)**(1.0/3.0)

        ecc_sqrt = sqrt(1.0 - self.e**2)
        
        #computation of SECULAR effects due to GR (see i.e. Will's book - TEGP - 1993) - added by A. Hees, 04/2016
        if GROrbit:
            # periastron precession
            dw=3.e-10*GM/(axis*cc.cm_in_au*(1.-self.e**2))/(cc.c**2) #no unit - secular domega/dt due to GR
            w=math.radians(self.w)+(epochs-epochs[0])*meanMotion*dw #in rad!

            # secular advance in the mean anomaly
            c_AU_yr = cc.c * 1.e5 * cc.sec_in_yr / cc.cm_in_au #speed of light in AU/yr
            M0=meanMotion * (epochs[0]-self.t0) #starting mean anomaly
            E0 = self.eccen_anomaly(array([M0]), self.e)
            cf0=(cos(E0)-self.e)/(1.-self.e*cos(E0))

            dn = meanMotion**3*axis**2/(c_AU_yr**2)*(-15.*((1.+self.e*cf0)/(1.-self.e**2))**2 +9.*(1.+self.e*cf0)/(1.-self.e**2) -3.)#rad per yr
            M = M0 + (epochs-epochs[0]) * (meanMotion + dn)
        
        
        else:
            w=math.radians(self.w)
            # Mean anomaly
            M = meanMotion * (epochs - self.t0)
            #print 't0: ', self.t0, ' meanMotion: ', meanMotion
            #print 'M should be ', meanMotion * (epochs[0] - self.t0)




        #----------
        # Now for each epoch we compute the x and y positions
        #----------



        # Eccentric anomaly
        E = self.eccen_anomaly(M, self.e)

        #for ee in range(len(E)):
        #    print 'Epoch[ee] = ', epochs[ee]
        #    print 'E: %7.4f\t meanAnom: %7.4f' % (E[ee], M[ee])

        #E = zeros(len(M), dtype=float64)
        #for ii in range(len(M)):
        #    E[ii] = self.eccen_anomaly2(M[ii], self.e)
        #    print E[ii]
            
        cos_E = cos(E)
        sin_E = sin(E)
        
        Edot = meanMotion / (1.0 - (self.e * cos_E))

        X = cos_E - self.e
        Y = ecc_sqrt * sin_E

        #----------
        # Calculate Thiele-Innes Constants
        #----------
        cos_bigOm = cos(math.radians(self.o))
        sin_bigOm = sin(math.radians(self.o))
        cos_i = cos(math.radians(self.i))
        sin_i = sin(math.radians(self.i))
        
        r = zeros((ecnt, 3), dtype=float64)
        v = zeros((ecnt, 3), dtype=float64)
        a = zeros((ecnt, 3), dtype=float64)


        cos_om = cos(w)
        sin_om = sin(w)
        
        self.conA = axis * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
        self.conB = axis * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
        self.conC = axis * (sin_om * sin_i)
        self.conF = axis * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
        self.conG = axis * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
        self.conH = axis * (cos_om * sin_i)


        r[:,0] = (self.conB * X) + (self.conG * Y)
        r[:,1] = (self.conA * X) + (self.conF * Y)
        r[:,2] = (self.conC * X) + (self.conH * Y)
        
        v[:,0] = Edot * ((-self.conB * sin_E) + (self.conG * ecc_sqrt * cos_E))
        v[:,1] = Edot * ((-self.conA * sin_E) + (self.conF * ecc_sqrt * cos_E))
        v[:,2] = Edot * ((-self.conC * sin_E) + (self.conH * ecc_sqrt * cos_E))
        

    
        # Calculate accleration
        for ii in range(ecnt):
            rmag_cm = sqrt( (r[ii,:]**2).sum() ) * cc.cm_in_au
            a[ii,:] = -GM * r[ii,:] * cc.cm_in_au / rmag_cm**3

        #Computation from relativistic redshift - added by A. Hees 04/2016
        if relRedshift:
            v2 = (v[:,0]*v[:,0] + v[:,1]*v[:,1] + v[:,2]*v[:,2])*(cc.cm_in_au/cc.sec_in_yr)**2 #cm^2/s^2
            GMr = GM/(cc.cm_in_au*sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] )) #cm^2/s^2
            vrel =   (0.5*v2+GMr)/(cc.c*1.e5) #cm/s
        
        # Unit conversions
        # r  from AU     to arcsec
        # v  from AU/yr  to mas/yr
        # a  from cm/s^2 to mas/yr^2
        # vrel from cm/s to mas/yr
        
        r /= cc.dist
        
        v *= 1000.0 / cc.dist
        a *= 1000.0 * cc.sec_in_yr**2 / (cc.cm_in_au * cc.dist)
        if relRedshift:
            vrel *= 1000.0 * cc.sec_in_yr / (cc.cm_in_au * cc.dist)
            v[:,2] = v[:,2] + vrel

        return (r, v, a)


    def eccen_anomaly(self, m, ecc, thresh=1e-10):
        """
        m - a numpy array of mean anomalies
        ecc - the eccentricity of the orbit (single float value from 0-1)
        """
        # set default values

        if (ecc < 0. or ecc >= 1.):
            print('Eccentricity must be 0<= ecc. < 1')
            
        #
        # Range reduction of m to -pi < m <= pi
        #
        mx = m.copy()

        ## ... m > pi
        zz = (where(mx > math.pi))[0]
        mx[zz] = mx[zz] % (2.0 * math.pi)
        zz = (where(mx > math.pi))[0]
        mx[zz] = mx[zz] - (2.0 * math.pi)

        # ... m < -pi
        zz = (where(mx <= -math.pi))[0]
        mx[zz] = mx[zz] % (2.0 * math.pi)
        zz = (where(mx <= -math.pi))[0]
        mx[zz] = mx[zz] + (2.0 * math.pi)

        #
        # Bail out for circular orbits...
        #
        if (ecc == 0.0):
            return mx

        aux   = (4.0 * ecc) + 0.50
        alpha = (1.0 - ecc) / aux

        beta = mx/(2.0*aux)
        aux = sqrt(beta**2 + alpha**3)
   
        z=beta+aux
        zz=(where(z <= 0.0))[0]
        z[zz]=beta[zz]-aux[zz]

        test=abs(z)**0.3333333333333333

        z =  test.copy()
        zz = (where(z < 0.0))[0]
        z[zz] = -z[zz]

        s0=z-alpha/z
        s1 = s0-(0.0780 * s0**5) / (1.0 + ecc)
        e0 = mx + ecc*((3.0 * s1) - (4.0 * s1**3))

        se0=sin(e0)
        ce0=cos(e0)

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

        zz = (where(eccanom >= 2.00*math.pi))[0]
        eccanom[zz]=eccanom[zz]-2.00*math.pi
        zz = (where(eccanom < 0.0))[0]
        eccanom[zz]=eccanom[zz]+2.00*math.pi

        # Now get more precise solution using Newton Raphson method
        # for those times when the Kepler equation is not yet solved
        # to better than 1e-10
        # (modification J. Wilms)

        mmm = mx.copy()
        ndx = (where(mmm < 0.))[0]
        mmm[ndx] += (2.0 * math.pi)
        diff = eccanom - ecc*sin(eccanom) - mmm

        ndx = (where(abs(diff) > 1e-10))[0]
        for i in ndx:
            # E-e sinE-M
            fe = eccanom[i]-ecc*sin(eccanom[i])-mmm[i]
            # f' = 1-e*cosE
            fs = 1.0 - ecc*cos(eccanom[i])
            oldval=eccanom[i]
            eccanom[i]=oldval-fe/fs

            loopCount = 0
            while (abs(oldval-eccanom[i]) > thresh):
                # E-e sinE-M
                fe = eccanom[i]-ecc*sin(eccanom[i])-mmm[i]
                # f' = 1-e*cosE
                fs = 1.0 - ecc*cos(eccanom[i])
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

    def oal2xy(self, epochs, mass=None, dist=None, accel=False):
        """
        This is for binaris orbiting each other and moving linearly together.
        Given a Orbit And a Linear motion, get the 2D position (x,y) at time epochs.
        Based on paper Koren et al. 2015
        
        Input:
        epochs -- epoch with detections

        Return:
        x -- x in arcsec
        y -- y in arcsec

        An example call is:

        orb = orbits.Orbit()
        orb.w = omega           # degree
        orb.o = bigOm           # degree
        orb.i = incl            # degree
        orb.e = e_mag             
        orb.p = p               # year
        orb.tp = tp             # the time of the periastron passage
        orb.aleph = aleph       # semi-major axis of photocenter in arcsec
        orb.vx = vx             # arcsec/yr 
        orb.vy = vy
        orb.x0 = x0             # arcsec
        orb.y0 = y0
        if accel=True:
            orb.ax = ax
            orb.ay = ay

        (x, y) = orb.oal2xy(array([refTime]))

        
        """
        if self.e <0 or self.e>1:
            return(-1e8, -1e8)
        cc = objects.Constants()

        if (mass != None):
            cc.mass = mass
        if (dist != None):
            cc.dist = dist
            cc.asy_to_kms = cc.dist * cc.cm_in_au / (1.0e5 * cc.sec_in_yr)

        GM = cc.mass * cc.msun * cc.G #cm^3/s^2

        ecnt = len(epochs)

        # meanMotion in radians per year
        meanMotion = 2.0 * math.pi / self.p
        
        #----------
        # Now for each epoch we compute the x and y positions from model
        #----------

        ecc_sqrt = sqrt(1.0 - self.e**2)

        # Mean anomaly
        M = meanMotion * (epochs - self.tp)
        
        # Eccentric anomaly
        E = self.eccen_anomaly(M, self.e)

        # True Anomaly
        eta = 2* np.arctan(np.sqrt((1+self.e)/(1-self.e)) * np.tan(E/2))

        

        # elliptical rectangular coordinates
        cos_E = cos(E)
        sin_E = sin(E)

        X = cos_E - self.e
        Y = ecc_sqrt * sin_E

        #----------
        # Calculate Thiele-Innes Constants
        #----------
        cos_bigOm = cos(math.radians(self.o))
        sin_bigOm = sin(math.radians(self.o))
        cos_i = cos(math.radians(self.i))
        sin_i = sin(math.radians(self.i))
        cos_om = cos(math.radians(self.w))
        sin_om = sin(math.radians(self.w))
        

        self.conA = self.aleph * (cos_om * cos_bigOm  - sin_om * sin_bigOm * cos_i)
        self.conB = self.aleph * (cos_om * sin_bigOm  + sin_om * cos_bigOm * cos_i)
        self.conC = self.aleph * (sin_om * sin_i)
        self.conF = self.aleph * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
        self.conG = self.aleph * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
        self.conH = self.aleph * (cos_om * sin_i)


        # fiducial time 
        tf = 1990

        # calculate x and y
        if accel:
            x = (self.conB * X) + (self.conG * Y) + self.ax * (epochs-tf)**2 + self.vx * (epochs-tf) + self.x0 
            y = (self.conA * X) + (self.conF * Y) + self.ay * (epochs-tf)**2 + self.vy * (epochs-tf) + self.y0
        else:
            x = (self.conB * X) + (self.conG * Y) + self.vx * (epochs-tf) + self.x0 
            y = (self.conA * X) + (self.conF * Y) + self.vy * (epochs-tf) + self.y0
            rv = self.K1 * (cos(math.radians(self.w) + eta) + self.e * cos(math.radians(self.w))) + self.gamma
            
        
        # Unit conversions
        # r  from AU     to arcsec
        # v  from AU/yr  to mas/yr
        
        #x /= cc.dist
        #y /= cc.dist

        return (x, y, rv)


class EccAnomalyError(Exception):
    def __init__(self, message):
        self.message = message
