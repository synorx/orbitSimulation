#!/usr/bin/env python
# coding: utf-8

from vpython import *

from astropy.time import Time
import numpy as np
from PIL import Image


# ## Define the Force Field of Earth In the Case of spherical Earth Model
# 
# The acceleration for a spherical Earth model is given as the gradient of the gravitational potential function.

#    \boldsymbol{a} = - \frac{GM}{r^2}\frac{\boldsymbol{r}}{r} = \nabla \frac{GM}{r}

# 
# 
# where $GM$ is the Earth gravitational constant. {r} is the position vector with respect to the mass center of Earth. 
# 
# 
# A Python class for representing the Speherical Earth acceleration model is given below. Note that the function a() returns the acceleration vector $[a_x,a_y,a_z]$ for an object at position $\boldsymbol{r}=[x,y,z]^T$ 
# represented in an inertial (quazi-inertial) reference frame centered at Earth's center of mass.

# An interface class to define a Force Field

class ForceField(object):
    def __init__(self,name):
        self.name = name
    '''
    return the acceleration vector given position r
    '''
    def a(self,r):
        pass

    
## A speherical Earth gravitional model
class GravitationalForceField(ForceField):
    def __init__(self):
        self.GM = 398600.4405e9
        self.R = 6371000
    
    def a(self,r):
        rr = np.sqrt(np.sum(r ** 2))
        
        return -self.GM/(rr ** 3) *r


# #  The Flattened Earth

# In the case of Flattened Earth The Force Field includes additional terms (J2). 
# The potential with respect to a flattened earth is defined as:
# Write down the $\nabla U$ which gives the gravitational acceleration.
 
# \nabla U(r) = \begin{bmatrix}
# \frac{\partial{U}}{\partial{x}} \\
# \frac{\partial{U}}{\partial{y}} \\
# \frac{\partial{U}}{\partial{z}}
# \end{bmatrix} = ?
 
# Given

# GM = 398600.4405x10^9 
# J_2 =  1.0826157x10^{-3}
# R = 6378135


class ForceFieldEllipsoidalEarth(GravitationalForceField):
    def __init__(self):
        self.GM = 398600.4405e9
        self.R = 6378135
        self.J2 = 1.0826157e-3
        
    def a(self, r):
        
        # related to the Spherical Earth Model
        a_sphere = GravitationalForceField.a(self,r)
        
        # Additional acceleration related to the J2 Term
        a_flat = np.array([self.partial_x(r), self.partial_y(r), self.partial_z(r)])
        
        return a_sphere + a_flat
    
    
    # This function returns the partial derivative with respect to x of the J2 term of the 
    # potential function.
    # pos is the position vector (pos = r = [x,y,z])
    
    def partial_x(self, pos):
        r = np.sqrt(np.sum(pos**2)) # length of the position vector
        x = pos[0] # x of the position vector 
        y = pos[1] # y of the position vector
        z = pos[2] # z of the position vector
        
        return -(self.GM*self.J2*self.R**2 *3*x*(x**2+y**2-4*z**2))/(2*r**7)
    
    # This function returns the partial derivative with respect to y of the J2 term of the 
    # potential function.
    # pos is the position vector (pos = r = [x,y,z])
    
    def partial_y(self, pos):
        r = np.sqrt(np.sum(pos**2)) # length of the position vector
        x = pos[0] # x of the position vector 
        y = pos[1] # y of the position vector
        z = pos[2] # z of the position vector
        
        return -(self.GM*self.J2*self.R**2 *3*y*(x**2+y**2-4*z**2))/(2*r**7)


    
    # This function returns the partial derivative of the J2 term of the 
    # potential function.
    # pos is the position vector (pos = r = [x,y,z])
    
    def partial_z(self, pos):
        r = np.sqrt(np.sum(pos**2)) # length of the position vector
        x = pos[0] # x of the position vector 
        y = pos[1] # y of the position vector
        z = pos[2] # z of the position vector

        return -(self.GM*self.J2*self.R**2 *3*z*(3*x**2+3*y**2-2*z**2))/(2*r**7)


# ## Orbit propagation

# Given the initial state of an artifical satellite,

# The initial geocentric Position vector.
# r_0 = [x_0, y_0, z_0]^T

# The initial velocity vector
# v_0 = [vx_0, vy_0, vz_0]^T

# the position at a given time can be obtained by integrating the orbital motion equations. 
# For this project  only consider the acceleration related to the gravitational attraction. 

# If the state of a satellite is specified as vector $s(t)=[r(t),v(t)]$  of position and velocity.


# The accelerations a_x(t,s), a_y(t,s) and a_z(t,s) are the partial derivatives of the gravitational potential function at the given position (s) .
# 
# ### Eulers method
# Then by using the Euler integration method. The given Ordinary differential Equation can be solved for given 
# Delta t increments using the initial condition as:
# 
# 
# s(t_0+\Delta t) = s(t_0) + s^\prime(t_0,s_0) \Delta t 
# 
# 
# Or in a general form
#  s_{i+1} = s_i + s^\prime_i(t_i,s_i) \Delta t 

# ### Runge Kutta Order 2
# In order to improve the stability of the Eulers method a midpoint is introduced in the numerical integration. 
# This makes the Runge-Kutta a second order numerical integration method.
# k_1 = s^\prime_i(t_i, s_i) \Delta t 
# k_2 = s^\prime_i(t_i+\frac{1}{2}\Delta t, s_i+\frac{1}{2}k_1) \Delta t 


# And finally:
# s_{i+1} = s_i + k_2 


# Base class for numerical integration
# Which implements Eulers method
class Integrator(object):
    def __init__(self):
        pass
    
    def initialize(self, state):
        self.state = state
        
        
    def next(self, t, dt):
        s = self.state.s(t)
        ds = self.state.ds(t,dt, s)
        return s+ dt*ds
    

## Runge Kutta 2 Numerical Integration implementation    
class RK2Integrator(Integrator):
    def __init__(self):
        pass
        
    def next(self, t, dt):
        
        # the current state s_i
        s = self.state.s(t)
        
        # the derivative s' at given time and pos
        k1 = dt * self.state.ds(t,dt,s)

        # the derivative s' at midpoint time and pos
        k2 = dt*self.state.ds(t,dt/2.0, s+k1/2.0)
        
        return s+k2

    
    
## Runge Kutta 4 Numerical Integration implementation    
class RK4Integrator(Integrator):
    def __init__(self):
        pass
        
    def next(self, t, dt):
        
        # the current state s_i
        s = self.state.s(t)
        
        
        k1 = dt * self.state.ds(t,dt,s)

        
        k2 = dt*self.state.ds(t,dt/2.0, s+k1/2.0)
        
        k3 = dt*self.state.ds(t,dt/2.0, s+k2/2.0)
        
        k4 = dt*self.state.ds(t, dt, s+k3)
        
        
        return s + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0


# ## The satellite state, Earth and Satellite Objects
# 
# The satellite state can be represented as a python class. 
# The class is initialized by a given initial position and velocity. The force field represents the gravitation acceleration field.
#  Additionally an Integrator is also given for the integration of the position and velocity to find the satellite position at later times.
# 
# 
# Note that there is an additional class method to obtain a satellite state given the keplerian elements.


class SatelliteState(object):
    def __init__(self, i_pos, i_vel, force_field, integ):
        self.position = np.array(i_pos)
        self.velocity = np.array(i_vel)
        self.force_field = force_field
        self.integrator = integ
        self.integrator.initialize(self)
        

    @classmethod
    def fromKeplerElements(cls, force_field, integ, a, e, incl, theta, omega, T, t):
        GM = force_field.GM
        
        incl = np.radians(incl)
        theta = np.radians(theta)
        omega = np.radians(omega)
        
        M = np.sqrt(GM/(a**3)) * (t-T)
        
        E = M
        
        for i in range(3):
            E = M + e*np.sin(E)
        
        v = 2*np.arctan( np.sqrt((1+e)/(1-e)) * np.tan(E/2.0))
        r = a*(1-e*np.cos(E))
        
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0,             0,              1]
            
        ])

        R3_theta = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0,             0,              1]
            
        ])

        R1_i = np.array([
            [1,             0,              1],
            [0, np.cos(incl), -np.sin(incl)],
            [0, np.sin(incl), np.cos(incl)],
            
        ])


        x_sat = np.array([r*np.cos(v), r*np.sin(v), 0])
        
        x_sat = np.dot(R3_omega, np.dot( R1_i, np.dot(R3_theta, x_sat)))
        
        p = a*(1-e**2)
        
        v_sat = np.array([-np.sqrt(GM/p)*np.sin(v), np.sqrt(GM/p)*(e+np.cos(v)), 0])
        
        v_sat = np.dot(R3_omega, np.dot( R1_i, np.dot(R3_theta, v_sat)))
        
        return SatelliteState(x_sat, v_sat, force_field, integ)
        
        
    
    
    #the position in delta t seconds 
    
    def next(self,t, dt):
        ns = self.integrator.next(t,dt)
        self.position = np.copy(ns[:3])
        self.velocity = np.copy(ns[3:])
    
    def s(self, t):
        return np.hstack( (self.position, self.velocity) )
    
    def ds(self ,t, dt, s):
        return np.hstack( (s[3:],self.force_field.a(s[:3]) ) )

    

#Visualization of satellite and its orbital propagation
class Satellite(object):
    def __init__(self,state):
        self.state = state
        self.obj = sphere ( pos=vector(state.position[0],state.position[1],state.position[2]), 
                            radius = 100000, visible=True, color=vector(1,0,0),
                            make_trail=True, trail_type="curve", interval=1, retain=5000
                          )
        

    def next(self,t, dt):
        
        
        ## Although the delta t increment may be larger than multiple of seconds.
        ## Here decompose the total delta t increment into two seconds of 
        ## inner increments to increase the precision.
        ## Loop Count of two seconds increments
        loop_count = dt // 2
        ## remaing seconds inrement
        remaining = dt % 2
        
        ## loop for two seconds increments
        for i in range(int(loop_count)):
            self.state.next(t,2)
            
        ## do the remaining increments.
        self.state.next(t, remaining)
        
        #update the position of the satellite for visualization
        self.obj.pos = vector(self.state.position[0],self.state.position[1],self.state.position[2])

        
        
## Visualization of Earth and its rotation around the z axis.
class Earth(object):
    def __init__(self):
        self.obj = sphere ( pos=vector(0,0,0),
                radius = 6371000, 
                visible=True, 
                texture=textures.earth,up=vector(0,0,1),
                flipx = False , shininess = 0.9,opacity=0.9)
        self.obj.rotate(angle=np.radians(90),axis=vector(0,0,1)) # X axis points to greenwhich meridian
        self.force_field = GravitationalForceField()
        sw = 100000
        hw=300000       
        self.arrow_z_GCRF = arrow(pos=vector(0,0,0),
                             axis=vector(0,0,6371000+800000), shaftwidth=sw,headwidth=hw,color=vector(0,0,1))


        self.arrow_y_GCRF = arrow(pos=vector(0,0,0),
                             axis=vector(0,6371000+800000,0),shaftwidth=sw,headwidth=hw,color=vector(0,1,0))

        self.arrow_x_GCRF = arrow(pos=vector(0,0,0),
                             axis=vector(6371000+800000,0,0), shaftwidth=sw,headwidth=hw,color=vector(1,0,0))

    
    def next(self, t, dt):

        delta_gast = self.calculate_EOP(t,dt)
        self.obj.rotate(angle = np.radians(delta_gast), axis=vector(0,0,1))
        
    def calculate_EOP(self, t, dt):
        return dt/86400.0*360


# ## Visualization of Orbital Propagation
# 
# The following code segment creates a VPython scene to display orbital propagation of artificial satellites.
# 
# 


scene.width = 800
scene.height = 600
scene.title = "Earth Rotation Sample"
scene.visible=False
running = False

t = Time('2012-11-20', scale='utc',location=('0','0'))

def Run(b):
    global running
    running = not running
    if running: b.text = "Pause"
    else: b.text = "Run"

              
button(text="Run", pos=scene.title_anchor, bind=Run)


scene.caption = "Vary the time increment: \n\n"

def setspeed(s):
    wt.text = '{:1.2f}'.format(s.value)
    
sl = slider(min=1, max=1440, value=1, length=220, bind=setspeed, right=15)

wt = wtext(text='{:1.2f}'.format(sl.value))
scene.waitfor("textures")
scene.visible = True  # show everything


dt = 1 # 1 seconds of increment in time

earth = Earth() ## Earth for visualization

earth.force_field = ForceFieldEllipsoidalEarth() ## change the force field to flattened

sats = [] # list of satellites for simulation, you can add as many


## Create a satellite with given altitude of 600000 meters assuming the equatorial 
## radius of earth is 6378135, with an inclination angle of 80 degrees.
## Note you can add more than one satellite with different orbit types.
## Note: you can also make the color a parameter in the construction of Satellite
## to display each orbit in different colors
own_sat = Satellite(SatelliteState.fromKeplerElements(
    earth.force_field, RK4Integrator(),
                                                     
                                                     6378135+600000,
                                                     0,
                                                      80,
                                                      0,
                                                      0,
                                                      0,
                                                      0
                                                     ))

sats.append(own_sat)





while True:
    rate(30) ## GUI refresh rate
    if (not running):
        continue # skip if not running
        
    dt = sl.value # Use the sliders value as delta t increment
    
    t = t + (dt/86400.0) # increment time to display JD
    
    wt.text = ' %s dt = %d seconds  JD=%f' % (str(t),sl.value,t.mjd)

    ## loop over every satellite to propagate their orbit in time
    for sat in sats:
        sat.next(t,dt)

    ## Rotate Earth.
    earth.next(t, dt)




