import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import scipy.integrate as spi
import bisect
import itertools
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import special

def get_simulation(dv=.001,dt=1e-4,tf = 0.1, verbose=False, update_method='exact', approx_order=None, tol=1e-8):

    # baseline and step function
    s0 = 90.0 
    tstep = tf/dt
    s1 = np.zeros(np.int(tstep))
    s2 = np.zeros(np.int(tstep))
    s1[200:600] = 180
    s2[700:1000] = 90
    s1 += s0
    s2 += s0
    print 's2: ',s2
    # Create simulation:
    b0 = ExternalPopulation(s1,dt) # exc
    b1 = ExternalPopulation(s2,dt) # inh
    e1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
    e2 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i2 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
    e3 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i3 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)

    # feedforward connection
    b0_e1 = Connection(b0, e1, 1, weights= .13, probs=1.,conn_type = 'ShortRange')
    b0_i1 = Connection(b0, i1, 1, weights= .12, probs=1.,conn_type = 'ShortRange')
    
    e1_e1 = Connection(e1, e1, 100, weights= .012, probs=1.,conn_type = 'ShortRange')
    i1_e1 = Connection(i1, e1, 100, weights= -.009, probs=1.,conn_type = 'ShortRange')
    e1_i1 = Connection(e1,i1, 100, weights= .008, probs=1.,conn_type = 'ShortRange')
    i1_i1 = Connection(i1, i1, 100, weights= -.003, probs=1.,conn_type = 'ShortRange')
 
    # feedforward connection
    b1_e2 = Connection(b1, e2, 1, weights= .13, probs=1.,conn_type = 'ShortRange')
    b1_i2 = Connection(b1, i2, 1, weights= .12, probs=1.,conn_type = 'ShortRange')
    # recurrent connection
    e2_e2 = Connection(e2, e2, 100, weights= .012, probs=1.,conn_type = 'ShortRange')
    i2_e2 = Connection(i2, e2, 100, weights= -.009, probs=1.,conn_type = 'ShortRange')
    e2_i2 = Connection(e2,i2, 100, weights= .008, probs=1.,conn_type = 'ShortRange')
    i2_i2 = Connection(i2, i2, 100, weights= -.003, probs=1.,conn_type = 'ShortRange')
    
    # feedforward connection
    b1_e3 = Connection(b1, e3, 1, weights= .13, probs=1.,conn_type = 'ShortRange')
    b1_i3 = Connection(b1, i3, 1, weights= .12, probs=1.,conn_type = 'ShortRange')
    # recurrent connection
    e3_e3 = Connection(e3, e3, 100, weights= .012, probs=1.,conn_type = 'ShortRange')
    i3_e3 = Connection(i3, e3, 100, weights= -.009, probs=1.,conn_type = 'ShortRange')
    e3_i3 = Connection(e3,i3, 100, weights= .008, probs=1.,conn_type = 'ShortRange')
    i3_i3 = Connection(i3, i3, 100, weights= -.003, probs=1.,conn_type = 'ShortRange')

    # long-range recurrent connection
    e1_e2 = Connection(e1, e2, 100, weights= .0463, probs=1.,conn_type = 'LongRange')
    e1_i2 = Connection(e1, i2, 100, weights= .0586, probs=1.,conn_type = 'LongRange')
    # long-range recurrent connection
    e2_e1 = Connection(e2, e1, 100, weights= .0460, probs=1.,conn_type = 'LongRange')
    e2_i1 = Connection(e2, i1, 100, weights= .0574, probs=1.,conn_type = 'LongRange')
    
    simulation = Simulation([b0,b1,e1,i1,i2,e2,i3,e3], [b0_e1,b0_i1,b1_i2,b1_e2,b1_i3,b1_e3,e1_e2,e1_i2,e1_e1,e1_i1,i1_i1,i1_e1,
                            e2_e2,e2_i2,i2_i2,i2_e2,e3_e3,e3_i3,i3_i3,i3_e3], verbose=verbose)

    return simulation


def example(show=True, save=False):

    # Settings:
    t0 = 0
    dt = .0001
    dv = .001
    tf = 0.15
    verbose = True
    update_method = 'approx'
    approx_order = 1
    tol = 1e-14


    # Run simulation:
    simulation = get_simulation(dv=dv, dt = dt,tf = tf,verbose=verbose, update_method=update_method, approx_order=approx_order, tol=tol)
    mm,mu = simulation.update(dt=dt, tf=tf, t0=t0)

    
    return mm,mu



mm,mu = example()
"""
tt = .001 * np.arange(0,len(mm[3,:]),1)
plt.figure(1)
plt.plot(tt,mm[2,:])
plt.plot(tt,mm[3,:])
plt.plot(tt,mm[4,:])
plt.plot(tt,mm[5,:])
plt.plot(tt,mm[6,:])
plt.plot(tt,mm[7,:])
plt.ylim([0,50])

plt.figure(12)
plt.plot(tt,mu[2,:])
plt.plot(tt,mu[3,:])
plt.plot(tt,mu[4,:])
plt.plot(tt,mu[5,:])
plt.plot(tt,mu[6,:])
plt.plot(tt,mu[7,:])

plt.ylim([0,1.0])
"""
