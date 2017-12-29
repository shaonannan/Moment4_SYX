def get_simulation(dv=.001,dt=1e-4,tf = 0.5,verbose=False, update_method='exact', approx_order=None, tol=1e-8):

    # baseline and step function
    s0 = 600.0 
    tstep = tf/dt
    s1 = np.zeros(np.int(tstep))
    s2 = np.zeros(np.int(tstep))
    s1[500:900] = 000
    s2[1000:1400] = 000
    s1 += s0/1.0
    s2 += s0/3.0
    print 's2: ',s2
    # Create simulation:
    b0 = ExternalPopulation(s1,dt) # exc
    b1 = ExternalPopulation(s2,dt) # inh
    e1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
    e2 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i2 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)

    # feedforward connection
    b0_e2 = Connection(b0, e2, 1, weights= 0.150, probs=1.,conn_type = 'ShortRange')
    e2_e2 = Connection(e2, e2, 100, weights= -0.022, probs=1.,conn_type = 'ShortRange')
    b1_e1 = Connection(b1, e1, 1, weights= 0.161, probs=1.,conn_type = 'ShortRange')
    e1_e2 = Connection(e1, e2, 100, weights= -0.0620, probs=1.,conn_type = 'LongRange')
    
    simulation = Simulation([b0,b1,e1,e2], [b0_e2,b1_e1,e2_e2,e1_e2], verbose=verbose)

    return simulation


def example(show=True, save=False):

    # Settings:
    t0 = 0
    dt = .00001
    dv = .01
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
tt = .00001 * np.arange(0,len(mm[3,:]),1)
plt.figure(11)
plt.plot(tt,mm[3,:])
"""
plt.plot(mm[3,:])
plt.plot(mm[4,:])
"""

plt.figure(3)
plt.plot(mu[1,:])
"""
plt.plot(mu[3,:])
plt.plot(mu[4,:])
plt.plot(mu[5,:])
"""
plt.ylim([0,1.0])