def get_simulation(dv=.001,dt=1e-4, verbose=False, update_method='exact', approx_order=None, tol=1e-8):

    # Create simulation:
    b0 = ExternalPopulation(5500,dt)
    b1 = ExternalPopulation(2000,dt)
    e1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)    
    i1 = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)
    b0_e1 = Connection(b0, e1, 1, weights=.01, probs=1.)
    e1_e1 = Connection(e1, e1, 2000, weights=0.1/2000.0, probs=1.)
    i1_i1 = Connection(i1, i1, 1000, weights=-0.2/1000.0, probs=1.)
    b1_i1 = Connection(b1, i1, 1, weights=.03, probs=1.)
    simulation = Simulation([b0,b1,e1,i1], [b0_e1,e1_e1,b1_i1,i1_i1], verbose=verbose)

    return simulation


def example(show=True, save=False):

    # Settings:
    t0 = 0
    dt = .0001
    dv = .001
    tf = 0.20
    verbose = True
    update_method = 'approx'
    approx_order = 1
    tol = 1e-14
    
    # Run simulation:
    simulation = get_simulation(dv=dv, dt = dt,verbose=verbose, update_method=update_method, approx_order=approx_order, tol=tol)
    mm = simulation.update(dt=dt, tf=tf, t0=t0)

    
    return mm

mm = example()
plt.figure(1)
plt.plot(mm[2,:])
plt.plot(mm[3,:])