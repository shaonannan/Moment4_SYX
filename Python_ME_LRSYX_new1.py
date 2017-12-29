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



"""
# Stimulation Configuration
t0 = .0
dt = 1e-4
Final_time = 3*1e-1
verbose = True
save = False
"""

"""
utility function
"""

def rho_EQ(Vs,D,V):
    
    Rv = np.copy(V)
    (vT,vR) = (1.0,0.0)
    tmpg = np.greater(V,vR)
    indp = (np.where(tmpg))
    sqrtD  = np.sqrt(D)

    
    intovT  = special.dawsn((vT-Vs)/sqrtD)*np.exp(np.square(vT-Vs)/D)
    intovSD = special.dawsn(-Vs/sqrtD)*np.exp(np.square(Vs)/D)

    # compute R with V>vR case:
    Rv[indp] = -special.dawsn((V[indp]-Vs)/sqrtD)+np.exp(-np.square(V[indp]-Vs)/D)*intovT
    if(indp[0][0]>1):
        Rv[0:indp[0][0]] = np.exp(-np.square(V[0:indp[0][0]]-Vs)/D)*(-intovSD + intovT)
    
    tmpl = np.less(V,-2.0/3.0)
    indp = np.where(tmpl)
    Rv[indp] = 0.0
    sum_c = (V[2]-V[1])*np.sum(Rv)
    """
    if np.isnan(sum_c):
        print 'SUM_C', sum_c
        print 'Error!'
        print 'Vs:', Vs,'D:',D
        pause(10)
    """
    # print 'sum_c',sum_c
    Rv = Rv/sum_c
    return (Rv,sum_c)

def optfun(lambda_u,mu,x,Pq,fin,gamma):
    lambda_u = lambda_u[:]
    k  = np.size(mu)
    # mu = np.reshape(mu,[k,1])
    tt = np.zeros(k+1)
    tt[0] = 1
    tt[1:k+1]  = mu[:]
    # print 'mu: ',tt
    dx = x[1]-x[0]
    # print 'dx: ',dx,'lambda: ',lambda_u
    # print 'lambda: ', lambda_u
    # lambda_u = lambda0[:]
    N  =np.size(lambda_u)
    # print N,np.shape(fin)
    
    p  = Pq*np.exp(np.dot(fin[:,0:N],lambda_u))
    f  = dx*np.sum(p)-np.dot(np.reshape(tt,[1,k+1]),lambda_u)   
    # print 'f: ',f
    return f

def fraction_overlap(a1, a2, b1, b2):
    '''Calculate the fractional overlap between range (a1,a2) and (b1,b2).
    
    Used to compute a reallocation of probability mass from one set of bins to
    another, assuming linear interpolation.
    '''
    if a1 >= b1:    # range of A starts after B starts
        if a2 <= b2:    
            return 1       # A is within B
        if a1 >= b2:
            return 0       # A is after B
        # overlap is from a1 to b2
        return (b2 - a1) / (a2 - a1)
    else:            # start of A is before start of B
        if a2 <= b1:
            return 0       # A is completely before B
        if a2 >= b2:
            # B is subsumed in A, but fraction relative to |A|
            return (b2 - b1) / (a2 - a1)
        # overlap is from b1 to a2
        return (a2 - b1) / (a2 - a1) 
    

def redistribute_probability_mass(A, B):
    '''Takes two 'edge' vectors and returns a 2D matrix mapping each 'bin' in B
    to overlapping bins in A. Assumes that A and B contain monotonically increasing edge values.
    '''
    
    mapping = np.zeros((len(A)-1, len(B)-1))
    newL = 0
    newR = newL
    
    # Matrix is mostly zeros -- concentrate on overlapping sections
    for L in range(len(A)-1):
        
        # Advance to the start of the overlap
        while newL < len(B) and B[newL] < A[L]:
            newL = newL + 1
        if newL > 0:
            newL = newL - 1
        newR = newL
        
        # Find end of overlap
        while newR < len(B) and B[newR] < A[L+1]:
            newR = newR + 1
        if newR >= len(B):
            newR = len(B) - 1

        # Calculate and store remapping weights
        for j in range(newL, newR):
            mapping[L][j] = fraction_overlap(A[L], A[L+1], B[j], B[j+1])

    return mapping

    
def flux_matrix(v, w, lam, p=1):
    'Compute a flux matrix for voltage bins v, weight w, firing rate lam, and probability p.'
    
    zero_bin_ind_list = get_zero_bin_list(v)
    
    # Flow back into zero bin:
    if w > 0:
        
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p

        # Threshold:
        flux_to_zero_vector = -A.sum(axis=0)
        for curr_zero_ind in zero_bin_ind_list:
            A[curr_zero_ind,:] += flux_to_zero_vector/len(zero_bin_ind_list)
    else:
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p
        
        
        missing_flux = -A.sum(axis=0)
        A[0,:] += missing_flux
        
        flux_to_zero_vector = np.zeros_like(A.sum(axis=0))

    return flux_to_zero_vector, A


def exact_update_method(J,rhov,dt = 1e-4):
    rhov = np.dot(spla.expm(J*dt),rhov)
    assert_probability_mass_conserved(rhov)
    return rhov


def approx_update_method_tol(J, pv, tol=2.2e-16, dt=.0001, norm='inf'):
    'Approximate the effect of a matrix exponential, with residual smaller than tol.'
    
    # No order specified:
    J *= dt
    curr_err = np.inf
    counter = 0.
    curr_del = pv
    pv_new = pv
    
    while curr_err > tol:
        counter += 1
        curr_del = J.dot(curr_del)/counter
        pv_new += curr_del
        curr_err = spla.norm(curr_del, norm)

    
    try:
        assert_probability_mass_conserved(pv)
    except:                                                                                                                                                     # pragma: no cover
        raise Exception("Probabiltiy mass error (p_sum=%s) at tol=%s; consider higher order, decrease dt, or increase dv" % (np.abs(pv).sum(), tol))            # pragma: no cover
    
    return pv_new

def approx_update_method_order(J, pv, dt=.0001, approx_order=2):
    'Approximate the effect of a matrix exponential, truncating Taylor series at order \'approx_order\'.'
    
    # Iterate to a specific order:
    coeff = 1.
    curr_del = pv
    pv_new = pv
    for curr_order in range(approx_order):
        coeff *= curr_order+1
        curr_del = J.dot(curr_del*dt)
        pv_new += (1./coeff)*curr_del
    
    try:
        assert_probability_mass_conserved(pv_new)
    except:                                                                                                                                                             # pragma: no cover
        raise Exception("Probabiltiy mass error (p_sum=%s) at approx_order=%s; consider higher order, decrease dt, or increase dv" % (np.abs(pv).sum(), approx_order))  # pragma: no cover

    return pv_new


def get_v_edges(v_min,v_max,dv):
    # Used for voltage-distribution and discretization
    edges = np.concatenate((np.arange(v_min,v_max,dv),[v_max]))
    edges[np.abs(edges) < np.finfo(np.float).eps] = 0
    return edges
def get_zero_bin_list(v):
    # find low boundary for vreset
    v = np.array(v) # cast to avoid mistake
    
    if(len(np.where(v==0)[0])>0):
        zero_edge_ind = np.where(v==0)[0][0]
        if zero_edge_ind == 0:
            return [0]
        else:
            return [zero_edge_ind-1,zero_edge_ind]
    else:
        return [bisect.bisect_right(v,0)-1]
    
def leak_matrix(v,tau): # original version (v,tau)
    # knew voltage-edge and design leaky-integrate-and-fire mode

    zero_bin_ind_list = get_zero_bin_list(v)   
    # initialize A transmit function for leaky integrate and fire
    A = np.zeros((len(v)-1,len(v)-1))
    
    # if dv/dt = right , right --> positive leak:
    delta_w_ind = -1
    for pre_ind in np.arange(max(zero_bin_ind_list)+1,len(v)-1):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind+1]-v[pre_ind]
        bump_rate = v[pre_ind+1]/(tau*dv)
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    # if dv/dt = right, right --> negative leak:
    delta_w_ind = 1
    for pre_ind in np.arange(0,min(zero_bin_ind_list)):
        post_ind = pre_ind + delta_w_ind
        dv = v[pre_ind]-v[pre_ind+1]
        # reverse
        bump_rate = v[pre_ind]/(tau*dv)  # always choose the smaller one
        A[pre_ind,pre_ind] -= bump_rate
        A[post_ind,pre_ind] += bump_rate
        
    return A

def assert_probability_mass_conserved(pv):
    'Assert that probability mass in control nodes sums to 1.'
    
    try:
        assert np.abs(np.abs(pv).sum() - 1) < 1e-12
    except:                                                                                 # pragma: no cover
        raise Exception('Probability mass below threshold: %s' % (np.abs(pv).sum() - 1))    # pragma: no cover


class ConnectionDistributionCollection(dict):
    def add_unique_connection(self,cd):
        if not cd.signature in self.keys():
            self[cd.signature] = cd


class Simulation(object):
    """
    Parameters:
    list :
        All sub-population (cluster)
        All connection (cluster)
        [type of both is 'List', which is changable variable, and could be changed]
        
    generate after initiate(by hand)
        connection_distribution
        connection_distribution_list
        [the differences between connection, connection_distribution and connection_distribution_list are
        connection ---> the component of 'connection_list', record all information and related information and object,like source and others
        connection_distribution --> this variable is a preparation variable for further processing, each 'connection' could generate a 
        class 'connecton_distribution' and then, using weight,syn,prob, could calculate flux_matrix and threshold
        each 'connection_distribution' item is defined by 'weight''syn ''prob', items with identical symbol will be classified to the same
        distribution
        connection_distribution_list --> this is a 'basket', store all unique connections(definition of unique: unique symbol
        'weight','syn','prob' no matter the target/source population)
    """
    def __init__(self,population_list,connection_list,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.rhov    = None
    
    def initialize(self,t0=0.0):
        """
        initialize by hand, first put all sub-population and connection-pair
        !!! put them on the same platform!!! simulationBridge
        """
        
        # An connection_distribution_list (store unique connection(defined by weight,syn,prob))
        self.connection_distribution_collection = ConnectionDistributionCollection() # this is 
        self.t = t0
        
        # put all subpopulation and all connections into the same platform
        for subpop in self.population_list:
            subpop.simulation = self
        for connpair in self.connection_list:
            connpair.simulation = self
            
        
            
        # initialize population_list, calculate 
        
        
        for p in self.population_list:
            p.initialize()      # 2   
        
        for c in self.connection_list:
            print 'initialize population'
            c.initialize()      # 1
            
    def update(self,t0 = 0.0,dt = 1e-3,tf = 1e-1):
        self.dt = dt
        self.tf = tf
        m_record = np.zeros((len(self.population_list),np.int(tf/dt)+10))
        mu_record = np.zeros((len(self.population_list),np.int(tf/dt)+10))
        
        
        # initialize:
        start_time = time.time()
        self.initialize(t0)
        self.initialize_time_period = time.time()-start_time
        
        # start_running
        start_time = time.time()
        counter = 0
        while self.t < self.tf:
            self.t+=self.dt
            ind_rec = 0
            if self.verbose: print 'time: %s' % self.t
            for p in self.population_list:
                p.update()
                m_record[ind_rec,counter] = p.curr_firing_rate
                mu_record[ind_rec,counter] = p.v1
                ind_rec += 1
                
                
            for c in self.connection_list:
                c.update()
            counter +=1
        return m_record,mu_record
                
 # Recurrent Connection
class ConnectionDistribution(object):
    """
    Parameters:
    which could define unique connection,
    like weight, nsyn and prob
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,edges,weights,probs,sparse = True):
        self.edges   = edges
        self.weights = weights
        self.probs   = probs
        """
        be remained!
        1) flux_matrix and threshold_flux_matrix,
        if connection has identical weight syn and prob, then the clux  matrix
        should be identical, this could be reuse --> connection_distribution
        """
        # mentioned above could be solved
        self.flux_matrix = None
        self.threshold_flux_vector = None
        self.fluxn_matrix = None
        self.threshold_fluxn_vector = None
        self.fluxk_matrix = None
        self.threshold_fluxk_matrix = None
        self.nmdajump = None
        # notice that threshold_flux_vector is !!! vector which 
        # could map to each voltage-center
        # self.simulation = None
        
        # reversal potential could be used in conductance based model
        self.reversal_potential = None
        if self.reversal_potential != None:
            assert NotImplementedError 
    def initialize(self):
        """
        if we already have those connectional properties, we could 
        calculate the flux_matrix and threshold_flux_matrix
        these matrix could be reused at identical connection_cluster as well as time steps
        """
        nv = len(self.edges)-1
        self.flux_matrix = np.zeros((nv,nv))
        self.threshold_flux_vector = np.zeros(nv)
        curr_threshold_flux_vector,curr_flux_matrix = flux_matrix(self.edges,self.weights,self.probs)
        self.flux_matrix = curr_flux_matrix
        self.threshold_flux_vector = curr_threshold_flux_vector
        self.fluxn_matrix = np.eye(nv)
        self.threshold_fluxn_vector = np.zeros(nv)

        self.fluxk_matrix = np.eye(nv)
        self.threshold_fluxk_vector = np.zeros(nv)
        curr_threshold_fluxk_vector,curr_fluxk_matrix = flux_matrix(self.edges,self.weights,self.probs)
        self.fluxk_matrix = curr_fluxk_matrix
        self.threshold_fluxk_vector = curr_threshold_fluxk_vector



    @property    
    def signature(self):
        """
        unique signature
        """
        return (tuple(self.edges),tuple([self.weights]),tuple([self.probs]))           
            
            
        
    
# Recurrent Connection
class Connection(object):
    """
    Parameters:
    pre-population
    post-population
    nsyn-population
    connection weight
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,pre,post,nsyn,weights,probs,conn_type):
        self.pre_population = pre
        self.post_population = post
        self.nsyn = nsyn  # Number of Pre(sender) population
        self.weights = weights
        self.probs = probs
        self.conn_type = conn_type
        # multiply probability of connection
        
        """
        1) connection_list should be classified into some unique population(cluster)
        which means, if 'weight''syn''prob' is identical,should be classified into identical 
        connection_distribution
        2) curr_firing_rate could be replace by ...
        3) simulation could be used to find original platform
        """
        # initialize None and Initialize when simulation
        self.firing_rate = 0.0
        self.simulation = None
        # long range
        self.inmda = 0.0
        """
        be remained!
        1) flux_matrix and threshold_flux_matrix,
        if connection has identical weight syn and prob, then the clux  matrix
        should be identical, this could be reuse --> connection_distribution
        """
    # initialize by hand! when start simulation
    def initialize(self):
        self.initialize_connection_distribution()
        self.initialize_firing_rate()
        self.initialize_I_nmda()
    
    def initialize_connection_distribution(self):
        CD = ConnectionDistribution(self.post_population.edges,self.weights,self.probs)
        CD.simulation = self.simulation
        self.simulation.connection_distribution_collection.add_unique_connection(CD)
        self.connection_distribution = self.simulation.connection_distribution_collection[CD.signature]
        
        
    def initialize_firing_rate(self):
        self.firing_rate = self.pre_population.curr_firing_rate
    # LONG RANGE 
    def initialize_I_nmda(self):
        self.inmda = self.pre_population.curr_Inmda
        
    def update(self):
        self.firing_rate = self.pre_population.curr_firing_rate
        self.inmda       = self.pre_population.curr_Inmda
        # initialize_firing_rate
    def update_flux_matrix(self,flux_matrix,threshold_flux_matrix):
        self.flux_matrix = flux_matrix
        self.threshold_flux_matrix = threshold_flux_matrix
    def update_connection(self,npre,npost,nsyn,**nkwargs):
        self.pre_population = [],
        self.pre_population = npre,
        self.post_population = [],
        self.post_population = npost,
        self.syn_population = [],
        self.syn_population = nsyn
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    @property
    def curr_Inmda(self):
        return self.inmda
    

        
        
# External Feedforward Input
class ExternalPopulation(object):
    """
    Parameters:
    etaE/I
    record:
    flag for recoding firing rate or not
    """
    def __init__(self,firing_rate,dt,record=False,**kwargs):
        self.firing_rate_stream = firing_rate
        self.firing_rate = 0.0
        # may adding function to generate spike/firing rate trial use lambdify!!!
        self.type = 'External'
        self.dt = dt
        # additional data/parameters
        self.metadata = kwargs
        self.inmda = 0.0
        self.hnmda = 0.0
        


        # for long-range connections
        self.tau_r = 0.002
        self.tau_d = 0.108

        self.v1 = 0.0
        
        
        # initialize in simulation
        self.simulation = None
    def initialize(self):
        self.initialize_firing_rate()
    def update(self):
        self.update_firing_rate()
        self.update_NMDA_midvar_syncurr()
        # print 'Ext NMDA: ',self.curr_Inmda,' HNMDA: ',self.hnmda
        
    def initialize_firing_rate(self):
        # current time --> in platform
        self.curr_t = self.simulation.t
        
        try:
            self.firing_rate = self.firing_rate_stream[np.int(self.curr_t/self.dt)]
        except:
            self.firing_rate = 0.0
    def update_firing_rate(self):
        self.curr_t = self.simulation.t
        
        try:
            self.firing_rate = self.firing_rate_stream[np.int(self.curr_t/self.dt)]
        except:
            self.firing_rate = 0.0
        """
        if self.curr_t >0.3:
            temp = 100+50*np.absolute(np.sin(40*self.curr_t))
            self.firing_rate = self.firing_rate_stream * 1.0
        else:
            self.firing_rate = self.firing_rate_stream * 1.0
        """
    # update own hNMDA and iNMDA, which only depends on curr_firing_rate 
    # in another words, a-subpopulation's hNMDA & iNMDA only depend on itself
    def update_NMDA_midvar_syncurr(self):
        ownfr = self.curr_firing_rate
        # parameters
        deltat = self.dt
        trise  = self.tau_r
        tdamp  = self.tau_d

        tr   = deltat/trise
        etr  = np.exp(-tr)
        td   = deltat/tdamp
        etd  = np.exp(-td)
        cst  = 1.0/(tdamp - trise)*(etd - etr) # trise/(tdamp - trise)*(etd - etr)

        self.inmda = self.inmda * etd + self.hnmda * cst
        self.hnmda = self.hnmda *etr + ownfr * self.dt

    @property
    def curr_firing_rate(self):
        curr_firing_rate = self.firing_rate
        return curr_firing_rate
    @property
    def curr_Inmda(self):
        curr_Inmda = self.inmda
        return curr_Inmda
    
class RecurrentPopulation(object):
    """
    Parameters:
    tau_m: time constant for membrane potential
    v_min: minimum voltage(default = -1.0)
    v_th/max: maximum/threshold 
    dv   : voltage domain discritization 
    record: flag(True/False)
    curr_firing_rate: firing rate of the corresponding recurrent population
    update_mode: str'approx' or 'exact'(default=')
    
    """
    
    def __init__(self,tau_m = 0.020,dt = 1e-4,v_min = -1.0,v_max = 1.0,dv = 1e-4,record = True,
                firing_rate = 0.0,update_method = 'exact',approx_order = None,tol = 1e-12,norm = np.inf,**kwargs):
        # transmit parameters
        self.tau_m = tau_m
        self.dt    = dt
        (self.v_min,self.v_max) = (v_min,v_max)
        self.dv = dv
        self.record = record
        self.firing_rate = 0.0 # firing_rate
        self.update_method = update_method
        self.approx_order = approx_order
        self.tol = tol
        self.norm = norm
        
        self.type = 'Recurrent'
        # additional parameters
        self.metadata = kwargs
        
        # before real initialization, voltage-edge and voltage-distribution
        # are all None, these setting should be initialized later by specific command
        
        self.edges = None
        self.rhov = None
        self.firing_rate_record = None # used for recording corresponding spike train
        self.t_record = None # time series
        self.leak_flux_matrix = None
        
        # simulation in identical platform
        self.simulation = None

        # if we use active release NMDA synaptic signal
        # once the sender(pre) population generated firing rate
        # it had ability to automatically release NMDA-type slow conductance
        # so it naturally has this property(without self.weights)
        self.hnmda = 0.0
        self.inmda = 0.0

        self.v1 = 0.0
        self.v2 = 0.0
        self.v3 = 0.0
        self.v4 = 0.0

        self.La0 = None
        self.fin = None
        
        self.total_fp_vslave = 0.0
        self.total_fp_sigv   = 0.0

        # for long-range connections
        self.tau_r = 0.002
        self.tau_d = 0.108

        self.nmda_vr = 0.0
        
    def initialize(self):
        """
        initialize some specific parameters and variables by hand
        with 
            1)voltage-edge/bin
            2)connection dictionary
            3)all about recorder
        """
        self.initialize_edges()
        self.initialize_prob()
        self.initialize_total_input_dict()

        self.initialize_fpmusigv_dict()
        
        self.initialize_nmdajump()

        
    """
    Code below is designed for some basic matrix or elements which might be initialized
    at the beginning and maintained unchanged during the whole data analysis, but if U 
    need update some connections or strurtures, U could still start the 'Update' function 
    to regenerate a new structure(connections)
    """   
    def initialize_edges(self):
        # initialize discreted voltage bins
        self.edges = get_v_edges(self.v_min,self.v_max,self.dv)
        leak_flux_matrix = leak_matrix(self.edges,self.tau_m)
        self.leak_flux_matrix = leak_flux_matrix

    def initialize_fpmusigv_dict(self):
        self.total_fpmu_dict = {}
        self.total_fpsig_dict = {}
        # identical for long-range connections
        for c in self.source_connection_list:
            if (c.conn_type =='ShortRange'):
                # if already initialize connection_distribution or not
                try:
                    curr_mu = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                    curr_sigv = self.total_fpsig_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_mu = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                    curr_sigv = self.total_fpsig_dict.setdefault(c.connection_distribution,0)
                self.total_fpmu_dict[c.connection_distribution] = curr_mu + c.curr_firing_rate * c.nsyn * c.weights
                self.total_fpsig_dict[c.connection_distribution] = curr_sigv + c.curr_firing_rate * c.nsyn * (c.weights**2)
            else:
                try:
                    # tmp_nmda_h = self.tmp_HNMDA_dict.setdefault(c.connection_distribution,0)
                    # tmp_nmda_i = self.tmp_INMDA_dict.setdefault(c.connection_distribution,0)
                    curr_nmda_i = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    # tmp_nmda_h = self.tmp_HNMDA_dict.setdefault(c.connection_distribution,0)
                    # tmp_nmda_i = self.tmp_INMDA_dict.setdefault(c.connection_distribution,0)

                    curr_nmda_i = self.total_fpmu_dict.setdefault(c.connection_distribution,0)

                self.total_fpmu_dict[c.connection_distribution] = curr_nmda_i + c.curr_Inmda *  c.nsyn * c.weights
                # no contribution to sigma
            
        # summation
        self.total_fp_vslave = 0.0
        for key,val in self.total_fpmu_dict.items():
            
            try:
                self.total_fp_vslave += val
            except:
                key.initialize()              
                self.total_fp_vslave += val
        self.total_fp_vslave  = self.total_fp_vslave * self.tau_m
                
        # summation
        self.total_fp_sigv = 0.0
        for key,val in self.total_fpsig_dict.items():
            try:
                self.total_fp_sigv += val
            except:
                key.initialize()
                self.total_fp_sigv += val
        self.total_fp_sigv  = self.total_fp_sigv * self.tau_m

    def initialize_prob(self):
        # initialize voltage-distribution
        self.rhov = np.zeros_like(self.edges[:-1])
        zero_bin_list = get_zero_bin_list(self.edges)
        for ii in zero_bin_list:
            self.rhov[ii] = 1./len(zero_bin_list)



    # also combine short-range and long-range input
    # short-range input relates to real firing rate of pre-population
    # while long-range input relates to slow-changed NMDA-type synaptic input
    # both have no relationship with weights!
    def initialize_total_input_dict(self):
        self.total_inputsr_dict = {}
        for c in self.source_connection_list:
            if(c.conn_type == 'ShortRange'):
                # if already initialize connection_distribution or not
                try:
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_input = self.total_inputsr_dict.setdefault(c.connection_distribution,0)
                self.total_inputsr_dict[c.connection_distribution] = curr_input + c.curr_firing_rate * c.nsyn
        self.total_inputlrk_dict = {}
        for c in self.source_connection_list:
            if(c.conn_type == 'LongRange'):
                # if already initialize connection_distribution or not
                try:
                    curr_input = self.total_inputlrk_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    # then have name and signature
                    curr_input = self.total_inputlrk_dict.setdefault(c.connection_distribution,0)
                self.total_inputlrk_dict[c.connection_distribution] = 0.0 +  c.nsyn * c.curr_Inmda


    def initialize_nmdajump(self):
        """
        long-range connections lift base-voltage, but didn't generate flux!!!
        """
        self.nmdakeep_dict = {}
        self.nmda_dict = {}
        for c in self.source_connection_list:
            if(c.conn_type == 'LongRange'):
                try:
                    curr_nmdakeep = self.nmdakeep_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_ditribution()
                    curr_nmdakeep = self.nmdakeep_dict.setdefault(C.connection_distribution,0)
                self.nmdakeep_dict[c.connection_distribution] = curr_nmdakeep 

            
    def update_total_flux_matrix(self):
        """
        long-range connections lift base-voltage, but didn't generate flux!!!
        """
        # update NMDA
        total_flux_matrix = self.leak_flux_matrix.copy()
        # this with Inmda
        for key,val in self.total_inputsr_dict.items():
            # print 'val: ',val
            try:
                total_flux_matrix += key.flux_matrix * val
            except:
                key.initialize()
                total_flux_matrix += key.flux_matrix * val
        print 'instan: ',val
        
        # sustained variable
        for key,val in self.total_inputlrk_dict.items():
            # print 'LR val: ',val
            try:
                total_flux_matrix += key.fluxk_matrix * val
                # print 'FLUX_MATRIX FOR NMDA: ',key.fluxk_matrix
            except:
                key.initialize()
                total_flux_matrix += key.fluxk_matrix * val
        
        print 'keep :  ',val
        return total_flux_matrix


    
    
    # short range and long rage input
    def update_total_input_dict(self):
        # for curr_CD in self.source_connection_list:
        # have already exist
        # short range input
        for curr_CD in self.total_inputsr_dict.keys():
            self.total_inputsr_dict[curr_CD] = 0.0
        # ok, already be zero
        for c in self.source_connection_list:
            if ( c.conn_type =='ShortRange'):
                self.total_inputsr_dict[c.connection_distribution] += c.curr_firing_rate * c.nsyn

        # for long range connections, the input stimuli will change step-by-step as well(base on firing rate of pre-population)
        # but will not immediately change Inmda (as well as hNMDA), because this function only care about total INPUT so, could reset and 
        # refresh step-by-step
        for curr_CD in self.total_inputlrk_dict.keys():
            self.total_inputlrk_dict[curr_CD] = 0.0
        for c in self.source_connection_list:
            if (c.conn_type == 'LongRange'):
                self.total_inputlrk_dict[c.connection_distribution] +=  c.nsyn * c.curr_Inmda

    def update_prob(self):
        J = self.update_total_flux_matrix()
        
        if self.update_method == 'exact':
            self.rhov = exact_update_method(J,self.rhov,self.simulation.dt)           
        elif self.update_method == 'approx':

            if self.approx_order == None:
                self.rhov = approx_update_method_tol(J, self.rhov, tol=self.tol, dt=self.simulation.dt, norm=self.norm)

            else:
                self.rhov = approx_update_method_order(J, self.rhov, approx_order=self.approx_order, dt=self.simulation.dt)
        
        else:
            raise Exception('Unrecognized population update method: "%s"' % self.update_method)  # pragma: no cover
        max_loc = np.where(self.rhov==np.amax(self.rhov))
        self.v1 = self.edges[max_loc[0][0]]
        
        # we could calculate firing rate here
    def update_firing_rate(self):
        flux_vector = reduce(np.add,[key.threshold_flux_vector * val for key,val in
                                    self.total_inputsr_dict.items()])
        # have long range or not
        flag = 0
        for c in self.source_connection_list:
            if (c.conn_type == 'LongRange'):
                flag = 1
        if (flag==1):
            
            fluxk_vector = reduce(np.add,[key.threshold_fluxk_vector * val for key,val in
                                    self.total_inputlrk_dict.items()])
            # adding together
            flux_vector += fluxk_vector
            
            
        self.firing_rate = np.dot(flux_vector,self.rhov)
        print 'firing rate: ',self.firing_rate

    # update own hNMDA and iNMDA, which only depends on curr_firing_rate 
    # in another words, a-subpopulation's hNMDA & iNMDA only depend on itself
    def update_NMDA_midvar_syncurr(self):
        ownfr = self.curr_firing_rate
        # parameters
        deltat = self.dt
        trise  = self.tau_r
        tdamp  = self.tau_d

        tr   = deltat/trise
        etr  = np.exp(-tr)
        td   = deltat/tdamp
        etd  = np.exp(-td)
        cst  = 1.0/(tdamp - trise)*(etd - etr) # trise/(tdamp - trise)*(etd - etr)

        self.inmda = self.inmda * etd + self.hnmda * cst
        self.hnmda = self.hnmda * etr + ownfr  * self.dt
        print 'release NMDA: ',self.inmda
        # print 'ownfr:  ',ownfr

    """
    then ! we use moment!!!!!!!
    """
       
    def update_total_fpmu_dict(self):
        # identical for each long-range connection
        # extract parameters
        deltat = self.dt
        trise  = self.tau_r
        tdamp  = self.tau_d

        tr  = deltat/trise
        etr = np.exp(-tr) 
        td  = deltat/tdamp
        etd = np.exp(-td)
        cst = trise/(tdamp-trise)

        # nmda should keep in memory which could not be reset to zerooooooo!!!
        """
        no resetting to zero --> go directly to refreshing !!! based on pre-value
        """
        # this type of variables have already been updated in update_NMDA_midvar_syncurr



        # for curr_CD in self.source_connection_list:
        # have already exist
        for c in self.source_connection_list:
           # if(c.conn_type == 'ShortRange'):
            self.total_fpmu_dict[c.connection_distribution] = 0.0
            """
            no matter short-/long-range connections should at first reset to zeros~~~
            in case not to be distubed by previous result
            """
        # have already clear up all the short range connections
        for c in self.source_connection_list:
            if(c.conn_type == 'ShortRange'):
                self.total_fpmu_dict[c.connection_distribution] += c.curr_firing_rate * c.nsyn * c.weights
                # print 'AMPA: ',c.curr_firing_rate * c.nsyn * c.weights
            else:
                self.total_fpmu_dict[c.connection_distribution] += c.curr_Inmda * c.nsyn * c.weights
                # print 'NMDA: ',c.curr_Inmda * c.nsyn * c.weights



        # summation
        self.total_fp_vslave = 0.0
        for key,val in self.total_fpmu_dict.items():
            
            try:
                self.total_fp_vslave += val
            except:
                key.initialize()
                self.total_fp_vslave += val
        self.total_fp_vslave = self.total_fp_vslave * self.tau_m

    
    def update_total_fpsig_dict(self):
        """
        update sigma(variance) for fokker-planck equation
        """
        for curr_CD in self.total_inputsr_dict.keys():
            self.total_fpsig_dict[curr_CD] = 0.0
        for c in self.source_connection_list:
            if(c.conn_type =='ShortRange'):
                self.total_fpsig_dict[c.connection_distribution] += (c.curr_firing_rate) * (c.weights**2) * c.nsyn

        # summation
        self.total_fp_sigv = 0.0
        for key,val in self.total_fpsig_dict.items():
            try:
                self.total_fp_sigv += val
            except:
                key.initialize()
                self.total_fp_sigv += val
        self.total_fp_sigv = self.total_fp_sigv * self.tau_m

    def update_fp_moment4(self):
        v1 = self.v1.copy()
        v2 = self.v2.copy()
        v3 = self.v3.copy()
        v4 = self.v4.copy()

        fr = self.curr_firing_rate
        vs = self.total_fp_vslave
        ds = self.total_fp_sigv

        dtgL = self.dt / self.tau_m
        gL   = 1.0/self.tau_m

        v1n = v1 + dtgL*(-fr/gL - (v1-vs))
        v2n = v2 + dtgL*(-fr/gL - 2.0*(v2-vs*v1-0.5*ds))
        v3n = v3 + dtgL*(-fr/gL - 3.0*(v3-vs*v2-ds*v1))
        v4n = v4 + dtgL*(-fr/gL - 4.0*(v4-vs*v3-1.5*ds*v2))

        self.v1 = v1n
        self.v2 = v2n
        self.v3 = v3n
        self.v4 = v4n            

        
    def update_ME_moment4(self):
        self.update_NMDA_midvar_syncurr()
        self.update_total_fpsig_dict()
        self.update_total_fpmu_dict()
        self.update_fp_moment4()
        # print 'Rec NMDA: ',self.curr_Inmda,self.inmda,' HNMDA: ',self.hnmda
        # print 'vs: ',self.total_fp_vslave ,'ds: ',self.total_fp_sigv

        vs = self.total_fp_vslave
        ds = self.total_fp_sigv
        La0 = self.La0
        fin  = self.fin
        gL   = 1.0/self.tau_m


        # print 'vs, ds, La0, fin,gL:',vs,ds,La0,fin,gL
        h = self.edges[2]-self.edges[1]
        # print 'H:',h
        vedges = self.edges
        vedges = 0.5*(vedges[0:-1] + vedges[1:])
        # print 'vslave" ',self.total_fp_vslave,'var: ',self.total_fp_sigv
        rhoEQ,sum_rhoEQ = rho_EQ(self.total_fp_vslave,self.total_fp_sigv,vedges)
        self.rhoEQ      = rhoEQ
        gamma = [1,self.v1,self.v2,self.v3,self.v4]
        fi    = np.transpose(gamma)
        F     = fi[1:3]
        (tmu,tx,tPEq,tfin,tgamma) = (F,vedges,self.rhoEQ,fin,1)
        a0    = La0
        res   = minimize(optfun,a0,args=(tmu,tx,tPEq,tfin,tgamma))
        La1   = res.x
        La0   = np.real(La1)
        # print 'La1:',La1,'rhoEQ: ',rhoEQ
        rhov   = rhoEQ*np.exp(np.dot(np.squeeze(fin[:,:]),La1))
        # print 'rho :',rhov
        # normalization
        rhov   = rhov/(h*np.sum(rhov))

        firing_rate  = gL*np.sqrt(ds)*np.exp(np.sum(La1))/sum_rhoEQ/2

        self.La0 = La0
        self.rhov = rhov
        self.firing_rate = firing_rate        
        
        
        
    # updating function set
    def update(self):
        self.update_total_input_dict()
        self.update_NMDA_midvar_syncurr()
        self.update_total_flux_matrix()
        self.update_firing_rate()
        self.update_prob()   
              

    @property
    def source_connection_list(self):
        return [c for c in self.simulation.connection_list if c.post_population == self]
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    @property
    def curr_rho_voltage(self):
        return self.rhov
    @property
    def curr_Inmda(self):
        return self.inmda
    @property
    def curr_Hnmda(self):
        return self.hnmda
    @property
    def n_bins(self):
        # number of voltage bins
        return len(self.edges)-1
        
    @property
    def n_edges(self):
        return len(self.edges)
