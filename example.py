import matplotlib.pyplot as plt
from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.simulation import Simulation
from dipde.internals.connection import Connection as Connection
import itertools

nsyn_background = {
    (0, 'bkg'): 1,
    (1, 'bkg'): 1
}

background_firing_rate = {
    (0,'bkg'): 180,
    (1,'bkg'): 120
}

background_start = {
    (0,'bkg'): 20,
    (1,'bkg'): 70
}

background_end = {
    (0,'bkg'): 60,
    (1,'bkg'): 110
}

internal_population_sizes = {
    (0, 'e'): 300,
    (0, 'i'): 100,
    (1, 'e'): 300,
    (1, 'i'): 100,
    (2, 'e'): 300,
    (2, 'i'): 100,
}

connection_weightss = {
    ((0,'e'),(0,'e')):.00234, ((1,'e'),(1,'e')):.00234, ((2,'e'),(2,'e')):.00234, ((3,'e'),(3,'e')):.00234,
    ((0,'i'),(0,'e')):-.00346, ((1,'i'),(1,'e')):-.00346, ((2,'i'),(2,'e')):-.00346, ((3,'i'),(3,'e')):-.00346, 
    ((0,'e'),(0,'i')):.00157, ((1,'e'),(1,'i')):.00157, ((2,'e'),(2,'i')):.00157, ((3,'e'),(3,'i')):.00157, 
    ((0,'i'),(0,'i')):-.00109, ((1,'i'),(1,'i')):-.00109, ((2,'i'),(2,'i')):-.00109, ((3,'i'),(3,'i')):-.00109, 


internal_population_settings = {'v_min': -1.0, 
                                'v_max': 1.0,
                                'dv':.001,
                                'update_method':'approx',
                                'tol':1e-14,
                                'tau_m':.01,
                                'record':True}

# Simulation settings:
t0 = 0.
dt = .0002
tf = .1
verbose = True
save = False

sstm = 90.0
sbase = 120.0

# Create visual stimuli
External_stimuli_dict = {}
for index, celltype in itertools.product([0,1],['bkg']):
    stm_tmp = np.zeros(np.int(tf/dt))
    stm_tmp[np.int(background_start[index,celltype]/dt):np.int(background_end[index,celltype]/dt)] = sstm
    stm_tmp += sbase
    External_stimuli_dict[index,celltype] = stm_tmp

# Create populations:
background_population_dict = {}
internal_population_dict = {}
for index, celltype in itertools.product([0,1], ['bkg']):    
    background_population_dict[index, celltype] = ExternalPopulation(External_stimuli[index,celltype],dt, record=False)
for index, celltype in itertools.product([0,1,2,3], ['e','i']):    
    internal_population_dict[index, celltype] = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method, approx_order=approx_order, tol=tol)

# Create background connections:
connection_list = []
for index, celltype in itertools.product([0], ['e', 'i']):
    source_population = background_population_dict[0,'bkg']
    target_population = internal_population_dict[layer, celltype]
    if celltype == 'e':
        background_delay = .005
    else:
        background_delay = 0.
    curr_connection = Connection(source_population, target_population, nsyn_background[layer, celltype], weights=[conn_weights['e']], probs=[1.], delay=background_delay) 
    connection_list.append(curr_connection)

# Create recurrent connections:
for source_layer, source_celltype in itertools.product([23, 4, 5, 6], ['e', 'i']):
    for target_layer, target_celltype in itertools.product([23, 4, 5, 6], ['e', 'i']):
        source_population = internal_population_dict[source_layer, source_celltype]
        target_population = internal_population_dict[target_layer, target_celltype]
        nsyn = connection_probabilities[(source_layer, source_celltype), (target_layer, target_celltype)]*internal_population_sizes[source_layer, source_celltype]
        weight = conn_weights[source_celltype]
        curr_connection = Connection(source_population, target_population, nsyn, weights=[weight], probs=[1.], delay=0)
        connection_list.append(curr_connection)

# Create simulation:
population_list = background_population_dict.values() + internal_population_dict.values()
simulation = Simulation(population_list, connection_list, verbose=True)

# Run simulation:
simulation.run(dt=dt, tf=tf, t0=t0)

# Visualize:
y_label_dict = {23:'2/3', 4:'4', 5:'5', 6:'6'}
fig, axes = plt.subplots(nrows=4, ncols=1, **{'figsize':(4,8)})
for row_ind, layer in enumerate([23, 4, 5, 6]):
    for plot_color, celltype in zip(['r', 'b'],['e', 'i']):
        curr_population = internal_population_dict[layer, celltype]
        axes[row_ind].plot(curr_population.t_record, curr_population.firing_rate_record, plot_color)

    axes[row_ind].set_xlim([0,tf])
    axes[row_ind].set_ylim(ymin=0)
    axes[row_ind].set_ylabel('Layer %s\nfiring rate (Hz)' % y_label_dict[layer])
    if layer == 5: axes[row_ind].legend(['Excitatory', 'Inhibitory'], prop={'size':10}, loc=4)

axes[3].set_xlabel('Time (seconds)')
fig.tight_layout()

if save == True: plt.savefig('./cortical_column.png')

plt.show()

