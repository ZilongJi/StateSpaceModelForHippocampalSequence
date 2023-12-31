#%%
from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal
from replay_trajectory_classification.misc import NumbaKDE

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# sleep duration threshold (in seconds)
sleep_duration_threshold = 100

# Data directories and definitions
#ROOT_DIR = join(abspath(dirname(__file__)), pardir)
ROOT_DIR = '/home/zilong/Desktop/replay_trajectory_paper'
#RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
RAW_DATA_DIR = '/media/zilong/4TB Storage HDD/ReplayTrajectoryAnalysis/Raw-Data'
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
    'cha': Animal(directory=join(RAW_DATA_DIR, 'Chapati'), short_name='cha'),
    'con': Animal(directory=join(RAW_DATA_DIR, 'Conley'), short_name='con'),
    'Cor': Animal(directory=join(RAW_DATA_DIR, 'Corriander'), short_name='Cor'),
    'dav': Animal(directory=join(RAW_DATA_DIR, 'Dave'), short_name='dav'),
    'dud': Animal(directory=join(RAW_DATA_DIR, 'Dudley'), short_name='dud'),
    'egy': Animal(directory=join(RAW_DATA_DIR, 'Egypt'), short_name='egy'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'Frank'), short_name='fra'),
    'gov': Animal(directory=join(RAW_DATA_DIR, 'Government'),
                  short_name='gov'),
    #'hig': Animal(directory=join(RAW_DATA_DIR, 'Higgs'), short_name='hig'),
    'remy': Animal(directory=join(RAW_DATA_DIR, 'Remy'), short_name='remy'),
}

#add by Zilong
MIN_N_NEURONS = 9

# ANIMALS = {
#     'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
# }

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']
_BRAIN_AREAS = ['CA1', 'CA2', 'CA3']

# Classifier Parameters
place_bin_size = 3.0
'''
In one time step (2 ms),
68% likely to be within 2.45 cm (12.25 m / s)
95% likely to be within 4.90 cm (24.5 m / s)
99.7% likely to be within 7.35 cm (36.75 m / s)
'''
movement_var = 6.0
replay_speed = 1
model = NumbaKDE
'''Bandwith is the standard deviation of the gaussian kernel. The first four
entries correspond to the bandwidth for the marks. The last two for the
position dimensions.
'''
model_kwargs = {
    'bandwidth': np.array([24.0, 24.0, 24.0, 24.0, 6.0, 6.0])
}
knot_spacing = 5
spike_model_penalty = 0.5

'''
1. Geometric mean of duration is: n_time_steps = 1 / (1 - p)
2. So p = 1 - (1 / n_time_steps).
3. Want `n_time_steps` to equal 100 ms.
4. If our timestep is 2 ms, then n_time_steps = 50
5. So p = 0.98
'''
discrete_diag = 0.98

continuous_transition_types = (
    [['random_walk', 'uniform', 'identity'],  # noqa
     ['uniform',     'uniform', 'uniform'],   # noqa
     ['random_walk', 'uniform', 'identity']])  # noqa

TRANSITION_TO_CATEGORY = {
    'identity': 'Hover',
    'random_walk': 'Continuous',
    'uniform': 'Fragmented',
    'inverse_random_walk': 'Fragmented',
    'random_walk_minus_identity': 'Continuous',
}

PROBABILITY_THRESHOLD = 0.8

STATE_ORDER = ['Hover', 'Hover-Continuous-Mix', 'Continuous',
               'Fragmented-Continuous-Mix', 'Fragmented', 'Unclassified']

SHORT_STATE_ORDER = ['Stat.', 'Stat.-Cont.-Mix', 'Cont.',
                     'Frag.-Cont.-Mix', 'Frag.', 'Unclass.']

# Plotting Colors
STATE_COLORS = {
    'Hover': '#9f043a',
    'Stat.': '#9f043a',
    'Stationary': '#9f043a',
    'Fragmented': '#ff6944',
    'Frag.': '#ff6944',
    'Continuous': '#521b65',
    'Cont.': '#521b65',
    'Hover-Continuous-Mix': '#61c5e6',
    'Stationary-Continuous-Mix': '#61c5e6',
    'Stat.-Cont.-Mix': '#61c5e6',
    'Fragmented-Continuous-Mix': '#2a586a',
    'Frag.-Cont.-Mix': '#2a586a',
    '': '#c7c7c7',
    'Unclassified': '#c7c7c7',
    'Unclass.': '#c7c7c7',
}

# %%

classifier_parameters_thetasweeps = {
    'movement_var': 25.0,
    'replay_speed': 1,
    'place_bin_size': 5.0,
    'continuous_transition_types': [['random_walk', 'uniform'],
                                    ['uniform',     'uniform']],
    'model_kwargs': {
        'bandwidth': np.array([20.0, 20.0, 20.0, 20.0, 8.0])}

}

discrete_state_transition = np.array([[0.968, 0.032],
                                      [0.968, 0.032]])
