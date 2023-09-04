#%%
import os
import pdb
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MAX_N_EXPOSURES, MIN_N_NEURONS)

from scripts.run_by_epoch import clusterless_thetasweeps
from tqdm.auto import tqdm

def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    neuron_info = make_neuron_dataframe(ANIMALS)
    n_neurons = (neuron_info
                    .groupby(['animal', 'day', 'epoch'])
                    .neuron_id
                    .agg(len)
                    .rename('n_neurons')
                    .to_frame())

    epoch_info = epoch_info.join(n_neurons)
    is_w_track = (epoch_info.environment
                    .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
    #is_animal = epoch_info.index.isin(['bon', 'fra', 'gov', 'dud', 'con'], level='animal')
    is_animal = epoch_info.index.isin(['bon', 'cha', 'con'], level='animal')
    
    valid_epochs = (is_w_track &
                    (epoch_info.n_neurons > MIN_N_NEURONS) &
                    (epoch_info.exposure <= MAX_N_EXPOSURES) &
                    is_animal
                    )
    
    # #print epoch_info.n_neurons and epoch_info.exposure if it is is_w_track
    # print(epoch_info.n_neurons)
    # print(epoch_info.exposure)
    # print(epoch_info.environment)

    
    #%%
    DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/Processed-Data'
    for epoch_key in tqdm(epoch_info[valid_epochs].index, desc='epochs'):
        animal, day, epoch = epoch_key
        
        if animal == 'cha' and day == 3 and epoch == 2:
            continue # skip this epoch because there is some error in the data will come back to this later
        if animal == 'cha' and day == 3 and epoch == 4:
            continue # skip this epoch because there is some error in the data will come back to this later
        
        # Check if this file has already been run
        replay_info_filename = os.path.join(
            DATA_DIR,
            'ThetaSweepTrajectories',
            f'{animal}_{day:02d}_{epoch:02d}_cv_classifier_clusterless_results.nc')

        if not os.path.isfile(replay_info_filename):
            animal, day, epoch = epoch_key
            print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
            clusterless_thetasweeps(epoch_key) #set brain_areas to ['CA1'] if only care about CA1
            
    # %% run single epoch_key
    # PROCESSED_DATA_DIR = '/home/zilong/Desktop/replay_trajectory_paper/ThetaSweep-Results'
    # epoch_key = ("bon", 8, 4)
    # print(f'Animal: {epoch_key[0]}, Day: {epoch_key[1]}, Epoch: {epoch_key[2]}')
    # clusterless_thetasweeps(epoch_key) 

if __name__ == '__main__':
    main()