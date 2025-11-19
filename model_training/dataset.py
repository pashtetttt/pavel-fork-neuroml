import os
import torch
from torch.utils.data import Dataset 
import h5py
import pathlib
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math 

class BrainToTextDataset(Dataset):
    '''
    Dataset for brain-to-text data
    
    Returns an entire batch of data instead of a single example
    '''

    def __init__(
            self, 
            trial_indicies,
            n_batches,
            split = 'train', 
            batch_size = 64, 
            days_per_batch = 1, 
            random_seed = -1,
            must_include_days = None,
            feature_subset = None
            ): 
        '''
        trial_indicies:  (dict)      - dictionary with day numbers as keys and lists of trial indices as values
        n_batches:       (int)       - number of random training batches to create
        split:           (string)    - string specifying if this is a train or test dataset
        batch_size:      (int)       - number of examples to include in batch returned from __getitem_()
        days_per_batch:  (int)       - how many unique days can exist in a batch; this is important for making sure that updates 
                                       to individual day layers in the GRU are not excesively noisy. Validation data will always have 1 day per batch
        random_seed:     (int)       - seed to set for randomly assigning trials to a batch. If set to -1, trial assignment will be random
        must_include_days ([int])    - list of days that must be included in every batch
        feature_subset  ([int])      - list of neural feature indicies that should be the only features included in the neural data 
         '''
        
        # Set random seed for reproducibility
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.split = split

        # Ensure the split is valid
        if self.split not in ['train', 'test']:
            raise ValueError(f'split must be either "train" or "test". Received {self.split}')
        
        self.days_per_batch = days_per_batch

        self.batch_size = batch_size

        self.n_batches = n_batches

        self.days = {}
        self.n_trials = 0 
        self.trial_indicies = trial_indicies
        self.n_days = len(trial_indicies.keys())

        self.feature_subset = feature_subset

        # Calculate total number of trials in the dataset
        for d in trial_indicies:
            self.n_trials += len(trial_indicies[d]['trials'])

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(f'must_include_days must be less than or equal to days_per_batch. Received {must_include_days} and days_per_batch {days_per_batch}')
        
        if must_include_days is not None and len(must_include_days) > self.n_days and split != 'train':
            raise ValueError(f'must_include_days is not valid for test data. Received {must_include_days} and but only {self.n_days} in the dataset')
        
        if must_include_days is not None:
            # Map must_include_days to correct indicies if they are negative
            for i, d in enumerate(must_include_days):
                if d < 0: 
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days    

        # Ensure that the days_per_batch is not greater than the number of days in the dataset. Raise error
        if self.split == 'train' and self.days_per_batch > self.n_days:
            raise ValueError(f'Requested days_per_batch: {days_per_batch} is greater than available days {self.n_days}.')
           
        
        if self.split == 'train':
            self.batch_index = self.create_batch_index_train()
        else: 
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index.keys()) # The validation data has a fixed amount of data 
    
    def __len__(self):
        ''' 
        How many batches are in this dataset. 
        Because training data is sampled randomly, there is no fixed dataset length, 
        however this method is required for DataLoader to work 
        '''
        return self.n_batches
    
    def __getitem__(self, idx):
        ''' 
        Gets an entire batch of data from the dataset, not just a single item
        '''
        batch = {
            'input_features' : [],
            'seq_class_ids' : [],
            'n_time_steps' : [],
            'phone_seq_lens' : [],
            'day_indicies' : [],
            'transcriptions' : [],
            'block_nums' : [],
            'trial_nums' : [],
        }

        index = self.batch_index[idx]

        # Iterate through each day in the index
        for d in index.keys():

            # Open the hdf5 file for that day
            with h5py.File(self.trial_indicies[d]['session_path'], 'r') as f:

                # For each trial in the selected trials in that day
                for t in index[d]:
                    
                    try: 
                        g = f[f'trial_{t:04d}']

                        # Remove features is neccessary 
                        input_features = torch.from_numpy(g['input_features'][:]) # neural data
                        if self.feature_subset:
                            input_features = input_features[:,self.feature_subset]

                        batch['input_features'].append(input_features)

                        batch['seq_class_ids'].append(torch.from_numpy(g['seq_class_ids'][:]))  # phoneme labels
                        batch['transcriptions'].append(torch.from_numpy(g['transcription'][:])) # character level transcriptions
                        batch['n_time_steps'].append(g.attrs['n_time_steps']) # number of time steps in the trial - required since we are padding
                        batch['phone_seq_lens'].append(g.attrs['seq_len']) # number of phonemes in the label - required since we are padding
                        batch['day_indicies'].append(int(d)) # day index of each trial - required for the day specific layers 
                        batch['block_nums'].append(g.attrs['block_num'])
                        batch['trial_nums'].append(g.attrs['trial_num'])
                    
                    except Exception as e:
                        print(f'Error loading trial {t} from session {self.trial_indicies[d]["session_path"]}: {e}')
                        continue

        # Pad data to form a cohesive batch
        batch['input_features'] = pad_sequence(batch['input_features'], batch_first = True, padding_value = 0)
        batch['seq_class_ids'] = pad_sequence(batch['seq_class_ids'], batch_first = True, padding_value = 0)

        batch['n_time_steps'] = torch.tensor(batch['n_time_steps']) 
        batch['phone_seq_lens'] = torch.tensor(batch['phone_seq_lens'])
        batch['day_indicies'] = torch.tensor(batch['day_indicies'])
        batch['transcriptions'] = torch.stack(batch['transcriptions'])
        batch['block_nums'] = torch.tensor(batch['block_nums'])
        batch['trial_nums'] = torch.tensor(batch['trial_nums'])

        return batch
    

    def create_batch_index_train(self):
        '''
        Create an index that maps a batch_number to batch_size number of trials

        Each batch will have days_per_batch unique days of data, with the number of trials for each day evenly split between the days
        (or as even as possible if batch_size is not divisible by days_per_batch)
        '''

        batch_index = {}

        # Precompute the days that are not in must_include_days and have trials
        if self.must_include_days is not None:
            # Filter must_include_days to only those with trials
            valid_must_include_days = [d for d in self.must_include_days if self.trial_indicies[d]['trials']]
            non_must_include_days_with_trials = [d for d in self.trial_indicies.keys() if d not in self.must_include_days and self.trial_indicies[d]['trials']]
        else:
            valid_must_include_days = []
            non_must_include_days_with_trials = [d for d in self.trial_indicies.keys() if self.trial_indicies[d]['trials']]

        # Check if we have enough days with trials to satisfy days_per_batch
        total_valid_days = len(valid_must_include_days) + len(non_must_include_days_with_trials)
        if total_valid_days < self.days_per_batch:
            available_days_str = ', '.join([f'Day {d}' for d in list(valid_must_include_days) + non_must_include_days_with_trials])
            raise ValueError(
                f"Cannot create batches with {self.days_per_batch} unique days per batch. "
                f"Only {total_valid_days} days have available trials: [{available_days_str}]. "
                f"Check if HDF5 files contain valid trials."
            )

        for batch_idx in range(self.n_batches):
            batch = {}

            # Which days will be used for this batch. Picked randomly without replacement from days that have trials
            # TODO: In the future we may want to consider sampling days in proportion to the number of trials in each day

            # If must_include_days is not empty (and valid), we will use those days and then randomly sample the rest from remaining days with trials
            if valid_must_include_days:
                # Ensure we don't try to sample more days than available in non_must_include_days_with_trials
                additional_days_needed = self.days_per_batch - len(valid_must_include_days)
                if additional_days_needed > 0:
                    if len(non_must_include_days_with_trials) < additional_days_needed:
                        # This case might occur if the requirement changes or data is very limited
                        # For now, raise an error if we cannot fulfill the request
                        # Or, adjust days_per_batch for this specific batch (less ideal)
                        raise ValueError(
                            f"Cannot select {self.days_per_batch} days for batch {batch_idx}. "
                            f"Must-include days: {valid_must_include_days}, "
                            f"Other available days: {non_must_include_days_with_trials}. "
                            f"Insufficient other days to meet days_per_batch requirement."
                        )
                    selected_other_days = np.random.choice(non_must_include_days_with_trials, size=additional_days_needed, replace=False).tolist()
                    days_for_batch = valid_must_include_days + selected_other_days
                else:
                    # days_per_batch is less than or equal to valid_must_include_days
                    # This scenario (days_per_batch < len(valid_must_include_days)) might be unusual but possible.
                    # Let's select the required number from valid_must_include_days
                    if self.days_per_batch <= len(valid_must_include_days):
                        days_for_batch = np.random.choice(valid_must_include_days, size=self.days_per_batch, replace=False).tolist()
                    else:
                        # Should not happen if the initial check passed, but safeguarding
                        raise RuntimeError("Internal logic error in batch creation.")
            # Otherwise we will select random days (that have trials) without replacement
            else:
                if len(non_must_include_days_with_trials) < self.days_per_batch:
                    # Should not happen if the initial check passed, but safeguarding
                    raise RuntimeError("Internal logic error in batch creation.")
                days_for_batch = np.random.choice(non_must_include_days_with_trials, size=self.days_per_batch, replace=False).tolist()

            # How many trials will be sampled from each day in this batch
            num_trials_per_day = math.ceil(self.batch_size / self.days_per_batch) # Use ceiling to make sure we get at least batch_size trials

            for d in days_for_batch:
                # Ensure the day has trials before sampling
                available_trials = self.trial_indicies[d]['trials']
                if not available_trials:
                    # This check should ideally not be needed if the day was selected correctly,
                    # but acts as a safeguard.
                    raise RuntimeError(f"Internal error: Selected day {d} for batch {batch_idx} has no trials.")
                # Trials are sampled with replacement, so if a day has less than num_trials_per_day trials, it won't be a problem
                trial_idxs = np.random.choice(available_trials, size=num_trials_per_day, replace=True)
                batch[d] = trial_idxs

            # Remove extra trials added due to ceiling calculation
            extra_trials = (num_trials_per_day * len(days_for_batch)) - self.batch_size

            # While we still have extra trials, remove one trial randomly from a random day in the batch
            while extra_trials > 0:
                # Choose a random day from the batch's days
                d = np.random.choice(days_for_batch)
                if len(batch[d]) > 0: # Ensure the day's trial list isn't already empty
                    batch[d] = np.delete(batch[d], -1) # Remove the last trial index
                    extra_trials -= 1
                else:
                    # This shouldn't happen if all days started with at least one trial,
                    # but safeguarding. If a day ends up empty before extra_trials are removed,
                    # it implies batch_size was too small relative to days_per_batch.
                    # In such a case, we might need to remove the day, but that changes batch composition.
                    # Let's assume for now that num_trials_per_day ensures at least one trial per day initially.
                    # If extra_trials removal leads to an empty day list *before* reaching 0 extra_trials,
                    # it indicates an edge case with batch_size/days_per_batch/trials_per_day.
                    # For this fix, we assume the logic holds for typical cases.
                    # A more robust fix might involve re-assigning trials between days or adjusting the batch structure,
                    # which is complex.
                    # Let's just break if we encounter this unlikely scenario during removal.
                    self.logger.warning(f"Warning: Attempting to remove extra trials but a day in batch {batch_idx} is already empty. Stopping removal.")
                    break # Exit the removal loop if a problem occurs


            batch_index[batch_idx] = batch

        return batch_index
    
    def create_batch_index_test(self):
        '''
        Create an index that is all validation/testing data in batches of up to self.batch_size

        If a day does not have at least self.batch_size trials, then the batch size will be less than self.batch_size

        This index will ensures that every trial in the validation set is seen once and only once
        '''
        batch_index = {}
        batch_idx = 0
        
        for d in self.trial_indicies.keys():

            # Calculate how many batches we need for this day
            num_trials = len(self.trial_indicies[d]['trials'])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size 
            
            # Create batches for this day
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                
                # Get the trial indices for this batch
                batch_trials = self.trial_indicies[d]['trials'][start_idx:end_idx]
                
                # Add to batch_index
                batch_index[batch_idx] = {d : batch_trials}
                batch_idx += 1
        
        return batch_index
        
def train_test_split_indicies(file_paths, test_percentage = 0.1, seed = -1, bad_trials_dict = None):
    '''
    Split data from file_paths into train and test splits 
    Returns two dictionaries that detail which trials in each day will be a part of that split:
    Example: 
        {
            0: trials[1,2,3], session_path: 'path'
            1: trials[2,5,6], session_path: 'path'
        }

    Args:
        file_paths (list): List of file paths to the hdf5 files containing the data
        test_percentage (float): Percentage of trials to use for testing. 0 will use all trials for training, 1 will use all trials for testing
        seed (int): Seed for reproducibility. If set to -1, the split will be random
        bad_trials_dict (dict): Dictionary of trials to exclude from the dataset. Formatted as:
            {
                'session_name_1': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                'session_name_2': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                ...
            }
    '''
    # Set seed for reporoducibility
    if seed != -1:
        np.random.seed(seed)

    # Get trials in each day
    trials_per_day = {}
    for i, path in enumerate(file_paths):
        # session = [s for s in path.split('/') if (s.startswith('t15.20') or s.startswith('t12.20'))][0]
        
        path_obj = pathlib.Path(path)
        session_dir_name = path_obj.parent.name

        if session_dir_name.startswith('t15.20') or session_dir_name.startswith('t12.20'):
            session = session_dir_name
        else:
            # Handle case where the parent directory doesn't match the expected pattern
            # This might indicate an incorrect path structure
            print(f"Warning: Path {path} does not seem to have a session directory matching 't15.20*' or 't12.20*' as its parent.")
            # You might want to raise an error or handle this differently
            # For now, let's assume it's an error in the path or naming convention
            raise ValueError(f"Session directory name '{session_dir_name}' in path '{path}' does not start with 't15.20' or 't12.20'. Check path format and session naming.")

        good_trial_indices = []

        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                num_trials = len(list(f.keys()))
                for t in range(num_trials):
                    key = f'trial_{t:04d}'
                    
                    block_num = f[key].attrs['block_num']
                    trial_num = f[key].attrs['trial_num']

                    if (
                        bad_trials_dict is not None
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        # print(f'Bad trial: {session}_{block_num}_{trial_num}')
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {'num_trials': len(good_trial_indices), 'trial_indices': good_trial_indices, 'session_path': path}

    # Pick test_percentage of trials from each day for testing and (1 - test_percentage) for training
    train_trials = {}
    test_trials = {}

    for day in trials_per_day.keys():

        num_trials = trials_per_day[day]['num_trials']

        # Generate all trial indices for this day (assuming 0-indexed)
        all_trial_indices = trials_per_day[day]['trial_indices']

        # If test_percentage is 0 or 1, we can just assign all trials to either train or test
        if test_percentage == 0:
            train_trials[day] = {'trials' : all_trial_indices, 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : [], 'session_path' : trials_per_day[day]['session_path']}
            continue
        
        elif test_percentage == 1:
            train_trials[day] = {'trials' : [], 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : all_trial_indices, 'session_path' : trials_per_day[day]['session_path']}
            continue    

        else:
            # Calculate how many trials to use for testing
            num_test = max(1, int(num_trials * test_percentage))
            
            # Randomly select indices for testing
            test_indices = np.random.choice(all_trial_indices, size=num_test, replace=False).tolist()
            
            # Remaining indices go to training
            train_indices = [idx for idx in all_trial_indices if idx not in test_indices]
            
            # Store the split indices
            train_trials[day] = {'trials' : train_indices, 'session_path' : trials_per_day[day]['session_path']}
            test_trials[day] = {'trials' : test_indices, 'session_path' : trials_per_day[day]['session_path']}
    
    return train_trials, test_trials