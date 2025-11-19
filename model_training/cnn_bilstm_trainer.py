import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle

from dataset import BrainToTextDataset, train_test_split_indicies
# from data_augmentations import gauss_smooth # You might need to adapt this for CNN input shape

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
torch.backends.cudnn.deterministic = True # makes training more reproducible
torch._dynamo.config.cache_size_limit = 64

from cnn_bilstm_model import CNNBiLSTMModel # Import the new model

class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder using CNN+BiLSTM
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']:
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Set seed if provided
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model
        self.model = CNNBiLSTMModel(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            n_cnn_layers = self.args['model']['n_cnn_layers'],
            cnn_dropout = self.args['model']['cnn_dropout'],
            n_lstm_layers = self.args['model']['n_lstm_layers'],
            lstm_dropout = self.args['model']['lstm_dropout'],
            input_dropout = self.args['model']['input_layer_dropout'],
        )

        # Call torch.compile to speed up training
        # self.logger.info("Using torch.compile")
        # self.model = torch.compile(self.model)

        self.logger.info(f"Initialized CNN+BiLSTM decoding model")
        self.logger.info("NOT using torch.compile for testing on CPU...")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day_' in name:
                day_params += param.numel()

        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        # Note: train_test_split_indicies expects full paths to data_train.hdf5 files
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_val.hdf5') for s in self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days (sessions)
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and test sets using the existing function
        # For training set: test_percentage = 0 means all data goes to train
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths,
            test_percentage = 0, # All for training
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None, # Add logic to load bad_trials_dict if needed
        )
        # For validation set: test_percentage = 1 means all data goes to test (validation)
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths,
            test_percentage = 1, # All for validation
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None, # Add logic to load bad_trials_dict if needed
        )


        # Save dictionaries to output directory to know which trials were train vs val
        # The keys in train_trials/val_trials are day indices (integers)
        # The values are dicts with 'trials' (list of trial nums) and 'session_path' (str)
        # Map day indices back to session names for logging if needed
        train_sessions_map = {day_idx: self.args['dataset']['sessions'][day_idx] for day_idx in train_trials.keys()}
        val_sessions_map = {day_idx: self.args['dataset']['sessions'][day_idx] for day_idx in val_trials.keys()}

        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f:
            json.dump({
                'train': {train_sessions_map[day]: train_trials[day] for day in train_trials},
                'val': {val_sessions_map[day]: val_trials[day] for day in val_trials}
            }, f)


        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] is not None:
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')

        # train dataset and dataloader
        # Note: split='train', days_per_batch, n_batches as per args
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials, # Pass the dictionary returned by split function
            split = 'train', # Use 'train' as per dataset.py
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None, # Add logic if needed
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, # Shuffling is handled internally by the dataset
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True
        )

        # val dataset and dataloader
        # Note: split='test' (as per dataset.py convention for validation), days_per_batch=None, n_batches=None
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, # Pass the dictionary returned by split function
            split = 'test', # Use 'test' as per dataset.py for validation
            days_per_batch = None, # Fixed to 1 day per batch for validation
            n_batches = None, # Calculated internally based on data size
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None, # Add logic if needed
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, # Shuffling is handled internally by the dataset
            num_workers = 0, # Typically 0 for validation to avoid potential issues with randomization
            pin_memory = True
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set CNN/BiLSTM and/or input layers to not trainable if specified
        for name, param in self.model.named_parameters():
            if not self.args['model']['cnn_trainable'] and 'cnn' in name:
                param.requires_grad = False
            elif not self.args['model']['lstm_trainable'] and 'lstm' in name:
                param.requires_grad = False
            elif not self.args['model']['input_network']['input_trainable'] and 'day_' in name:
                param.requires_grad = False

        # Send model to device
        self.model.to(self.device)

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups
        Biases and day weights should not be decayed
        Day weights should have a separate learning rate
        '''
        bias_params = [p for name, p in self.model.named_parameters() if 'bias' in name and 'day_' not in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]

        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay
            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)

            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # biases
                lambda step: lr_lambda(
                    step,
                    lr_min_day / lr_max_day,
                    lr_decay_steps_day,
                    lr_warmup_steps_day,
                    ), # day params
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # biases
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")

        return LambdaLR(optim, lr_lambdas, -1)

    def load_model_checkpoint(self, load_path):
        '''
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)

        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint
        '''

        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }

        torch.save(checkpoint, save_path)

        self.logger.info("Saved model to checkpoint: " + save_path)

        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    # Note: The `transform_data` method might need adjustments for CNNs (e.g., smoothing might be applied differently)
    # For now, keeping it similar to the RNN trainer, but consider if augmentations like warping are applied per-feature or per-timepoint.
    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''
        # This is a simplified version. You might need to adapt data augmentation logic
        # for CNN inputs (e.g., ensure features are correctly handled after transformations).
        # The original RNN trainer's `gauss_smooth` function might need to be checked for compatibility.
        # For now, assume it works or adapt it.

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1] # Assuming (B, T, C) format

        # We only apply these augmentations in training
        if mode == 'train':
            # Example: add white noise
            if self.args['dataset']['data_transforms']['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.args['dataset']['data_transforms']['white_noise_std']

            # Example: add constant offset noise
            if self.args['dataset']['data_transforms']['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.args['dataset']['data_transforms']['constant_offset_std']

            # Example: add random walk noise
            if self.args['dataset']['data_transforms']['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.args['dataset']['data_transforms']['random_walk_std'], dim = 1) # Assuming axis=1 is time

            # Example: randomly cutoff part of the data timecourse
            if self.args['dataset']['data_transforms']['random_cut'] > 0:
                cut = np.random.randint(0, self.args['dataset']['data_transforms']['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data
        # This is done in both training and validation
        if self.args['dataset']['data_transforms']['smooth_data']:
            # Placeholder - you need to implement or adapt gauss_smooth for CNN inputs
            # Ensure gauss_smooth handles (B, T, C) correctly or convert if needed.
            # from data_augmentations import gauss_smooth
            # features = gauss_smooth(
            #     inputs = features,
            #     device = self.device,
            #     smooth_kernel_std = self.args['dataset']['data_transforms']['smooth_kernel_std'],
            #     smooth_kernel_size= self.args['dataset']['data_transforms']['smooth_kernel_size'],
            # )
            pass # Placeholder - implement or adapt as needed

        return features, n_time_steps


    def train(self):
        '''
        Train the model
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        # training params
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()

        # train for specified number of batches
        for i, batch in enumerate(self.train_loader):

            self.model.train()
            self.optimizer.zero_grad()

            # Train step
            start_time = time.time()

            # Move data to device
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Use autocast for efficiency
            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):

                # Apply augmentations to the data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                # Calculate adjusted lens for CTC loss after model processing
                # THIS IS THE CRITICAL PART TO FIX FOR CNN+BiLSTM
                # The model might change the sequence length. You need to determine the output length.
                # Example: If CNN has no padding and stride 1, and BiLSTM is unidirectional, output_len = input_len
                # If CNN has stride > 1, pooling, or BiLSTM is bidirectional, output_len might differ.
                # For now, assume output length is same as input length (n_time_steps) - THIS NEEDS VERIFICATION
                # If your CNN+BiLSTM reduces length, you need to calculate the actual output length.
                # Let's assume the model outputs the same length for simplicity, but check the model!
                # If the model output is shorter, adjust n_time_steps accordingly.
                # For example, if the last layer reduces sequence length by a factor of 2:
                # adjusted_lens = (n_time_steps / 2).to(torch.int32)
                # For now, using n_time_steps as a placeholder:
                adjusted_lens = n_time_steps.to(torch.int32)

                # Get phoneme predictions
                logits = self.model(features, day_indicies)
                # Check the shape of logits: (batch_size, output_seq_len, n_classes)
                model_output_len = logits.shape[1]
                # If model_output_len != adjusted_lens, you need to adjust adjusted_lens or the model
                # For CTC, input_lengths should match the sequence length dimension of log_probs
                # If model_output_len is different from n_time_steps, adjust adjusted_lens
                # For example, if model consistently outputs half the length:
                # adjusted_lens = (n_time_steps // 2).to(torch.int32)
                # Or, if the model outputs a fixed length based on its architecture, calculate it.
                # For now, assume model output length matches the input length (after any potential patching in RNN was applied).
                # If patching is not used in CNN, and the model outputs the same length as input_features.shape[1],
                # then adjusted_lens should be n_time_steps.
                # If you add pooling or strided convs, adjust this logic.
                # Example: If the model outputs length T_out, then:
                # adjusted_lens = torch.full_like(n_time_steps, T_out) # Or calculate based on input T_in
                # For now, let's assume the model output length is the same as the input feature length.
                # So, if features is (B, T, C), logits is (B, T, n_classes), then adjusted_lens = n_time_steps.
                # This is often true if no downsampling is applied in the temporal dimension.
                # If you apply downsampling (e.g., stride=2 conv, maxpool), you need to calculate the resulting length.
                # For a simple CNN+BiLSTM without temporal downsampling, this might be correct.
                # Let's keep it as n_time_steps for now, but this is the most likely place for an error.
                # You might need to inspect the model's forward pass or add a method to get output length.

                # Calculate CTC Loss
                # CTC expects [T, B, C] for log_probs, [B, S] for targets
                log_probs_transposed = torch.permute(logits.log_softmax(2), [1, 0, 2]) # [T, B, C]
                loss = self.ctc_loss(
                    log_probs = log_probs_transposed,
                    targets = labels,
                    input_lengths = adjusted_lens, # This should be the length after model processing
                    target_lengths = phone_seq_lens
                )

                loss = torch.mean(loss) # take mean loss over batches

            loss.backward()

            # Clip gradient
            if self.args['grad_norm_clip_value'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm = self.args['grad_norm_clip_value'],
                                               error_if_nonfinite = True,
                                               foreach = True
                                               )

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            # Save training metrics
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                        f'loss: {(loss.detach().item()):.2f} ' +
                        f'grad norm: {grad_norm:.2f} '
                        f'time: {train_step_duration:.3f}')

            # Incrementally run a test step
            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")

                # Calculate metrics on val data
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time


                # Log info
                self.logger.info(f'Val batch {i}: ' +
                        f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                        f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                        f'time: {val_step_duration:.3f}')

                if self.args['log_individual_day_val_PER']:
                    # Note: val_metrics['day_PERs'] keys are day indices (integers)
                    for day_idx in val_metrics['day_PERs'].keys():
                        session_name = self.args['dataset']['sessions'][day_idx]
                        per = val_metrics['day_PERs'][day_idx]['total_edit_distance'] / val_metrics['day_PERs'][day_idx]['total_seq_length']
                        self.logger.info(f"{session_name} val PER: {per:0.4f}")

                # Save metrics
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                # Determine if new best day. Based on if PER is lower, or in the case of a PER tie, if loss is lower
                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss):
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:

                    # Checkpoint if metrics have improved
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)

                    # save validation metrics to pickle file
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0

                else:
                    val_steps_since_improvement +=1

                # Optionally save this validation checkpoint, regardless of performance
                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}', val_metrics['avg_PER'])

                # Early stopping
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break

        # Log final training steps
        training_duration = time.time() - train_start_time


        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model
        if self.args['save_final_model']:
            # OLD LINE - Missing 'loss' argument
            # self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])
            # NEW LINE - Pass both PER and the corresponding loss
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1], val_losses[-1]) # Pass PER and corresponding loss

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}

        # Record metrics
        if return_logits:
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data:
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0
        total_seq_length = 0

        # Calculate PER for each specific day
        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            # Note: Validation uses 'test' split, so dataset_probability_val might not apply directly
            # The val_dataset internally handles day-specific batches
            # We initialize for all days present in val_trials
            if d in self.val_dataset.trial_indicies: # Check if day 'd' exists in validation data
                 day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Note: The validation dataset handles day-specific batching internally.
            # All trials in a batch come from the same day (or a fixed number per args).
            # We don't need the dataset_probability_val logic here as it's handled by the dataset itself.
            # The first day index in the batch represents the day for this whole batch.
            day = day_indicies[0].item() # Get the day index for this batch (they should all be the same in val)

            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    # Calculate adjusted lens for CTC loss (same as in training)
                    # This needs to match the logic used in the train loop.
                    # Placeholder: assume same length as input, adjust if needed based on model architecture.
                    adjusted_lens = n_time_steps.to(torch.int32)

                    logits = self.model(features, day_indicies)

                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]), # [T, B, C]
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)

                metrics['losses'].append(loss.cpu().detach().numpy())

                # Calculate PER per day and also avg over entire validation set
                batch_edit_distance = 0
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    # Decode using argmax and CTC rules (remove blanks and repeats)
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1) # Remove repeats
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0]) # Remove blanks (ID 0)

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )

                    batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)

                    decoded_seqs.append(decoded_seq)

            # Update metrics for the day corresponding to this batch
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            # Record metrics
            if return_logits:
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data:
                metrics['input_features'].append(batch['input_features'].cpu().numpy())

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics
