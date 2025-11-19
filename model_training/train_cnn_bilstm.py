from omegaconf import OmegaConf
from cnn_bilstm_trainer import BrainToTextDecoder_Trainer # Import from the new trainer file

# Load arguments from the new YAML file
args = OmegaConf.load('cnn_bilstm_args.yaml')

# Initialize the trainer with the new arguments
trainer = BrainToTextDecoder_Trainer(args)

# Start the training process
metrics = trainer.train()

print("Training completed.")
print(f"Final best validation PER: {trainer.best_val_PER}")
print(f"Final best validation loss: {trainer.best_val_loss}")

# Optionally, save training stats
import pickle
with open(f'{args.checkpoint_dir}/train_stats.pkl', 'wb') as f:
    pickle.dump(metrics, f)
