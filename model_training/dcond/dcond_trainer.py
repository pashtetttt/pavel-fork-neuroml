# dcond_trainer.py
from rnn_trainer import BrainToTextDecoder_Trainer  # Reuse most logic
import torch

class DCoND_Trainer(BrainToTextDecoder_Trainer):
    def __init__(self, args):
        # Initialize parent class (loads data, builds model, etc.)
        super().__init__(args)

    def create_model(self):
        """Override to use DCoND model"""
        from dcond_model import DCoND_GRUDecoder
        self.model = DCoND_GRUDecoder(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_classes=self.args['dataset']['n_classes'],
            n_layers=self.args['model']['n_layers'],
            rnn_dropout=self.args['model']['rnn_dropout'],
            input_dropout=self.args['model']['input_layer_dropout'],
            n_days=len(self.args['dataset']['sessions'])
        )

    def train_step(self, batch):
        """Override training step to use diphone+monophone loss"""
        features = batch['input_features'].to(self.device)
        labels = batch['seq_class_ids'].to(self.device)
        n_time_steps = batch['n_time_steps'].to(self.device)
        phone_seq_lens = batch['phone_seq_lens'].to(self.device)
        day_indices = batch['day_indicies'].to(self.device)

        monophone_logits, diphone_logits = self.model(features, day_indices)

        # CTC Loss on monophones (main loss)
        monophone_log_probs = torch.log_softmax(monophone_logits, dim=-1).permute(1, 0, 2)
        loss_mono = self.ctc_loss(monophone_log_probs, labels, n_time_steps, phone_seq_lens)

        # Optional: CTC Loss on diphones (auxiliary loss)
        # You'd need diphone-level labels (not implemented here)
        loss_diphone = 0.0

        # Combined loss (α=0.6 as in DCoND paper)
        alpha = 0.6
        total_loss = alpha * loss_mono.mean() + (1 - alpha) * loss_diphone

        return total_loss, monophone_logits
