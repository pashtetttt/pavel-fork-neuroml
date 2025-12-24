from omegaconf import OmegaConf
from dcond_trainer import DCoND_Trainer

if __name__ == "__main__":
    args = OmegaConf.load('dcond_args.yaml')
    trainer = DCoND_Trainer(args)
    trainer.train()
