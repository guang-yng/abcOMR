import json
import torch
from fire import Fire
from data import PretrainingDataset
from smt_trainer import SMTPP_Trainer
from ExperimentConfig import ExperimentConfig, experiment_config_from_dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class GradNormMonitor(Callback):
    def on_after_backward(self, trainer, pl_module):
        # Calculate total gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Log gradient norm
        pl_module.log("grad_norm", total_norm, on_step=True, prog_bar=True)

torch.set_float32_matmul_precision('high')

def main(config:ExperimentConfig):

    data = PretrainingDataset(config=config)
    w2i = data.train_dataset.w2i
    pad_token = w2i['<pad>'] if '<pad>' in w2i else w2i['<|pad|>']
    model_wrapper = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=2048, out_categories=len(data.train_dataset.w2i), 
                                  padding_token=pad_token, in_channels=1, w2i=data.train_dataset.w2i, i2w=data.train_dataset.i2w, 
                                  d_model=256, dim_ff=256, num_dec_layers=8)
    
    wandb_logger = WandbLogger(project='SMTABC', group=f"musescoreabc-314", name=f"SMTABC-3lr", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/pretraining/", filename=f"SMTABC-sqrtd-scheduler-3lr", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=3, verbose=True)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    grad_norm_monitor = GradNormMonitor()

    trainer = Trainer(max_epochs=10000, 
                      gradient_clip_val=1.0,
                      check_val_every_n_epoch=1, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping, lr_monitor, grad_norm_monitor], precision='bf16-mixed')

    trainer.fit(model_wrapper, datamodule=data)
    
    model_wrapper = SMTPP_Trainer.load_from_checkpoint(checkpointer.best_model_path)
    
    model_wrapper.model.save_pretrained("SMTABC", variant="MuseScore_ABC_pretrain")
    

def launch(config_path:str):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config)

if __name__ == "__main__":
    Fire(launch)
