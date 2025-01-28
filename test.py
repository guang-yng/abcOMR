import torch
import json
from data import FinetuningDataset, PretrainingDataset
from smt_trainer import SMTPP_Trainer
from ExperimentConfig import ExperimentConfig, experiment_config_from_dict
from fire import Fire

from lightning.pytorch import Trainer
import wandb
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

def main(config:ExperimentConfig):
    model_wrapper = SMTPP_Trainer.load_from_checkpoint('weights/pretraining/SMTABC-sqrtd-scheduler-3lr-v2.ckpt')
    wandb_logger = WandbLogger(project='SMTABC', group=f"musescoreabc-314", name=f"SMTABC-3lr", log_model=False)

    trainer = Trainer(max_epochs=100000, 
                      logger=wandb_logger,
                      check_val_every_n_epoch=1, precision='bf16-mixed')

    wandb_logger.watch(model_wrapper.model, log='all')

    
    data = PretrainingDataset(config=config)

    test_results = trainer.test(model_wrapper, datamodule=data)
    wandb.log({"test_results": test_results})
    
    # model_wrapper.model.save_pretrained("SMTABC", variant="Mozarteum_BeKern_fold0")
    
def launch(config_path:str):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)

    main(config=config)

if __name__ == "__main__":
    Fire(launch)