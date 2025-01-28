import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import lightning.pytorch as L
from transformers import GenerationConfig, get_linear_schedule_with_warmup

from torchinfo import summary
from eval_functions import compute_poliphony_metrics_abc
from smt_model.configuration_smt import SMTConfig
from smt_model.modeling_smt import SMTModelForCausalLM
from typing import Any, Dict, List, Union

class SMTPP_Trainer(L.LightningModule):
    def __init__(self, maxh, maxw, maxlen, out_categories, padding_token, in_channels, w2i, i2w, d_model=256, dim_ff=256, num_dec_layers=8):
        super().__init__()
        config = SMTConfig(maxh=maxh, maxw=maxw, maxlen=maxlen, out_categories=out_categories,
                           padding_token=padding_token, in_channels=in_channels, 
                           w2i=w2i, i2w=i2w,
                           d_model=d_model, dim_ff=dim_ff, num_dec_layers=num_dec_layers)
        self.model = SMTModelForCausalLM(config)
        self.padding_token = padding_token
        
        self.preds = []
        self.grtrs = []
        
        self.worst_loss = -1
        self.worst_image = None
        self.best_loss = np.inf
        self.best_image = None

        summary(self, input_size=[(1,1,config.maxh,config.maxw), (1,config.maxlen)], 
                dtypes=[torch.float, torch.long])
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), lr=3e-4, amsgrad=False, weight_decay=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=31000,
            num_training_steps=620000
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def forward(self, input, last_preds) -> Any:
        return self.model(input, last_preds)
    
    def training_step(self, batch):
        x, di, y, = batch
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        
        if loss.item() > self.worst_loss:
            self.worst_image = x
            self.worst_loss = loss.item()
        
        if loss.item() < self.best_loss:
            self.best_image = x
            self.best_loss = loss.item()
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.logger.experiment.log({"worst_training_loss": [wandb.Image(worst_image.cpu().numpy()) for worst_image in self.worst_image]})
        self.logger.experiment.log({"best_training_loss": [wandb.Image(best_image.squeeze(0).cpu().numpy()) for best_image in self.best_image]})
        
        self.worst_loss = -1
        self.worst_image = None
        self.best_loss = np.inf
        self.best_image = None
        
    
    def validation_step(self, val_batch):
        x, dec_in, y = val_batch
        predicted_sequences, _ = self.model.predict(input=x)
        
        # dec = "".join(predicted_sequence)
        # dec = dec.replace("<t>", "\t")
        # dec = dec.replace("<b>", "\n")
        # dec = dec.replace("<s>", " ")
        dec = [[token for token in predicted_sequence if token not in ['<|pad|>', '<|begin_of_abc|>', '<|end_of_abc|>']] for predicted_sequence in predicted_sequences]

        # gt = "".join([self.model.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        # gt = gt.replace("<t>", "\t")
        # gt = gt.replace("<b>", "\n")
        # gt = gt.replace("<s>", " ")
        gt = [[self.model.i2w[token.item()] for token in tokens if self.model.i2w[token.item()] not in ['<|pad|>', '<|begin_of_abc|>', '<|end_of_abc|>']] for tokens in y] 

        self.preds.extend(dec)
        self.grtrs.extend(gt)
        
    def on_validation_epoch_end(self, metric_name="val") -> None:
        cer, ser, ler = compute_poliphony_metrics_abc(self.preds, self.grtrs)
        
        random_index = random.randint(0, len(self.preds)-1)
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        print(f"[Prediction] - {''.join(predtoshow)}")
        print(f"[GT] - {''.join(gttoshow)}")
        
        self.log(f'{metric_name}_CER', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER', ler, on_epoch=True, prog_bar=True)
        
        self.preds = []
        self.grtrs = []
        
        return ser
    
    def test_step(self, test_batch) -> Union[torch.Tensor, Dict[str, Any], None]:
        return self.validation_step(test_batch)
    
    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end("test")