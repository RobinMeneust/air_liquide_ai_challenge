import lightning.pytorch as pl
import torch
from torch import nn

class TorchLightningModule(pl.LightningModule):
    def __init__(self, model_class, model_params, lr=0.001, device="cuda"):
        super().__init__()
        self.lr = lr
        self.model = model_class(**model_params).to(device)
        self.loss_function = nn.L1Loss()
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        
        y_pred = self.model(X)
        loss_train = self.loss_function(y_pred, y)
        
        self.log('train_loss', loss_train, prog_bar=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        X, y = batch
        
        y_pred = self.model(X)
        loss_val = self.loss_function(y_pred, y)
        
        self.log('val_loss', loss_val, prog_bar=True)
        return loss_val
    
    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        
        return self.get_test_metrics(y_pred, y)
    
    def on_test_epoch_end(self, outputs):
        torch.cat([output["y_pred"] for output in outputs])
        for key in self.test_scores:
            self.test_scores[key] /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
            self.log(f"test_{key}", self.test_scores[key])
    
    @staticmethod
    def get_test_metrics(y_pred, y):
        mse_function = nn.MSELoss()
        mse_value = mse_function(y_pred, y)
        return {
            "mse": mse_value.item(),
            "mae": torch.mean(torch.abs(y_pred - y)).item(),
            "rmse": torch.sqrt(mse_value).item(),
        }
