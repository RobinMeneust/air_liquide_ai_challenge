import lightning.pytorch as pl
import torch
from torch import nn

class TorchLightningModule(pl.LightningModule):
    """PyTorch Lightning module for forecasting
    
    Attributes:
        lr (float): Learning rate
        model (nn.Module): Model
        loss_function (nn.Module): Loss function
    """
    def __init__(self, model_class, model_params, lr=0.001, device="cuda"):
        """Initialize the Lightning module
        
        Args:
            model_class (nn.Module): Model class
            model_params (dict): Model parameters
            lr (float): Learning rate
            device (str): Device to use
        """
        super().__init__()
        self.lr = lr
        self.model = model_class(**model_params).to(device)
        self.loss_function = nn.L1Loss()
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        """Create the optimizer
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        """Training step
        
        Args:
            batch (tuple): Batch containing the input and target data
            batch_idx (int): Batch index
        
        Returns:
            torch.Tensor: Loss value
        """
        X, y = batch
        
        y_pred = self.model(X)
        loss_train = self.loss_function(y_pred, y)
        
        self.log('train_loss', loss_train, prog_bar=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        """Validation step
        
        Args:
            batch (tuple): Batch containing the input and target data
        
        Returns:
            torch.Tensor: Loss value
        """
        X, y = batch
        
        y_pred = self.model(X)
        loss_val = self.loss_function(y_pred, y)
        
        self.log('val_loss', loss_val, prog_bar=True)
        return loss_val
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Output data
        """
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        """Test step
        
        Args:
            batch (tuple): Batch containing the input and target data
            batch_idx (int): Batch index
        
        Returns:
            dict: Dictionary containing the test metrics
        """
        X, y = batch
        y_pred = self.model(X)
        
        return self.get_test_metrics(y_pred, y)
    
    def on_test_epoch_end(self, outputs):
        """Test epoch end (aggregate and log test scores)
        
        Args:
            outputs (list): List of outputs
        """
        torch.cat([output["y_pred"] for output in outputs])
        for key in self.test_scores:
            self.test_scores[key] /= len(self.trainer.datamodule.test_dataloader()) # Divide by the number of batches
            self.log(f"test_{key}", self.test_scores[key])
    
    @staticmethod
    def get_test_metrics(y_pred, y):
        """Get the test metrics
        
        Args:
            y_pred (torch.Tensor): Predicted values
            y (torch.Tensor): True values
        
        Returns:
            dict: Dictionary containing the test metrics (mse, mae, rmse)
        """
        mse_function = nn.MSELoss()
        mse_value = mse_function(y_pred, y)
        return {
            "mse": mse_value.item(),
            "mae": torch.mean(torch.abs(y_pred - y)).item(),
            "rmse": torch.sqrt(mse_value).item(),
        }
