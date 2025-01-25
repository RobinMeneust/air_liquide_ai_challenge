import optuna
import lightning.pytorch as pl
from electricity_price_forecast.model.torch_lightning_module import TorchLightningModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from electricity_price_forecast.runner.torch_runner import TorchRunner
from electricity_price_forecast.model.transformers_model import TransformersModel


class LSTMRunner(TorchRunner):
    """Runner for the LSTM model
    
    Attributes:
        model_name (str): Model name
    """
    def __init__(self):
        """Initialize the LSTMRunner"""
        super().__init__('transformers')
        
    def get_best_params(self, datamodule, horizon, n_trials=50):
        """Get the best parameters for the LSTM model
        
        Args:
            datamodule (Datamodule): Datamodule
            horizon (int): Horizon (number of points to forecast)
            n_trials (int): Number of trials
        
        Returns:
            dict: Best parameters
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.train_model(
            datamodule,
            horizon,
            lr=trial.suggest_float("lr", 1e-5, 1e-1),
            n_epochs=30,
            n_layers=trial.suggest_int("n_layers", 1, 10),
        )[-1]["val_loss"], n_trials=n_trials)
        return study.best_params
    
    def train_model(self, datamodule, horizon, early_stopping=True, num_heads=1, lr=0.001, n_epochs=50, n_layers=1, device="cuda"):
        """Train the LSTM model
        
        Args:
            datamodule (Datamodule): Datamodule
            horizon (int): Horizon (number of points to forecast)
            early_stopping (bool): Whether to use early stopping
            lr (float): Learning rate
            n_epochs (int): Number of epochs
            n_layers (int): Number of layers
            device (str): Device
        
        Returns:
            Tuple[nn.Module, dict]: Trained model and metrics
        """
        train_dataloader = datamodule.train_dataloader()
        X_batch, _ = next(iter(train_dataloader))
        input_dim = X_batch.shape[-1]
        
        model_params = {
            "input_dim": input_dim,
            "n_layers": n_layers,
            "output_dim": horizon,
            "num_heads": num_heads,
        }
        
        model = TorchLightningModule(TransformersModel, model_params, lr=lr, device=device)
        
        if early_stopping:
            callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False)]
        else:
            callbacks = []
            
        if device == "cuda":
            trainer = pl.Trainer(max_epochs=n_epochs, devices=-1, accelerator="cuda", callbacks=callbacks, enable_checkpointing=False, logger=False)
        else:
            trainer = pl.Trainer(max_epochs=n_epochs, callbacks=callbacks, enable_checkpointing=False, logger=False)
        
        trainer.fit(model, datamodule)
                
        return model, {"val_loss": trainer.callback_metrics["val_loss"].item(), "train_loss": trainer.callback_metrics["train_loss"].item()}


if __name__ == "__main__":
    params = [
        {'lr': 0.08914550925407072, 'n_layers': 4}, # without synthetic data & without normalization
        {'lr': 0.08914550925407072, 'n_layers': 4}, # with synthetic data & without normalization
        {'lr': 0.04016963835331307, 'n_layers': 6}, # without synthetic data & with normalization
        {'lr': 0.04016963835331307, 'n_layers': 6} # with synthetic data & with normalization
    ]

    # params = None
    LSTMRunner().run_all(params)