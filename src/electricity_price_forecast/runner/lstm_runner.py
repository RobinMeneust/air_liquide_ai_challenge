import optuna
import lightning.pytorch as pl
from electricity_price_forecast.model.lstm_model import LSTMModel
from electricity_price_forecast.model.torch_lightning_module import TorchLightningModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from electricity_price_forecast.model.lstm_model import MULTI_STEP
from electricity_price_forecast.runner.torch_runner import TorchRunner


class LSTMRunner(TorchRunner):
    """Runner for the LSTM model
    
    Attributes:
        model_name (str): Model name
    """
    def __init__(self):
        """Initialize the LSTMRunner"""
        super().__init__('lstm')
        
    def get_best_params(self, datamodule, horizon, n_trials=50):
        """Get the best parameters for the LSTM model
        
        Args:
            datamodule (Datamodule): Datamodule
            horizon (int): Horizon
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
            hidden_dim=trial.suggest_int("hidden_dim", 16, 512),
            n_layers=trial.suggest_int("n_layers", 1, 10),
        )[-1]["val_loss"], n_trials=n_trials)
        return study.best_params
    
    def train_model(self, datamodule, horizon, early_stopping=True, lr=0.001, n_epochs=50, hidden_dim=32, n_layers=1, device="cuda"):
        """Train the LSTM model
        
        Args:
            datamodule (Datamodule): Datamodule
            horizon (int): Horizon (number of points to forecast)
            early_stopping (bool): Whether to use early stopping
            lr (float): Learning rate
            n_epochs (int): Number of epochs
            hidden_dim (int): Hidden dimension
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
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "output_dim": horizon,
            "multi_step": MULTI_STEP.ALL_AT_ONCE
        }
        
        model = TorchLightningModule(LSTMModel, model_params, lr=lr, device=device)
        
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
        {'lr': 0.07425688743431168, 'hidden_dim': 274, 'n_layers': 2}, # without synthetic data & without normalization
        {'lr': 0.07425688743431168, 'hidden_dim': 274, 'n_layers': 2}, # with synthetic data & without normalization
        {'lr': 0.04576315047600192, 'hidden_dim': 30, 'n_layers': 8}, # without synthetic data & with normalization
        {'lr': 0.04576315047600192, 'hidden_dim': 30, 'n_layers': 8} # with synthetic data & with normalization
    ]

    # params = None
    LSTMRunner().run_all(params)