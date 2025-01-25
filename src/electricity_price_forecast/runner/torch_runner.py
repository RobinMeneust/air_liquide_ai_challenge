import os
import pandas as pd
from tqdm import tqdm
import torch
from electricity_price_forecast.data.dataset import DatasetWithWindow
from electricity_price_forecast.model.torch_lightning_module import TorchLightningModule
from electricity_price_forecast.data.datamodule import Datamodule
from electricity_price_forecast.data.data_visualization import plot_predictions_vs_real
from electricity_price_forecast.data.data_processing import DataNormalizer, get_predict_data, get_splits
from abc import abstractmethod
from copy import copy
from electricity_price_forecast.runner.abstract_runner import AbstractRunner


class TorchRunner(AbstractRunner):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._val_ratio = 0.2
        self._features = ["dayofweek", "hourofday", "month", "price"]
        self._test_size = self._tested_horizons[-1] + self._window_size  # to have enough data to predict the last horizon (168 h)
        self._n_trials_params_search = 5
        self._max_synthetic_data_fetched = 20 # a 1:20 factor of data augmentation is already a lot


    @abstractmethod
    def get_best_params(self, datamodule, horizon, n_trials=50):
        pass

    @abstractmethod
    def train_model(self, datamodule, horizon, early_stopping=True, lr=0.001, n_epochs=50, hidden_dim=32, n_layers=1, device="cuda"):
        pass

    def eval_model(self, model, dataloader, device="cuda"):
        predictions = []
        ground_truth = []
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            for X, y in tqdm(dataloader):
                y_pred = model(X)
                predictions.append(y_pred)
                ground_truth.append(y)
        
        predictions = torch.cat(predictions).squeeze()
        ground_truth = torch.cat(ground_truth).squeeze()
        return TorchLightningModule.get_test_metrics(predictions, ground_truth)

    def predict(self, model, x, device="cuda"):
        model.eval()
        model.to(device)
        y_pred = None
        
        with torch.no_grad():
            y_pred = model(x)
        
        return y_pred   
        
    def run(self, use_synthetic_data=False, data_normalizer=None, params=None):
        all_results = {}
        true_data = self.load_true_data()
        synthetic_data = None
        if use_synthetic_data:
            data_path = os.path.join(self._data_root_path, "scenarios synthetiques", "prix")
            synthetic_data = self.load_synthetic_data(data_path, max_num_fetched=self._max_synthetic_data_fetched, data_normalizer=data_normalizer, initial_df=self.load_true_data()[-self._test_size:])
        true_data = data_normalizer.transform_df(true_data) if data_normalizer else true_data
        
        save_file_prefix = f"{self._model_name}{'_synthetic' if use_synthetic_data else ''}{'_normalized' if data_normalizer else ''}_{self._window_size}_w_{self._window_step}_s"
        
        for i, horizon in enumerate(self._tested_horizons):
            print(f"Step {i+1}/{len(self._tested_horizons)}...")
            
            true_dataset = DatasetWithWindow(true_data, self._window_size, self._window_step, horizon, self._features, "price")
            train_split, val_split, test_split = get_splits(true_dataset, self._test_size, self._val_ratio)
            
            save_plot_path_file = os.path.join(self._save_path_root, f"{save_file_prefix}_{horizon}_h.png")
              
            if use_synthetic_data:                
                synthetic_dataset = DatasetWithWindow(synthetic_data, self._window_size, self._window_step, horizon, self._features, "price") if use_synthetic_data else None

                # overwrite train and val but keep thee true data for the test
                train_split, val_split, _ = get_splits(synthetic_dataset, self._test_size, self._val_ratio)

            predict_dates, predict_y, predict_x = get_predict_data(test_split)
            
            datamodule = Datamodule(train_split, val_split, test_split, batch_size=32)
                        
            best_params = params if params else self.get_best_params(datamodule, horizon, n_trials=self._n_trials_params_search)
            model, train_results = self.train_model(datamodule, horizon, n_epochs=50, early_stopping=True, **best_params)
            
            test_dataloader = datamodule.test_dataloader()
            test_results = self.eval_model(model, test_dataloader)

            # add dim before (for batch)
            predictions = self.predict(model, predict_x.unsqueeze(0), device="cuda")
            
            # take last window ([-1, :])
            predictions = predictions.cpu().numpy()[-1, :]
            
            plot_predictions_vs_real(predict_dates, predict_y, predictions, save_path=save_plot_path_file, data_normalizer=data_normalizer)
            
            results = train_results
            for k, v in test_results.items():
                results[f"test_{k}"] = v
                
            all_results[horizon] = results
        
        save_csv_path_file = os.path.join(self._save_path_root, f"{save_file_prefix}.csv")
        
        results_df = []
        for horizon, res_dict in all_results.items():
            temp = copy(res_dict)
            temp["horizon"] = horizon
            results_df.append(temp)
        
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(save_csv_path_file, index=False)
        print("Done")
            
    def run_all(self, params=None):
        if type(params) is not list or len(params) != 4:
            params = [None]*4
            print("Params must be a list of 4 elements or a dict. So it will be reset to None and found using grid search instead.")
        
        self.run(use_synthetic_data=False, params=params[0])
        self.run(use_synthetic_data=True, params=params[1])
        
        # self.run(use_synthetic_data=False, data_normalizer=DataNormalizer(), params=params[2])
        # self.run(use_synthetic_data=True, data_normalizer=DataNormalizer(), params=params[3])
        
