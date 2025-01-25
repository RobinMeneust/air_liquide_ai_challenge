import optuna
import lightning.pytorch as pl
from electricity_price_forecast.model.lstm_model import LSTMModel
from electricity_price_forecast.model.torch_lightning_module import TorchLightningModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from electricity_price_forecast.model.lstm_model import MULTI_STEP
from electricity_price_forecast.runner.torch_runner import AbstractRunner
from electricity_price_forecast.model.transformers_model import TransformersModel

import os
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from electricity_price_forecast.eval import get_test_metrics
from electricity_price_forecast.data.data_visualization import plot_prices, plot_predictions_vs_real
from electricity_price_forecast.data.data_processing import preprocess_true_data, preprocess_synthetic_data


from typing import List


class LSTMRunner(AbstractRunner):
    def __init__(self):
        super().__init__('transformers')
        
    def get_x_y_point_by_point(df, x_keys: List[str], y_key: str):
        X = df[x_keys]
        y = df[y_key]
        return X, y
    
    def get_x_y_window(df, y_key: str, window_size: int, window_step: int, horizon: int):
        X, y  = [], []

        for i in range(0, len(df[y_key]) - window_size - horizon, window_step):
            X.append(df[y_key][i:i+window_size])
            y.append(df[y_key][i+window_size:i+window_size+horizon])
            
        return X, y
    
    def split_data(df, horizon):
        train_df = df[:-horizon]
        test_df = df[-horizon:]
        return train_df, test_df
        
    def get_best_params(X_train, y_train):
        xgb_model = xgb.XGBRegressor()
        reg_cv = GridSearchCV(xgb_model, {'max_depth': [1,5,10], 'n_estimators': [50, 200, 500], 'learning_rate': [0.001, 0.01, 0.1]}, verbose=1)
        reg_cv.fit(X_train, y_train)
        return reg_cv.best_params_
        
    def train_model(X_train, y_train, params=None):
        if params is None:
            xgb_model = xgb.XGBRegressor()
        else:
            xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train,y_train)
        
        return xgb_model
    
    def run(self, use_synthetic_data=False, params=None):
        train_data, test_data = split_data(df_price_preprocessed, 24)
        X_train, y_train = get_x_y_point_by_point(train_data, ["dayofweek", "hourofday", "dayofseries"], "price")
        X_test, y_test = get_x_y_point_by_point(test_data, ["dayofweek", "hourofday", "dayofseries"], "price")

        # params = get_best_params(X_train, y_train)
        params = {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200}

        xgb_model = train_model(X_train, y_train, params)
        
        predictions = xgb_model.predict(X_test)

        results = get_test_metrics(predictions, y_test)

        print("Results:", results)
        plot_predictions_vs_real(X_test["date"].values, y_test, predictions)



        all_results = {}
        true_data = self.load_true_data()
        
        synthetic_data = None
        if use_synthetic_data:
            data_path = os.path.join(self._data_root_path, "scenarios synthetiques")
            synthetic_data = self.load_synthetic_data(data_path, max_num_fetched=self._max_synthetic_data_fetched, initial_df=true_data)
        
        save_file_prefix = f"xgboost{'_synthetic' if use_synthetic_data else ''}{'_normalized' if data_normalizer else ''}_{self._window_size}_w_{self._window_step}_s"
        
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
            
            
            print(f"len predict_dates: {len(predict_dates)}, len predict_y: {len(predict_y)}, len predict_x: {len(predict_x)}")       
            
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
        

if __name__ == "__main__":
    params = [{'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200}] * 2

    # params = None
    LSTMRunner().run_all(params)