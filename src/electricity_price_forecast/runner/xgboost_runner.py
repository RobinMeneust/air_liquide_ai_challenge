from copy import copy
import os
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from electricity_price_forecast.eval import get_test_metrics
from electricity_price_forecast.data.data_visualization import plot_predictions_vs_real
from electricity_price_forecast.data.data_processing import preprocess_true_data, preprocess_synthetic_data
from electricity_price_forecast.runner.abstract_runner import AbstractRunner
from typing import List


class XGBoostRunner(AbstractRunner):
    def __init__(self):
        super().__init__('xgboost')
        self._n_print_before = self._window_size
        self._features_by_point = ["dayofweek", "hourofday", "dayofseries"]
        
    def get_x_y_point_by_point(self, df, x_keys: List[str], y_key: str):
        X = df[x_keys]
        y = df[y_key]
        return X, y
    
    def get_x_y_window(self, df, y_key: str, window_size: int, window_step: int, horizon: int):
        X, y  = [], []

        for i in range(0, len(df[y_key]) - window_size - horizon, window_step):
            X.append(df[y_key][i:i+window_size])
            y.append(df[y_key][i+window_size:i+window_size+horizon])
            
        return X, y
    
    def split_data(self, df, test_size):
        train_df = df[:-test_size]
        test_df = df[-test_size:]
        return train_df, test_df
        
    def get_best_params(self, X_train, y_train):
        xgb_model = xgb.XGBRegressor()
        reg_cv = GridSearchCV(xgb_model, {'max_depth': [1,5,10], 'n_estimators': [50, 200, 500], 'learning_rate': [0.001, 0.01, 0.1]}, verbose=1)
        reg_cv.fit(X_train, y_train)
        return reg_cv.best_params_
        
    def train_model(self, X_train, y_train, params=None):
        if params is None:
            xgb_model = xgb.XGBRegressor()
        else:
            xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train,y_train)
        
        return xgb_model
    
    def run(self, use_synthetic_data=False, use_window=False, params=None):
        all_results = {}
        true_data = self.load_true_data()
        
        synthetic_data = None
        if use_synthetic_data:
            data_path = os.path.join(self._data_root_path, "scenarios synthetiques", "prix")
            # self._window_size + max(self._tested_horizons) + 1 -> to avoid using the same data for training and testing
            synthetic_data = self.load_synthetic_data(data_path, max_num_fetched=self._max_synthetic_data_fetched, initial_df=true_data[:self._window_size + max(self._tested_horizons) + 1])
        
        if use_window:
            save_file_prefix = f"xgboost{'_synthetic' if use_synthetic_data else ''}_window_mode_{self._window_size}_w_{self._window_step}_s"
        else:
            save_file_prefix = f"xgboost{'_synthetic' if use_synthetic_data else ''}_no_window"
        
        for i, horizon in enumerate(self._tested_horizons):
            print(f"Step {i+1}/{len(self._tested_horizons)}...")
            
            if use_window:
                train_data, test_data = self.split_data(true_data, self._window_size + horizon + 1)
                X_train, y_train = self.get_x_y_window(train_data, "price", self._window_size, self._window_step, horizon)
                X_test, y_test = self.get_x_y_window(test_data, "price", self._window_size, self._window_step, horizon)
            else:
                train_data, test_data = self.split_data(true_data, horizon+self._n_print_before)
                X_train, y_train = self.get_x_y_point_by_point(train_data, self._features_by_point, "price")
                X_test, y_test = self.get_x_y_point_by_point(test_data, self._features_by_point, "price")
                            
            save_plot_path_file = os.path.join(self._save_path_root, f"{save_file_prefix}_{horizon}_h.png")
              
            if use_synthetic_data:
                # overwrite train and val but keep the true data for the test
                if use_window:
                    X_train, y_train = self.get_x_y_window(synthetic_data, "price", self._window_size, self._window_step, horizon)
                else:
                    X_train, y_train = self.get_x_y_point_by_point(synthetic_data, self._features_by_point, "price")

            best_params = params if params else self.get_best_params(X_train, y_train)
            
            xgb_model = self.train_model(X_train, y_train, best_params)
            
            if use_window:
                start, end = X_test[0].index[0], y_test[0].index[-1]
                X_test_hourofseries = list(range(start, end+1))
                X_test_dates = [true_data["date"].iloc[i] for i in X_test_hourofseries]
                y_test_true = [true_data["price"].iloc[i] for i in X_test_hourofseries]
                
                predictions = xgb_model.predict(X_test)
                
                results = get_test_metrics(predictions[0], y_test[0].values)
                plot_predictions_vs_real(X_test_dates, y_test_true, predictions[0], save_path=save_plot_path_file)
            else:
                predictions = xgb_model.predict(X_test[self._n_print_before:])
                results = get_test_metrics(predictions, y_test.values[self._n_print_before:])
                

                plot_predictions_vs_real(true_data["date"][X_test.index].values, y_test.values, predictions, save_path=save_plot_path_file)
            
            all_results[horizon] = {f"test_{k}": v for k, v in results.items()}
        
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
        
        self.run(use_synthetic_data=False, use_window=False, params=params[0])
        self.run(use_synthetic_data=True, use_window=False, params=params[1])
        
        # self.run(use_synthetic_data=False, use_window=True, params=params[2])
        # self.run(use_synthetic_data=True, use_window=True, params=params[3])
        

if __name__ == "__main__":
    params = [{'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200}] * 4
    XGBoostRunner().run_all(params)