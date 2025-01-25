from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def get_test_metrics(predictions, ground_truth):
    mae = mean_absolute_error(ground_truth, predictions).item()
    mse = mean_squared_error(ground_truth, predictions).item()
    rmse = math.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}