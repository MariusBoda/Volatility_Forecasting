import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


class VolatilityEvaluator:
    """
    Comprehensive evaluation framework for volatility forecasting models.
    
    Implements industry-standard metrics and statistical tests for model comparison.
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_forecast_errors(self, actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
        """
        Calculate basic forecast error metrics.
        
        Args:
            actual (pd.Series): Actual realized volatility
            forecast (pd.Series): Forecasted volatility
            
        Returns:
            Dict[str, float]: Dictionary of error metrics
        """
        # Align series and remove NaN values
        aligned_data = pd.concat([actual, forecast], axis=1).dropna()
        if len(aligned_data) == 0:
            return {}
            
        y_true = aligned_data.iloc[:, 0]
        y_pred = aligned_data.iloc[:, 1]
        
        errors = y_pred - y_true
        
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs(errors / y_true)) * 100,
            'Mean_Error': np.mean(errors),
            'Std_Error': np.std(errors),
            'Min_Error': np.min(errors),
            'Max_Error': np.max(errors)
        }
    
    def qlike_loss(self, actual: pd.Series, forecast: pd.Series) -> float:
        """
        Calculate QLIKE (Quasi-Likelihood) loss function.
        
        QLIKE is the standard loss function for volatility forecasting:
        QLIKE = log(forecast) + actual/forecast
        
        Args:
            actual (pd.Series): Actual realized volatility (squared returns or variance)
            forecast (pd.Series): Forecasted volatility (variance)
            
        Returns:
            float: QLIKE loss (lower is better)
        """
        # Align series and remove NaN values
        aligned_data = pd.concat([actual, forecast], axis=1).dropna()
        if len(aligned_data) == 0:
            return np.nan
            
        y_true = aligned_data.iloc[:, 0]
        y_pred = aligned_data.iloc[:, 1]
        
        # Ensure positive values (add small epsilon to avoid log(0))
        y_pred = np.maximum(y_pred, 1e-8)
        y_true = np.maximum(y_true, 1e-8)
        
        qlike = np.mean(np.log(y_pred) + y_true / y_pred)
        return qlike
    
    def r2_log_loss(self, actual: pd.Series, forecast: pd.Series) -> float:
        """
        Calculate R² log loss (Patton, 2011).
        
        Args:
            actual (pd.Series): Actual realized volatility
            forecast (pd.Series): Forecasted volatility
            
        Returns:
            float: R² log loss (higher is better, max = 1)
        """
        # Align series and remove NaN values
        aligned_data = pd.concat([actual, forecast], axis=1).dropna()
        if len(aligned_data) == 0:
            return np.nan
            
        y_true = aligned_data.iloc[:, 0]
        y_pred = aligned_data.iloc[:, 1]
        
        # Ensure positive values
        y_pred = np.maximum(y_pred, 1e-8)
        y_true = np.maximum(y_true, 1e-8)
        
        # Calculate R² log
        numerator = np.sum((np.log(y_true) - np.log(y_pred))**2)
        denominator = np.sum((np.log(y_true) - np.mean(np.log(y_true)))**2)
        
        r2_log = 1 - (numerator / denominator)
        return r2_log
    
    def hit_rate(self, actual: pd.Series, forecast: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate hit rate (directional accuracy).
        
        Args:
            actual (pd.Series): Actual realized volatility changes
            forecast (pd.Series): Forecasted volatility changes
            threshold (float): Threshold for defining "hit"
            
        Returns:
            float: Hit rate (proportion of correct directional predictions)
        """
        # Calculate changes
        actual_changes = actual.diff().dropna()
        forecast_changes = forecast.diff().dropna()
        
        # Align series
        aligned_data = pd.concat([actual_changes, forecast_changes], axis=1).dropna()
        if len(aligned_data) == 0:
            return np.nan
            
        y_true_change = aligned_data.iloc[:, 0]
        y_pred_change = aligned_data.iloc[:, 1]
        
        # Calculate hit rate
        correct_direction = ((y_true_change > threshold) & (y_pred_change > threshold)) | \
                          ((y_true_change <= threshold) & (y_pred_change <= threshold))
        
        return np.mean(correct_direction)
    
    def diebold_mariano_test(self, actual: pd.Series, forecast1: pd.Series, 
                           forecast2: pd.Series, loss_function: str = 'mse') -> Dict[str, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        
        Args:
            actual (pd.Series): Actual values
            forecast1 (pd.Series): First model forecasts
            forecast2 (pd.Series): Second model forecasts
            loss_function (str): Loss function ('mse', 'mae', or 'qlike')
            
        Returns:
            Dict[str, float]: Test statistic and p-value
        """
        # Align all series
        aligned_data = pd.concat([actual, forecast1, forecast2], axis=1).dropna()
        if len(aligned_data) < 10:
            return {'dm_stat': np.nan, 'p_value': np.nan}
            
        y_true = aligned_data.iloc[:, 0]
        y_pred1 = aligned_data.iloc[:, 1]
        y_pred2 = aligned_data.iloc[:, 2]
        
        # Calculate loss differences
        if loss_function == 'mse':
            loss1 = (y_true - y_pred1) ** 2
            loss2 = (y_true - y_pred2) ** 2
        elif loss_function == 'mae':
            loss1 = np.abs(y_true - y_pred1)
            loss2 = np.abs(y_true - y_pred2)
        elif loss_function == 'qlike':
            y_pred1 = np.maximum(y_pred1, 1e-8)
            y_pred2 = np.maximum(y_pred2, 1e-8)
            y_true = np.maximum(y_true, 1e-8)
            loss1 = np.log(y_pred1) + y_true / y_pred1
            loss2 = np.log(y_pred2) + y_true / y_pred2
        else:
            raise ValueError("loss_function must be 'mse', 'mae', or 'qlike'")
        
        loss_diff = loss1 - loss2
        
        # Calculate DM statistic
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)
        n = len(loss_diff)
        
        dm_stat = mean_diff / np.sqrt(var_diff / n)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return {
            'dm_stat': dm_stat,
            'p_value': p_value,
            'mean_loss_diff': mean_diff
        }
    
    def model_confidence_set(self, actual: pd.Series, forecasts_dict: Dict[str, pd.Series], 
                           alpha: float = 0.1) -> Dict[str, bool]:
        """
        Model Confidence Set (Hansen, Lunde, Nason, 2011).
        
        Args:
            actual (pd.Series): Actual values
            forecasts_dict (Dict[str, pd.Series]): Dictionary of model forecasts
            alpha (float): Significance level
            
        Returns:
            Dict[str, bool]: Whether each model is in the confidence set
        """
        # This is a simplified implementation
        # Full MCS requires bootstrap procedures
        
        model_names = list(forecasts_dict.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {name: True for name in model_names}
        
        # Calculate pairwise DM tests
        dm_results = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    dm_result = self.diebold_mariano_test(
                        actual, forecasts_dict[model1], forecasts_dict[model2]
                    )
                    dm_results[(model1, model2)] = dm_result['p_value']
        
        # Simple rule: include models that are not significantly worse than any other
        confidence_set = {}
        for model in model_names:
            is_in_set = True
            for (m1, m2), p_val in dm_results.items():
                if model == m1 and p_val < alpha:
                    # Model is significantly worse than m2
                    is_in_set = False
                    break
                elif model == m2 and p_val < alpha:
                    # Model is significantly worse than m1
                    is_in_set = False
                    break
            confidence_set[model] = is_in_set
        
        return confidence_set
    
    def comprehensive_evaluation(self, actual: pd.Series, 
                               forecasts_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Comprehensive evaluation of multiple models.
        
        Args:
            actual (pd.Series): Actual realized volatility
            forecasts_dict (Dict[str, pd.Series]): Dictionary of model forecasts
            
        Returns:
            pd.DataFrame: Comprehensive evaluation results
        """
        results = []
        
        for model_name, forecast in forecasts_dict.items():
            # Basic error metrics
            error_metrics = self.calculate_forecast_errors(actual, forecast)
            
            # Volatility-specific metrics
            qlike = self.qlike_loss(actual**2, forecast**2)  # Use squared values for QLIKE
            r2_log = self.r2_log_loss(actual, forecast)
            hit_rate = self.hit_rate(actual, forecast)
            
            # Combine all metrics
            model_results = {
                'Model': model_name,
                'RMSE': error_metrics.get('RMSE', np.nan),
                'MAE': error_metrics.get('MAE', np.nan),
                'MAPE': error_metrics.get('MAPE', np.nan),
                'QLIKE': qlike,
                'R2_Log': r2_log,
                'Hit_Rate': hit_rate,
                'Mean_Error': error_metrics.get('Mean_Error', np.nan),
                'Std_Error': error_metrics.get('Std_Error', np.nan)
            }
            
            results.append(model_results)
        
        results_df = pd.DataFrame(results)
        results_df.set_index('Model', inplace=True)
        
        # Add rankings
        for metric in ['RMSE', 'MAE', 'MAPE', 'QLIKE']:
            if metric in results_df.columns:
                results_df[f'{metric}_Rank'] = results_df[metric].rank()
        
        for metric in ['R2_Log', 'Hit_Rate']:
            if metric in results_df.columns:
                results_df[f'{metric}_Rank'] = results_df[metric].rank(ascending=False)
        
        return results_df
    
    def plot_forecast_comparison(self, actual: pd.Series, 
                               forecasts_dict: Dict[str, pd.Series],
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot actual vs forecasted volatility for multiple models.
        
        Args:
            actual (pd.Series): Actual realized volatility
            forecasts_dict (Dict[str, pd.Series]): Dictionary of model forecasts
            start_date (str, optional): Start date for plotting
            end_date (str, optional): End date for plotting
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Filter data by date range if specified
        plot_actual = actual.copy()
        plot_forecasts = forecasts_dict.copy()
        
        if start_date:
            plot_actual = plot_actual[plot_actual.index >= start_date]
            plot_forecasts = {k: v[v.index >= start_date] for k, v in plot_forecasts.items()}
        
        if end_date:
            plot_actual = plot_actual[plot_actual.index <= end_date]
            plot_forecasts = {k: v[v.index <= end_date] for k, v in plot_forecasts.items()}
        
        # Plot time series
        axes[0].plot(plot_actual.index, plot_actual.values, 
                    label='Actual', color='black', linewidth=2)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(plot_forecasts)))
        for i, (model_name, forecast) in enumerate(plot_forecasts.items()):
            axes[0].plot(forecast.index, forecast.values, 
                        label=model_name, color=colors[i], alpha=0.7)
        
        axes[0].set_ylabel('Volatility')
        axes[0].set_title('Volatility Forecasts vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot forecast errors
        for i, (model_name, forecast) in enumerate(plot_forecasts.items()):
            aligned_data = pd.concat([plot_actual, forecast], axis=1).dropna()
            if len(aligned_data) > 0:
                errors = aligned_data.iloc[:, 1] - aligned_data.iloc[:, 0]
                axes[1].plot(aligned_data.index, errors, 
                           label=f'{model_name} Error', color=colors[i], alpha=0.7)
        
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Forecast Error')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Forecast Errors')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, actual: pd.Series, 
                              forecasts_dict: Dict[str, pd.Series],
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot distribution of forecast errors.
        
        Args:
            actual (pd.Series): Actual realized volatility
            forecasts_dict (Dict[str, pd.Series]): Dictionary of model forecasts
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        errors_data = []
        model_names = []
        
        for model_name, forecast in forecasts_dict.items():
            aligned_data = pd.concat([actual, forecast], axis=1).dropna()
            if len(aligned_data) > 0:
                errors = aligned_data.iloc[:, 1] - aligned_data.iloc[:, 0]
                errors_data.extend(errors.values)
                model_names.extend([model_name] * len(errors))
        
        if errors_data:
            error_df = pd.DataFrame({
                'Error': errors_data,
                'Model': model_names
            })
            
            # Box plot
            sns.boxplot(data=error_df, x='Model', y='Error', ax=axes[0])
            axes[0].set_title('Forecast Error Distribution')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Histogram
            for model_name in forecasts_dict.keys():
                model_errors = error_df[error_df['Model'] == model_name]['Error']
                axes[1].hist(model_errors, alpha=0.6, label=model_name, bins=30)
            
            axes[1].set_xlabel('Forecast Error')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Error Histogram')
            axes[1].legend()
        
        plt.tight_layout()
        return fig
