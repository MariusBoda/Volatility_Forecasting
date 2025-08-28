import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, Normal, StudentsT
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    This implementation uses the ARCH library for professional-grade GARCH modeling
    with proper maximum likelihood estimation and model diagnostics.
    """
    
    def __init__(self, p: int = 1, q: int = 1, distribution: str = 'normal'):
        """
        Initialize GARCH model.
        
        Args:
            p (int): Order of GARCH terms (default: 1)
            q (int): Order of ARCH terms (default: 1) 
            distribution (str): Error distribution ('normal' or 't')
        """
        self.p = p
        self.q = q
        self.distribution = distribution
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def fit(self, returns: pd.Series, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit GARCH model to return series.
        
        Args:
            returns (pd.Series): Daily returns (not annualized)
            verbose (bool): Print fitting information
            
        Returns:
            Dict[str, Any]: Model fitting results and diagnostics
        """
        # Remove any NaN values
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 100:
            raise ValueError("Need at least 100 observations for reliable GARCH estimation")
        
        # Convert to percentage returns for numerical stability
        returns_pct = returns_clean * 100
        
        # Create GARCH model
        self.model = arch_model(
            returns_pct,
            vol='GARCH',
            p=self.p,
            q=self.q,
            rescale=False
        )
        
        # Fit the model
        try:
            self.fitted_model = self.model.fit(disp='off' if not verbose else 'final')
            self.is_fitted = True
            
            if verbose:
                print(f"GARCH({self.p},{self.q}) model fitted successfully")
                print(f"Log-likelihood: {self.fitted_model.loglikelihood:.2f}")
                print(f"AIC: {self.fitted_model.aic:.2f}")
                print(f"BIC: {self.fitted_model.bic:.2f}")
                
        except Exception as e:
            print(f"GARCH model fitting failed: {str(e)}")
            self.is_fitted = False
            return {'success': False, 'error': str(e)}
        
        # Calculate model diagnostics
        diagnostics = self._calculate_diagnostics(returns_pct)
        
        return {
            'success': True,
            'loglikelihood': self.fitted_model.loglikelihood,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'parameters': dict(self.fitted_model.params),
            'diagnostics': diagnostics
        }
    
    def _calculate_diagnostics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate model diagnostics."""
        if not self.is_fitted:
            return {}
            
        # Get standardized residuals
        std_resid = self.fitted_model.std_resid
        
        # Ljung-Box test on standardized residuals (should be no autocorrelation)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_stat = acorr_ljungbox(std_resid, lags=10, return_df=True)
        
        # Ljung-Box test on squared standardized residuals (should be no ARCH effects)
        lb_stat_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        
        return {
            'ljung_box_pvalue': lb_stat['lb_pvalue'].iloc[-1],
            'ljung_box_squared_pvalue': lb_stat_sq['lb_pvalue'].iloc[-1],
            'mean_abs_std_resid': np.mean(np.abs(std_resid)),
            'std_resid_skewness': std_resid.skew(),
            'std_resid_kurtosis': std_resid.kurtosis()
        }
    
    def forecast(self, horizon: int = 1, method: str = 'simulation', 
                simulations: int = 1000) -> Dict[str, Any]:
        """
        Generate volatility forecasts.
        
        Args:
            horizon (int): Forecast horizon in days
            method (str): Forecasting method ('analytical' or 'simulation')
            simulations (int): Number of simulations for simulation method
            
        Returns:
            Dict[str, Any]: Forecast results including mean, variance, and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if method == 'analytical':
            # Analytical forecast (faster, exact for GARCH)
            forecast_result = self.fitted_model.forecast(horizon=horizon, method='analytic')
        else:
            # Simulation-based forecast (more flexible, includes full distribution)
            forecast_result = self.fitted_model.forecast(
                horizon=horizon, 
                method='simulation',
                simulations=simulations
            )
        
        # Extract forecasts (convert back from percentage scale)
        variance_forecast = forecast_result.variance.iloc[-1] / 10000  # Convert from %^2 to decimal^2
        volatility_forecast = np.sqrt(variance_forecast * 252)  # Annualized volatility
        
        # Calculate confidence intervals for simulation method
        if method == 'simulation':
            # Get simulation paths
            sim_data = forecast_result.simulations
            vol_sims = np.sqrt(sim_data.variance.values / 10000 * 252)  # Annualized
            
            confidence_intervals = {}
            for conf_level in [0.05, 0.25, 0.75, 0.95]:
                confidence_intervals[f'q_{conf_level}'] = np.quantile(vol_sims, conf_level, axis=0)
        else:
            confidence_intervals = {}
        
        return {
            'horizon': horizon,
            'method': method,
            'volatility_forecast': volatility_forecast,
            'variance_forecast': variance_forecast,
            'confidence_intervals': confidence_intervals
        }
    
    def rolling_forecast(self, returns: pd.Series, window_size: int = 252, 
                        refit_frequency: int = 21) -> pd.DataFrame:
        """
        Perform rolling out-of-sample forecasts.
        
        Args:
            returns (pd.Series): Full return series
            window_size (int): Rolling window size for estimation
            refit_frequency (int): How often to refit the model (in days)
            
        Returns:
            pd.DataFrame: DataFrame with dates, actual volatility, and forecasts
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < window_size + 10:
            raise ValueError(f"Need at least {window_size + 10} observations for rolling forecast")
        
        forecasts = []
        actual_vol = []
        dates = []
        
        # Start forecasting after initial window
        start_idx = window_size
        refit_counter = 0
        
        for i in range(start_idx, len(returns_clean)):
            current_date = returns_clean.index[i]
            
            # Refit model if needed
            if refit_counter % refit_frequency == 0:
                # Use rolling window of data
                train_data = returns_clean.iloc[i-window_size:i]
                
                # Create and fit new model instance
                temp_model = GARCHModel(p=self.p, q=self.q, distribution=self.distribution)
                fit_result = temp_model.fit(train_data, verbose=False)
                
                if not fit_result['success']:
                    print(f"Failed to fit model at {current_date}")
                    continue
                    
                current_fitted_model = temp_model
            
            # Generate 1-day ahead forecast
            try:
                forecast_result = current_fitted_model.forecast(horizon=1, method='analytical')
                vol_forecast = forecast_result['volatility_forecast'][0]
                
                # Calculate actual realized volatility (21-day rolling)
                if i + 21 < len(returns_clean):
                    actual_vol_val = returns_clean.iloc[i:i+21].std() * np.sqrt(252)
                else:
                    actual_vol_val = np.nan
                
                forecasts.append(vol_forecast)
                actual_vol.append(actual_vol_val)
                dates.append(current_date)
                
            except Exception as e:
                print(f"Forecast failed at {current_date}: {str(e)}")
                continue
            
            refit_counter += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual_Vol': actual_vol,
            'GARCH_Forecast': forecasts
        })
        results_df.set_index('Date', inplace=True)
        
        print(f"Rolling forecast completed: {len(results_df)} forecasts generated")
        return results_df
    
    def get_model_summary(self) -> str:
        """Get formatted model summary."""
        if not self.is_fitted:
            return "Model not fitted"
        
        return str(self.fitted_model.summary())
    
    def get_parameters(self) -> Dict[str, float]:
        """Get fitted model parameters."""
        if not self.is_fitted:
            return {}
        
        return dict(self.fitted_model.params)
