import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def fetch_yahoo_finance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance and calculate realized volatility.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'SPY' for S&P 500 ETF).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: DataFrame with daily OHLCV data and realized volatility.
    """
    # Fetch historical data from Yahoo Finance with auto_adjust=False
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    # Flatten multi-level column index if it exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns]

    # Print the first few rows to inspect the data
    print(data.head())

    # Handle column names - look for patterns with ticker suffix
    adj_close_cols = [col for col in data.columns if 'Adj Close' in col]
    close_cols = [col for col in data.columns if 'Close' in col and 'Adj Close' not in col]

    if adj_close_cols:
        data['Adj Close'] = data[adj_close_cols[0]]
    elif close_cols:
        data['Adj Close'] = data[close_cols[0]]
    else:
        raise KeyError("'Adj Close' or 'Close' column not found in the data.")

    # Also extract other OHLC columns if they exist
    for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
        cols = [col for col in data.columns if col_type in col]
        if cols:
            data[col_type] = data[cols[0]]
    
    # Calculate daily returns
    data['Daily_Return'] = data['Adj Close'].pct_change()
    
    # Calculate realized volatility (standard deviation of daily returns)
    data['Realized_Volatility'] = data['Daily_Return'].rolling(window=21).std() * (252 ** 0.5)  # Annualized volatility
    
    return data


def validate_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Validate and clean financial data.
    
    Args:
        data (pd.DataFrame): Raw financial data
        ticker (str): Ticker symbol for logging
    
    Returns:
        pd.DataFrame: Cleaned and validated data
    """
    print(f"Validating data for {ticker}...")
    
    # Check for missing values
    missing_pct = data.isnull().sum() / len(data) * 100
    if missing_pct.any():
        print(f"Missing data percentages:\n{missing_pct[missing_pct > 0]}")
    
    # Remove rows with missing adjusted close prices
    initial_len = len(data)
    data = data.dropna(subset=['Adj Close'])
    if len(data) < initial_len:
        print(f"Removed {initial_len - len(data)} rows with missing Adj Close prices")
    
    # Check for zero or negative prices (data quality issue)
    invalid_prices = (data['Adj Close'] <= 0).sum()
    if invalid_prices > 0:
        print(f"Warning: Found {invalid_prices} zero or negative prices")
        data = data[data['Adj Close'] > 0]
    
    # Detect and handle extreme returns (potential data errors)
    returns = data['Daily_Return'].dropna()
    extreme_threshold = 0.5  # 50% daily return threshold
    extreme_returns = (abs(returns) > extreme_threshold).sum()
    if extreme_returns > 0:
        print(f"Warning: Found {extreme_returns} extreme returns (>{extreme_threshold*100}%)")
    
    print(f"Data validation complete. Final dataset: {len(data)} observations")
    return data


def calculate_volatility_features(data: pd.DataFrame, windows: list = [5, 10, 21, 63]) -> pd.DataFrame:
    """
    Calculate various volatility measures and features.
    
    Args:
        data (pd.DataFrame): DataFrame with price and return data
        windows (list): List of rolling window sizes for volatility calculation
    
    Returns:
        pd.DataFrame: DataFrame with additional volatility features
    """
    data = data.copy()
    
    # Calculate returns if not already present
    if 'Daily_Return' not in data.columns:
        data['Daily_Return'] = data['Adj Close'].pct_change()
    
    # Rolling volatilities (annualized)
    for window in windows:
        col_name = f'RV_{window}d'
        data[col_name] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
    
    # Exponentially weighted moving average volatility
    data['EWMA_Vol'] = data['Daily_Return'].ewm(span=21).std() * np.sqrt(252)
    
    # Garman-Klass volatility (if OHLC data available)
    if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
        # Garman-Klass estimator
        data['GK_Vol'] = np.sqrt(
            0.5 * (np.log(data['High'] / data['Low'])) ** 2 - 
            (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open'])) ** 2
        ) * np.sqrt(252)
    
    # Parkinson volatility (High-Low estimator)
    if all(col in data.columns for col in ['High', 'Low']):
        data['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(data['High'] / data['Low'])) ** 2
        ) * np.sqrt(252)
    
    # Volatility of volatility (second moment)
    data['Vol_of_Vol'] = data['RV_21d'].rolling(window=21).std()
    
    return data


def prepare_model_data(data: pd.DataFrame, target_col: str = 'RV_21d', 
                      lookback: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for volatility modeling with lagged features.
    
    Args:
        data (pd.DataFrame): DataFrame with volatility features
        target_col (str): Target variable column name
        lookback (int): Number of lagged periods to include
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable
    """
    # Create lagged features for HAR-type models
    features_df = pd.DataFrame(index=data.index)
    
    # Add lagged volatility (HAR structure: daily, weekly, monthly)
    if target_col in data.columns:
        features_df[f'{target_col}_lag1'] = data[target_col].shift(1)  # Daily
        features_df[f'{target_col}_lag5'] = data[target_col].rolling(5).mean().shift(1)  # Weekly
        features_df[f'{target_col}_lag22'] = data[target_col].rolling(22).mean().shift(1)  # Monthly
    
    # Add other volatility measures as features
    vol_cols = [col for col in data.columns if 'Vol' in col and col != target_col]
    for col in vol_cols:
        features_df[f'{col}_lag1'] = data[col].shift(1)
    
    # Add return-based features
    if 'Daily_Return' in data.columns:
        features_df['Return_lag1'] = data['Daily_Return'].shift(1)
        features_df['Abs_Return_lag1'] = abs(data['Daily_Return']).shift(1)
        features_df['Return_squared_lag1'] = (data['Daily_Return'] ** 2).shift(1)
    
    # Remove rows with NaN values
    target = data[target_col]
    combined_data = pd.concat([features_df, target], axis=1).dropna()
    
    X = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1]
    
    print(f"Model data prepared: {len(X)} observations, {len(X.columns)} features")
    return X, y


def train_test_split_timeseries(X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time series aware train-test split.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
    
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")
    print(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
    
    return X_train, X_test, y_train, y_test
