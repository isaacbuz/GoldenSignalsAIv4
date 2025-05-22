import logging
from typing import List, Dict, Any, Optional
import pandas as pd

class CorrelationAgent:
    """
    Agent to compute correlation matrices and rolling correlations between asset time series.
    Supports Pearson, Spearman, Kendall, and rolling window correlations.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_correlation(self, data: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Compute correlation matrix for the given data using the specified method.
        Args:
            data (pd.DataFrame): DataFrame where columns are asset names and rows are time series values.
            method (str): 'pearson', 'spearman', or 'kendall'.
        Returns:
            Dict[str, Any]: Correlation matrix as dict, or error message.
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty or data.shape[1] < 2:
                self.logger.error("Invalid input data for correlation computation.")
                return {"error": "Input must be a DataFrame with at least two columns."}
            corr = data.corr(method=method)
            self.logger.info(f"Computed {method} correlation matrix.")
            return {"correlation_matrix": corr.to_dict()}
        except Exception as e:
            self.logger.exception("Correlation computation failed.")
            return {"error": str(e)}

    def compute_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int = 30, method: str = 'pearson') -> Dict[str, Any]:
        """
        Compute rolling correlation between two series.
        Args:
            series1, series2 (pd.Series): Time series data.
            window (int): Rolling window size.
            method (str): Only 'pearson' is supported for rolling.
        Returns:
            Dict[str, Any]: Rolling correlation as list, or error message.
        """
        try:
            if not (isinstance(series1, pd.Series) and isinstance(series2, pd.Series)):
                self.logger.error("Invalid input series for rolling correlation.")
                return {"error": "Inputs must be pandas Series objects."}
            if len(series1) != len(series2):
                self.logger.error("Series length mismatch for rolling correlation.")
                return {"error": "Series must be of equal length."}
            if len(series1) < window:
                self.logger.error("Series too short for rolling window.")
                return {"error": "Series length must be at least as large as window size."}
            if method != 'pearson':
                self.logger.error("Only Pearson correlation is supported for rolling correlation in pandas.")
                return {"error": "Only Pearson correlation is supported for rolling correlation."}
            rolling_corr = series1.rolling(window=window).corr(series2)
            self.logger.info(f"Computed rolling Pearson correlation with window={window}.")
            return {"rolling_correlation": rolling_corr.tolist()}
        except Exception as e:
            self.logger.exception("Rolling correlation computation failed.")
            return {"error": str(e)}
