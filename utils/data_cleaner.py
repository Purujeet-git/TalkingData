import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from utils.logger import AppLogger

logger = AppLogger.get_logger("DataCleaner")

class DataCleaner:
    """
    Advanced Data Cleaning and Profiling Engine for TalkingData.
    
    This class handles the automatic sanitization of uploaded datasets,
    including dropping duplicates, removing completely empty columns/rows,
    and generating insightful profiling metrics to populate the UI dashboard.
    
    Attributes:
        df_raw (pd.DataFrame): The original, untouched pandas DataFrame.
        df_cleaned (pd.DataFrame): The sanitized pandas DataFrame.
        metrics (Dict[str, Any]): A dictionary containing health metrics of the dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataCleaner with a raw dataframe.
        
        Args:
            df (pd.DataFrame): The raw dataframe uploaded by the user.
        """
        logger.info(f"Initializing DataCleaner with DataFrame of shape {df.shape}")
        self.df_raw = df.copy()
        self.df_cleaned = df.copy()
        self.metrics = {
            "original_rows": len(df),
            "original_columns": len(df.columns),
            "cleaned_rows": 0,
            "cleaned_columns": 0,
            "duplicates_removed": 0,
            "empty_rows_removed": 0,
            "empty_cols_removed": 0,
            "missing_values_remaining": 0,
            "memory_usage_mb": 0.0
        }
        
    def execute_cleaning_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes the full automated cleaning pipeline.
        
        The pipeline consists of:
        1. Removing exact duplicate rows.
        2. Removing rows that are 100% missing (NaN).
        3. Removing columns that are 100% missing (NaN).
        4. Calculating post-cleaning metrics.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The cleaned DataFrame and a dictionary of metrics.
        """
        logger.info("Starting automated cleaning pipeline...")
        
        self._remove_duplicates()
        self._remove_empty_rows()
        self._remove_empty_columns()
        self._calculate_final_metrics()
        
        logger.info(f"Cleaning complete. Final shape: {self.df_cleaned.shape}")
        return self.df_cleaned, self.metrics
        
    def _remove_duplicates(self) -> None:
        """Removes duplicate rows and updates metrics."""
        initial_count = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.drop_duplicates()
        removed_count = initial_count - len(self.df_cleaned)
        self.metrics["duplicates_removed"] = removed_count
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows.")
            
    def _remove_empty_rows(self) -> None:
        """Removes rows where all values are NaN and updates metrics."""
        initial_count = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.dropna(how='all')
        removed_count = initial_count - len(self.df_cleaned)
        self.metrics["empty_rows_removed"] = removed_count
        if removed_count > 0:
            logger.info(f"Removed {removed_count} entirely empty rows.")
            
    def _remove_empty_columns(self) -> None:
        """Removes columns where all values are NaN and updates metrics."""
        initial_count = len(self.df_cleaned.columns)
        self.df_cleaned = self.df_cleaned.dropna(axis=1, how='all')
        removed_count = initial_count - len(self.df_cleaned.columns)
        self.metrics["empty_cols_removed"] = removed_count
        if removed_count > 0:
            logger.info(f"Removed {removed_count} entirely empty columns.")
            
    def _calculate_final_metrics(self) -> None:
        """Calculates memory usage, remaining NaNs, and final shape."""
        self.metrics["cleaned_rows"] = len(self.df_cleaned)
        self.metrics["cleaned_columns"] = len(self.df_cleaned.columns)
        self.metrics["missing_values_remaining"] = int(self.df_cleaned.isna().sum().sum())
        
        # Calculate memory usage in Megabytes
        mem_bytes = self.df_cleaned.memory_usage(deep=True).sum()
        self.metrics["memory_usage_mb"] = round(mem_bytes / (1024 * 1024), 2)
        
        logger.debug(f"Final metrics calculated: {self.metrics}")

    def get_column_types(self) -> Dict[str, List[str]]:
        """
        Analyzes the dataframe to categorize columns by their data type.
        This is useful for the Data Explorer page to filter by numeric vs categorical.
        
        Returns:
            Dict[str, List[str]]: Dictionary with keys 'numeric', 'categorical', 'datetime'.
        """
        logger.info("Analyzing column data types...")
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = self.df_cleaned.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        
        # All other columns are considered categorical/text for simplicity
        all_cols = set(self.df_cleaned.columns)
        categorical_cols = list(all_cols - set(numeric_cols) - set(datetime_cols))
        
        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        }
        
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generates a statistical summary (describe) of the numerical columns,
        transposed for better UI rendering.
        
        Returns:
            pd.DataFrame: Transposed summary statistics DataFrame.
        """
        logger.info("Generating numerical summary statistics.")
        try:
            summary = self.df_cleaned.describe().T
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {e}")
            return pd.DataFrame()
