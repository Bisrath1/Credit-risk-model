"""
data_processing.py
This script transforms the raw Xente dataset into a model-ready format by performing
feature engineering steps: aggregation, temporal feature extraction, imputation,
categorical encoding, and numerical scaling. The processed data is saved to data/processed/.
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer to compute customer-level aggregate features."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Computing aggregate features for CustomerId")
            agg_df = X.groupby('CustomerId').agg({
                'Amount': ['sum', 'mean', 'count', 'std'],
            }).reset_index()
            
            agg_df.columns = [
                'CustomerId',
                'total_transaction_amount',
                'avg_transaction_amount',
                'transaction_count',
                'std_transaction_amount'
            ]
            
            X = X.merge(agg_df, on='CustomerId', how='left')
            logger.info("Aggregate features computed successfully")
            return X
        except Exception as e:
            logger.error(f"Error in AggregateFeatures: {str(e)}")
            raise

class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer to extract temporal features from TransactionStartTime."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Extracting temporal features from TransactionStartTime")
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            X['transaction_hour'] = X['TransactionStartTime'].dt.hour
            X['transaction_day'] = X['TransactionStartTime'].dt.day
            X['transaction_month'] = X['TransactionStartTime'].dt.month
            X['transaction_year'] = X['TransactionStartTime'].dt.year
            logger.info("Temporal features extracted successfully")
            return X
        except Exception as e:
            logger.error(f"Error in TemporalFeatures: {str(e)}")
            raise

# Define imputers for numerical and categorical features
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Define categorical encoder
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Define scaler for numerical features
numerical_scaler = StandardScaler()

# Define WoE transformer (to be used after proxy variable is created in Task 4)
woe_transformer = WOE()

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline for feature engineering."""
    numerical_cols = [
        'Amount', 'Value', 'total_transaction_amount',
        'avg_transaction_amount', 'transaction_count',
        'std_transaction_amount', 'transaction_hour',
        'transaction_day', 'transaction_month', 'transaction_year'
    ]
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', numerical_imputer),
                ('scaler', numerical_scaler)
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', categorical_imputer),
                ('encoder', categorical_encoder)
            ]), categorical_cols)
        ])
    
    pipeline = Pipeline([
        ('aggregate', AggregateFeatures()),
        ('temporal', TemporalFeatures()),
        ('preprocessor', preprocessor),
        # ('woe', woe_transformer)  # Uncomment after Task 4
    ])
    
    return pipeline

def process_data(input_path, output_path):
    """Process raw data and save the transformed dataset."""
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info("Creating and applying preprocessing pipeline")
        pipeline = create_preprocessing_pipeline()
        transformed_data = pipeline.fit_transform(df)
        
        # Get feature names for the transformed data
        numerical_cols = [
            'Amount', 'Value', 'total_transaction_amount',
            'avg_transaction_amount', 'transaction_count',
            'std_transaction_amount', 'transaction_hour',
            'transaction_day', 'transaction_month', 'transaction_year'
        ]
        categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
        transformed_cols = (
            numerical_cols +
            list(pipeline.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['encoder']
                 .get_feature_names_out(categorical_cols))
        )
        
        # Convert to DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=transformed_cols)
        transformed_df['CustomerId'] = df['CustomerId'].values
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transformed_df.to_csv(output_path, index=False)
        
        logger.info("Data processing completed successfully")
        return transformed_df
    
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = r"C:\10x AIMastery\Credit-risk-model\data\raw\data.csv"
    output_path = r"C:\10x AIMastery\Credit-risk-model\data\processed\processed_data.csv"
    processed_df = process_data(input_path, output_path)
    print(f"Processed data saved to: {output_path}")


class RFMCalculator(BaseEstimator, TransformerMixin):
    """Custom transformer to compute RFM metrics for each CustomerId."""
    def __init__(self, snapshot_date='2025-07-01'):
        self.snapshot_date = snapshot_date
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Computing RFM metrics for CustomerId")
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            snapshot = pd.to_datetime(self.snapshot_date)

            # Calculate RFM
            rfm_df = X.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (snapshot - x.max()).days,  # Recency
                'TransactionId': 'count',  # Frequency
                'Value': 'sum'  # Monetary
            }).reset_index()

            rfm_df.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

            # Save RFM data for debugging
            os.makedirs('data/processed', exist_ok=True)
            rfm_df.to_csv('data/processed/rfm.csv', index=False)
            logger.info("RFM metrics saved to data/processed/rfm.csv")

            # Merge RFM back to original data
            X = X.merge(rfm_df, on='CustomerId', how='left')
            logger.info("RFM metrics computed and merged successfully")
            return X
        except Exception as e:
            logger.error(f"Error in RFMCalculator: {str(e)}")
            raise