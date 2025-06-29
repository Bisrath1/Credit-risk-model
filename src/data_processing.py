import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
import os


from sklearn.base import BaseEstimator, TransformerMixin

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure X is a DataFrame
        X = X.copy()
        
        # Group by CustomerId and compute aggregates
        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            'CustomerId',
            'total_transaction_amount',
            'avg_transaction_amount',
            'transaction_count',
            'std_transaction_amount'
        ]
        
        # Merge aggregates back to original dataframe
        X = X.merge(agg_df, on='CustomerId', how='left')
        
        return X
    

class TemporalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert TransactionStartTime to datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Extract temporal features
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        
        return X