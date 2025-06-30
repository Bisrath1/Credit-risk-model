"""
data_processing.py
This script transforms the raw Xente dataset into a model-ready format by performing
feature engineering steps: aggregation, temporal feature extraction, RFM calculation,
clustering, risk labeling, imputation, categorical encoding, and numerical scaling.
The processed data is saved to data/processed/.
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE
import joblib

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
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime']).dt.tz_localize(None)

            snapshot = pd.to_datetime(self.snapshot_date)  # already naive


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

class RFMPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer to scale RFM features for clustering."""
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        try:
            logger.info("Fitting RFM scaler")
            rfm_features = X[['Recency', 'Frequency', 'Monetary']]
            self.scaler.fit(rfm_features)
            joblib.dump(self.scaler, 'data/processed/rfm_scaler.joblib')
            logger.info("RFM scaler saved to data/processed/rfm_scaler.joblib")
            return self
        except Exception as e:
            logger.error(f"Error in RFMPreprocessor fit: {str(e)}")
            raise
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Scaling RFM features")
            rfm_features = X[['Recency', 'Frequency', 'Monetary']]
            rfm_features['Monetary'] = np.log1p(rfm_features['Monetary'].clip(lower=0))  # Handle skewness
            X[['Recency', 'Frequency', 'Monetary']] = self.scaler.transform(rfm_features)
            logger.info("RFM features scaled successfully")
            return X
        except Exception as e:
            logger.error(f"Error in RFMPreprocessor transform: {str(e)}")
            raise

class RFMClusterer(BaseEstimator, TransformerMixin):
    """Custom transformer to cluster customers based on RFM features."""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
    
    def fit(self, X, y=None):
        try:
            logger.info("Fitting K-Means clustering model")
            rfm_features = X[['Recency', 'Frequency', 'Monetary']]
            self.kmeans.fit(rfm_features)
            joblib.dump(self.kmeans, 'data/processed/kmeans.joblib')
            logger.info("K-Means model saved to data/processed/kmeans.joblib")
            return self
        except Exception as e:
            logger.error(f"Error in RFMClusterer fit: {str(e)}")
            raise
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Applying K-Means clustering")
            rfm_features = X[['Recency', 'Frequency', 'Monetary']]
            X['Cluster'] = self.kmeans.predict(rfm_features)
            logger.info("Clustering completed successfully")
            return X
        except Exception as e:
            logger.error(f"Error in RFMClusterer transform: {str(e)}")
            raise

class RiskLabeler(BaseEstimator, TransformerMixin):
    """Custom transformer to assign high-risk labels based on RFM clusters."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X = X.copy()
            logger.info("Assigning high-risk labels based on clusters")
            
            # Analyze cluster characteristics
            cluster_summary = X.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).reset_index()

            # Try to find high-risk cluster
            risky_cluster_df = cluster_summary[
                (cluster_summary['Recency'] == cluster_summary['Recency'].max()) &
                (cluster_summary['Frequency'] == cluster_summary['Frequency'].min()) &
                (cluster_summary['Monetary'] == cluster_summary['Monetary'].min())
            ]

            if risky_cluster_df.empty:
                logger.warning("No cluster matched all risk conditions — falling back to cluster with max Recency")
                high_risk_cluster = cluster_summary.loc[cluster_summary['Recency'].idxmax(), 'Cluster']
            else:
                high_risk_cluster = risky_cluster_df['Cluster'].iloc[0]

            X['is_high_risk'] = (X['Cluster'] == high_risk_cluster).astype(int)
            logger.info("High-risk labels assigned successfully")
            return X

        except Exception as e:
            logger.error(f"Error in RiskLabeler: {str(e)}")
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
        'transaction_day', 'transaction_month', 'transaction_year',
        'Recency', 'Frequency', 'Monetary'
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
        ('rfm', RFMCalculator(snapshot_date='2025-07-01')),
        ('rfm_preprocess', RFMPreprocessor()),
        ('rfm_cluster', RFMClusterer(n_clusters=3, random_state=42)),
        ('risk_label', RiskLabeler()),
        ('preprocessor', preprocessor),
        # ('woe', woe_transformer)  # Uncomment after Task 4 completion
    ])
    
    return pipeline

def process_data(input_path, output_path):
    """Process raw data and save the transformed dataset."""
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        logger.info("Creating preprocessing pipeline")
        # Step-by-step apply pipeline up to 'risk_label'
        pipeline = create_preprocessing_pipeline()

        # Manually break apart the pipeline to access intermediate outputs
        df_step1 = pipeline.named_steps['aggregate'].transform(df)
        df_step2 = pipeline.named_steps['temporal'].transform(df_step1)
        df_step3 = pipeline.named_steps['rfm'].transform(df_step2)
        df_step4 = pipeline.named_steps['rfm_preprocess'].fit_transform(df_step3)
        df_step5 = pipeline.named_steps['rfm_cluster'].fit_transform(df_step3)
        df_step6 = pipeline.named_steps['risk_label'].transform(df_step5)

        # Store `is_high_risk` before preprocessing
        is_high_risk = df_step6['is_high_risk'].values
        customer_ids = df_step6['CustomerId'].values

        # Final transformation step: encoding & scaling
        final_df = pipeline.named_steps['preprocessor'].fit_transform(df_step6)

        # Get column names
        numerical_cols = [
            'Amount', 'Value', 'total_transaction_amount',
            'avg_transaction_amount', 'transaction_count',
            'std_transaction_amount', 'transaction_hour',
            'transaction_day', 'transaction_month', 'transaction_year',
            'Recency', 'Frequency', 'Monetary'
        ]
        categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
        encoded_cols = list(pipeline.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['encoder']
                            .get_feature_names_out(categorical_cols))
        transformed_cols = numerical_cols + encoded_cols

        # Rebuild final DataFrame
        transformed_df = pd.DataFrame(final_df, columns=transformed_cols)
        transformed_df['CustomerId'] = customer_ids
        transformed_df['is_high_risk'] = is_high_risk

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






