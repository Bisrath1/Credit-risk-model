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
    

# Define imputers for numerical and categorical features
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')


# Define categorical encoder
categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# Define scaler for numerical features
numerical_scaler = StandardScaler()

# Define WoE transformer (to be used after proxy variable is created)
woe_transformer = WOE()




def create_preprocessing_pipeline():
    # Define numerical and categorical columns
    numerical_cols = ['Amount', 'Value', 'total_transaction_amount', 
                      'avg_transaction_amount', 'transaction_count', 
                      'std_transaction_amount', 'transaction_hour', 
                      'transaction_day', 'transaction_month', 'transaction_year']
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # Create ColumnTransformer for numerical and categorical features
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
    
    # Full pipeline
    pipeline = Pipeline([
        ('aggregate', AggregateFeatures()),
        ('temporal', TemporalFeatures()),
        ('preprocessor', preprocessor),
        # ('woe', woe_transformer)  # Uncomment after Task 4
    ])
    
    return pipeline

def process_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    transformed_data = pipeline.fit_transform(df)
    
    # Convert output to DataFrame (since OneHotEncoder returns numpy array)
    transformed_cols = (
        numerical_cols + 
        [col for col in pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['encoder']
                .get_feature_names_out(categorical_cols)]
    )
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_cols)
    
    # Add CustomerId back (not transformed)
    transformed_df['CustomerId'] = df['CustomerId']
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_df.to_csv(output_path, index=False)
    
    return transformed_df

if __name__ == "__main__":
    input_path = "data/raw/training.csv"
    output_path = "data/processed/processed_data.csv"
    processed_df = process_data(input_path, output_path)
    print("Processed data saved to:", output_path)