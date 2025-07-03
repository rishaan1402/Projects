import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass
import pickle
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for creating the preprocessing pipeline
        """
        try:
            logging.info("Creating data transformation pipeline")
            
            # Define categorical and numerical columns
            categorical_columns = [
                'workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'native-country'
            ]
            
            numerical_columns = [
                'age', 'fnlwgt', 'education-num', 'capital-gain', 
                'capital-loss', 'hours-per-week'
            ]
            
            # Create preprocessing pipelines
            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
            ])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            logging.info("Data transformation pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            logging.error("Exception occurred in get_data_transformer_object")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process
        """
        try:
            logging.info("Starting data transformation process")
            
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Check for any remaining missing values
            train_missing = train_df.isnull().sum().sum()
            test_missing = test_df.isnull().sum().sum()
            
            if train_missing > 0 or test_missing > 0:
                logging.warning(f"Missing values found - Train: {train_missing}, Test: {test_missing}")
                # Drop rows with missing values
                train_df = train_df.dropna()
                test_df = test_df.dropna()
                logging.info(f"After dropping missing values - Train: {train_df.shape}, Test: {test_df.shape}")
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Define target column
            target_column_name = "income"
            
            # Separate features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # Separate features and target for test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing on training and test dataframes")
            
            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Convert sparse matrix to dense if needed
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, 'toarray'):
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            logging.info(f"Transformed train features shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed test features shape: {input_feature_test_arr.shape}")
            
            # Encode target variable
            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df)
            
            logging.info(f"Encoded train target shape: {target_feature_train_encoded.shape}")
            logging.info(f"Encoded test target shape: {target_feature_test_encoded.shape}")
            
            # Ensure both arrays have the same number of samples
            if input_feature_train_arr.shape[0] != target_feature_train_encoded.shape[0]:
                raise CustomException(
                    f"Mismatch in number of samples: features {input_feature_train_arr.shape[0]} vs target {target_feature_train_encoded.shape[0]}", 
                    sys
                )
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")
            
            logging.info("Preprocessing completed successfully")
            
            # Save preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump({
                    'preprocessor': preprocessing_obj,
                    'label_encoder': label_encoder
                }, f)
            
            logging.info(f"Preprocessing object saved to {self.data_transformation_config.preprocessor_obj_file_path}")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # For testing purposes
    train_data_path = "artifacts/train.csv"
    test_data_path = "artifacts/test.csv"
    
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print(f"Data transformation completed. Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
    else:
        print("Train and test data files not found. Please run data ingestion first.")