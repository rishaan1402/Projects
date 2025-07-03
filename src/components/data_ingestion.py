import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        This method initiates the data ingestion process
        """
        logging.info("Data ingestion process started")
        
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Load data from UCI Adult dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            logging.info(f"Loading data from URL: {url}")
            
            df = pd.read_csv(url, header=None)
            logging.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Set column names
            df.columns = [
                "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"
            ]
            logging.info("Column names assigned successfully")
            
            # Replace missing values represented as ' ?'
            df = df.replace(' ?', pd.NA)
            logging.info("Missing values marked as NaN")
            
            # Drop rows with missing values
            initial_shape = df.shape
            df.dropna(inplace=True)
            logging.info(f"Dropped missing values. Shape changed from {initial_shape} to {df.shape}")
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Split the data into train and test sets
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42, 
                stratify=df['income']
            )
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Train set saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test set saved to {self.ingestion_config.test_data_path}")
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Data ingestion completed. Train data: {train_data}, Test data: {test_data}")