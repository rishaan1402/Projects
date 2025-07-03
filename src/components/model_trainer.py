import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_file_path: str = os.path.join('artifacts', 'model_report.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def evaluate_model(self, y_true, y_pred, y_prob=None):
        """
        Evaluate model performance using multiple metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred)
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            return metrics
            
        except Exception as e:
            logging.error("Exception occurred in evaluate_model")
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        This function initiates the model training process
        """
        try:
            logging.info("Starting model training process")
            
            # Split the data into features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            logging.info(f"Training target shape: {y_train.shape}")
            logging.info(f"Test target shape: {y_test.shape}")
            
            # Ensure target arrays are 1D
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            
            logging.info(f"After ravel - Training target shape: {y_train.shape}")
            logging.info(f"After ravel - Test target shape: {y_test.shape}")
            
            # Define models to train
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
                "SVM Linear": SVC(probability=True, kernel='linear', random_state=42),
                "SVM RBF": SVC(probability=True, kernel='rbf', random_state=42),
                "Naive Bayes": GaussianNB(),
                "MLP Neural Network": MLPClassifier(
                    hidden_layer_sizes=(64, 32), 
                    max_iter=600, 
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            }
            
            # Dictionary to store model results
            model_results = {}
            
            logging.info("Training and evaluating models")
            
            for model_name, model in models.items():
                try:
                    logging.info(f"Training {model_name}")
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Get probability predictions if available
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = None
                    
                    # Evaluate model
                    metrics = self.evaluate_model(y_test, y_pred, y_prob)
                    
                    # Store results
                    model_results[model_name] = {
                        'model': model,
                        'metrics': metrics
                    }
                    
                    logging.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
                    
                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Find the best model based on F1 score
            if not model_results:
                logging.error("No models were successfully trained")
                raise CustomException("No models were successfully trained", sys)
            
            best_model_name = max(model_results.keys(), 
                                key=lambda x: model_results[x]['metrics']['f1_score'])
            best_model = model_results[best_model_name]['model']
            best_f1_score = model_results[best_model_name]['metrics']['f1_score']
            
            logging.info(f"Best model: {best_model_name} with F1 Score: {best_f1_score:.4f}")
            
            # Check if the best model meets minimum performance threshold
            if best_f1_score < 0.6:
                logging.warning(f"Best model F1 score {best_f1_score:.4f} is below threshold 0.6")
                # Don't raise exception, just log warning and continue
                logging.info("Proceeding with best available model despite low performance")
            
            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            model_data = {
                'model': best_model,
                'model_name': best_model_name,
                'f1_score': best_f1_score,
                'all_metrics': model_results[best_model_name]['metrics']
            }
            
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")
            
            # Create model comparison report
            results_list = []
            for model_name, result in model_results.items():
                metrics = result['metrics']
                results_list.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1_score'],
                    'ROC AUC': metrics.get('roc_auc', 'N/A')
                })
            
            # Sort by F1 score
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
            
            # Save model report
            results_df.to_csv(self.model_trainer_config.model_report_file_path, index=False)
            logging.info(f"Model comparison report saved to {self.model_trainer_config.model_report_file_path}")
            
            logging.info("Model training completed successfully")
            
            return best_f1_score
            
        except Exception as e:
            logging.error("Exception occurred in initiate_model_trainer")
            raise CustomException(e, sys)

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning for the best performing models
    """
    try:
        logging.info("Starting hyperparameter tuning")
        
        # Define parameter grids for top models
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for model_name, model in models.items():
            logging.info(f"Tuning {model_name}")
            
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
            
            logging.info(f"{model_name} best score: {grid_search.best_score_:.4f}")
        
        return best_model, best_score
        
    except Exception as e:
        logging.error("Exception occurred in hyperparameter_tuning")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # For testing purposes - example of how to use with actual data
    import numpy as np
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation
    
    try:
        # Run the complete pipeline
        logging.info("Starting complete ML pipeline")
        
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # Model Training
        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Pipeline completed successfully! Best F1 Score: {best_score:.4f}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        logging.error(f"Pipeline failed: {str(e)}")
    
    print("Model trainer module execution completed")