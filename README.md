MLAlgos on UCI Dataset
A comprehensive implementation of core machine learning algorithms on the UCI Adult Census dataset for income classification.

Project Overview
This project serves as a learning and experimentation platform for understanding, implementing, and comparing various machine learning algorithms. It utilizes the UCI Adult Income dataset to predict whether an individual's income exceeds $50K per year based on demographic and employment-related attributes.

The project includes:

End-to-end ML pipeline from data ingestion to model evaluation

Modular code architecture with reusable components

Implementation of key algorithms including XGBoost, SVM, Naive Bayes, MLP, PCA, and K-Means
📁 Directory Structure
graphql
Copy
Edit
mlalgos/
│
├── artifacts/              # Saved models, transformers, processed data
├── data/                   # Raw and interim data
├── notebooks/              # EDA and experimentation notebooks
├── src/
│   └── components/         # Modular scripts: ingestion, transformation, training
├── venv/                   # Virtual environment (excluded in .gitignore)
├── config.yaml             # Configuration file for pipeline parameters
├── requirements.txt        # Project dependencies
├── main.py                 # Pipeline execution entry point
└── README.md               # Project documentation
📂 Dataset
Source: UCI Adult Dataset

Records: ~32,000

Features: age, education, occupation, relationship, race, sex, hours-per-week, native-country, etc.

Target: income (<=50K or >50K)

🔍 Exploratory Data Analysis (EDA)
Addressed missing values represented as "?"

Performed label encoding and one-hot encoding

Visualized class imbalance and key feature distributions

Observed significant correlation between education level and income

Notebooks for EDA are available in the notebooks/ directory.

🧪 Models & Evaluation with Performance Metrics
Implemented and evaluated the following algorithms:

Model,Accuracy,Precision,Recall,F1 Score,ROC AUC
XGBoost,0.8630863583623405,0.7686804451510334,0.6438082556591211,0.7007246376811594,0.9226621548668574
Random Forest,0.8486656721365822,0.7295401402961809,0.6231691078561917,0.6721723518850987,0.90237250061053
Logistic Regression,0.8475053870379579,0.7362012987012987,0.6038615179760319,0.6634967081199707,0.9021297579832497
MLP Neural Network,0.8480026520802254,0.7399507793273175,0.6005326231691078,0.6629915472252849,0.9093152042402964
SVM RBF,0.8488314271506713,0.7517064846416383,0.5865512649800266,0.6589379207180255,0.8974757705535561
SVM Linear,0.8465108569534229,0.7478485370051635,0.5785619174434088,0.6524024024024024,0.8998955560172694
K-Nearest Neighbors,0.8279462953754351,0.6710914454277286,0.6058588548601864,0.6368089573128062,0.8792317959927483
Decision Tree,0.8140228741919443,0.6254953764861294,0.6304926764314248,0.6279840848806366,0.7526773688932671
Naive Bayes,0.5624067628045748,0.357035175879397,0.9460719041278296,0.5184239328712149,0.7820749410555661

⚙️ Pipeline Overview
Each stage is implemented as a separate module under src/components/:

data_ingestion.py: Loads and splits data

data_transformation.py: Cleans, encodes, and transforms features

model_trainer.py: Trains multiple ML models

main.py: Orchestrates the pipeline

Configuration is managed via config.yaml, and logs are automatically captured for debugging.

🚀 How to Run the Project
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/mlalgos.git
cd mlalgos
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the pipeline:

bash
Copy
Edit
python main.py
🧰 Technologies Used
Python 3.13

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn

XGBoost

YAML for configuration

Logging & Exception handling

✅ Future Improvements
Hyperparameter tuning (GridSearchCV, Optuna)

Model explainability using SHAP or LIME

Deployment via Streamlit or FastAPI

Unit testing and CI/CD integration
