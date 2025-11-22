Airbnb Price Prediction with MLflow and AWS S3

Project Overview:

This project develops a machine learning solution for Airbnb dataset, a global vacation rental platform, to predict optimal nightly prices for Airbnb listings. The system helps hosts set competitive rates by analyzing factors such as location, amenities, and reviews.

Business Context:

Listing prices on StayWise vary significantly, even among similar properties. This predictive model enables:
Hosts: Set competitive and profitable pricing
Business Team Understand pricing dynamics across markets
Platform: Improve listing quality and booking rates

Key Features:

-  Experiment Tracking: MLflow for comprehensive experiment management
-  Cloud Storage: AWS S3 integration for data storage
-  Multiple Models: Comparison of Linear Regression, Ridge, Random Forest, Gradient Boosting, and XGBoost
-  Model Registry: Centralized model versioning and deployment tracking
-  Reproducible Pipeline: End-to-end ML workflow with version control

Problem Statement:

Build a modeling pipeline that:
1. Handles noisy data from AWS S3 (missing values, outliers, categorical fields)
2. Trains and compares multiple regression models
3. Tracks experiments systematically using MLflow
4. Evaluates model performance with standardized metrics
5. Registers and versions the best-performing models


📁 Repository Structure


airbnb-price-prediction/
│
├── data/                          
│   ├── raw/                       
│   └── processed/                 
│
├── notebooks/                     
│   └── airbnbmain.ipynb          
│
├── src/                           
│   ├── data_preprocessing.py       
│   ├── model_training.py          
│   └── utils.py                   
│
├── mlruns/                        
│
├── models/                        
│
├── screenshots/                   
│   ├── MLFLOW.png                
│   └── MLFLOW_2.png              
│
├── requirements.txt               
├── .gitignore                     
└── README.md                      

To start the project

Prerequisites

- Python 3.8 or higher
- AWS account with S3 access
- pip package manager

Installation

1.Clone the repository
   ```bash
   git clone https://github.com/mansithakkar2604/Airbnbyouwillbemissed
   cd airbnb-price-prediction
   ```

2.Create a virtual environment
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure AWS credentials
   ```bash
   aws configure
   # AWS Access Key ID, Secret Access Key, and region
   ```

5. Set up MLflow tracking
   ```bash
   # Start MLflow UI 
   mlflow ui --port 5000
   


Workflow Description

1. Data Acquisition
- Fetch Airbnb listing data from AWS S3
- Initial data exploration and quality assessment

2. Data Preprocessing
- Missing Value Handling: Imputation strategies for numerical and categorical features
- Outlier Detection: Statistical methods to identify and treat anomalies
- Feature Engineering:
  - One-hot encoding for categorical variables
  - Scaling/normalization of numerical features
  - Creation of derived features (e.g., price per bedroom)

3. Model Training & Experimentation
Five regression models are trained and compared:
- Linear Regression: Baseline model
- Ridge Regression: Regularized linear model
- Random Forest: Ensemble tree-based model
- Gradient Boosting: Sequential boosting algorithm
- XGBoost: Optimized gradient boosting

4. Experiment Tracking with MLflow
Each experiment logs:
- Parameters: Hyperparameters, preprocessing steps
- Metrics: RMSE, MAE, MSE, R²
- Artifacts: Model files, feature importance plots, preprocessing pipelines
- Tags: Model type, dataset version, experiment notes

5. Model Evaluation
Performance metrics:
- RMSE (Root Mean Squared Error): Measures prediction accuracy
- MAE (Mean Absolute Error): Average absolute deviation
- MSE (Mean Squared Error): Squared error metric
- R² Score: Proportion of variance explained

6. Model Registry
- Best models registered in MLflow Model Registry
- Version tracking for production deployment
- Model stage transitions (Staging → Production)

MLflow UI Screenshots
(MLFLOW.png)
Experiment Runs Comparison

Key Observations:
- Linear Regression achieved RMSE of 63.86
- Random Forest :showed best performance with RMSE of 63.85
- XGBoost: demonstrated competitive results with RMSE of 54.71
- Gradient Boosting achieved RMSE of 55.82
- Ridge Regression produced RMSE of 54.19

Model Registry View
(MLFLOW 2.png)

Registered Models:
- `xgboost_model` - MAE: 41.16, MSE: 2993.11, R²: 0.36
- `gradient_boosting_model` - MAE: 42.69, MSE: 3115.44, R²: 0.33
- `random_forest_model` - MAE: 40.77, MSE: 2936.08, R²: 0.37
- `ridge_model` - MAE: 50.76, MSE: 4077.21, R²: 0.12
- `linear_regression_model` - MAE: 50.76, MSE: 4077.48, R²: 0.12

Key Insights and Observations

Model Performance

1. Random Forest emerged as the best performer:
   - Highest R² score of 0.37
   - Lowest MSE of 2936.08
   - Strong ability to capture non-linear relationships

2. XGBoost showed competitive results:
   - Lowest MAE of 41.16
   - Efficient training time
   - Good generalization capability

3. Linear Models (Ridge & Linear Regression):
   - Similar performance (R² = 0.12)
   - Struggled with complex feature interactions
   - Suitable as baseline models

Data :

- Feature importance analysis reveals location and property size as key price drivers
- Amenities and review scores show moderate correlation with pricing
- Seasonal patterns and booking trends affect price predictions

Recommendations

1. For Production: Deploy Random Forest or XGBoost based on inference latency requirements
2. Feature Engineering: Further exploration of location-based features could improve predictions
3. Hyperparameter Tuning: Grid search or Bayesian optimization for optimal parameters
4. Ensemble Methods: Stacking multiple models could boost overall performance

Usage

Running the Complete Pipeline

```bash
# Execute the main notebook
jupyter notebook notebooks/Airbnbmain.ipynb
```

Training Individual Models

```python
import mlflow
from src.model_training import train_model

#Start MLflow experiment
mlflow.set_experiment("airbnb-price-prediction")

# Train a model
with mlflow.start_run(run_name="xgboost-experiment"):
    model = train_model(model_type='xgboost', X_train, y_train)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")


Viewing MLflow UI

```bash
# Launch MLflow UI
mlflow ui --port 5000

# Open browser to: http://127.0.0.1:5000
```

Loading a Registered Model

```python
import mlflow.pyfunc

# Load production model
model_uri = "models:/random_forest_model/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
predictions = model.predict(X_test)
```

Dependencies

Key libraries used:
- MLflow: Experiment tracking and model registry
- scikit-learn: Machine learning algorithms
- XGBoost: Gradient boosting framework
- pandas: Data manipulation
- numpy: Numerical computing
- boto3: AWS S3 integration
- matplotlib/seaborn: Visualization

 `requirements.txt` for complete list.

AWS S3 Configuration

Setting Up S3 Access

1. Create an S3 bucket for your data
2. Upload dataset to the bucket
3. Configure IAM permissions for access
4. Set environment variables:

```bash
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION
export S3_BUCKET_NAME
```

 Future Enhancements

-  Implement deep learning models 
-  Add real-time price prediction API
-  Create interactive dashboard for model monitoring
-  Integrate A/B testing framework
-  Expand feature engineering with external data sources
-  Automate model retraining pipeline
-  Deploy models to AWS SageMaker










