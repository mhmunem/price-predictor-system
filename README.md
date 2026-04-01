# Housing Price Prediction with ZenML & MLflow

This project implements a machine learning pipeline to predict housing prices using the **[Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)**, orchestrated with **[ZenML](https://zenml.io/)** and tracked with **[MLflow](https://github.com/mlflow/mlflow)**. It includes data preprocessing, model training, evaluation and deployment in a reproducible and scalable way.

---

## Features

- **End-to-end ML Pipeline**: From data ingestion to model serving.
- **Modular Design**: Clean separation of concerns using design patterns (Strategy, Factory, Template).
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis in Jupyter notebooks.
- **Missing Value & Outlier Handling**: Robust preprocessing pipeline.
- **Feature Engineering**: Automated transformations and engineering.
- **Model Training & Evaluation**: Train and compare models with MLflow tracking.
- **Deployment Ready**: Model can be served locally or via REST API.
- **Reproducibility**: Config-driven with `config.yaml` and MLflow logging.

---

## Technologies Used

- **Python** – Core programming
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn** – Machine learning models
- **Matplotlib, Seaborn** – Data visualization
- **MLflow** – Model tracking and registry
- **ZenML** – MLOps orchestration (inferred from step structure)
- **Jupyter Notebook** – EDA and prototyping

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mhmunem/prices-predictor-system.git
cd prices-predictor-system
```

### 2. Install ZenML
Before proceeding, install **ZenML** globally:

```bash
pip install zenml
```

> Installation Guide: [https://docs.zenml.io/getting-started/installation](https://docs.zenml.io/getting-started/installation)

### 3. Create and Activate a Virtual Environment

Watch this guide for help: [link](https://youtu.be/GZbeL5AcTgw?si=uj7B8-10kbyEytKo)

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
# OR
venv\Scripts\activate       # Windows
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Setup ZenML with MLflow

This project uses **MLflow** for experiment tracking and model deployment via **ZenML**.

### Install Required Integrations

```bash
zenml integration install mlflow -y
```

### Register Components and Stack

Run the following commands to set up a local MLflow-enabled ZenML stack:

```bash
zenml login --local  # opens the dashboard locally 
```

```bash
# Register MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register MLflow model deployer
zenml model-deployer register mlflow --flavor=mlflow

# Register and set the active stack
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

You now have a fully functional ZenML stack with:
- Experiment tracking via MLflow
- Model deployment via MLflow
- Artifacts stored locally

> 💡 MLflow UI can be accessed with: `mlflow ui` after running the pipeline.


## Running the Pipelines

### 1. Run the Training Pipeline

```bash
python run_pipeline.py
```

This will:
- Ingest and preprocess the data
- Handle missing values and outliers
- Perform feature engineering
- Split the data
- Train a model
- Evaluate performance
- Log experiments to MLflow

### 2. Run the Deployment Pipeline

```bash
python run_deployment.py
```

Deploys the trained model using MLflow. The model will be served locally (or containerized, depending on configuration).

> Ensure MLflow integration is installed and stack is properly configured.

### 3. Make Sample Predictions

After deployment, test the model:

```bash
python sample_predict.py
```

Sends a sample input to the prediction service and prints the result.

---

## Explore Experiments with MLflow

After running the pipeline, launch the MLflow UI to view runs, parameters, and metrics:

```bash
     mlflow ui --backend-store-uri 'file:/<your-file-path>/.config/zenml/local_stores/2ac1bb25-cce8-4463-9b07-58f2fe5837e9/mlruns'
```

The file path will be displayed when you run the `run_pipeline.py` file

---

## Project Structure (Summary)

```
├── analysis/               # EDA scripts and notebooks
├── data/                   # Raw zipped dataset
├── extracted_data/         # Unzipped CSV: AmesHousing.csv
├── pipelines/              # Training & deployment pipelines
├── src/                    # Core modules for data & modeling
├── steps/                  # ZenML pipeline steps
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Run training pipeline
├── run_deployment.py       # Run deployment pipeline
├── sample_predict.py       # Make predictions
└── README.md
```

---


## License

MIT License – see `LICENSE` for details.

---

## Acknowledgments

- Dataset: Dean De Cock, *Ames Housing Data*
- Inspired by Kaggle House Prices competition
- Orchestration: [ZenML Framework](https://zenml.io)

---

## Contact

Mohammad Munem
[GitHub](https://github.com/mhmunem) | [LinkedIn](https://linkedin.com/in/mhmunem)

