
# Wildfire Cause Prediction

## Project Overview

This project aims to model and predict the cause of wildfires in the USA using the dataset provided by Kaggle. The goal is to optimize the weighted ROC AUC score for all classes, ensuring accurate prediction of wildfire causes.

## Dataset

The dataset used for this project can be found on Kaggle: [188 Million US Wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires).

## Project Guidelines

The project is part of the Applied Competitive Lab and requires:
- A notebook representing the entire workflow.
- A detailed report in PDF format explaining the methodologies and decisions.
- A requirements.txt file for the necessary libraries.
- Submission of up-to-date notebooks every two weeks.

## Project Structure

1. **Exploring and Cleaning the Data**
   - Identified and removed leakage features.
   - Addressed multiple class incidents.
   - Explored and cleaned various features, including handling missing values and transforming features.
   - Developed ideas for new features and searched for relevant datasets.

2. **Feature Extraction and Engineering**
   - Extracted features such as IS_WEEKDAY, SEASON, and DISCOVERY_YEAR/MONTH/DAY.
   - Engineered features based on demographic, weather, and geo-location information.
   - Evaluated feature importance using baseline models.

3. **Model Selection**
   - Baselined several models: Logistic Regression, K-Nearest Neighbors, Random Forest, and XGBoost.
   - Compared model performance using metrics like ROC AUC, accuracy, precision, recall, and F1 score.
   - Selected XGBoost as the best performing model for further tuning.

4. **Hyperparameter Tuning and Cross-Validation**
   - Performed hyperparameter tuning on XGBoost to optimize performance.
   - Validated the final model using cross-validation.

## Installation

To run the project, clone this repository and install the necessary libraries using the requirements.txt file:

```bash
git clone https://github.com/sositon/wildfire-cause-prediction.git
cd wildfire-cause-prediction
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook to see the entire workflow:

```bash
jupyter notebook Wildfires_Prediction.ipynb
```

## Dependencies

The project requires the following libraries, as listed in requirements.txt:
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- geopandas
- shapely

## Acknowledgements

I would like to thank my instructor and classmates for their contribution and feedback throughout this project.
