# Income Prediction with Machine Learning (UCI Adult Dataset)

Predicting whether a person earns more than $50K a year from census data, built as a
best-practice rework of a public beginner tutorial, as part of the Data Mining and Machine
Learning module of my MSc in Data Analytics at National College of Ireland.

## Problem
Binary classification on the [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
(32,561 people, 14 features). The target is imbalanced: 76% earn <=50K and 24% earn >50K,
so accuracy alone is misleading.

## What I improved over the original tutorial
| Issue in the original | What this project does |
|---|---|
| Scaler fitted on the test set (data leakage) | All preprocessing inside a scikit-learn `Pipeline`, fitted on training data only |
| `LabelEncoder` on every column, including numeric ones | One-hot encoding for categorical features, scaling for numeric ones |
| Missing values replaced with the most common value | Missing categories kept as an `Unknown` category |
| Plain random split, accuracy only | Stratified split, baseline model, precision/recall/F1/ROC-AUC, 5-fold cross-validation |
| No outlier analysis | Tukey's IQR and Local Outlier Factor (LOF) compared, with a justified decision |
| Broken variable names, meaningless plots | Clean, documented notebook that runs top to bottom |

## Results (test set)
| Metric | Baseline (always <=50K) | Original tutorial | This project |
|---|---|---|---|
| Recall (>50K) | 0.00 | 0.43 | **0.86** |
| F1 (>50K) | 0.00 | 0.54 | **0.69** |
| Accuracy | 0.76 | 0.82 | 0.81 |
| ROC-AUC | 0.50 | not reported | **0.91** |

5-fold cross-validation F1: 0.678 ± 0.007.

The model finds twice as many high earners as the original. Accuracy drops slightly because
the original mostly predicted the majority class.

## Outlier analysis
- **Tukey's IQR** flagged ~40% of rows, mostly non-zero capital gains/losses (most people have 0),
  so it is unsuitable for these skewed columns.
- **LOF** flagged the most unusual 1% of rows using all numeric features together.
- Outliers were kept: they are genuine values that carry information about high earners.

## How to run
1. Download `adult.data` from the UCI link above into an `adult/` folder next to the notebook.
2. Install the requirements: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `Income_Prediction_with_ML.ipynb` and run all cells.

## Tech stack
Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn

## Credits
Original tutorial: *Income Prediction with ML* by RinKul on Hashnode.
Dataset: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository.

## Author
Prathmesh Arya - MSc Data Analytics, National College of Ireland
