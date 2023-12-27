# 5G user forecast
# Machine Learning Model Comparison
This repository contains code for training and evaluating three different machine learning models: XGBoost Classifier, Gradient Boosting Classifier, and Decision Tree Classifier. The models are trained on a given dataset, and their performance is compared based on accuracy, AUC (Area Under the ROC Curve), and F1 score.

## Dataset

rain.csv: The dataset used for training and evaluation.
model_comparison.ipynb: Jupyter Notebook containing the code for model training and evaluation.
README.md: This document providing an overview and instructions.



## Usage

Dataset: Place your dataset in CSV format in the same directory and update the file path in the code.

Requirements: Make sure you have the required libraries installed. You can install them using the following:

```python
pip install pandas scikit-learn xgboost matplotlib seaborn
```
 run 

```python
python 220.py
```


### Model Comparison
Accuracy: The accuracy of each model is computed and displayed.
AUC (Area Under the ROC Curve): The AUC scores for each model are computed and displayed in a ROC curve plot.
F1 Score: The F1 scores for each model are computed and displayed in a bar plot for comparison.
You can train your model through the following command line:

### Results
The results of the model comparison, including accuracy, AUC, and F1 score, are visualized in bar plots. The ROC curve provides a visual representation of the trade-off between true positive rate and false positive rate for each model.

Feel free to adapt the code to your specific dataset and requirements. If you have any questions or suggestions, please open an issue.

# Citation
If you find this project useful, please cite our github link.

   
