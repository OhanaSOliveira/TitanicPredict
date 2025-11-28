# Titanic Predict  

## Machine Learning Model to Predict Titanic Passengers Survival  

**Data:** [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)  

One of the classic Kaggle challenges is the Titanic Survival Prediction, which aims to predict whether a passenger survived the disaster based on features such as class, age, sex, fare, and other variables.  

The idea is to train a Machine Learning model using the provided training and test datasets, generating a binary classification — 1 for survived, 0 for did not survive.  

In this project, I used **NumPy, Pandas, and Scikit-Learn**, applying the **Random Forest Classifier** to perform predictions.  

---

## Project Steps  

### 1. Initial Import and Setup  
I imported the required libraries and set the seed (`random_state=0`) to standardize results and ensure reproducibility.  
Then, I loaded the `train.csv` and `test.csv` files from Kaggle, also saving the `PassengerId` for the final submission file.  

---

### 2. Feature Analysis  
An exploratory analysis was conducted to understand which features had the greatest influence on survival.  
The main features selected were:  

- **Numerical:** Age, Fare, and FamilySize (number of family members onboard);  
- **Categorical:** Pclass (ticket class), Sex, Embarked (port of embarkation), Title (extracted from Name), Cabin, and isAlone (indicates if traveling alone).  

I aimed for a balance between **complexity and interpretability**, prioritizing relevant features with a direct impact on the model.  

---

## Preprocessing  

### Extraction of Title (`Title`)  
The `Name` column contained the passenger’s title along with the name, separated by a comma and a period.  
A function was created to **extract the title** (e.g., Mr, Miss, Mrs, Dr).  

Additional adjustments:
- Replaced Mlle and Ms with Miss, and Mme with Mrs;  
- Grouped rare titles into a "Rare" category, including:  
  'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'.  

### Creation of New Features  
- **FamilySize**: sum of SibSp (siblings/spouses onboard) + Parch (parents/children onboard) + 1 (the passenger themselves).  
- **isAlone**: binary variable indicating if the passenger was alone.  

### Preprocessing Pipeline  
Separate **pipelines for numerical and categorical features** were created:  
- Numerical: missing values replaced by the **median**;  
- Categorical: missing values replaced by the **most frequent category**, followed by **One-Hot Encoding**.  

These transformations are applied **in memory only**, without modifying the original files, ensuring consistency and reproducibility.  

---

## Machine Learning Model  

I used the **RandomForestClassifier**, an ensemble learning model composed of multiple decision trees.  
Each tree makes independent predictions, and the final result is determined by the majority vote of all trees.  

**Key parameters used:**
- `n_estimators=100` → number of trees in the forest;  
- `max_depth=None` → maximum depth of the trees (unlimited);  
- `min_samples_split=2` → minimum number of samples required to split a node;  
- `random_state=42` → ensures reproducibility.  

Evaluation was performed using **Cross-Validation**, specifically `StratifiedKFold` with 5 splits.  
This technique provides a more reliable performance estimate, avoiding bias from a single train-test split.  

---

## Results  

- **Cross-Validation Accuracy:** ~82%  
- **Kaggle Submission Accuracy:** 0.77  

These results indicate good generalization of the model, with no significant signs of overfitting.  

---

## Learnings  

During this project, I consolidated knowledge in:
- Data preprocessing (handling missing values and encoding categorical variables);  
- Feature engineering and manipulation;  
- Application and evaluation of supervised classification models;  
- Using pipelines for a clean, reproducible workflow;  
- Practical understanding of cross-validation and the importance of reproducibility.  

---

## Next Steps  

- Test other models such as **XGBoost**, **Logistic Regression**, and **Gradient Boosting**;  
- Optimize hyperparameters using **GridSearchCV** or **Optuna**;  
- Implement **model interpretation** with SHAP or LIME;  
- Create an **interactive dashboard** to visualize predictions.  

