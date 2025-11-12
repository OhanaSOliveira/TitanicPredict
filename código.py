import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 0

# 1) Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['PassengerId']

# 2) Simple feature engineering
def extract_title(name):
    m = re.search(r',\s*([^\.]+)\.', name)
    return m.group(1).strip() if m else 'Unknown'

for df in (train, test):
    df['Title'] = df['Name'].map(extract_title)
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss').replace(['Mme'],'Mrs')
    rare = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].apply(lambda t: 'Rare' if t in rare else t)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['CabinKnown'] = (~df['Cabin'].isnull()).astype(int)


num_features = ['Age', 'Fare', 'FamilySize']
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'CabinKnown', 'IsAlone']

X = train[num_features + cat_features].copy()
y = train['Survived'].copy()
X_test = test[num_features + cat_features].copy()


num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

# 5) Model pipeline
rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1)
pipe = Pipeline([
    ('preproc', preprocessor),
    ('clf', rf)
])

# 6) Cross-validation (stratified)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("CV accuracy scores:", np.round(scores,4))
print("Mean CV accuracy:", np.round(scores.mean(),4))

# 7) Fit and predict
pipe.fit(X, y)
preds = pipe.predict(X_test).astype(int)

# 8) Save submission
submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")

