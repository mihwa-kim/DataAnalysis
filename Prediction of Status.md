# Prediction of Status
```python
import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv('GEF_projects (1).csv')

# Display the first few rows of the data
data.head()
```

* Title: The title of the project.
* ID: An identifier for the project.
* Countries: The country associated with the project.
* Focal Areas: The main focus areas of the project.
* Type: The type of project.
* Agencies: The agency responsible for the project.
* GEF Grant: The grant amount from GEF.
* Cofinancing: Cofinancing amount (if any).
* Status: The status of the project (e.g., "Project Approved").
* Approval FY: The fiscal year of approval.
* Funding Source (indexed field): The source of the funding.
* Non-Grant Instrument (indexed field): Information about the non-grant instrument.
* Capacity-building Initiative for Transparency: Information about the capacity-building initiative for transparency.
* GEF Period: The period of GEF support.
* Before proceeding, we should decide on the objective of our machine learning task. Here are a few possibilities:

[v] Classification: Predicting the Status of a project based on other features.
[v] Regression: Predicting the GEF Grant amount based on other features.
* Clustering: Grouping projects based on their similarities.
* Anomaly Detection: Finding projects that deviate from the norm.

## Feature Engineering
# Check for missing values in each column
```python
missing_values = data.isnull().sum()

missing_values
```

```python
# Handle missing values based on the suggested approach

# 1. Fill 'Focal Areas' with 'Unknown'
data['Focal Areas'].fillna('Unknown', inplace=True)

# 2. Fill 'GEF Grant' with its median after removing commas and converting it to float
data['GEF Grant'] = data['GEF Grant'].str.replace(',', '').astype(float)
data['GEF Grant'].fillna(data['GEF Grant'].median(), inplace=True)

# 3. Fill 'Cofinancing' with its median (after converting it to float, or 0 if not convertible)
def convert_to_float_or_zero(value):
    try:
        return float(str(value).replace(',', ''))
    except ValueError:
        return 0.0

data['Cofinancing'] = data['Cofinancing'].apply(convert_to_float_or_zero)
data['Cofinancing'].fillna(data['Cofinancing'].median(), inplace=True)

# 4. Drop rows with missing 'Status'
data.dropna(subset=['Status'], inplace=True)

# 5. Fill 'Approval FY' with its median
data['Approval FY'].fillna(data['Approval FY'].median(), inplace=True)

# Check if all missing values are handled
missing_values_updated = data.isnull().sum()
missing_values_updated

```

```python
# Drop the 'Title' and 'ID' columns as they are unique identifiers
data = data.drop(columns=['Title', 'ID'])

# Extract the target variable 'Status'
y = data['Status']
data = data.drop(columns=['Status'])

# Convert categorical columns to one-hot encoded columns
X = pd.get_dummies(data)

# Splitting the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

```
* The data has been successfully split into training and testing sets. We have:

* 4,639 samples in the training set
* 1,160 samples in the testing set
* 1,151 features after one-hot encoding
* For the classification task, we can use various algorithms. Some commonly used classifiers include:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosted Trees
- Support Vector Machines
- Neural Networks

## Machine Learning
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
```

### Feature Importance
```python
# Reload the data and preprocess it
data = pd.read_csv('GEF_projects (1).csv')

# Handle missing values as previously described
data['Focal Areas'].fillna('Unknown', inplace=True)
data['GEF Grant'] = data['GEF Grant'].str.replace(',', '').astype(float)
data['GEF Grant'].fillna(data['GEF Grant'].median(), inplace=True)
data['Cofinancing'] = data['Cofinancing'].apply(convert_to_float_or_zero)
data['Cofinancing'].fillna(data['Cofinancing'].median(), inplace=True)
data.dropna(subset=['Status'], inplace=True)
data['Approval FY'].fillna(data['Approval FY'].median(), inplace=True)

# Drop 'Title' and 'ID' columns, and extract target variable
y = data['Status']
data = data.drop(columns=['Title', 'ID', 'Status'])

# Convert categorical columns to one-hot encoded columns
X = pd.get_dummies(data)

# Recreate the training and testing splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a smaller Random Forest model to get feature importances
small_rf = RandomForestClassifier(n_estimators=10, random_state=42)
small_rf.fit(X_train, y_train)

# Get feature importances and identify top 20 features
feature_importances = small_rf.feature_importances_
top_features = sorted(list(zip(X.columns, feature_importances)), key=lambda x: x[1], reverse=True)[:20]
top_feature_names = [feature[0] for feature in top_features]

# Select only the top features for training and testing sets
X_train_reduced = X_train[top_feature_names]
X_test_reduced = X_test[top_feature_names]

top_feature_names

```

#### Machine Accuracy Comparison
```python
# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
}

# Train and evaluate classifiers on the reduced dataset
accuracies_reduced = {}
for name, clf in classifiers.items():
    # Train classifier
    clf.fit(X_train_reduced, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test_reduced)
    
    # Calculate accuracy
    accuracies_reduced[name] = accuracy_score(y_test, y_pred)

accuracies_reduced

```

#### Param Tuning
```python
# Remove 'Status' related features from the top features list
top_feature_names = [feature for feature in top_feature_names if 'Status' not in feature]

# Select only the revised top features for training and testing sets
X_train_revised = X_train[top_feature_names]
X_test_revised = X_test[top_feature_names]

# Initialize classifiers with adjustments to reduce complexity
classifiers_revised = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42), # Regularization with C=0.5
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42), # Limiting tree depth
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), # Reducing number of trees and depth
    "Gradient Boosted Trees": GradientBoostingClassifier(max_depth=5, random_state=42), # Limiting tree depth
    "SVM": SVC(random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, alpha=0.005, random_state=42) # Regularization with alpha=0.005
}

# Train and evaluate classifiers on the revised dataset
accuracies_revised = {}
for name, clf in classifiers_revised.items():
    # Train classifier
    clf.fit(X_train_revised, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test_revised)
    
    # Calculate accuracy
    accuracies_revised[name] = accuracy_score(y_test, y_pred)

accuracies_revised

```

#### Test with the sample data
```python
# Randomly select a subset of the test data for predictions
sample_data = X_test_revised.sample(n=10, random_state=42)

# Get predictions from each model
predictions = {}
for name, clf in classifiers_revised.items():
    predictions[name] = clf.predict(sample_data)

# Convert predictions to a DataFrame for better visualization
predictions_df = pd.DataFrame(predictions)
predictions_df

```

###### I suggest using Gradient Boosted Trees for predicting the Status of Future Projects. With this ML process, each organisations could predict the planned project's results.
