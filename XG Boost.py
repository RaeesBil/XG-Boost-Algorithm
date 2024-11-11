import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load training and testing datasets
train_data = pd.read_csv('train.csv')  # replace with actual file path
test_data = pd.read_csv('test.csv')    # replace with actual file path

# Drop Employee ID column from both datasets
train_data = train_data.drop('Employee ID', axis=1)
test_data = test_data.drop('Employee ID', axis=1)

# Initialize Label Encoder
le = LabelEncoder()

# Ordinal columns that require Label Encoding
ordinal_cols = ['Gender', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 
                'Overtime', 'Education Level', 'Marital Status', 'Job Level', 
                'Company Size', 'Remote Work', 'Leadership Opportunities', 
                'Innovation Opportunities', 'Company Reputation', 'Employee Recognition', 'Attrition']

# Apply Label Encoding to train dataset and handle unseen labels in the test set
for col in ordinal_cols:
    train_data[col] = le.fit_transform(train_data[col])  # Fit and transform on training data
    test_data[col] = le.transform(test_data[col]) if set(test_data[col]) <= set(le.classes_) else -1

# One-Hot Encoding for nominal categorical features
train_data = pd.get_dummies(train_data, columns=['Job Role'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Job Role'], drop_first=True)

# Align columns in test data with train data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Feature-target split for train and test datasets
X_train = train_data.drop('Attrition', axis=1)
y_train = train_data['Attrition'].apply(lambda x: 1 if x == 'Left' else 0)  # Convert Attrition to binary

X_test = test_data.drop('Attrition', axis=1)
y_test = test_data['Attrition'].apply(lambda x: 1 if x == 'Left' else 0)    # Convert Attrition to binary

# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0, 1],  # L2 regularization
    'alpha': [0, 0.5]   # L1 regularization
}

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best parameters: ", grid_search.best_params_)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Cross-validate on the training set to avoid overfitting
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f'Cross-validated accuracy on training data: {cv_scores.mean()}')

# Test the best model on the test dataset
y_pred = best_model.predict(X_test)

# Evaluate the model's performance on the test data
print(f'Accuracy on test data: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, and F1-Score for "Left" (Attrition = 1)
print(f"Precision (Left): {precision_score(y_test, y_pred)}")
print(f"Recall (Left): {recall_score(y_test, y_pred)}")
print(f"F1-Score (Left): {f1_score(y_test, y_pred)}")

# Feature importance plot
importances = best_model.feature_importances_
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(train_data.columns[sorted_idx], importances[sorted_idx], color='steelblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
