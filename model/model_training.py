import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib



# Load the dataset
df = pd.read_csv('model/loan_data.csv')

# update person_gender column as male=1 and female=0
df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})
# update person_education 
df['person_education'] = df['person_education'].map({'High School': 0, 'Associate': 1,'Bachelor': 2, 'Master': 3,'Doctorate': 4})

# update previous_loan_defaults_on_file column as Yes=1 and No=0
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding (Convert to Binary Vectors)

columns_to_encode = ['person_home_ownership', 'loan_intent']

df = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)

# convert boolean columns to integers
bool_cols = df.select_dtypes(include=bool).columns
df[bool_cols] = df[bool_cols].astype(int)

# Define features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

combined_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
combined_df.to_csv("test.csv", index=False, encoding="utf-8")

 

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Save the model using joblib
filename = f'model/logistic_regression.joblib'
joblib.dump(log_reg, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_log = log_reg.predict(X_test)

# # Evaluate the model
# results = []

# results.append({
# "Model": "Logistic Regression",
# "Accuracy":accuracy_score(y_test, y_pred_log),
# "AUC Score": roc_auc_score(y_test, y_pred_log),
# "Precision": precision_score(y_test, y_pred_log),
# "Recall": recall_score(y_test, y_pred_log),
# "F1 Score": f1_score(y_test, y_pred_log),
# "MCC Score": matthews_corrcoef(y_test, y_pred_log)})

# Initialize Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/decision_tree.joblib'
joblib.dump(dt_classifier, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_dt = dt_classifier.predict(X_test)

# # Evaluate the model
# results.append({
# "Model": "Decision Tree",
# "Accuracy":accuracy_score(y_test, y_pred_dt),  
# "AUC Score": roc_auc_score(y_test, y_pred_dt),
# "Precision": precision_score(y_test, y_pred_dt),
# "Recall": recall_score(y_test, y_pred_dt),
# "F1 Score": f1_score(y_test, y_pred_dt),
# "MCC Score": matthews_corrcoef(y_test, y_pred_dt)})

# Initialize K-Nearest Neighbors model
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/knn.joblib'
joblib.dump(knn_classifier, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_knn = knn_classifier.predict(X_test)

# # Evaluate the model
# results.append({
# "Model": "K-Nearest Neighbors",
# "Accuracy":accuracy_score(y_test, y_pred_knn),  
# "AUC Score": roc_auc_score(y_test, y_pred_knn),
# "Precision": precision_score(y_test, y_pred_knn),
# "Recall": recall_score(y_test, y_pred_knn),
# "F1 Score": f1_score(y_test, y_pred_knn),
# "MCC Score": matthews_corrcoef(y_test, y_pred_knn)})

# Initialize Naive Bayes model
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/naive_bayes.joblib'
joblib.dump(nb_classifier, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_nb = nb_classifier.predict(X_test)

# # Evaluate the model
# results.append({
# "Model": "Naive Bayes",
# "Accuracy":accuracy_score(y_test, y_pred_nb),  
# "AUC Score": roc_auc_score(y_test, y_pred_nb),
# "Precision": precision_score(y_test, y_pred_nb),
# "Recall": recall_score(y_test, y_pred_nb),
# "F1 Score": f1_score(y_test, y_pred_nb),
# "MCC Score": matthews_corrcoef(y_test, y_pred_nb)})

# Initialize Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/random_forest.joblib'
joblib.dump(rf_classifier, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_rf = rf_classifier.predict(X_test)

# # Evaluate the model
# results.append({
# "Model": "Random Forest (Ensemble)",
# "Accuracy":accuracy_score(y_test, y_pred_rf),  
# "AUC Score": roc_auc_score(y_test, y_pred_rf),
# "Precision": precision_score(y_test, y_pred_rf),
# "Recall": recall_score(y_test, y_pred_rf),
# "F1 Score": f1_score(y_test, y_pred_rf),
# "MCC Score": matthews_corrcoef(y_test, y_pred_rf)})

# Initialize Gradient Boosting model
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gb_classifier.fit(X_train, y_train)

# Save the model using joblib

filename = f'model/gradient_boosting.joblib'
joblib.dump(gb_classifier, filename)
print(f"Saved {filename}")

# # Predict on the test set
# y_pred_gb = gb_classifier.predict(X_test)

# # Evaluate the model
# results.append({
# "Model": "Gradient Boosting (Ensemble)",
# "Accuracy":accuracy_score(y_test, y_pred_gb),
# "AUC Score": roc_auc_score(y_test, y_pred_gb),
# "Precision": precision_score(y_test, y_pred_gb),
# "Recall": recall_score(y_test, y_pred_gb),
# "F1 Score": f1_score(y_test, y_pred_gb),
# "MCC Score": matthews_corrcoef(y_test, y_pred_gb)})

# Convert results to DataFrame for better visualization
# results_df = pd.DataFrame(results) 

# print(results_df)


print("All models saved successfully.")