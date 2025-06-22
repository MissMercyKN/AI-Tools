# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Optional: View first few rows
print("Initial data sample:")
print(df.head())

# Step 2: Handle missing values (if any)
# For demonstration, let's introduce some missing values manually (optional step)
df.loc[0:2, 'sepal length (cm)'] = np.nan

# Use SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Step 3: Encode labels (target)
# In this case, the dataset is already encoded (0, 1, 2), but if it were strings:
# encoder = LabelEncoder()
# df['species'] = encoder.fit_transform(df['species'])

# Step 4: Split dataset into training and testing sets
X = df.iloc[:, :-1]  # Features
y = df['species']    # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # macro = average per class
recall = recall_score(y_test, y_pred, average='macro')

# Display evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
