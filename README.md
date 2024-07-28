# Bank Dataset Analysis and Decision Tree Classifier

This README provides an overview of the steps taken to analyze the bank dataset, including data cleaning, exploratory data analysis (EDA), and building a decision tree classifier for predicting the target variable. The guide also includes visualizations to help understand the results.
**Table of Contents**

1.Setup
2.Data Loading
3.Data Cleaning
4.Exploratory Data Analysis (EDA)
5.Decision Tree Classifier
6.Visualizations

**Setup**

Ensure you have the required libraries installed. You can install them using pip:

```sh
pip install pandas scikit-learn matplotlib
```
## Data Loading

Load the dataset from an Excel file. Ensure the file path is correct.

```sh
import pandas as pd

x = "/home/pulicherla/Documents/bank.xlsx"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df = pd.read_excel(x)
print(df.head())
```

## Data Cleaning
**Removing Missing Values**

Remove rows with missing values and count the number of missing values.

```sh
# Remove missing values
df_cleaned = df.dropna()
print(df_cleaned.head())

# Count missing values
missing_values_count = df.isnull().sum()
print(missing_values_count)
```
**Identifying Duplicate Data**

Find duplicate rows in the dataset.

```sh
# Check for duplicates
duplicates = df.duplicated()
print(duplicates.head())
```

## Exploratory Data Analysis (EDA)

Perform EDA to understand the distribution and relationships within the data.

```sh
# Summary statistics
print(df.describe())

# Check data types
print(df.dtypes)

# Countplot for the target variable 'y'
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=df)
plt.title('Distribution of Target Variable y')
plt.show()
```

## Decision Tree Classifier

Build and evaluate a decision tree classifier to predict the target variable y.

```sh
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset
x = "/home/pulicherla/Documents/bank.xlsx"
df = pd.read_excel(x)

# Preprocess the data: Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Define the feature matrix X and the target vector y
X = df.drop(columns=['y_yes'])
y = df['y_yes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=12)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.show()
```

## Visualizations

Visualizations help understand the structure and performance of the decision tree.
**Decision Tree Visualization**

```sh
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.show()
```

## Conclusion

This guide outlines the key steps for loading, cleaning, analyzing, and building a decision tree classifier using the bank dataset. The decision tree model helps predict the likelihood of a target variable, providing valuable insights for further decision-making.


