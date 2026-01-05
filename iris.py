import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')

print(df.head())
print(df.info())



# Check missing values
print(df.isnull().sum())

# Remove missing values (if any)
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()



# Statistical summary
print(df.describe())

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# Histograms
df[numerical_cols].hist(figsize=(10,6))
plt.show()

# Correlation matrix (FIX)
corr = df[numerical_cols].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()



# Scatter plot
plt.scatter(df['sepal_length'], df['petal_length'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Bar chart
df['species'].value_counts().plot(kind='bar')
plt.show()

# Boxplot
sns.boxplot(x='species', y='sepal_length', data=df)
plt.show()

# Pairplot
sns.pairplot(df, hue='species')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

