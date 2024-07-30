import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
df = pd.read_csv('HR_Analytics.csv')

# Display basic information and statistics
print(df.info())
print(df.describe())
print(df.head())

# Handle Missing Values
df = df.dropna()

# Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Attrition_Yes', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns.drop('Attrition_Yes'))

# Split the Data
X = df_scaled
y = df['Attrition_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('Accuracy:', accuracy_score(y_test, predictions))

# Visualize the Results
# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Pairplot for selected features
selected_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'Attrition_Yes']
sns.pairplot(df[selected_features], hue='Attrition_Yes')
plt.title('Pairplot for Selected Features')
plt.show()
