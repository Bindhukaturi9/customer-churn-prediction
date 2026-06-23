import pandas as pd
# Load the data set
#Read Dataset
df = pd.read_csv("C:/Users/KATURI BINDU/OneDrive/Desktop/ML project/ML project2/Churn.csv")
#Read Dataset
print(df.head())
#Read Dataset
print(df.isnull().sum())
df.drop('customerID', axis=1, inplace=True)

#converting text to numerical using LabelEncoder because machine doens't understand the text 
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].median()
)
from sklearn.preprocessing import LabelEncoder
for col in df.columns:
    if col != 'TotalCharges':
        if df[col].dtype != 'int64' and df[col].dtype != 'float64':

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

            print("\n",col)
            print(
                dict(
                    zip(
                        le.classes_,
                        le.transform(le.classes_)
                    )
                )
            )
print(df)
# splitting the data
X = df.drop('Churn', axis=1)

y = df['Churn'] # inputs for prediction

#output for prediction
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.2,random_state=42)

#creating a model 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42
)
#train the model 
model.fit(X_train,y_train)
#predict the model
predictions = model.predict(X_test)
#checking the accuracy 
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,predictions)

print("Accuracy:",accuracy)
# where the model is correct where the model is wrong [[TN FP][FN TP]]
# True = Correct

# False = Wrong

# Positive = Leave

# Negative = Stay
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions))
from sklearn.metrics import classification_report
# "Classification Report provides Precision, Recall, F1-Score, and Accuracy to evaluate classification model performance."
print(classification_report(y_test,predictions))

print(df.select_dtypes(include=['number']).corr()) # taking only numeric columns 
print(df.dtypes)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

# SAVE FEATURE IMPORTANCE

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print(
    "Feature Importance Saved Successfully!"
)
import matplotlib.pyplot as plt

importance.plot(
    x='Feature',
    y='Importance',
    kind='bar'
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.show()
import joblib

feature_list = X.columns

joblib.dump((model, feature_list), "churn_model.pkl")

print("Model Saved Successfully!")

print(X.columns)
print(len(X.columns))
print(X.columns)