import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

# Streamlit app
st.title('Admission Prediction')

# Load dataset
st.subheader('Load and Preview Dataset')
df = pd.read_csv('admission_predict.csv')
st.dataframe(df.head())

# Basic Data Info
st.subheader('Dataset Information')
st.write('Shape of the dataset:', df.shape)
st.write('Columns:', df.columns.tolist())
st.write('Data Types:', df.dtypes)
st.write('Missing Values:', df.isnull().sum())
st.write('Basic Statistics:')
st.write(df.describe().T)

# Rename columns
df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'Probability'})
st.write('Renamed Columns Preview:')
st.dataframe(df.head())

# Visualization
st.subheader('Data Visualizations')
features = ['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
for feature in features:
    fig, ax = plt.subplots()
    ax.hist(df[feature], rwidth=0.7)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Data Preprocessing
df.drop('Serial No.', axis='columns', inplace=True)
df_copy = df.copy(deep=True)
df_copy[['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA']] = df_copy[['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA']].replace(0, np.NaN)

# Splitting the dataset
X = df_copy.drop('Probability', axis='columns')
y = df_copy['Probability']

# Model Selection using GridSearchCV
st.subheader('Model Selection with GridSearchCV')
def find_best_model(X, y):
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'parameters': {'normalize': [True, False]}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(),
            'parameters': {'criterion': ['mse', 'friedman_mse'], 'splitter': ['best', 'random']}
        },
        'Random Forest': {
            'model': RandomForestRegressor(),
            'parameters': {'n_estimators': [5, 10, 15, 20]}
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'parameters': {'n_neighbors': [2, 5, 10, 20]}
        }
    }

    scores = []
    for model_name, mp in models.items():
        gs = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False)
        gs.fit(X, y)
        scores.append({'model': model_name, 'best_parameters': gs.best_params_, 'score': gs.best_score_})
    return pd.DataFrame(scores)

result = find_best_model(X, y)
st.write(result)

# Model Evaluation with Cross-Validation
st.subheader('Cross-Validation')
scores = cross_val_score(LinearRegression(normalize=True), X, y, cv=5)
st.write('Highest Accuracy: {:.2f}%'.format(sum(scores) * 100 / len(scores)))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
st.write(f'Train Set Size: {len(X_train)}, Test Set Size: {len(X_test)}')

# Linear Regression Model
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
st.write('Model Test Score:', score)

# Predictions
st.subheader('Make Predictions')
GRE = st.number_input('GRE Score', min_value=0, max_value=340, value=320)
TOEFL = st.number_input('TOEFL Score', min_value=0, max_value=120, value=110)
University_Rating = st.number_input('University Rating', min_value=1, max_value=5, value=3)
SOP = st.number_input('SOP', min_value=0.0, max_value=5.0, value=3.5)
LOR = st.number_input('LOR', min_value=0.0, max_value=5.0, value=3.5)
CGPA = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0)
Research = st.selectbox('Research', [0, 1])

if st.button('Predict Chance of Admission'):
    prediction = model.predict([[GRE, TOEFL, University_Rating, SOP, LOR, CGPA, Research]])
    st.write(f'Chance of Admission: {round(prediction[0] * 100, 2)}%')
