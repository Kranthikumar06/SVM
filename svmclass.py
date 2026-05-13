import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Page configuration
st.set_page_config(page_title="Cancer Detection System", layout="centered")

# Title
st.title("🏥 Cancer Detection System")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Breatscancer_dataset2.csv')
    return df

# Data preprocessing
@st.cache_resource
def preprocess_and_train_model():
    df = load_data()
    df_processed = df.copy()
    
    # Remove outliers using IQR method
    outliers = {'Resistin', 'Adiponectin', 'MCP.1', 'Leptin', 'Insulin', 'Glucose', 'HOMA'}
    for column in outliers:
        if column in df_processed.columns:
            Q1 = df_processed[column].quantile(0.25)
            Q3 = df_processed[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[column] = df_processed[column].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
    
    # Fill null values with median
    for column in df_processed.columns:
        median_value = df_processed[column].median()
        df_processed[column].fillna(median_value, inplace=True)
    
    # Train-test split
    X = df_processed.drop(['Classification', 'ID'], axis=1)
    y = df_processed['Classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM model
    model = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    
    return model, X_train.columns, X_train

# Load model and data
model, feature_columns, X_train = preprocess_and_train_model()

st.write("Enter patient values to check for cancer risk:")
st.divider()

col1, col2, col3 = st.columns(3)
input_values = {}

with col1:
    for feature in feature_columns[:3]:
        input_values[feature] = st.number_input(f"{feature}", value=float(X_train[feature].mean()), step=0.1)

with col2:
    for feature in feature_columns[3:6]:
        input_values[feature] = st.number_input(f"{feature}", value=float(X_train[feature].mean()), step=0.1)

with col3:
    for feature in feature_columns[6:]:
        input_values[feature] = st.number_input(f"{feature}", value=float(X_train[feature].mean()), step=0.1)

st.divider()

if st.button("Check Cancer Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([input_values])
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("⚠️ **CANCER DETECTED** - High Risk")
    else:
        st.success("✅ **NO CANCER** - Low Risk")
