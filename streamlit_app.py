import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.title("Fraud Detection App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # Select target column
    target_column = st.selectbox("Select the target column", df.columns)

    if target_column:
        # Visualize class distribution
        st.subheader("1. Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=target_column, ax=ax)
        st.pyplot(fig)

        # Model selection
        st.subheader("2. Choose a Classification Model")
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # Prepare features and labels
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        

        # Encode categorical features
        X = pd.get_dummies(X)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model.fit(X_train, y_train)
        
        import os
import joblib

if os.path.exists('fraud_detection_model.pkl'):
    model = joblib.load('fraud_detection_model.pkl')
    st.success("Model loaded successfully")
else:
    st.warning("Model file not found. Please train and save the model first.")


        # Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        st.subheader("3. Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        # Classification Report
        st.subheader("4. Classification Report")
        st.text(classification_report(y_test, y_pred))

        # ROC-AUC Score
        st.subheader("5. ROC-AUC Score")
        st.write(f"{roc_auc_score(y_test, y_proba):.2f}")




