import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Title
st.title("Fraud Detection System with Anomaly Detection")

# Sidebar navigation
menu = ["Home", "Train Model", "Predict Fraud", "Visualizations"]
choice = st.sidebar.selectbox("Menu", menu)

# Load and preprocess data
@st.cache_data
def load_data():
    data_path = "Fraud data FY 2023-24 for B&CC (1).xlsx"
    data = pd.read_excel(data_path, sheet_name='Fraud data')
    return data

# Preprocessing pipeline
def preprocess_data(data):
    label_encoders = {}
    imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()

    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    X = data.drop(columns=['Fraud Category'])
    y = data['Fraud Category']
    y = LabelEncoder().fit_transform(y)  # Ensure y is encoded consistently
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    return X, y, label_encoders, scaler, imputer, data.columns.drop('Fraud Category')

# Train model with hyperparameter tuning and anomaly detection
def train_model(data):
    X, y, label_encoders, scaler, imputer, feature_columns = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    iso_forest = IsolationForest(random_state=42, contamination=0.1)
    iso_forest.fit(X)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == best_model.n_classes_:
        auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
    else:
        auc_score = "N/A"

    return best_model, iso_forest, label_encoders, accuracy, auc_score, X_test, y_test, scaler, imputer, feature_columns

# Load the model and preprocessors
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

# Main logic
if choice == "Home":
    st.write("Welcome to the Fraud Detection System with Anomaly Detection.")
    st.write("Navigate through the menu to train the model, predict fraud, or visualize the data.")

elif choice == "Train Model":
    data = load_data()
    st.subheader("Training Data")
    st.write(data.head())  # Display a preview of the data

    if st.button("Train Model"):
        model, iso_forest, label_encoders, accuracy, auc_score, X_test, y_test, scaler, imputer, feature_columns = train_model(data)
        st.success(f"Model trained successfully with an accuracy of {accuracy:.2f} and ROC AUC Score of {auc_score}")

        # Save the model and preprocessors
        joblib.dump((model, iso_forest, label_encoders, scaler, imputer, feature_columns), "fraud_detection_model.pkl")

        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

elif choice == "Predict Fraud":
    st.subheader("Enter Policy Details to Predict Fraud and Anomaly Detection")
    assured_age = st.number_input("Assured Age", 18, 60, step=1)
    nominee_relation = st.selectbox("Nominee Relation", ['Mother', 'Wife', 'Husband', 'Son', 'Daughter'])
    occupation = st.selectbox("Occupation", ['Service', 'Business', 'Agriculturist', 'Self-Employed', 'Housewife'])
    policy_sum_assured = st.number_input("Policy Sum Assured", 50000, 5000000, step=5000)
    premium = st.number_input("Premium Amount", 5000, 100000, step=100)
    payment_mode = st.selectbox("Premium Payment Mode", ['Monthly', 'Quarterly', 'Yearly'])
    annual_income = st.number_input("Annual Income", 100000, 5000000, step=10000)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married'])

    if st.button("Predict Fraud"):
        model, iso_forest, label_encoders, scaler, imputer, feature_columns = load_model()

        input_data = pd.DataFrame([{
            'ASSURED_AGE': assured_age,
            'NOMINEE_RELATION': nominee_relation,
            'OCCUPATION': occupation,
            'POLICY_SUMASSURED': policy_sum_assured,
            'Premium': premium,
            'PREMIUMPAYMENTMODE': payment_mode,
            'Annual_Income': annual_income,
            'HOLDERMARITALSTATUS': marital_status
        }])

        # Ensure all necessary columns are present with default values
        default_values = {col: 0 for col in feature_columns}
        input_data = input_data.assign(**default_values)
        input_data = input_data[feature_columns]

        for col, le in label_encoders.items():
            if col in input_data:
                try:
                    input_data[col] = le.transform(input_data[col].astype(str))
                except ValueError:
                    input_data[col] = le.transform([le.classes_[0]])[0]

        input_data = scaler.transform(imputer.transform(input_data.values))

        prediction = model.predict(input_data)
        anomaly_score = iso_forest.predict(input_data)
        is_anomaly = "Yes" if anomaly_score[0] == -1 else "No"

        st.success(f"The policy is predicted as: {'Fraud' if prediction[0] == 1 else 'Genuine'}")
        st.warning(f"Anomaly Detection Result: {is_anomaly}")

elif choice == "Visualizations":
    st.subheader("Visualize the Data")
    data = load_data()

    # Plot Age Distribution by Fraud Status
    st.subheader("Age Distribution by Fraud Status")
    fig, ax = plt.subplots()
    sns.histplot(data, x='ASSURED_AGE', hue='Fraud Category', multiple="stack", ax=ax)
    st.pyplot(fig)
