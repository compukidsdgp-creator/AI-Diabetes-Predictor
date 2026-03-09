# pip install streamlit pandas scikit-learn plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes AI System", layout="wide")

st.title("🩺 Diabetes Prediction System")
st.markdown("AI based clinical decision support tool")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# TRAIN MODEL
@st.cache_resource
def train_model(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    return model, x_test, y_test

model, x_test, y_test = train_model(df)

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Dataset Analysis", "Upload CSV", "Manual Prediction", "Model Performance"]
)

# DASHBOARD
if page == "Dashboard":

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", len(df))
    col2.metric("Avg Glucose", round(df["Glucose"].mean(),2))
    col3.metric("Avg BMI", round(df["BMI"].mean(),2))

    fig = px.pie(df, names="Outcome", title="Diabetes Distribution")
    st.plotly_chart(fig, use_container_width=True)

# DATASET ANALYSIS
elif page == "Dataset Analysis":

    st.subheader("Dataset Preview")
    st.dataframe(df)

    fig = px.histogram(df, x="Glucose", color="Outcome")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="Age", y="BMI", color="Outcome")
    st.plotly_chart(fig2, use_container_width=True)

# CSV PREDICTION
elif page == "Upload CSV":

    st.subheader("Upload Patient CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        user_data = pd.read_csv(uploaded_file)

        st.write(user_data)

        prediction = model.predict(user_data)

        probability = model.predict_proba(user_data)

        user_data["Prediction"] = prediction
        user_data["Risk %"] = probability[:,1]*100

        user_data["Result"] = user_data["Prediction"].map(
            {0:"Not Diabetic",1:"Diabetic"}
        )

        st.write(user_data)

        st.download_button(
            "Download Results",
            user_data.to_csv(index=False),
            "prediction_results.csv"
        )

# MANUAL PREDICTION
elif page == "Manual Prediction":

    st.subheader("Enter Patient Data")

    col1, col2 = st.columns(2)

    preg = col1.number_input("Pregnancies",0,20)
    glucose = col1.number_input("Glucose",0,200)
    bp = col1.number_input("Blood Pressure",0,150)
    skin = col1.number_input("Skin Thickness",0,100)

    insulin = col2.number_input("Insulin",0,900)
    bmi = col2.number_input("BMI",0.0,70.0)
    dpf = col2.number_input("Diabetes Pedigree Function",0.0,3.0)
    age = col2.number_input("Age",1,120)

    if st.button("Predict"):

        data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])

        pred = model.predict(data)
        prob = model.predict_proba(data)

        if pred[0]==1:
            st.error(f"High Risk of Diabetes : {round(prob[0][1]*100,2)}%")
        else:
            st.success(f"Low Risk : {round(prob[0][1]*100,2)}%")

# MODEL PERFORMANCE
elif page == "Model Performance":

    acc = accuracy_score(y_test, model.predict(x_test))

    st.metric("Model Accuracy", str(round(acc*100,2))+" %")
