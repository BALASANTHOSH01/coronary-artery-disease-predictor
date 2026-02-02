import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import os
import shap
import plotly.express as px

# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(name):
    return os.path.join(BASE_DIR, name)

# ---------------- LOAD MODEL ---------------- #

with open(load_file("best_model.pkl"), "rb") as f:
    model = pkl.load(f)

with open(load_file("cbe_encoder.pkl"), "rb") as f:
    encoder = pkl.load(f)

data = pd.read_csv(load_file("brfss2022_data_wrangling_output.zip"), compression="zip")
data["heart_disease"] = data["heart_disease"].map({"yes":1,"no":0})

# ---------------- PAGE CONFIG ---------------- #

icon = Image.open(load_file("heart_disease.jpg"))

st.set_page_config(
    page_title="AI Heart Disease Predictor",
    layout="wide",
    page_icon=icon
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🫀 Navigation")

section = st.sidebar.radio(
    "Go to:",
    ["Introduction", "Core App", "Team"]
)

st.sidebar.markdown("---")
st.sidebar.write("AI Powered Medical Risk Tool")

# ---------------- INTRO ---------------- #

if section == "Introduction":

    st.title("Coronary Artery Disease Predictor")

    st.markdown("""
    This AI-powered system predicts **heart disease risk percentage** 
    using advanced Machine Learning models trained on health survey data.

    ### Features
    - Accurate Risk Score
    - SHAP Explainability
    - Personalized Recommendations
    - Interactive Dashboard
    """)

# ---------------- TEAM ---------------- #

elif section == "Team":

    st.title("👨‍💻 Authors / Team")

    st.markdown("""
    **Project Team**

    • Bala Santhosh – ML & Full Stack Developer  
    • Contributors – AI Research  

    This project focuses on preventive healthcare using AI.
    """)

# ---------------- CORE APP ---------------- #

else:

    st.title("AI Heart Disease Risk Assessment")

    st.write("---")

    col1,col2,col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender",["female","male","nonbinary"])
        race = st.selectbox("Race",[
            "white_only_non_hispanic",
            "black_only_non_hispanic",
            "asian_only_non_hispanic",
            "american_indian_or_alaskan_native_only_non_hispanic",
            "multiracial_non_hispanic",
            "hispanic",
            "native_hawaiian_or_other_pacific_islander_only_non_hispanic"
        ])
        age_category = st.selectbox("Age Group",[
            "Age_18_to_24","Age_25_to_29","Age_30_to_34","Age_35_to_39",
            "Age_40_to_44","Age_45_to_49","Age_50_to_54","Age_55_to_59",
            "Age_60_to_64","Age_65_to_69","Age_70_to_74","Age_75_to_79",
            "Age_80_or_older"
        ])

    with col2:
        general_health = st.selectbox("General Health",
            ["excellent","very_good","good","fair","poor"])
        heart_attack = st.selectbox("Heart Attack History",["yes","no"])
        stroke = st.selectbox("Stroke History",["yes","no"])
        kidney_disease = st.selectbox("Kidney Disease",["yes","no"])
        diabetes = st.selectbox("Diabetes",["yes","no","no_prediabetes","yes_during_pregnancy"])

    with col3:
        bmi = st.selectbox("BMI",[
            "underweight_bmi_less_than_18_5",
            "normal_weight_bmi_18_5_to_24_9",
            "overweight_bmi_25_to_29_9",
            "obese_bmi_30_or_more"
        ])
        smoking_status = st.selectbox("Smoking Status",[
            "never_smoked",
            "former_smoker",
            "current_smoker_some_days",
            "current_smoker_every_day"
        ])
        exercise_status = st.selectbox("Exercise Past 30 Days",["yes","no"])

    sleep_category = st.selectbox("Sleep Category",[
        "very_short_sleep_0_to_3_hours",
        "short_sleep_4_to_5_hours",
        "normal_sleep_6_to_8_hours",
        "long_sleep_9_to_10_hours",
        "very_long_sleep_11_or_more_hours"
    ])

    drinks_category = st.selectbox("Alcohol Intake",[
        "did_not_drink",
        "very_low_consumption_0.01_to_1_drinks",
        "low_consumption_1.01_to_5_drinks",
        "moderate_consumption_5.01_to_10_drinks",
        "high_consumption_10.01_to_20_drinks",
        "very_high_consumption_more_than_20_drinks"
    ])

    binge_drinking_status = st.selectbox("Binge Drinking",["yes","no"])

    depressive_disorder = st.selectbox("Depressive Disorder",["yes","no"])

    physical_health = st.selectbox("Physical Health Status",[
        "zero_days_not_good",
        "1_to_13_days_not_good",
        "14_plus_days_not_good"
    ])

    mental_health = st.selectbox("Mental Health Status",[
        "zero_days_not_good",
        "1_to_13_days_not_good",
        "14_plus_days_not_good"
    ])

    walking = st.selectbox("Difficulty Walking",["yes","no"])

    length_of_time_since_last_routine_checkup = st.selectbox(
        "Last Routine Checkup",
        ["past_year","past_2_years","past_5_years","5+_years_ago","never"]
    )

    could_not_afford_to_see_doctor = st.selectbox(
        "Couldn't Afford Doctor",["yes","no"]
    )

    health_care_provider = st.selectbox(
        "Healthcare Provider",
        ["yes_only_one","more_than_one","no"]
    )

    asthma = st.selectbox(
        "Asthma Status",
        ["never_asthma","current_asthma","former_asthma"]
    )

    # ---------------- INPUT DICT ---------------- #

    input_data = {
        'gender': gender,
        'race': race,
        'general_health': general_health,
        'health_care_provider': health_care_provider,
        'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
        'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
        'ever_diagnosed_with_heart_attack': heart_attack,
        'ever_diagnosed_with_a_stroke': stroke,
        'ever_told_you_had_a_depressive_disorder': depressive_disorder,
        'ever_told_you_have_kidney_disease': kidney_disease,
        'ever_told_you_had_diabetes': diabetes,
        'BMI': bmi,
        'difficulty_walking_or_climbing_stairs': walking,
        'physical_health_status': physical_health,
        'mental_health_status': mental_health,
        'asthma_Status': asthma,
        'smoking_status': smoking_status,
        'binge_drinking_status': binge_drinking_status,
        'exercise_status_in_past_30_Days': exercise_status,
        'age_category': age_category,
        'sleep_category': sleep_category,
        'drinks_category': drinks_category
    }

    # ---------------- PREDICTION ---------------- #

    if st.button("Get Risk Assessment"):

        df = pd.DataFrame([input_data])
        encoded = encoder.transform(df)

        risk = model.predict_proba(encoded)[0][1] * 100

        st.success(f"Predicted Heart Disease Risk: {risk:.2f}%")

        # ---------------- SHAP FIXED ---------------- #

        lgbm_model = model.estimators_[0].steps[-1][1]
        explainer = shap.TreeExplainer(lgbm_model)

        shap_values = explainer.shap_values(encoded)

        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        importance = np.abs(shap_vals).sum(axis=0)
        importance = importance / importance.sum() * 100

        imp_df = pd.DataFrame({
            "Feature": encoded.columns,
            "Importance": importance
        }).sort_values(by="Importance",ascending=False)

        top = imp_df.head(6)

        fig = px.pie(top,names="Feature",values="Importance")
        st.plotly_chart(fig)

        # ---------------- RECOMMENDATIONS ---------------- #

        st.write("### Personalized Recommendations")

        for _,row in top.iterrows():

            f = row["Feature"]

            if "smoking" in f:
                st.write("🚭 Quit smoking to reduce heart disease risk.")
            elif "BMI" in f:
                st.write("⚖ Maintain healthy weight through diet & exercise.")
            elif "exercise" in f:
                st.write("🏃 Increase physical activity regularly.")
            elif "sleep" in f:
                st.write("😴 Aim for 7-9 hours sleep daily.")
            elif "diabetes" in f:
                st.write("🩺 Control blood sugar levels carefully.")
            else:
                st.write("❤️ Follow healthy lifestyle habits.")

        st.write("---")
        st.write("⚠ This tool is not a medical diagnosis. Consult doctors.")

