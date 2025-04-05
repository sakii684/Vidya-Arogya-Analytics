# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# import joblib

# # Load dataset
# df = pd.read_csv("indian_student_dataset.csv")

# # Convert categorical variable 'Gender' to numerical
# df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# # Define features and target variables
# features = ["Age", "Gender", "Hours_Sleep", "Screen_Time", "Physical_Activity", "Healthy_Diet_Score", "Study_Hours"]
# X = df[features]
# y_performance = df["Academic_Performance"]
# y_stress = df["Stress_Level"]

# # Normalize data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data into training and testing sets
# X_train, X_test, y_perf_train, y_perf_test = train_test_split(X_scaled, y_performance, test_size=0.2, random_state=42)
# X_train, X_test, y_stress_train, y_stress_test = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)

# # Hyperparameter tuning for Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }

# rf = RandomForestRegressor(random_state=42)
# perf_model = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# stress_model = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# perf_model.fit(X_train, y_perf_train)
# stress_model.fit(X_train, y_stress_train)

# # Save models and scaler
# joblib.dump(perf_model.best_estimator_, "performance_model.pkl")
# joblib.dump(stress_model.best_estimator_, "stress_model.pkl")
# joblib.dump(scaler, "scaler.pkl")

# # Function for prediction and visualization
# def predict_performance_and_stress(age, gender, sleep, screen_time, activity, diet_score, study_hours):
#     gender = 0 if gender.lower() == "male" else 1
#     input_data = np.array([[age, gender, sleep, screen_time, activity, diet_score, study_hours]])
    
#     scaler = joblib.load("scaler.pkl")
#     input_data_scaled = scaler.transform(input_data)
    
#     performance_model = joblib.load("performance_model.pkl")
#     stress_model = joblib.load("stress_model.pkl")
    
#     predicted_performance = performance_model.predict(input_data_scaled)[0]
#     predicted_stress = stress_model.predict(input_data_scaled)[0]
    
#     # Categorize GPA
#     def categorize_gpa(gpa):
#         if gpa >= 3.7:
#             return "Excellent"
#         elif gpa >= 3.3:
#             return "Good"
#         elif gpa >= 3.0:
#             return "Satisfactory"
#         elif gpa >= 2.5:
#             return "Average"
#         else:
#             return "Needs Improvement"
    
#     gpa_category = categorize_gpa(predicted_performance)
    
#     return round(predicted_performance, 2), round(predicted_stress, 2), gpa_category


# # Streamlit app
# st.set_page_config(page_title="Student AI Prediction", layout="wide")

# # Sidebar navigation
# page = st.sidebar.radio("Navigation", ["Guidelines", "Prediction"])

# if page == "Guidelines":

#     st.title("Guidelines for AI Usage")
#     st.header("Good AI vs. Bad AI")
#     st.write("### **Good AI:**")
#     st.write("- Ethical, unbiased, and transparent.")
#     st.write("- Helps in decision-making without discrimination.")
#     st.write("- Enhances productivity and well-being.")
#     st.write("- Provides accurate and fair predictions.")
    
#     st.write("### **Bad AI:**")
#     st.write("- Biased, unfair, or discriminatory.")
#     st.write("- Misuses personal data.")
#     st.write("- Generates misleading or harmful predictions.")
#     st.write("- Lacks accountability and transparency.")
    
#     st.write("By understanding these aspects, we can develop AI responsibly for student well-being.")

   

# elif page == 'Prediction':
#     st.title("Vidya Arogya Analytics")

#     # Input fields
#     age = st.number_input("Age", min_value=18, max_value=25, value=20, step=1)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     sleep = st.number_input("Hours of Sleep", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
#     screen_time = st.number_input("Screen Time (hours)", min_value=1.0, max_value=8.0, value=4.0, step=0.1)
#     activity = st.number_input("Physical Activity (hours per week)", min_value=0.0, max_value=7.0, value=3.0, step=0.1)
#     diet_score = st.number_input("Healthy Diet Score (1-10)", min_value=1, max_value=10, value=7, step=1)
#     study_hours = st.number_input("Study Hours per Day", min_value=1.0, max_value=6.0, value=4.5, step=0.1)

#     # Button to trigger prediction
#     if st.button("Predict"):
#         predicted_gpa, predicted_stress, gpa_category = predict_performance_and_stress(
#             age, gender, sleep, screen_time, activity, diet_score, study_hours
#         )

#         st.success(f"Predicted GPA: {predicted_gpa:.2f} ({gpa_category})")
#         st.success(f"Predicted Stress Level: {predicted_stress:.2f}")


        
    
   
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Streamlit app settings
st.set_page_config(page_title="Student AI Prediction", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("indian_student_dataset.csv")
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df

df = load_data()

# Define features & target variables
features = ["Age", "Gender", "Hours_Sleep", "Screen_Time", "Physical_Activity", "Healthy_Diet_Score", "Study_Hours"]
X = df[features]
y_performance = df["Academic_Performance"]
y_stress = df["Stress_Level"]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models if they don't exist
if not os.path.exists("performance_model.pkl") or not os.path.exists("stress_model.pkl"):
    X_train, X_test, y_perf_train, y_perf_test = train_test_split(X_scaled, y_performance, test_size=0.2, random_state=42)
    X_train, X_test, y_stress_train, y_stress_test = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)

    rf_perf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_stress = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    rf_perf.fit(X_train, y_perf_train)
    rf_stress.fit(X_train, y_stress_train)

    joblib.dump(rf_perf, "performance_model.pkl")
    joblib.dump(rf_stress, "stress_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# Load models & scaler once using caching
@st.cache_resource
def load_models():
    return (
        joblib.load("performance_model.pkl"),
        joblib.load("stress_model.pkl"),
        joblib.load("scaler.pkl"),
    )

performance_model, stress_model, scaler = load_models()

# Prediction function
def predict_performance_and_stress(age, gender, sleep, screen_time, activity, diet_score, study_hours):
    gender = 0 if gender.lower() == "male" else 1
    input_data = np.array([[age, gender, sleep, screen_time, activity, diet_score, study_hours]])
    
    input_data_scaled = scaler.transform(input_data)

    predicted_performance = performance_model.predict(input_data_scaled)[0]
    predicted_stress = stress_model.predict(input_data_scaled)[0]

    def categorize_gpa(gpa):
        if gpa >= 3.7: return "Excellent"
        elif gpa >= 3.3: return "Good"
        elif gpa >= 3.0: return "Satisfactory"
        elif gpa >= 2.5: return "Average"
        else: return "Needs Improvement"

    return round(predicted_performance, 2), round(predicted_stress, 2), categorize_gpa(predicted_performance)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Guidelines", "Prediction"])

if page == "Guidelines":
    st.title("Guidelines for AI Usage")
    st.header("Good AI vs. Bad AI")
    st.write("### **Good AI:**")
    st.write("- Ethical, unbiased, and transparent.")
    st.write("- Helps in decision-making without discrimination.")
    st.write("- Enhances productivity and well-being.")
    st.write("- Provides accurate and fair predictions.")
    
    st.write("### **Bad AI:**")
    st.write("- Biased, unfair, or discriminatory.")
    st.write("- Misuses personal data.")
    st.write("- Generates misleading or harmful predictions.")
    st.write("- Lacks accountability and transparency.")
    
    st.write("By understanding these aspects, we can develop AI responsibly for student well-being.")
    
elif page == 'Prediction':
    st.title("Vidya Arogya Analytics")

    age = st.number_input("Age", min_value=18, max_value=25, value=20, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sleep = st.number_input("Hours of Sleep", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
    screen_time = st.number_input("Screen Time (hours)", min_value=1.0, max_value=8.0, value=4.0, step=0.1)
    activity = st.number_input("Physical Activity (hours per week)", min_value=0.0, max_value=7.0, value=3.0, step=0.1)
    diet_score = st.number_input("Healthy Diet Score (1-10)", min_value=1, max_value=10, value=7, step=1)
    study_hours = st.number_input("Study Hours per Day", min_value=1.0, max_value=6.0, value=4.5, step=0.1)

    if st.button("Predict"):
        predicted_gpa, predicted_stress, gpa_category = predict_performance_and_stress(
            age, gender, sleep, screen_time, activity, diet_score, study_hours
        )
        st.success(f"Predicted GPA: {predicted_gpa:.2f} ({gpa_category})")
        st.success(f"Predicted Stress Level: {predicted_stress:.2f}")


