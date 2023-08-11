import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder


model_file_path_catboost = os.path.join(os.path.dirname(__file__), 'catboost_model.joblib')

if os.path.exists(model_file_path_catboost):
    model_catboost = joblib.load(model_file_path_catboost)
else:
    print(f"Error: The file '{model_file_path_catboost}' does not exist.")

model_file_path_gradient = os.path.join(os.path.dirname(__file__), 'gradient_boosting_model.joblib')

if os.path.exists(model_file_path_gradient):
    model_gradient = joblib.load(model_file_path_gradient)
else:
    st.error(f"The file '{model_file_path_gradient}' does not exist.")

csv_file_path = os.path.join(os.path.dirname(__file__), 'cleaned1_catboost.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
else:
    print(f"Error: The file '{csv_file_path}' does not exist.")
    
unique_locations = df['location'].unique()
unique_furnished = df['furnished'].unique()
unique_property_type = df['property_type'].unique()

st.title("KL/Selangor Rent Prediction with Regression Models")

st.sidebar.title("Choose Model")

model_choice = st.sidebar.radio("Models:", ("CatBoost", "Gradient Boosting"))

categorical_columns = ['location', 'furnished', 'property_type']

categorical_feature_indices = [0, 1, 2]

justified_text = """
<div style="text-align: justify; line-height: 2.0;">
    This is a simple project to predict the monthly rate for high-rise accommodation within the region of Selangor and Kuala Lumpur, Malaysia. Regression models such as CatBoost and Decision Tree have been tested on this dataset.
</div>

<div style="text-align: justify; line-height: 2.0;">
    However, the results are still not viable in a real-life scenario as all the models have shown signs of overfitting, even after applying various data preprocessing techniques and hyperparameter tuning. This is most likely due to the inconsistency of the dataset itself.
</div>

<div style="text-align: justify; line-height: 2.0;">
    The predictions tend to fall off for units having 'premium' features and are supposedly to have expensive rents.
</div>

<div style="text-align: justify; line-height: 2.0;">
    Further contributions are welcomed to improve the model in the GitHub link.
</div>
"""
st.sidebar.markdown(justified_text, unsafe_allow_html = True)

github_link = "[GitHub Link](https://github.com/clifford96/ML-rent)"

st.sidebar.markdown(github_link, unsafe_allow_html = True)

option1_location = st.selectbox("Location", unique_locations)

option2_furnished = st.selectbox("Furnished Status", unique_furnished)

option3_property_type = st.selectbox("Property Type", unique_property_type)

option4_size_sqf = st.slider("Size (Square Feet)", min_value = 400, max_value = 4000, value = 1000)

option5_completion_year = st.slider("Completion Year", min_value = 1970, max_value = 2025, value = 2020)

option6_number_facilities = st.slider("Number of Facilities", min_value = 0, max_value = 14, value = 1)

option7_rooms = st.slider("Number of Rooms", min_value = 1, max_value = 9, value = 1)

option8_bathrooms = st.slider("Number of Bathrooms", min_value = 1, max_value = 9, value = 1)

option9_parking = st.slider("Number of Parking", min_value = 0, max_value = 3, value = 1)

option10_public_transport = st.slider("Nearby Public Transport", min_value = 0, max_value = 1, value = 0)

if model_choice == "CatBoost":
    model_path = model_file_path_catboost
    categorical_columns_for_model = categorical_feature_indices
    categorical_encoder = None
elif model_choice == "Gradient Boosting":
    model_path = model_file_path_gradient
    categorical_columns_for_model = categorical_columns
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_encoder.fit(df[categorical_columns_for_model])
else:
    st.error("Invalid model choice")

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"The file '{model_path}' does not exist.")
    
input_data_catboost = [[
                   option1_location,
                   option2_furnished,
                   option3_property_type,
                   option4_size_sqf,
                   option5_completion_year,
                   option6_number_facilities,
                   option7_rooms,
                   option8_bathrooms,
                   option9_parking,
                   option10_public_transport]]

input_df_ohe = None

if model_choice == "CatBoost":
    input_data = input_data_catboost
else:
    input_data_ohe = {
        'size_sqf': option4_size_sqf,
        'completion_year': option5_completion_year,
        'number_facilities': option6_number_facilities,
        'rooms': option7_rooms,
        'bathrooms': option8_bathrooms,
        'parking': option9_parking,
        'public_transport': option10_public_transport
    }
    input_df_ohe = pd.DataFrame([input_data_ohe])

    if categorical_encoder:
        categorical_data = {
            'location': [option1_location],
            'furnished': [option2_furnished],
            'property_type': [option3_property_type]
        }
        input_categorical_df = pd.DataFrame(categorical_data)
        input_categorical_encoded = categorical_encoder.transform(input_categorical_df).toarray()
        input_categorical_encoded_df = pd.DataFrame(input_categorical_encoded)
        input_df_ohe = pd.concat([input_categorical_encoded_df, input_df_ohe], axis=1)

if st.button("Predict Rent"):
    if model_choice == "CatBoost":
        predicted_price = model.predict(input_data_catboost)[0]
    else:
        predicted_price = model.predict(input_df_ohe.values)[0]
    st.write(f"Predicted Price: RM{predicted_price:.2f} per month")






