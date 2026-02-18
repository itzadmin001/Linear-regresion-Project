import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Scaler ---
# Make sure these files ('KNN.pkl' and 'scaler.pkl') are in the same directory
# as your Streamlit app file (e.g., app.py) or provide the full path.
try:
    model = pickle.load(open('knn_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Sorry, model or scaler files not found. Please ensure 'knn_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

# --- Custom CSS for a more modern look ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSidebar .stSlider {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSelectbox>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .css-1d3w5wq {
        background-color: #1a1a1a; /* Sidebar background color */
        color: white;
    }
    .css-1lcbmhc, .css-1aumxmz {
        color: #333333; /* Text color in main content */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- Title and Description ---
st.title('🏡 California House Price Predictor')
st.markdown('Use this application to estimate house prices in California.')
st.write('---')

# --- Sidebar for User Input ---
st.sidebar.header('Input Features')
st.sidebar.markdown('Use sliders and options to adjust house details.')

def user_input_features():
    # Using actual min/max values from df.describe() and mean/median for default
    st.sidebar.markdown('**Longitude**')
    longitude = st.sidebar.slider(' ', -124.35, -114.31, -119.57, step=0.01)
    st.sidebar.markdown('**Latitude**')
    latitude = st.sidebar.slider(' ', 32.54, 41.95, 35.63, step=0.01)
    st.sidebar.markdown('**Housing Median Age**')
    housing_median_age = st.sidebar.slider(' ', 1.0, 52.0, 29.0, step=1.0)
    st.sidebar.markdown('**Total Rooms**')
    total_rooms = st.sidebar.slider(' ', 2.0, 39320.0, 2127.0, step=10.0)
    st.sidebar.markdown('**Total Bedrooms**')
    total_bedrooms = st.sidebar.slider(' ', 1.0, 6445.0, 435.0, step=1.0)
    st.sidebar.markdown('**Population**')
    population = st.sidebar.slider(' ', 3.0, 35682.0, 1166.0, step=1.0)
    st.sidebar.markdown('**Households**')
    households = st.sidebar.slider(' ', 1.0, 6082.0, 409.0, step=1.0)
    st.sidebar.markdown('**Median Income**')
    median_income = st.sidebar.slider(' ', 0.4999, 15.0001, 3.5348, step=0.0001)
    
    # Derived features - crucial to ensure they are calculated consistently
    # Use default values if households or total_rooms are zero to avoid division by zero
    rooms_per_household = total_rooms / households if households != 0 else 0
    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms != 0 else 0
    population_per_household = population / households if households != 0 else 0

    # Ocean Proximity selection (matching one-hot encoded columns)
    ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    st.sidebar.markdown('**Ocean Proximity**')
    selected_ocean_proximity = st.sidebar.selectbox(' ', ocean_proximity_options)

    # Prepare the input dictionary
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household
    }
    
    # Add one-hot encoded columns for ocean_proximity
    # Ensure all possible one-hot columns are present and initialized to 0
    one_hot_cols = [
        'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 
        'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 
        'ocean_proximity_NEAR OCEAN'
    ]
    for col in one_hot_cols:
        data[col] = 0 # Initialize all to 0
    
    # Set the selected ocean_proximity to 1
    if f'ocean_proximity_{selected_ocean_proximity}' in data:
        data[f'ocean_proximity_{selected_ocean_proximity}'] = 1

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Feature name labels for user clarity ---
feature_labels = {
    'longitude': 'Longitude (geographic coordinate)',
    'latitude': 'Latitude (geographic coordinate)',
    'housing_median_age': 'Housing Median Age (years)',
    'total_rooms': 'Total Rooms',
    'total_bedrooms': 'Total Bedrooms',
    'population': 'Population',
    'households': 'Households',
    'median_income': 'Median Income (median income for block; original dataset scale)',
    'rooms_per_household': 'Rooms per Household',
    'bedrooms_per_room': 'Bedrooms per Room',
    'population_per_household': 'Population per Household',
    'ocean_proximity_<1H OCEAN': "Ocean Proximity: <1H OCEAN",
    'ocean_proximity_INLAND': 'Ocean Proximity: INLAND',
    'ocean_proximity_ISLAND': 'Ocean Proximity: ISLAND',
    'ocean_proximity_NEAR BAY': 'Ocean Proximity: NEAR BAY',
    'ocean_proximity_NEAR OCEAN': 'Ocean Proximity: NEAR OCEAN'
}

with st.sidebar.expander('Feature descriptions', expanded=False):
    for col, desc in feature_labels.items():
        st.write(f"**{col}**: {desc}")

# --- Display User Input ---
st.subheader('Your Input Features')
st.dataframe(input_df.style.set_properties(**{'background-color': '#e6f7ff', 'color': 'black'}), hide_index=True)
st.write('---')

# --- Prediction Button and Logic ---
if st.button('Predict House Price', key='predict_button'):
    # Scale the input features using the loaded scaler
    # Ensure the order of columns matches the training data
    # The order of columns must be consistent with X_train.columns used during training
    # You might need to explicitly reorder `input_df` columns if the model expects a specific order
    # For this example, assuming input_df column order matches X_train.columns
    
    # A more robust way to ensure column order:
    # Get original column order from a dummy dataframe or X_train.columns if available
    # For now, let's assume `input_df` has the same column order as `X` during training
    
    # Manually reordering columns to match the training data's X.columns
    # This is crucial for correct predictions
    expected_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
        'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
        'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
    ]
    # Add missing columns with 0 if any (e.g., if a new one-hot category appears that wasn't in training)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns] # Reorder to match training

    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)
    pred_value = float(prediction[0])
    clamped = False
    if pred_value < 0:
        clamped = True
        pred_value = 0.0

    st.subheader('💰 Estimated House Price')
    if clamped:
        st.warning('Model predicted a negative price; displayed value set to $0. This may indicate a model/scaler mismatch or inputs outside the training distribution.')
    st.success(f'Based on the features you provided, the estimated price is: **${pred_value:,.2f}**')
    st.balloons()

# --- Footer ---
st.markdown('---')
st.info('This app is based on a KNN (K-Nearest Neighbors) model trained on the California housing dataset.')
