import streamlit as st
import numpy as np
import pandas as pd
# Import ALL seven classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier 
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image # Essential for image manipulation
from sklearn.preprocessing import LabelEncoder 

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION AND STYLING (The New Theme) ---
st.set_page_config(
    page_title="🌱 Agri Master Crop Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Theme Colors based on your website's look
DARK_BACKGROUND = "#0D1117"
MAIN_GREEN = "#00AA44"
LIGHT_GREEN = "#A3D9AA"

# Custom CSS for Dark Theme and Gradient Styling
st.markdown(f"""
<style>
    /* Main Background and Text */
    .main {{
        background-color: {DARK_BACKGROUND}; /* Dark Theme Background */
        color: #C9D1D9;
    }}

    /* Sidebar Styling: Use a subtle gradient and light green border */
    .st-emotion-cache-1ldf15l {{ /* Target the sidebar background */
        background-color: #161B22; /* Slightly darker background for depth */
        border-right: 3px solid {MAIN_GREEN};
        border-radius: 0 10px 10px 0;
        color: {LIGHT_GREEN}; 
    }}
    
    /* Sidebar Header Text */
    .st-emotion-cache-1ldf15l h2, .st-emotion-cache-1ldf15l label {{
        color: {LIGHT_GREEN} !important;
    }}

    /* Button Styling: Gradient fill and strong shadow */
    .st-emotion-cache-vk3wp9 button {{ /* Target the button container */
        background-image: linear-gradient(to right, {MAIN_GREEN}, #00CC55);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px #007733;
    }}
    .st-emotion-cache-vk3wp9 button:hover {{
        background-image: linear-gradient(to right, #00CC55, {MAIN_GREEN});
        box-shadow: 0 6px #007733;
        transform: translateY(-2px);
    }}
    
    /* Input Fields Styling (Tightly integrated dark theme) */
    .st-emotion-cache-vj1c9o {{ 
        background-color: #161B22; 
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #30363D;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }}

    /* Header Styling: Main Green */
    h1 {{
        color: {MAIN_GREEN};
        font-weight: 900;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }}
    
    /* Success Message Box Styling (Matches Light Green Accent) */
    [data-testid="stSuccess"] {{
        background-color: #10301A;
        border-left: 5px solid {MAIN_GREEN};
        color: {LIGHT_GREEN};
        padding: 15px;
        border-radius: 8px;
    }}
    
    /* Metric Card Styling to match the dark theme */
    [data-testid="stMetric"] {{
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        color: {LIGHT_GREEN};
        text-align: center; /* Center text in the metric for better look */
    }}

    /* Small font size for model names in metrics */
    [data-testid="stMetricLabel"] > div:first-child {{
        font-size: 11px; 
        overflow: hidden;
        text-overflow: ellipsis;
    }}
</style>
""", unsafe_allow_html=True)


# --- 2. DISPLAY IMAGE (Image Resizing Implemented) ---

st.markdown(f"<h1>🌱 Agri Master: Smart Crop Recommendations</h1>", unsafe_allow_html=True)

try:
    # Use PIL to open the image
    img = Image.open('cp.jpg')
    
    # NEW IMAGE RESIZING LOGIC
    # Define a target width for the resized image (e.g., 600 pixels)
    TARGET_WIDTH = 600
    
    # Calculate the new height to maintain the aspect ratio
    original_width, original_height = img.size
    
    # Check if image is wider than target. Only resize down.
    if original_width > TARGET_WIDTH:
        target_height = int((original_height / original_width) * TARGET_WIDTH)
        resized_img = img.resize((TARGET_WIDTH, target_height))
    else:
        # If image is already small, use the original image
        resized_img = img

    # Use a column structure to help center the image on the page
    col1, col2, col3 = st.columns([1, 4, 1]) # 1/6 blank, 4/6 image, 1/6 blank
    with col2:
        # Display the resized image
        st.image(resized_img, caption='Crop Prediction Visual', use_container_width=True)

except FileNotFoundError:
    st.warning("Image file 'cp.jpg' not found in the application directory. Displaying placeholder text instead.")
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; border: 2px dashed #00AA44; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #A3D9AA;">CROP PREDICTION SYSTEM</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"An unexpected error occurred while loading the image: {e}")


# --- 3. MODEL LOADING AND TRAINING (KNN Fix Implemented) ---

try:
    df = pd.read_csv('Crop_recommendation.csv')
    X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    y_raw = df['label'] # Keep raw labels for mapping

    # Initialize and fit the Label Encoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw) # Convert text labels (y_raw) to numerical labels (y)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

    # Dictionary to hold models and their accuracies
    MODELS = {}

    # 1. Random Forest (RF)
    RF = RandomForestClassifier(n_estimators=20, random_state=5)
    RF.fit(Xtrain, Ytrain)
    MODELS["Random Forest"] = (RF, metrics.accuracy_score(Ytest, RF.predict(Xtest)))

    # 2. Decision Tree (DT)
    DT = DecisionTreeClassifier(random_state=5)
    DT.fit(Xtrain, Ytrain)
    MODELS["Decision Tree"] = (DT, metrics.accuracy_score(Ytest, DT.predict(Xtest)))

    # 3. Naive Bayes (NB)
    NB = GaussianNB()
    NB.fit(Xtrain, Ytrain)
    MODELS["Naive Bayes"] = (NB, metrics.accuracy_score(Ytest, NB.predict(Xtest)))
    
    # 4. K-Nearest Neighbors (KNN)
    KNN = KNeighborsClassifier(n_neighbors=5)
    # FIX: Corrected Ytest to Ytrain here to resolve inconsistent samples error
    KNN.fit(Xtrain, Ytrain) 
    MODELS["K-Nearest Neighbors"] = (KNN, metrics.accuracy_score(Ytest, KNN.predict(Xtest)))

    # 5. Support Vector Machine (SVM) 
    SVM = SVC(kernel='linear', random_state=5)
    SVM.fit(Xtrain, Ytrain)
    MODELS["Support Vector Machine"] = (SVM, metrics.accuracy_score(Ytest, SVM.predict(Xtest)))
    
    # 6. Logistic Regression (LR)
    LR = LogisticRegression(max_iter=1000, random_state=5)
    LR.fit(Xtrain, Ytrain)
    MODELS["Logistic Regression"] = (LR, metrics.accuracy_score(Ytest, LR.predict(Xtest)))
    
    # 7. XGBoost Classifier (XGB)
    XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=5)
    XGB.fit(Xtrain, Ytrain)
    MODELS["XGBoost"] = (XGB, metrics.accuracy_score(Ytest, XGB.predict(Xtest)))


except FileNotFoundError:
    st.error("Error: 'Crop_recommendation.csv' file not found. Please ensure the CSV file is in the same directory.")
    st.stop()
except Exception as e:
    # Display the specific error message to the user
    st.error(f"An error occurred during model setup: {e}")
    st.stop()


@st.cache_data
def predict_crop(model_key, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Makes a prediction using the selected model."""
    global le # Access the global LabelEncoder
    
    model, _ = MODELS[model_key] # Retrieve the model object
    input_array = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1)
    
    # Predicts the numerical label (e.g., 0, 1, 2...)
    numerical_prediction = model.predict(input_array)
    
    # Inverse transform to get the crop name (e.g., 'apple', 'banana')
    crop_name = le.inverse_transform(numerical_prediction) 
    
    return crop_name[0] # Return the actual crop name

# --- 4. STREAMLIT APP INTERFACE (Model Accuracies and Selection) ---
def main():
    
    # Display model accuracies in a compact layout
    st.markdown("### Model Accuracies Comparison", unsafe_allow_html=True)
    
    # Use two rows for the 7 models for better layout on smaller screens
    model_keys = list(MODELS.keys())
    
    # Row 1: 4 columns
    cols1 = st.columns(4)
    for i in range(4):
        name = model_keys[i]
        acc = MODELS[name][1]
        with cols1[i]:
            st.metric(label=name, value=f"{acc*100:.2f}%")

    # Row 2: 3 columns (centered look)
    cols2 = st.columns([1, 1, 1, 1, 1]) # 5 columns total, with 1/5 blank space on each side
    for i in range(4, 7): # Iterate through models 4, 5, 6
        name = model_keys[i]
        acc = MODELS[name][1]
        with cols2[i-3]: # Use index 1, 2, 3 for the middle columns
            st.metric(label=name, value=f"{acc*100:.2f}%")
    
    st.markdown("---")


    # Sidebar Inputs
    st.sidebar.title("🌱 Enter Soil & Climate Details")
    
    # Model Selection Dropdown
    st.sidebar.markdown("### Choose Prediction Model")
    selected_model = st.sidebar.selectbox(
        "Select Model for Prediction:",
        list(MODELS.keys()),
        index=0 # Default to Random Forest
    )
    
    st.sidebar.markdown("### Input Parameters")
    # Input parameter fields remain the same
    nitrogen = st.sidebar.number_input("Nitrogen (N) [0-140]", 0.0, 140.0, 90.0, 0.1)
    phosphorus = st.sidebar.number_input("Phosphorus (P) [0-145]", 0.0, 145.0, 42.0, 0.1)
    potassium = st.sidebar.number_input("Potassium (K) [0-205]", 0.0, 205.0, 43.0, 0.1)
    temperature = st.sidebar.number_input("Temperature (°C) [8-43]", 8.0, 43.0, 20.8, 0.1)
    humidity = st.sidebar.number_input("Humidity (%) [14-99]", 14.0, 99.0, 82.0, 0.1)
    ph = st.sidebar.number_input("pH Level [3.5-9.9]", 3.5, 9.9, 6.5, 0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm) [20-298]", 20.0, 298.0, 202.9, 0.1)
    
    if st.sidebar.button("✨ Predict Crop"):
        inputs_filled = all(isinstance(val, (int, float)) and val > 0 for val in 
                            [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])
        if not inputs_filled:
            st.error("Please ensure all input fields are filled with non-zero values.")
        else:
            with st.spinner(f'Calculating best crop using **{selected_model}**...'):
                prediction = predict_crop(selected_model, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            
            # Display Prediction and Summary
            st.success(f"The Smart Recommendation using **{selected_model}** is: **{prediction.upper()}**")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Input Summary")
                st.dataframe(pd.DataFrame({
                    'Factor': ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall'],
                    'Value': [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                }).set_index('Factor'))
            with col2:
                st.markdown("### 💡 Farming Insight")
                st.info(f"The recommended crop, **{prediction.upper()}**, is optimal for these soil and climate conditions.")

if __name__ == '__main__':
    main()