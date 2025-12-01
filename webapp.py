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
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image  # Essential for image manipulation
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 0. TRANSLATION DICTIONARY ---------------------------------------

TEXTS = {
    "English": {
        "title": "🌱 Agri Master: Smart Crop Recommendations",
        "model_acc_title": "Model Accuracies Comparison",
        "sidebar_title": "🌱 Enter Soil & Climate Details",
        "choose_model": "Choose Prediction Model",
        "select_model": "Select Model for Prediction:",
        "input_params": "Input Parameters",
        "nitrogen": "Nitrogen (N) [0-140]",
        "phosphorus": "Phosphorus (P) [0-145]",
        "potassium": "Potassium (K) [0-205]",
        "temperature": "Temperature (°C) [8-43]",
        "humidity": "Humidity (%) [14-99]",
        "ph": "pH Level [3.5-9.9]",
        "rainfall": "Rainfall (mm) [20-298]",
        "predict_btn": "✨ Predict Crop",
        "error_inputs": "Please ensure all input fields are filled with non-zero values.",
        "spinner_text": "Calculating best crop using",
        "result_text": "The Smart Recommendation using",
        "input_summary": "📊 Input Summary",
        "farming_insight": "💡 Farming Insight",
        "insight_text": "The recommended crop, {crop}, is optimal for these soil and climate conditions.",
        "image_caption": "Crop Prediction Visual",
        "image_missing": "Image file 'cp.jpg' not found in the application directory. Displaying placeholder text instead.",
        "image_placeholder_title": "CROP PREDICTION SYSTEM",
        "lang_label": "Language / ಭಾಷೆ",
    },
    "Kannada": {
        "title": "🌱 ಅಗ್ರಿ ಮಾಸ್ಟರ್: ಸ್ಮಾರ್ಟ್ ಬೆಳೆ ಶಿಫಾರಸುಗಳು",
        "model_acc_title": "ಮಾದರಿಗಳ ನಿಖರತೆಯ ಹೋಲಿಕೆ",
        "sidebar_title": "🌱 ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ",
        "choose_model": "ಅನುವಾಂಶಿಕ ಮಾದರಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "select_model": "ಭವಿಷ್ಯವಾಣಿ ಮಾಡಲು ಮಾದರಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        "input_params": "ಇನ್ಪುಟ್ ಪರಿಮಾಣಗಳು",
        "nitrogen": "ನೈಟ್ರೊಜನ್ (N) [0-140]",
        "phosphorus": "ಫಾಸ್ಫರಸ್ (P) [0-145]",
        "potassium": "ಪೊಟ್ಯಾಸಿಯಂ (K) [0-205]",
        "temperature": "ತಾಪಮಾನ (°C) [8-43]",
        "humidity": "ಆರ್ದ್ರತೆ (%) [14-99]",
        "ph": "pH ಮಟ್ಟ [3.5-9.9]",
        "rainfall": "ವರ್ಷಾಕಾಲ (ಮಿಮೀ) [20-298]",
        "predict_btn": "✨ ಬೆಳೆ ಅನ್ನು ಊಹಿಸಿ",
        "error_inputs": "ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಇನ್ಪುಟ್ ಮೌಲ್ಯಗಳು ಶೂನ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ಇರುವಂತೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
        "spinner_text": "ಉತ್ತಮ ಬೆಳೆ ಲೆಕ್ಕ ಹಾಕಲಾಗುತ್ತಿದೆ (ಮಾದರಿ:",
        "result_text": "ಸ್ಮಾರ್ಟ್ ಶಿಫಾರಸು (ಮಾದರಿ ಬಳಸಿ)",
        "input_summary": "📊 ಇನ್ಪುಟ್ ಸಾರಾಂಶ",
        "farming_insight": "💡 ಕೃಷಿ ಸಲಹೆ",
        "insight_text": "ಶಿಫಾರಸಾದ ಬೆಳೆ **{crop}** ಈ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ ಸೂಕ್ತವಾಗಿದೆ.",
        "image_caption": "ಬೆಳೆ ಭವಿಷ್ಯವಾಣಿ ದೃಶ್ಯ",
        "image_missing": "ಚಿತ್ರ ಫೈಲ್ 'cp.jpg' ಕಂಡುಬಂದಿಲ್ಲ. ತಾತ್ಕಾಲಿಕ ಪಠ್ಯ ಪ್ರದರ್ಶಿಸಲಾಗುತ್ತಿದೆ.",
        "image_placeholder_title": "ಬೆಳೆ ಭವಿಷ್ಯವಾಣಿ ವ್ಯವಸ್ಥೆ",
        "lang_label": "ಭಾಷೆ / Language",
    },
}

def T(lang: str, key: str, **kwargs) -> str:
    """Helper to fetch translated text."""
    text = TEXTS[lang][key]
    if kwargs:
        text = text.format(**kwargs)
    return text

# --- 1. CONFIGURATION AND STYLING (The New Theme) ---
st.set_page_config(
    page_title="🌱 Agri Master Crop Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define Theme Colors based on your website's look
DARK_BACKGROUND = "#0D1117"
MAIN_GREEN = "#00AA44"
LIGHT_GREEN = "#A3D9AA"

# Custom CSS for Dark Theme and Gradient Styling
st.markdown(
    f"""
<style>
    .main {{
        background-color: {DARK_BACKGROUND};
        color: #C9D1D9;
    }}

    .st-emotion-cache-1ldf15l {{
        background-color: #161B22;
        border-right: 3px solid {MAIN_GREEN};
        border-radius: 0 10px 10px 0;
        color: {LIGHT_GREEN}; 
    }}
    
    .st-emotion-cache-1ldf15l h2, .st-emotion-cache-1ldf15l label {{
        color: {LIGHT_GREEN} !important;
    }}

    .st-emotion-cache-vk3wp9 button {{
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
    
    .st-emotion-cache-vj1c9o {{ 
        background-color: #161B22; 
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #30363D;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }}

    h1 {{
        color: {MAIN_GREEN};
        font-weight: 900;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }}
    
    [data-testid="stSuccess"] {{
        background-color: #10301A;
        border-left: 5px solid {MAIN_GREEN};
        color: {LIGHT_GREEN};
        padding: 15px;
        border-radius: 8px;
    }}
    
    [data-testid="stMetric"] {{
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        color: {LIGHT_GREEN};
        text-align: center;
    }}

    [data-testid="stMetricLabel"] > div:first-child {{
        font-size: 11px; 
        overflow: hidden;
        text-overflow: ellipsis;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# --- 2. MODEL LOADING AND TRAINING -----------------------------------

try:
    df = pd.read_csv("crop_recommendation.csv")
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y_raw = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    MODELS = {}

    RF = RandomForestClassifier(n_estimators=20, random_state=5)
    RF.fit(Xtrain, Ytrain)
    MODELS["Random Forest"] = (RF, metrics.accuracy_score(Ytest, RF.predict(Xtest)))

    DT = DecisionTreeClassifier(random_state=5)
    DT.fit(Xtrain, Ytrain)
    MODELS["Decision Tree"] = (DT, metrics.accuracy_score(Ytest, DT.predict(Xtest)))

    NB = GaussianNB()
    NB.fit(Xtrain, Ytrain)
    MODELS["Naive Bayes"] = (NB, metrics.accuracy_score(Ytest, NB.predict(Xtest)))

    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(Xtrain, Ytrain)
    MODELS["K-Nearest Neighbors"] = (KNN, metrics.accuracy_score(Ytest, KNN.predict(Xtest)))

    SVM = SVC(kernel="linear", random_state=5)
    SVM.fit(Xtrain, Ytrain)
    MODELS["Support Vector Machine"] = (SVM, metrics.accuracy_score(Ytest, SVM.predict(Xtest)))

    LR = LogisticRegression(max_iter=1000, random_state=5)
    LR.fit(Xtrain, Ytrain)
    MODELS["Logistic Regression"] = (LR, metrics.accuracy_score(Ytest, LR.predict(Xtest)))

    XGB = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=5)
    XGB.fit(Xtrain, Ytrain)
    MODELS["XGBoost"] = (XGB, metrics.accuracy_score(Ytest, XGB.predict(Xtest)))

except FileNotFoundError:
    st.error(
        "Error: 'crop_recommendation.csv' file not found. Please ensure the CSV file is in the same directory."
    )
    st.stop()
except Exception as e:
    st.error(f"An error occurred during model setup: {e}")
    st.stop()


@st.cache_data
def predict_crop(
    model_key, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
):
    model, _ = MODELS[model_key]
    input_array = np.array(
        [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    ).reshape(1, -1)
    numerical_prediction = model.predict(input_array)
    crop_name = le.inverse_transform(numerical_prediction)
    return crop_name[0]


# --- 3. STREAMLIT APP INTERFACE --------------------------------------

def main():
    # Language selector (stored in session_state so you can reuse across pages)
    if "lang" not in st.session_state:
        st.session_state["lang"] = "English"

    lang = st.sidebar.selectbox(
        TEXTS["English"]["lang_label"],  # label always visible to both
        ["English", "Kannada"],
        index=0 if st.session_state["lang"] == "English" else 1,
    )
    st.session_state["lang"] = lang  # save choice

    # Title
    st.markdown(f"<h1>{T(lang, 'title')}</h1>", unsafe_allow_html=True)

    # --- Display Image (centered) ---
    try:
        img = Image.open("cp.jpg")

        TARGET_WIDTH = 600
        original_width, original_height = img.size

        if original_width > TARGET_WIDTH:
            target_height = int((original_height / original_width) * TARGET_WIDTH)
            resized_img = img.resize((TARGET_WIDTH, target_height))
        else:
            resized_img = img

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(
                resized_img,
                caption=T(lang, "image_caption"),
                use_container_width=True,
            )

    except FileNotFoundError:
        st.warning(T(lang, "image_missing"))
        st.markdown(
            f"""
            <div style="text-align: center; padding: 40px; border: 2px dashed #00AA44; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: #A3D9AA;">{T(lang, "image_placeholder_title")}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the image: {e}")

    # --- Model Accuracies Section ---
    st.markdown(f"### {T(lang, 'model_acc_title')}", unsafe_allow_html=True)

    model_keys = list(MODELS.keys())

    cols1 = st.columns(4)
    for i in range(4):
        name = model_keys[i]
        acc = MODELS[name][1]
        with cols1[i]:
            st.metric(label=name, value=f"{acc * 100:.2f}%")

    cols2 = st.columns([1, 1, 1, 1, 1])
    for i in range(4, 7):
        name = model_keys[i]
        acc = MODELS[name][1]
        with cols2[i - 3]:
            st.metric(label=name, value=f"{acc * 100:.2f}%")

    st.markdown("---")

    # --- Sidebar Inputs ---
    st.sidebar.title(T(lang, "sidebar_title"))

    st.sidebar.markdown(f"### {T(lang, 'choose_model')}")
    selected_model = st.sidebar.selectbox(
        T(lang, "select_model"),
        list(MODELS.keys()),
        index=0,
    )

    st.sidebar.markdown(f"### {T(lang, 'input_params')}")
    nitrogen = st.sidebar.number_input(T(lang, "nitrogen"), 0.0, 140.0, 90.0, 0.1)
    phosphorus = st.sidebar.number_input(T(lang, "phosphorus"), 0.0, 145.0, 42.0, 0.1)
    potassium = st.sidebar.number_input(T(lang, "potassium"), 0.0, 205.0, 43.0, 0.1)
    temperature = st.sidebar.number_input(T(lang, "temperature"), 8.0, 43.0, 20.8, 0.1)
    humidity = st.sidebar.number_input(T(lang, "humidity"), 14.0, 99.0, 82.0, 0.1)
    ph = st.sidebar.number_input(T(lang, "ph"), 3.5, 9.9, 6.5, 0.1)
    rainfall = st.sidebar.number_input(T(lang, "rainfall"), 20.0, 298.0, 202.9, 0.1)

    if st.sidebar.button(T(lang, "predict_btn")):
        inputs_filled = all(
            isinstance(val, (int, float)) and val > 0
            for val in [
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
            ]
        )

        if not inputs_filled:
            st.error(T(lang, "error_inputs"))
        else:
            with st.spinner(f"{T(lang, 'spinner_text')} **{selected_model}**..."):
                prediction = predict_crop(
                    selected_model,
                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall,
                )

            st.success(
                f"{T(lang, 'result_text')} **{selected_model}**: **{prediction.upper()}**"
            )
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {T(lang, 'input_summary')}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Factor": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
                            "Value": [
                                nitrogen,
                                phosphorus,
                                potassium,
                                temperature,
                                humidity,
                                ph,
                                rainfall,
                            ],
                        }
                    ).set_index("Factor")
                )

            with col2:
                st.markdown(f"### {T(lang, 'farming_insight')}")
                st.info(
                    T(lang, "insight_text", crop=prediction.upper())
                )


if __name__ == "__main__":
    main()
