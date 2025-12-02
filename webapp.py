# app.py (optimized for faster deploys)
import os
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib  # faster for model I/O than pickle
import warnings
warnings.filterwarnings("ignore")

# LIGHTWEIGHT model imports (defer heavy optional imports)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Try to import XGBoost only if available on the system (deferred)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(page_title="🌱 Agri Master (fast)", layout="wide")

MODELS_FILENAME = "models.joblib"   # file to save/load models dict + labelencoder
CSV_FILE = "crop_recommendation.csv"  # dataset filename
SAMPLE_FRAC_QUICK = 0.20  # fraction to use for quick training on first-run

# --- UTILS: translations omitted here, use your existing TEXTS / T() if needed ---
def safe_read_csv(path):
    return pd.read_csv(path)

# --- CACHE: load dataset quickly and reuse across reruns ---
@st.cache_data(ttl=3600)
def load_dataset(path=CSV_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found.")
    df = safe_read_csv(path)
    return df

# --- CACHE: load models from disk to reuse (fast) ---
@st.cache_resource
def load_models_from_disk(filename=MODELS_FILENAME):
    if Path(filename).exists():
        try:
            obj = joblib.load(filename)
            # Expecting dict: {"models": MODELS_DICT, "label_encoder": le}
            return obj.get("models", {}), obj.get("label_encoder", None)
        except Exception as e:
            st.warning(f"Failed to load saved models: {e}")
            return {}, None
    return {}, None

# --- TRAIN QUICK: train a small set of lightweight models on sampled data ---
def train_quick_models(df, sample_frac=SAMPLE_FRAC_QUICK):
    # sample to speed up training (quick)
    if sample_frac < 1.0:
        df_train = df.sample(frac=sample_frac, random_state=42)
    else:
        df_train = df.copy()

    X = df_train[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_raw = df_train['label']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)

    MODELS = {}

    # Keep models small / fast
    RF = RandomForestClassifier(n_estimators=10, random_state=5, n_jobs=1)  # small forest
    RF.fit(Xtrain, Ytrain)
    MODELS["Random Forest"] = (RF, metrics.accuracy_score(Ytest, RF.predict(Xtest)))

    DT = DecisionTreeClassifier(random_state=5)
    DT.fit(Xtrain, Ytrain)
    MODELS["Decision Tree"] = (DT, metrics.accuracy_score(Ytest, DT.predict(Xtest)))

    NB = GaussianNB()
    NB.fit(Xtrain, Ytrain)
    MODELS["Naive Bayes"] = (NB, metrics.accuracy_score(Ytest, NB.predict(Xtest)))

    LR = LogisticRegression(max_iter=500, random_state=5)
    LR.fit(Xtrain, Ytrain)
    MODELS["Logistic Regression"] = (LR, metrics.accuracy_score(Ytest, LR.predict(Xtest)))

    # KNN kept but with small k
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(Xtrain, Ytrain)
    MODELS["K-Nearest Neighbors"] = (KNN, metrics.accuracy_score(Ytest, KNN.predict(Xtest)))

    # SVM with linear kernel (can be slower on many samples)
    SVM = SVC(kernel='linear', random_state=5)
    SVM.fit(Xtrain, Ytrain)
    MODELS["Support Vector Machine"] = (SVM, metrics.accuracy_score(Ytest, SVM.predict(Xtest)))

    # Optionally include XGBoost if available (kept small)
    if XGB_AVAILABLE:
        try:
            XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=20, random_state=5)
            XGB.fit(Xtrain, Ytrain)
            MODELS["XGBoost"] = (XGB, metrics.accuracy_score(Ytest, XGB.predict(Xtest)))
        except Exception:
            pass

    return MODELS, le

# --- SAVE models to disk (joblib) ---
def save_models_to_disk(models, le, filename=MODELS_FILENAME):
    joblib.dump({"models": models, "label_encoder": le}, filename)

# --- PREDICT function cached for speed (reads cached models resource) ---
@st.cache_data
def predict_crop_cached(model_key, input_array, models_obj):
    model = models_obj[model_key][0]
    pred_num = model.predict(np.array(input_array).reshape(1, -1))
    return pred_num

# --- APP UI ---
def main():
    st.title("🌱 Agri Master (fast deploy)")

    # 1) Load dataset (but cached)
    try:
        df = load_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # 2) Try to load models from disk (fast)
    MODELS, LE = load_models_from_disk()

    # 3) If no saved models, show quick-train option and auto quick train (fast)
    if not MODELS:
        st.info("No saved models found. Quick-training a small model set (fast).")
        with st.spinner("Quick training models (this may take a few seconds)..."):
            MODELS, LE = train_quick_models(df, sample_frac=SAMPLE_FRAC_QUICK)
            # Save quick models for subsequent runs
            try:
                save_models_to_disk(MODELS, LE)
                st.success("Quick models trained and saved to disk (models.joblib).")
            except Exception as e:
                st.warning(f"Could not save models to disk: {e}")

    # 4) Show model accuracies
    st.subheader("Model Accuracies (quick set)")
    cols = st.columns(3)
    keys = list(MODELS.keys())
    for i, k in enumerate(keys):
        acc = MODELS[k][1]
        with cols[i % len(cols)]:
            st.metric(label=k, value=f"{acc*100:.2f}%")

    st.sidebar.header("Inputs")
    selected_model = st.sidebar.selectbox("Model", keys, index=0)

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 90.0, 0.1)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 42.0, 0.1)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 43.0, 0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", 8.0, 43.0, 20.8, 0.1)
    humidity = st.sidebar.number_input("Humidity (%)", 14.0, 99.0, 82.0, 0.1)
    ph = st.sidebar.number_input("pH", 3.5, 9.9, 6.5, 0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 20.0, 298.0, 202.9, 0.1)

    if st.sidebar.button("Predict (fast)"):
        inputs = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        # ensure positive values
        if not all(isinstance(v, (int, float)) and v > 0 for v in inputs):
            st.error("Please fill positive numeric values for all inputs.")
        else:
            with st.spinner("Predicting..."):
                pred_num = predict_crop_cached(selected_model, inputs, MODELS)
                crop_name = LE.inverse_transform(pred_num)[0]
                st.success(f"Recommended crop: {crop_name.upper()}")

    st.sidebar.markdown("---")
    st.sidebar.write("Advanced options:")

    # Button to train full models (manual action) - warns user it's slower
    if st.sidebar.button("Train full models now (slow)"):
        st.warning("Training full models can take several minutes on free containers. Recommended only once.")
        with st.spinner("Training full models..."):
            # Train using the full dataset (no sampling)
            MODELS_FULL, LE_FULL = train_quick_models(df, sample_frac=1.0)  # reuse function but with full data
            # Optionally add heavier config here (increase n_estimators etc.)
            try:
                save_models_to_disk(MODELS_FULL, LE_FULL)
                st.success("Full models trained and saved. Next starts will be fast.")
                # update in-memory
                MODELS.update(MODELS_FULL)
                LE = LE_FULL
            except Exception as e:
                st.error(f"Could not save full models: {e}")

    # Optionally show dataset head
    if st.checkbox("Show dataset sample (for debugging)"):
        st.dataframe(df.head())

if __name__ == "__main__":
    main()
