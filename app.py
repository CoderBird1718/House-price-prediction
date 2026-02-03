import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

# --------------------
# App Config (mobile‑friendly)
# --------------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------
# Safe model loading
# --------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    try:
        model = joblib.load("house_price_model.pkl")
        columns = joblib.load("model_columns.pkl")
        return model, list(columns), None
    except Exception as e:
        return None, None, traceback.format_exc()

model, columns, load_error = load_artifacts()

if load_error:
    st.error("🚨 Failed to load model files")
    st.code(load_error)
    st.stop()

# --------------------
# Helpers
# --------------------
def encode_yes_no(val: str) -> int:
    return 1 if val == "Yes" else 0


def format_currency(val: float) -> str:
    return f"₹{val:,.0f}"

# --------------------
# UI
# --------------------
st.title("🏠 House Price Prediction")
st.caption("Stable · Validated · Cloud‑ready")

with st.form("inputs"):
    area = st.number_input("Area (sq ft)", min_value=300, max_value=20000, step=100)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    stories = st.slider("Stories", 1, 4, 1)

    c1, c2 = st.columns(2)
    with c1:
        mainroad = st.selectbox("Main Road", ["Yes", "No"])
        guestroom = st.selectbox("Guest Room", ["Yes", "No"])
        basement = st.selectbox("Basement", ["Yes", "No"])
    with c2:
        hotwater = st.selectbox("Hot Water Heating", ["Yes", "No"])
        aircon = st.selectbox("Air Conditioning", ["Yes", "No"])
        prefarea = st.selectbox("Preferred Area", ["Yes", "No"])

    parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])
    furnishing = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"],
    )

    submitted = st.form_submit_button("Predict Price")

# --------------------
# Validation + Prediction
# --------------------
if submitted:
    if area < 300:
        st.error("Area must be at least 300 sq ft")
        st.stop()

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": encode_yes_no(mainroad),
        "guestroom": encode_yes_no(guestroom),
        "basement": encode_yes_no(basement),
        "hotwaterheating": encode_yes_no(hotwater),
        "airconditioning": encode_yes_no(aircon),
        "parking": parking,
        "prefarea": encode_yes_no(prefarea),
        "furnishingstatus_semi-furnished": 0,
        "furnishingstatus_unfurnished": 0,
    }

    if furnishing == "semi-furnished":
        input_data["furnishingstatus_semi-furnished"] = 1
    elif furnishing == "unfurnished":
        input_data["furnishingstatus_unfurnished"] = 1

    try:
        final_input = np.array([[input_data.get(col, 0) for col in columns]])
        prediction = float(model.predict(final_input)[0])
    except Exception as e:
        st.error("Prediction failed")
        st.code(traceback.format_exc())
        st.stop()

    st.success(f"Estimated House Price: {format_currency(prediction)}")

    # --------------------
    # Chart
    # --------------------
    st.subheader("📊 Area vs Price")

    areas = np.linspace(area * 0.6, area * 1.4, 8)
    preds = []

    area_idx = columns.index("area")
    for a in areas:
        temp = final_input.copy()
        temp[0, area_idx] = a
        preds.append(model.predict(temp)[0])

    df = pd.DataFrame({"Area (sq ft)": areas.astype(int), "Predicted Price": preds})

    fig, ax = plt.subplots()
    ax.plot(df["Area (sq ft)"], df["Predicted Price"])
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    with st.expander("🔎 Input Summary"):
        st.json(input_data)
