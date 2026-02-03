import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# App Config
# --------------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------
# Load model artifacts safely
# --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("house_price_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, columns = load_artifacts()

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
st.caption("Mobile‑friendly · Validated inputs · Visual insights")

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
# Validation
# --------------------
if submitted:
    errors = []
    if area < 300:
        errors.append("Area must be at least 300 sq ft")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # --------------------
    # Build model input
    # --------------------
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

    final_input = np.array([[input_data.get(col, 0) for col in columns]])

    # --------------------
    # Prediction
    # --------------------
    prediction = float(model.predict(final_input)[0])

    st.success(f"Estimated House Price: {format_currency(prediction)}")

    # --------------------
    # Simple Explainability Chart
    # --------------------
    st.subheader("📊 Price Sensitivity (Area Impact)")

    areas = np.linspace(area * 0.6, area * 1.4, 8)
    preds = []
    for a in areas:
        temp = final_input.copy()
        idx = list(columns).index("area")
        temp[0, idx] = a
        preds.append(model.predict(temp)[0])

    df = pd.DataFrame({"Area (sq ft)": areas.astype(int), "Predicted Price": preds})

    fig, ax = plt.subplots()
    ax.plot(df["Area (sq ft)"], df["Predicted Price"])
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    with st.expander("🔎 Input Summary"):
        st.json(input_data)
