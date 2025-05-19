import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Mock training data setup (adjusted with realistic model) ---
def create_mock_model():
    n = 200
    np.random.seed(42)
    data = pd.DataFrame({
        'Year': np.random.choice(range(2008, 2023), size=n),
        'Make': ['Hino'] * n,
        'Model': ['268'] * n,
        'Mileage': np.random.randint(100000, 400000, size=n),
        'Engine': np.random.choice(['J08E', 'J08E-VB', 'J08E-WU'], size=n),
        'Condition Rating': np.random.randint(1, 6, size=n),
        'Box Type': np.random.choice(['Dry Van', 'Reefer', 'Flatbed'], size=n),
        'Length': np.random.choice([20, 22, 24, 26], size=n),
        'Width': np.random.choice([96, 102], size=n),
        'Height': np.random.choice([90, 96, 102], size=n),
        'Lift Maker': np.random.choice(['Maxon', 'Anthony', 'Waltco'], size=n),
        'Location': np.random.choice(['Northeast', 'South', 'West', 'Midwest'], size=n),
        'Engine Hours': np.random.randint(3000, 15000, size=n)
    })

    data['Price Paid at Auction'] = (
        50000 - (2024 - data['Year']) * 1200
        - data['Mileage'] * 0.06
        - data['Engine Hours'] * 0.6
        + data['Condition Rating'] * 2500
        + np.random.normal(0, 2500, size=n)
    ).clip(lower=5000)

    X = data.drop(columns=['Price Paid at Auction'])
    y = data['Price Paid at Auction']

    categorical_cols = ['Engine', 'Box Type', 'Lift Maker', 'Location']
    numeric_cols = ['Year', 'Mileage', 'Condition Rating', 'Length', 'Width', 'Height', 'Engine Hours']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model

model = create_mock_model()

# --- Streamlit App UI ---
st.set_page_config(page_title="Hino 268 Auction Price Estimator")
st.title("Hino 268 Auction Price Estimator")

st.markdown("Enter truck specifications below to get a suggested auction price.")

# --- Input Form ---
year = st.number_input("Year", min_value=2005, max_value=2024, value=2018)
mileage = st.number_input("Mileage", min_value=50000, max_value=500000, step=1000, value=250000)
engine = st.selectbox("Engine", ['J08E', 'J08E-VB', 'J08E-WU'])
condition = st.slider("Condition Rating (1 = Poor, 5 = Excellent)", 1, 5, value=3)
box_type = st.selectbox("Box Type", ['Dry Van', 'Reefer', 'Flatbed'])
length = st.selectbox("Length (ft)", [20, 22, 24, 26])
width = st.selectbox("Width (in)", [96, 102])
height = st.selectbox("Height (in)", [90, 96, 102])
lift_maker = st.selectbox("Lift Maker", ['Maxon', 'Anthony', 'Waltco'])
location = st.selectbox("Location", ['Northeast', 'South', 'West', 'Midwest'])
engine_hours = st.number_input("Engine Hours", min_value=0, max_value=20000, step=100, value=8500)

# --- Prediction ---
if st.button("Estimate Price"):
    input_df = pd.DataFrame([{
        'Year': year,
        'Make': 'Hino',
        'Model': '268',
        'Mileage': mileage,
        'Engine': engine,
        'Condition Rating': condition,
        'Box Type': box_type,
        'Length': length,
        'Width': width,
        'Height': height,
        'Lift Maker': lift_maker,
        'Location': location,
        'Engine Hours': engine_hours
    }])

    price = model.predict(input_df)[0]
    st.success(f"Estimated Auction Price: ${price:,.2f}")
