from sklearn.linear_model import LinearRegression
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import streamlit as st
import pandas as pd
import numpy as np


class QuantityPredictor:
    def __init__(self, data):
        self.df = data
        self.pipeline = None
        self.preprocess_data()

    def preprocess_data(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.dropna(inplace=True)
        self.df["DayOfWeek"] = self.df["Date"].dt.day_name()
        self.df["Month"] = self.df["Date"].dt.month_name()
        self.df["Date of month"] = self.df["Date"].dt.day

    def fit_model(self, model_type="linear", country=None):
        if country:
            df_filtered = self.df[self.df["Country"] == country]
        else:
            df_filtered = self.df

        X = df_filtered.drop(["Quantity", "Date"], axis=1)
        y = df_filtered["Quantity"]

        categorical_features = ["Month", "Country", "ProductName", "DayOfWeek"]
        numerical_features = ["Date of month"]
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", one_hot_encoder, categorical_features),
                ("num", "passthrough", numerical_features),
            ]
        )

        if model_type == "linear":
            model = LinearRegression()

        #  Use any other model here in elif cases
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )

        self.pipeline.fit(X, y)

    def predict_sales(self, input_date, product_name, country_name):
        next_monday = input_date + timedelta(days=(7 - input_date.weekday()))
        pred_dates = [next_monday + timedelta(days=i) for i in range(7)]
        pred_data = {
            "Month": [date.strftime("%B") for date in pred_dates],
            "Country": [country_name] * 7,
            "ProductName": [product_name] * 7,
            "DayOfWeek": [date.strftime("%A") for date in pred_dates],
            "Date of month": [date.day for date in pred_dates],
        }

        pred_df = pd.DataFrame(pred_data)
        predictions = self.pipeline.predict(pred_df)
        total_quantity = sum(predictions)
        return int(total_quantity)


def predict(data, date, country, product):
    predictor = QuantityPredictor(data)
    predictor.fit_model(model_type="linear")
    input_date = pd.to_datetime(date)
    total_quantity = predictor.predict_sales(input_date, product, country)
    print(total_quantity)
    return total_quantity


def predict_quantity(data):
    with st.expander(
        "🚀 Discover the future of shopping with our cutting-edge Product Forecast Tool! 🛍️"
    ):
        st.text(
            """
                By harnessing the power of Artificial Neural Networks and Linear Regression, 
                we've turned the unpredictable into the predictable. Enter a date, country, and 
                product name, and voila! Our tool magically predicts the quantity of the product 
                that will fly off the shelves in the coming week. 📈 Whether you're a business 
                aiming for flawless inventory management or a shopper craving a seamless 
                experience, our innovation ensures you're always a step ahead. Say goodbye to 
                overstocking woes and hello to just-in-time restocking. 🌟 Dive into a world where 
                market trends meet precision forecasting, making every shopping journey an adventure 
                in efficiency!
                """
        )
    countries_list = data["Country"].unique()
    countries_list = np.array(countries_list)
    select = np.array(["Select"])
    countries_list = np.concatenate((select, countries_list))

    predict_country = st.selectbox(
        "Which country do you want to select?",
        countries_list,
    )

    if predict_country != "Select":
        data_country = data[data["Country"] == predict_country].copy()

        productList = data_country["ProductName"].unique()
        productList = np.array(productList)
        select = np.array(["Select"])
        productList = np.concatenate((select, productList))

        predict_productName = st.selectbox("What do you wish to buy?", productList)
        if predict_productName != "Select":
            date = st.date_input("Enter the date: ", value="today", format="YYYY-MM-DD")

            if st.button("Predict"):
                with st.spinner("Running the algorithm...."):
                    quantity = predict(
                        data, str(date), predict_country, predict_productName
                    )
                    st.text("Successfully Predicted 📈")
                st.text("🌟 Get ready to meet demand with precision: ")
                st.success(
                    f"You're set to sell around :red[_{quantity}_] of these gems in the next 7 days! 🚀"
                )
