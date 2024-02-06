import streamlit as st
import pandas as pd
import plotly.express as px


def exploratory_data_analysis(df: pd.DataFrame):
    st.header("Graphical Insights")

    option = st.selectbox("Which country do you want to select", df["Country"].unique())

    df_by_month = (
        df[df["Country"] == option]
        .groupby("month")
        .sum(numeric_only=True)
        .reset_index()
    )

    fig1 = px.area(
        df_by_month,
        x="month",
        y="Quantity",
        title="Average Monthly Sales Quantity Trends (Season Trend) for " + option,
        labels={"month": "Month", "Quantity": "Average Total Quantity Sold"},
    )

    # Update x-axis ticks to month names
    fig1.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    # Optionally add gridlines
    fig1.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)

    st.plotly_chart(fig1, use_container_width=True)

    product_index = df["ProductName"].value_counts().nlargest(10).index
    df_by_product_month = (
        df[df["Country"] == option]
        .groupby(["ProductName", "month"])[["Price"]]
        .mean()
        .reset_index()
    )
    product_monthly_price = df_by_product_month[
        df_by_product_month["ProductName"].isin(product_index)
    ]

    # Assuming product_monthly_price is your DataFrame and it has columns 'month', 'Price', and 'ProductName'
    fig2 = px.area(
        product_monthly_price,
        x="month",
        y="Price",
        color="ProductName",  # This differentiates the lines by product
        title="Monthly Price by Product (Top 10) for " + option,
        labels={
            "month": "Month",
            "Price": "Total Quantity Sold",
            "ProductName": "Product Name",
        },
    )

    # Update x-axis ticks to month names
    fig2.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    # Optionally add gridlines and adjust the legend
    fig2.update_layout(
        xaxis_showgrid=True, yaxis_showgrid=True, legend_title_text="Product Name"
    )

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("See explanation"):
        st.write(
            """When examining the monthly sales data, we observe a notable upward trend in quantities sold, 
                    particularly after October. 📈 This increase can be attributed to several factors, 
                    including lower product prices post-October 🏷️ and festive season shopping sprees 🎉. 
                    These elements combined lead to a surge in consumer purchases, showcasing the impact of 
                    pricing strategies and seasonal festivities on sales volumes. 🛍️✨"""
        )

    df["Revenue"] = df["Price"] * df["Quantity"]
    top_10_products = df["ProductName"].value_counts().nlargest(10).index
    df_top_10_products = df[df["Country"] == option][
        df["ProductName"].isin(top_10_products)
    ]
    monthly_revenue = (
        df_top_10_products.groupby(["ProductName", "month"])["Revenue"]
        .sum()
        .reset_index()
    )
    revenue_comparison = (
        monthly_revenue.groupby(["ProductName"])
        .apply(
            lambda x: pd.Series(
                {
                    "Jan_to_Sep_Revenue": x[x["month"] <= 9]["Revenue"].sum(),
                    "Oct_to_Dec_Revenue": x[x["month"] > 9]["Revenue"].sum(),
                }
            )
        )
        .reset_index()
    )
    # Step 3: Compare the total revenue for each product in these two periods
    revenue_comparison["Revenue_Increase"] = (
        revenue_comparison["Oct_to_Dec_Revenue"]
        - revenue_comparison["Jan_to_Sep_Revenue"]
    )
    revenue_comparison["Revenue_Increase_Percentage"] = (
        revenue_comparison["Revenue_Increase"]
        / revenue_comparison["Jan_to_Sep_Revenue"]
    ) * 100
    annual_revenue = (
        monthly_revenue.groupby(["ProductName"])["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "Annual_Revenue"})
    )
    revenue_comparison = revenue_comparison.merge(annual_revenue, on="ProductName")
    revenue_comparison["Oct_to_Dec_Percentage_of_Annual"] = (
        revenue_comparison["Oct_to_Dec_Revenue"] / revenue_comparison["Annual_Revenue"]
    ) * 100

    st.header("Seasonal Revenue Contribution Analysis from October to December")
    st.write(
        revenue_comparison[
            [
                "ProductName",
                "Annual_Revenue",
                "Oct_to_Dec_Revenue",
                "Oct_to_Dec_Percentage_of_Annual",
            ]
        ]
    )

    with st.expander("See Insights"):
        st.write(
            """Dive into the heart of holiday shopping trends with our insightful table! 🌟 
                    It showcases the remarkable journey of select products from their annual sales journey to their 
                    spectacular performance from October to December. 📊🎁 Whether it's the enchanting 
                    "Assorted Colour Bird Ornament" capturing nearly 30% of its annual revenue for _UK_ in these festive months, 
                    or the "Popcorn Holder" stealing the show In _Norway_ with a staggering 75%, each product tells a story of 
                    seasonal success. 🎄💡 This table not only highlights the seasonal allure of products but also 
                    provides a strategic snapshot for businesses aiming to leverage these golden months for maximum impact. 
                    🚀📈 It's a testament to the power of seasonal patterns in driving consumer spending, making it an 
                    essential tool for marketers and strategists alike"""
        )
