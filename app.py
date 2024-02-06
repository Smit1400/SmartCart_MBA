import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from eda_app import exploratory_data_analysis
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from collaborative_filtering import CollaborativeFiltering
import multiprocessing


def recommend(rules, current_cart):
    """
    Generates a list of recommendations based on the current cart contents, adjusted for string representation of antecedents and consequents.

    Parameters:
    rules (DataFrame): The DataFrame containing the association rules.
    current_cart (set): A set of items currently in the shopping cart.

    Returns:
    list: A list of recommended items.
    """
    recommendations = set()
    for _, rule in rules.iterrows():
        antecedents = set(rule["antecedents"].split(", "))
        consequents = set(rule["consequents"].split(", "))
        if current_cart.issubset(antecedents):
            recommendations.update(consequents - current_cart)
    return list(recommendations)


def fpgrowth_multiprocessing(queue, data, min_support):
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    queue.put(frequent_itemsets)


@st.cache_data
def load_data():
    df = pd.read_csv("./data.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["CustomerNo"].fillna(-1, inplace=True)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    return df

@st.cache_resource
def initialize_collaborative_filtering(df):
    cf = CollaborativeFiltering(df)
    return cf
    


@st.cache_resource
def run_fpgrowth(dataframe, min_support=0.01):
    dataframe.dropna(subset=["ProductName", "Quantity"], inplace=True)
    basket = (
        dataframe.groupby(["TransactionNo", "ProductName"])["Quantity"]
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index("TransactionNo")
    )
    basket = (basket > 0).astype(int)
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    min_confidence = 0.5
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules


st.title("Welcome to SmartCart! 🛒")
st.caption(
    "Transform your shopping with 🛒 SmartCart: Unlock predictive insights for a personalized e-commerce experience like never before! 💡🚀"
)

menu = st.sidebar.selectbox(
    "Select Options",
    ("Market Basket Analysis", "Exploratory Data Analysis", "Predict Quantity"),
)

df = load_data()

countries_list = df["Country"].unique()
countries_list = np.array(countries_list)
select = np.array(["Select"])
countries_list = np.concatenate((select, countries_list))

if "country" not in st.session_state:
    st.session_state.country = "Select"

if "season" not in st.session_state:
    st.session_state.season = "Select"

if "min_support" not in st.session_state:
    st.session_state.min_support = 0.01

if "run" not in st.session_state:
    st.session_state.run = False


def run_button():
    st.session_state.run = True


def change_button_status():
    st.session_state.run = False


if menu == "Market Basket Analysis":

    toggle = st.sidebar.toggle("MBA on Full Data")

    if not toggle:
        st.session_state.country = st.selectbox(
            "Which country do you want to select",
            countries_list,
            on_change=change_button_status,
        )

        st.session_state.season = st.selectbox(
            "Select Season",
            ("Select", "JAN - SEPT", "OCT - DEC"),
            on_change=change_button_status,
        )

        with st.expander(
            "`min_support` (What is this and How to choose a value for it)"
        ):
            st.write(
                f"""
                `min_support` refers to the minimum support threshold used in market basket analysis and frequent itemset 
                mining algorithms like Apriori and FP-growth. It is a parameter that determines the lowest level of support 
                at which an itemset is considered frequent. Support, in this context, is a measure of how often an itemset 
                appears in the dataset. Specifically, it is the proportion of transactions in the dataset that contain a 
                particular itemset.

                For example, if min_support is set to 0.01, it means that only itemsets that appear in at least 1% of all 
                transactions in the dataset will be considered frequent and included in the analysis for generating association 
                rules.
                """
            )

        st.session_state.min_support = st.number_input(
            "Choose min support",
            min_value=0.01,
            max_value=1.0,
            value="min",
            format="%.2f",
            on_change=change_button_status,
        )

        if (
            st.session_state.country != "Select"
            and st.session_state.season != "Select"
            and st.session_state.min_support > 0.0
        ):

            df_country = df[df["Country"] == st.session_state.country].copy()

            if st.session_state.season == "JAN - SEPT":
                fdf = df_country[
                    df_country["month"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])
                ].copy()
            else:
                fdf = df_country[df_country["month"].isin([10, 11, 12])].copy()

            with st.expander("Note: "):
                st.write(
                    f"""
                         If the data is too small keep the support factor high (around 0.2) for it to run, or else it might not stop running.
                
                        For this country and season, shape of the data is {fdf["TransactionNo"].nunique()}
                         """
                )
            run = st.button("Run", on_click=run_button)

            if st.session_state.run:

                with st.spinner("Running FPGrowth......"):
                    rules = run_fpgrowth(fdf, st.session_state.min_support)
                    st.success("FPGrwoth implemented successfully...")
                    st.caption("These are the rules....")
                    st.write(rules)

                productList = fdf["ProductName"].unique()
                productList = np.array(productList)
                select = np.array(["Select"])
                productList = np.concatenate((select, productList))

                productName = st.selectbox("What do you wish to buy?", productList)
                if productName != "Select":
                    current_cart = {productName}
                    recommended_items = recommend(rules, current_cart)

                    if len(recommended_items) == 0:
                        st.info("No Recommendations", icon="🚨")
                    else:
                        st.success(
                            f"Since you bought _{productName}_ you can also buy "
                        )
                        with st.container(border=True, height=200):
                            for index, items in enumerate(recommended_items):
                                st.text(f"{index+1}. {items}")

        elif (
            st.session_state.country == "Select" or st.session_state.season == "Select"
        ):
            st.text("Please select a Country and a Season to Proceed......")
    else:
        st.text("You can get product recommendation considering the whole data")
        algorithm = st.selectbox(
            "Which algorithm do you wish to proceed with?",
            ("Select", "FPGrowth", "Collaborative Filtering"),
        )

        if algorithm == "FPGrowth":
            with st.spinner("Running FPGrowth.."):
                rules = run_fpgrowth(df)
                st.success("Done")

            productList = df["ProductName"].unique()
            productList = np.array(productList)
            select = np.array(["Select"])
            productList = np.concatenate((select, productList))
            productName = st.selectbox("What do you wish to buy?", productList)

            if productName != "Select":
                current_cart = {productName}
                recommended_items = recommend(rules, current_cart)

                if len(recommended_items) == 0:
                    st.text("No Recommendations for this product.")
                else:
                    st.success(f"Since you bought _{productName}_ you can also buy ")
                    with st.container(border=True, height=200):
                        for index, items in enumerate(recommended_items):
                            st.text(f"{index+1}. {items}")
                            
        elif algorithm == "Collaborative Filtering":
            with st.spinner("Initializing Collabortive Filtering Class...."):
                collaborative_filtering = initialize_collaborative_filtering(df)
                st.success("Successfully Initialised")
            
            with st.expander("What is Collaborative Filtering?"):
                with st.container():
                    st.text("""
                            Collaborative filtering is like having a bunch of friends who share their shopping lists with you! 
                            🛍️ Imagine you and your friends all shop at the same giant online store. You all buy different things, 
                            but sometimes you buy things that others have bought too. Collaborative filtering takes note of what 
                            you and others buy, and when it spots a pattern—like you and your friend both buying funky socks and 
                            a cool mug—it thinks, "Aha! Since you both liked those, maybe you'll like this other quirky notebook 
                            your friend bought!" 📓✨
                            """)
                
            with st.expander("How does it work for our data?"):
                st.text("""
                        For our e-commerce data, think of each row as a shopping trip by a customer. 
                        The collaborative filtering algorithm plays the role of a super-observant friend
                        who remembers every item bought by every customer. It then uses this massive shopping 
                        diary to recommend products. If Customer A and Customer B both bought wooden crates and 
                        vintage tins, and Customer A also bought a vintage clock, the algorithm might suggest 
                        that vintage clock to Customer B. It's all about connecting the dots between customers' 
                        purchases to find hidden gems you might love! 🕵️‍♂️💎
                        """)
            
            userIdList = df["CustomerNo"].unique()
            userIdList = np.array(userIdList)
            select = np.array(["Select"])
            userIdList = np.concatenate((select, userIdList))
            userId = st.selectbox("Select any User by their User ID", userIdList)
            
            if userId != "Select":
                recommended_items = collaborative_filtering.recommend_products_for_user(user_id=float(userId), top_n=5)

                if len(recommended_items) == 0:
                    st.text("No Recommendations for this product.")
                else:
                    st.success(f"Some recommended items for this user are as follows: ")
                    with st.container(border=True, height=200):
                        for index, items in enumerate(recommended_items):
                            st.text(f"{index+1}. {items}")


elif menu == "Exploratory Data Analysis":
    tab1, tab2 = st.tabs(["📈 Charts", "🗃 Data"])

    with tab1:
        exploratory_data_analysis(df)

    with tab2:
        st.write(df)

else:
    st.write("Predict Quantity")
