import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class CollaborativeFiltering:
    def __init__(self, data):
        self.data = data

        # Create a user-item matrix where rows represent users and columns represent items
        self.user_item_matrix = data.pivot_table(
            index="CustomerNo", columns="ProductNo", values="Quantity", fill_value=0
        )

        # Convert the user-item matrix into a sparse matrix format for efficient calculations
        self.sparse_user_item = csr_matrix(self.user_item_matrix.values)

        # Calculate cosine similarity between users
        self.user_similarity = cosine_similarity(self.sparse_user_item)

        # Convert cosine similarity matrix to a DataFrame for easier handling
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend_products_for_user(self, user_id, top_n=5):
        # Find the top N similar users
        similar_users = (
            self.user_similarity_df[user_id]
            .sort_values(ascending=False)
            .iloc[1 : top_n + 1]
            .index
        )

        # Get the products bought by the similar users
        products_bought_by_similar_users = (
            self.user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)
        )

        # Filter out the products already bought by the target user
        products_already_bought = self.user_item_matrix.loc[user_id]
        recommended_products = products_bought_by_similar_users.index[
            ~products_bought_by_similar_users.index.isin(
                products_already_bought[products_already_bought > 0].index
            )
        ]

        recommended_products_name = self.data[
            self.data["ProductNo"].isin(list(recommended_products[:top_n]))
        ]["ProductName"].unique()

        return list(recommended_products_name)


# # Load the dataset
# data_path = './data.csv'
# data = pd.read_csv(data_path)

# collaborative_filtering = CollaborativeFiltering(data)

# # Example usage
# user_id = 581478.0 # Example user ID
# print(collaborative_filtering.recommend_products_for_user(user_id))
