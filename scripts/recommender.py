import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import sys

# Suppress UserWarning globally
warnings.simplefilter("ignore", category=UserWarning)

class MovieRecommender:
    def __init__(self, ratings):
        self.ratings = ratings
        self.similarity_matrix = self.calculate_similarity_matrix()

    def calculate_similarity_matrix(self):
        num_items = self.ratings.shape[1]
        similarity_matrix = np.zeros((num_items, num_items))

        for i in range(num_items):
            for j in range(i, num_items):
                if i != j:
                    item1_ratings = self.ratings[:, i]
                    item2_ratings = self.ratings[:, j]

                    # Check for constant input arrays
                    if not np.all(item1_ratings == item1_ratings[0]) and not np.all(item2_ratings == item2_ratings[0]):
                        common_users = np.logical_and(item1_ratings != 0, item2_ratings != 0)

                        # Redirect stderr to null temporarily
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            sys.stderr = open('nul', 'w')  # On Windows, use 'nul'; on Unix-like systems, use '/dev/null'
                            try:
                                if np.sum(common_users) > 1:  # At least 2 common users for Pearson coefficient
                                    similarity, _ = pearsonr(item1_ratings[common_users], item2_ratings[common_users])
                                    similarity_matrix[i, j] = similarity
                                    similarity_matrix[j, i] = similarity
                            except Warning:
                                pass  # Ignore the warning and continue
                            finally:
                                sys.stderr = sys.__stderr__  # Restore stderr
        return similarity_matrix

    # The rest of the class remains unchanged

    def predict_weighted_average(self, user_ratings, item_index, N):
        # Get N most similar items
        similar_items = np.argsort(self.similarity_matrix[item_index])[-N:]

        # Filter out items without user ratings
        valid_similar_items = similar_items[user_ratings[similar_items] != 0]

        # Calculate weighted average prediction
        numerator = np.sum(self.similarity_matrix[item_index, valid_similar_items] * user_ratings[valid_similar_items])
        denominator = np.sum(np.abs(self.similarity_matrix[item_index, valid_similar_items]))

        if denominator == 0:
            return 0  # Avoid division by zero

        prediction = numerator / denominator

        # Replace NaN with 0
        return np.nan_to_num(prediction)
    
    def predict_weighted_average_adjusted(self, user_ratings, item_index, N):
        # Get N most similar items
        similar_items = np.argsort(self.similarity_matrix[item_index])[-N:]

        # Filter out items without user ratings
        valid_similar_items = similar_items[user_ratings[similar_items] != 0]

        # Calculate adjusted weighted average prediction
        user_avg = np.mean(user_ratings)
        numerator = np.sum((self.similarity_matrix[item_index, valid_similar_items] * (user_ratings[valid_similar_items] - user_avg)))
        denominator = np.sum(np.abs(self.similarity_matrix[item_index, valid_similar_items]))

        if denominator == 0:
          return user_avg  # Return user's average rating if denominator is 0

        prediction = user_avg + (numerator / denominator)

        #Replace NaN with 0
        return np.nan_to_num(prediction)

    def predict_weighted_average_variance(self, user_ratings, item_index, N):
        # Get N most similar items
        similar_items = np.argsort(self.similarity_matrix[item_index])[-N:]

        # Filter out items without user ratings
        valid_similar_items = similar_items[user_ratings[similar_items] != 0]

        # Calculate weights based on the variance of ratings
        weights = np.var(self.ratings[:, valid_similar_items], axis=0)

        # Calculate weighted average prediction with variance-based weights
        numerator = np.sum(self.similarity_matrix[item_index, valid_similar_items] * user_ratings[valid_similar_items] * weights)
        denominator = np.sum(np.abs(self.similarity_matrix[item_index, valid_similar_items]) * weights)

        if denominator == 0:
            return 0  # Avoid division by zero

        prediction = numerator / denominator

        # Replace NaN with 0
        return np.nan_to_num(prediction)
    

    def predict_weighted_average_common_users(self, user_ratings, item_index, N):
        # Get N most similar items
        similar_items = np.argsort(self.similarity_matrix[item_index])[-N:]

        # Filter out items without user ratings
        valid_similar_items = similar_items[user_ratings[similar_items] != 0]

        # Calculate weights based on the number of common users
        common_users_count = np.sum(self.ratings[:, valid_similar_items] != 0, axis=0)

        # Check if there are common users
        if np.any(common_users_count):
            weights = common_users_count / np.max(common_users_count)  # Normalize weights
        else:
            # If there are no common users, assign equal weights to all items
            weights = np.ones_like(common_users_count) / len(common_users_count)

        # Calculate weighted average prediction with common users-based weights
        numerator = np.sum(self.similarity_matrix[item_index, valid_similar_items] * user_ratings[valid_similar_items] * weights)
        denominator = np.sum(np.abs(self.similarity_matrix[item_index, valid_similar_items]) * weights)

        if denominator == 0:
            return 0  # Avoid division by zero

        prediction = numerator / denominator

        # Replace NaN with 0
        return np.nan_to_num(prediction)


    def evaluate(self, test_ratings, N, prediction_function):
        all_predictions = []
        all_ground_truth = []

        for user_index in range(test_ratings.shape[0]):
            user_ratings = test_ratings[user_index]

            for item_index in range(test_ratings.shape[1]):
                if user_ratings[item_index] != 0:
                    # Use the specified prediction function
                    prediction = prediction_function(user_ratings, item_index, N)
                    all_predictions.append(prediction)
                    all_ground_truth.append(user_ratings[item_index])

        # Replace NaN with 0 in predictions and ground truth
        all_predictions = np.nan_to_num(all_predictions)
        all_ground_truth = np.nan_to_num(all_ground_truth)

        mae = mean_absolute_error(all_ground_truth, all_predictions)
        precision = precision_score(np.array(all_ground_truth) >= 3, np.array(all_predictions) >= 3, average='macro')
        recall = recall_score(np.array(all_ground_truth) >= 3, np.array(all_predictions) >= 3, average='macro')

        return mae, precision, recall


# Load data from the CSV file
ratings_data = pd.read_csv('ratings-reduced.csv')

# Convert the data into a ratings matrix
ratings_matrix = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0).values

# Unique N values to be tested
N_values = [5, 10, 15, 20, 25]

# Run the experiment for each N value with a print statement
for i, N in enumerate(N_values, start=1):
    print(f"\nExperiment for N={N} ({i}/{len(N_values)}):")

    # Split the data into training and testing sets
    train_ratings, test_ratings = train_test_split(ratings_matrix, test_size=0.2, random_state=None)  # Use a random state for reproducibility

    recommender = MovieRecommender(train_ratings)

    # Evaluate using the original prediction function
    mae, precision, recall = recommender.evaluate(test_ratings, N, recommender.predict_weighted_average)
    print(f"    Original Weighted Average (MAE): {mae}")
    print(f"    Original Weighted Average Precision: {precision}")
    print(f"    Original Weighted Average Recall: {recall}")

    # Evaluate using the adjusted prediction function
    mae, precision, recall = recommender.evaluate(test_ratings, N, recommender.predict_weighted_average_adjusted)
    print(f"    Weighted Average with Adjustment (MAE): {mae}")
    print(f"    Weighted Average with Adjustment Precision: {precision}")
    print(f"    Weighted Average with Adjustment Recall: {recall}")

    # Evaluate using the variance-based prediction function
    mae, precision, recall = recommender.evaluate(test_ratings, N, recommender.predict_weighted_average_variance)
    print(f"    Weighted Average with Variance (MAE): {mae}")
    print(f"    Weighted Average with Variance Precision: {precision}")
    print(f"    Weighted Average with Variance Recall: {recall}")

    # Evaluate using the common users-based prediction function
    mae, precision, recall = recommender.evaluate(test_ratings, N, recommender.predict_weighted_average_common_users)
    print(f"    Weighted Average with Common Users (MAE): {mae}")
    print(f"    Weighted Average with Common Users Precision: {precision}")
    print(f"    Weighted Average with Common Users Recall: {recall}")