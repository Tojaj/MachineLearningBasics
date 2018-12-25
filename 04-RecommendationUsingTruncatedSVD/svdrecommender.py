"""
Memory-based recommendation using Truncated Singular Value Decomposition (Truncated SVD)

In this example I use Truncated SVD [1] to get significant dimensions for movies based on user ratings
and then using correlation coefficient to suggest similar movies based on similarities in their
ratings (Collaborative filtering).

[1] https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD
[2] https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
[3] https://medium.com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-svd-162ac796c27d
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


MOVIE_TITLE = "Back to the Future (1985)"  # Movie we are interested in


# 1) Load data

ratings = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=["userId", "movieId", "rating"])
movies = pd.read_csv("../data/ml-latest-small/movies.csv", usecols=["movieId", "title"], index_col="movieId")


# 2) Prepare the data

data = ratings.join(movies, on="movieId")

# Pivot table
crosstab = pd.pivot_table(data, index="title", columns="userId", values="rating", fill_value=0)

# SVD decomposition: A = U x S x V
# A - input matrix we want to decompose (m x n)
# U - orthogonal matrix (m x m) - represent rows from input matrix (movies in our case)
# S - non-negative diagonal matrix (m x n)
# V - orthogonal matrix (n x n) - represent columns from input matrix (user ratings in our case)
# Truncated means that some dimensions will be removed from the U S V matrices.
# Because of this we won't be able to get A from them but only A' which is similar to A (A ~ A').
# Benefit is that the truncated matrices are smaller and should contain only important dimensions.

svd = TruncatedSVD(
    n_components=15,
    random_state=2,
)
res_matrix = svd.fit_transform(crosstab)  # (res_matrix) T = U x S
# The res_matrix contains significant features of movies (U)

# Create a matrix with correlation coefficients
corr_matrix = np.corrcoef(res_matrix)

# List of movie names we have
movie_names = crosstab.index  # Movie names in crosstab

# Get row with coefficients of the movie we are interested in
movie_row = movie_names.get_loc(MOVIE_TITLE)  # Position of the movie in crosstab
movie_coeffs = corr_matrix[movie_row]  # Coefficients of the movie

# Find and print similar movies
similar_movies = pd.DataFrame(movie_coeffs)
similar_movies = pd.concat([similar_movies, pd.Series(movie_names)], axis=1)
similar_movies.columns = ["coeff", "title"]
similar_movies.sort_values("coeff", ascending=False, inplace=True)
similar_movies = similar_movies[similar_movies["coeff"] < 1.0][:10][["title", "coeff"]]

print(f"\nSimilar movies to '{MOVIE_TITLE}':\n")
for movie in similar_movies.itertuples():
    print(f"{movie.title:70.70} {movie.coeff:.4}")