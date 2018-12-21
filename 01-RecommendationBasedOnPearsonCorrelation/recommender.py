"""
Recommendation based on Pearson Correlation

A simple example of collaborative filtering
that provides recommendations based on user rating.

The system is based on Pearson correlation coefficient [1].
It provides items that are similar to the item chosen by user.

[1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
"""

import tempfile
import webbrowser

import pandas as pd


MOVIE_ID = 2712  # ID of "Eyes Wide Shut (1999)" in movies.csv


# Load data
ratings = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=["userId", "movieId", "rating"])
movies = pd.read_csv("../data/ml-latest-small/movies.csv", usecols=["movieId", "title"], index_col="movieId")

# Number of ratings per movie
ratings_count = pd.DataFrame(ratings.groupby("movieId")['rating'].count())
ratings_count.rename({"rating": "rating_count"}, axis=1, inplace=True)

# Pivot table (Kontingenční tabulka)
# Rows - Users, Columns - Movies
crosstab = pd.pivot_table(ratings, values="rating", index="userId", columns="movieId", aggfunc="mean")

# Ratings of the movie of our interest
movie_ratings = crosstab[MOVIE_ID]

# Calculate correlation with other movies
similar_movies = crosstab.corrwith(movie_ratings)
similar_movies = pd.DataFrame(similar_movies, columns=["pearson"])

# Clean the table and add ratings
similar_movies.drop(MOVIE_ID, inplace=True)
similar_movies.dropna(inplace=True)
similar_movies = similar_movies.join(ratings_count)

# Filter out movies that are not relevant or don't have enough reviews
best_matches = similar_movies[similar_movies["pearson"] > 0.5]
best_matches = best_matches[best_matches["rating_count"] > 15]

# Sort the list
best_matches.sort_values(["pearson", "rating_count", "movieId"], ascending=False, inplace=True)

# Keep only first best matches and add movie titles
best_matches = best_matches.head(50)
best_matches = best_matches.join(movies, "movieId")

# Create an html file with the results
_, tmpfn = tempfile.mkstemp()
with open(tmpfn, "w") as f:
    f.write(best_matches.to_html())
webbrowser.open(tmpfn)