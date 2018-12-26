"""
Classification using Logistic Regression

An example of logistic regression [1] classifier that's based on set of
features of a customer decides if we should contact him or not.

[1] https://en.wikipedia.org/wiki/Logistic_regression
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


CLIENT = [
    30,    # age
    False, # default
    1984,  # balance
    True,  # housing
    False, # loan
    261,   # duration
    1,     # campaign
    -1,    # pdays
    0,     # previous
    1,     # job_admin.
    0,     # job_blue - collar
    0,     # job_entrepreneur
    0,     # job_housemaid
    0,     # job_management
    0,     # job_retired
    0,     # job_self - employed
    0,     # job_services
    0,     # job_student
    0,     # job_technician
    0,     # job_unemployed
    0,     # job_unknown
    0,     # marital_divorced
    0,     # marital_married
    1,     # marital_single
    0,     # education_primary
    0,     # education_secondary
    1,     # education_tertiary
    0,     # education_unknown
    0,     # poutcome_failure
    0,     # poutcome_other
    0,     # poutcome_success
    1,     # poutcome_unknown
]


# 1) Load data

# Client data:
# * age
# * job: type of job
# * marital: marital status
# * education
# * default: has credit in default?
# * balance: average yearly balance, in euros
# * housing: has housing loan?
# * loan: has personal loan?
# related with the last contact of the current campaign:
# * contact: contact communication type
# * day: last contact day of the month
# * month: last contact month of year
# * duration: last contact duration, in seconds
# * campaign: number of contacts performed during this campaign and for this client
# * pdays: number of days that passed by after the client was last contacted from a previous campaign
# * previous: number of contacts performed before this campaign and for this client
# * poutcome: outcome of the previous marketing campaign
#
# Output variable:
# * y - has the client subscribed a term deposit?

data = pd.read_csv("../data/bank-marketing-data-set/bank-full.csv",
                   sep=";",
                   usecols=["age",
                            "job",
                            "marital",
                            "education",
                            "default",
                            "balance",
                            "housing",
                            "loan",
                            #"contact",
                            #"day",
                            #"month",
                            "duration",
                            "campaign",
                            "pdays",
                            "previous",
                            "poutcome",
                            "y"
                            ]
                   )


# 2) Prepare the data

# One-hot encoding
data = pd.get_dummies(data, columns=["job", "marital", "education", "poutcome"])

# Convert boolean fields from string to boolean
data.replace({
    "default": {"yes": True, "no": False},
    "housing": {"yes": True, "no": False},
    "loan": {"yes": True, "no": False},
    "y": {"yes": True, "no": False},
}, inplace=True)


# 3) Prepare params for logistic regression

# Boolean dependent variable (The variable we want to predict)
y = data.loc[:, "y"]

# Set of customer features - independent variables / predictors (Values we use for prediction)
X = data.iloc[:, :]
del X["y"]


# 4) Split the data to training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y)


# 5) Calculate a logistic regression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# 6) Predict

# Will our client accept the offer?
prediction = log_reg.predict([CLIENT])
probability = log_reg.predict_proba([CLIENT])
print(f"Will our client accept the offer: {prediction}")
print(f"Probability estimates: {probability}\n")


# Appendix 1) Measure the score

score = log_reg.score(X_test, y_test)
print(f"Score (mean accuracy) for testing data: {score}\n")


# Appendix 2) Classification report

y_pred = log_reg.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))


# Appendix 2) Plot part of the input data

# TODO