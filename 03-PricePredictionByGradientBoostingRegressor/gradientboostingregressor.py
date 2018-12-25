"""
Model-based price prediction using Gradient Boosting Regressor

A gradient boosting [1] predictor trained with some core features of
cars and based on them it provides car price predictions.

[1] https://en.wikipedia.org/wiki/Gradient_boosting
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# We are going to predict value of the Fabia
FABIA = [
      180,  # months_old
       55,  # power
    65000,  # kms
        0,  # make_Alfa
        0,  # make_Audi
        0,  # make_Bentley
        0,  # make_Bmw
        0,  # make_Chevrolet
        0,  # make_Chrysler
        0,  # make_Citroen
        0,  # make_Dacia
        0,  # make_Ds
        0,  # make_Fiat
        0,  # make_Ford
        0,  # make_Honda
        0,  # make_Hyundai
        0,  # make_Iveco
        0,  # make_Jaguar
        0,  # make_Jeep
        0,  # make_Kia
        0,  # make_Land
        0,  # make_Mazda
        0,  # make_Mercedes-Benz
        0,  # make_Mini
        0,  # make_Mitsubishi
        0,  # make_Nissan
        0,  # make_Opel
        0,  # make_Peugeot
        0,  # make_Porsche
        0,  # make_Renault
        0,  # make_Saab
        0,  # make_Seat
        1,  # make_Skoda
        0,  # make_Smart
        0,  # make_Ssangyong
        0,  # make_Suzuki
        0,  # make_Toyota
        0,  # make_Volkswagen
        0,  # make_Volvo
        0,  # model_116
        0,  # model_118
        0,  # model_120
        0,  # model_2008
        0,  # model_206
        0,  # model_207
        0,  # model_208
        0,  # model_216
        0,  # model_218
        0,  # model_220
        0,  # model_3
        0,  # model_3008
        0,  # model_307
        0,  # model_308
        0,  # model_316
        0,  # model_318
        0,  # model_320
        0,  # model_325
        0,  # model_330
        0,  # model_407
        0,  # model_420
        0,  # model_500
        0,  # model_5008
        0,  # model_500C
        0,  # model_500L
        0,  # model_500X
        0,  # model_508
        0,  # model_520
        0,  # model_525
        0,  # model_530
        0,  # model_6
        0,  # model_640
        0,  # model_730
        0,  # model_9-3
        0,  # model_911
        0,  # model_A1
        0,  # model_A180
        0,  # model_A200
        0,  # model_A3
        0,  # model_A4
        0,  # model_A4Allroad
        0,  # model_A5
        0,  # model_A6
        0,  # model_A7
        0,  # model_A8
        0,  # model_Accord
        0,  # model_Adam
        0,  # model_Alhambra
        0,  # model_Altea
        0,  # model_AlteaXl
        0,  # model_Antara
        0,  # model_Astra
        0,  # model_Asx
        0,  # model_Ateca
        0,  # model_Auris
        0,  # model_AutomobilesDs3
        0,  # model_AutomobilesDs4
        0,  # model_AutomobilesDs5
        0,  # model_Avensis
        0,  # model_Aveo
        0,  # model_Aygo
        0,  # model_B-Max
        0,  # model_B180
        0,  # model_B200
        0,  # model_Beetle
        0,  # model_Berlingo
        0,  # model_Bipper
        0,  # model_Boxster
        0,  # model_C-Elys�e
        0,  # model_C-Max
        0,  # model_C2
        0,  # model_C200
        0,  # model_C220
        0,  # model_C3
        0,  # model_C3Aircross
        0,  # model_C3Picasso
        0,  # model_C4
        0,  # model_C4Cactus
        0,  # model_C4Picasso
        0,  # model_C5
        0,  # model_Caddy
        0,  # model_Captiva
        0,  # model_Captur
        0,  # model_Carens
        0,  # model_Carnival
        0,  # model_Cayenne
        0,  # model_Cc
        0,  # model_Cee'D
        0,  # model_Citan
        0,  # model_Civic
        0,  # model_Cla200
        0,  # model_Cla220
        0,  # model_Clio
        0,  # model_Cls350
        0,  # model_Combo
        0,  # model_Compass
        0,  # model_Continental
        0,  # model_Cooper
        0,  # model_CooperD
        0,  # model_Cordoba
        0,  # model_Corsa
        0,  # model_Cr-V
        0,  # model_Cruze
        0,  # model_Cx-5
        0,  # model_Daily
        0,  # model_Doblo
        0,  # model_Ducato
        0,  # model_Duster
        0,  # model_E220
        0,  # model_E320
        0,  # model_E350
        0,  # model_Ecosport
        0,  # model_Eos
        0,  # model_Espace
        0,  # model_Exeo
        0,  # model_Expert
        1,  # model_Fabia
        0,  # model_Fiesta
        0,  # model_Fiorino
        0,  # model_Focus
        0,  # model_Forfour
        0,  # model_Fortwo
        0,  # model_Gla200
        0,  # model_Golf
        0,  # model_GolfGti
        0,  # model_GolfSportsvan
        0,  # model_GolfVariant
        0,  # model_GrandC-Max
        0,  # model_GrandC4Picasso
        0,  # model_GrandCherokee
        0,  # model_GrandScenic
        0,  # model_GrandVitara
        0,  # model_GrandVoyager
        0,  # model_I10
        0,  # model_I20
        0,  # model_I30
        0,  # model_I40
        0,  # model_Ibiza
        0,  # model_Insignia
        0,  # model_Ix20
        0,  # model_Ix35
        0,  # model_Juke
        0,  # model_Jumper
        0,  # model_Jumpy
        0,  # model_Ka/Ka+
        0,  # model_Kadjar
        0,  # model_Kangoo
        0,  # model_Kuga
        0,  # model_Laguna
        0,  # model_LandCruiser
        0,  # model_Leon
        0,  # model_Macan
        0,  # model_Megane
        0,  # model_Meriva
        0,  # model_Micra
        0,  # model_Ml320
        0,  # model_Ml350
        0,  # model_Mokka
        0,  # model_Mondeo
        0,  # model_Montero
        0,  # model_Mustang
        0,  # model_Note
        0,  # model_Octavia
        0,  # model_Outlander
        0,  # model_Panamera
        0,  # model_Panda
        0,  # model_Partner
        0,  # model_Passat
        0,  # model_PassatVariant
        0,  # model_Picanto
        0,  # model_Polo
        0,  # model_Pulsar
        0,  # model_Punto
        0,  # model_Q2
        0,  # model_Q3
        0,  # model_Q5
        0,  # model_Q7
        0,  # model_Qashqai
        0,  # model_Qashqai+2
        0,  # model_Rapid/Spaceback
        0,  # model_Rav4
        0,  # model_Renegade
        0,  # model_Rexton
        0,  # model_Rio
        0,  # model_Rodius
        0,  # model_RomeoGiulietta
        0,  # model_RomeoMito
        0,  # model_RoverDefender
        0,  # model_RoverDiscovery
        0,  # model_RoverFreelander
        0,  # model_RoverRangeRover
        0,  # model_RoverRangeRoverEvoque
        0,  # model_RoverRangeRoverSport
        0,  # model_S-Max
        0,  # model_S40
        0,  # model_S60
        0,  # model_Sandero
        0,  # model_SantaFe
        0,  # model_Scenic
        0,  # model_Scirocco
        0,  # model_Sharan
        0,  # model_Slk200
        0,  # model_Spacetourer
        0,  # model_Sportage
        0,  # model_Sprinter
        0,  # model_Superb
        0,  # model_T5Multivan
        0,  # model_Talisman
        0,  # model_Tiguan
        0,  # model_Tipo
        0,  # model_Toledo
        0,  # model_Touareg
        0,  # model_Touran
        0,  # model_Trafic
        0,  # model_Transit
        0,  # model_TransitConnect
        0,  # model_Transporter
        0,  # model_Tt
        0,  # model_Tucson
        0,  # model_Twingo
        0,  # model_V220
        0,  # model_V40
        0,  # model_V40Cc
        0,  # model_V60
        0,  # model_Vectra
        0,  # model_Verso
        0,  # model_Viano
        0,  # model_Vito
        0,  # model_Vivaro
        0,  # model_Voyager
        0,  # model_Wrangler
        0,  # model_X-Trail
        0,  # model_X1
        0,  # model_X3
        0,  # model_X4
        0,  # model_X5
        0,  # model_X6
        0,  # model_Xc60
        0,  # model_Xc90
        0,  # model_Xe
        0,  # model_Xf
        0,  # model_Xsara
        0,  # model_XsaraPicasso
        0,  # model_Yaris
        0,  # model_Z4
        0,  # model_Zafira
        0,  # model_ZafiraTourer
        0,  # sale_type_almost_new
        0,  # sale_type_classic
        0,  # sale_type_demo
        0,  # sale_type_km_0
        0,  # sale_type_new
        1,  # sale_type_used
        0,  # gear_type_automatic
        1,  # gear_type_manual
        0,  # gear_type_semi-automatic
        0,  # fuel_type_CNG
        0,  # fuel_type_LPG
        0,  # fuel_type_diesel
        0,  # fuel_type_electric
        0,  # fuel_type_etanol
        1,  # fuel_type_gasoline
        0,  # fuel_type_hybrid
  ]


# 1) Load data

data = pd.read_csv("../data/online-adds-of-used-cars/data.csv",
                   sep=";",
                   usecols=["make",
                            "model",
                            "months_old",
                            "power",
                            "sale_type",
                            "gear_type",
                            "fuel_type",
                            "kms",
                            "price"],
                   index_col=False,
                   error_bad_lines=False,
                   )


# 2) Prepare the data

data.dropna(inplace=True)
data.months_old = data.months_old.astype("int32")
data.power = data.power.astype("int32")
data.kms = data.kms.astype("int32")
data.price = data.price.astype("int32")
data = data[data["power"] >= 30]      # Ignore too weak "cars"
data = data[data["price"] <= 600000]  # Ignore too expensive cards

# One-hot encoding
X = pd.get_dummies(data, columns=["make", "model", "sale_type", "gear_type", "fuel_type"])

# Finalize X and y
del X["price"]
y = data["price"]


# 3) Split the data to training and testing sets and randomize them

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 4) Train the model

model = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth = 8,
    min_samples_leaf = 6,
    max_features = 0.1
)
model.fit(X_train, y_train)

# 5) Predict value for my Skoda Fabia

prediction = model.predict([FABIA])
print(f"Predictions for Skoda Fabia:\n{prediction}\n")


# Appendix 1) Find good hyperparameter values
# Let computer to find good hyperparams for our use-case (data).

#from sklearn.model_selection import GridSearchCV

# Hyperparam combinations we want to try
#param_grid = {
#    "n_estimators": [1000, 1500],
#    "max_depth": [8],
#    "min_samples_leaf": [6, 9],
#    "max_features": [0.1]
#}

#clf = GridSearchCV(model, param_grid, n_jobs=4)
#clf.fit(X_train, y_train)

#print(f"Good hyperparams: {clf.best_params_}\n")


# Appendix 2) Investigate importance of individual features
# Features that are not important for our model should be removed.

feature_importances = model.feature_importances_
feature_labels = X.columns.values
sorted_indexes = feature_importances.argsort()

for index in sorted_indexes[::-1]:
    print(f"{feature_labels[index]:20}: {feature_importances[index]}")
print()


# Appendix 3) Check the error rate
# The MAE value is a mean error between predicted rating and the real one.

mae_train = mean_absolute_error(y_train, model.predict(X_train))
print(f"Mean absolute error for train set: {mae_train}")

mae_test = mean_absolute_error(y_test, model.predict(X_test))
print(f"Mean absolute error for test set: {mae_test}\n")


# Appendix 4) Example how to store/load model
# Training of model is time consuming, we need to be able to store
# and load already trained ones.

#import joblib
#joblib.dump(model, "model.pickle")   # Save
#model = joblib.load("model.pickle")  # Load


# Appendix 5) Plot a graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # because of projection='3d'

# Prepare data for the graph
sample = data.sample(n=300)
old_axe = sample["months_old"]
price_axe = sample["price"]
kms_axe = sample["kms"]

# 2D Plot
plt.scatter(old_axe, price_axe, color='blue')
plt.title('Car prices', fontsize=16)
plt.xlabel('months_old', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(old_axe, kms_axe, price_axe)
ax.set_xlabel('months_old')
ax.set_ylabel('kms')
ax.set_zlabel('price')
plt.show()