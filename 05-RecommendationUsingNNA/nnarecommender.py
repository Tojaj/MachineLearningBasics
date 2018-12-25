"""
Memory-based content-based recommendation using Nearest neighbour algorithm

A Nearest neighbour algorithm [1] used for similar car recommendation based on
car params (make, power, gear_type, fuel_type).

[1] https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors


DREAM_CAR_SPECS = [
#   power  make_Alfa  make_Audi  make_Bentley  make_Bmw  make_Chevrolet  make_Chrysler  make_Citroen  make_Dacia  make_Ds  make_Fiat  make_Ford  make_Honda  make_Hyundai  make_Iveco  make_Jaguar  make_Jeep  make_Kia  make_Land  make_Mazda  make_Mercedes-Benz  make_Mini  make_Mitsubishi  make_Nissan  make_Opel  make_Peugeot  make_Porsche  make_Renault  make_Saab  make_Seat  make_Skoda  make_Smart  make_Ssangyong  make_Suzuki  make_Toyota  make_Volkswagen  make_Volvo  gear_type_automatic  gear_type_manual  gear_type_semi-automatic  fuel_type_CNG  fuel_type_LPG  fuel_type_diesel  fuel_type_electric  fuel_type_etanol  fuel_type_gasoline  fuel_type_hybrid
   100.0,         0,         0,            0,        0,              0,             0,            0,          0,       0,         0,         0,          0,            0,          0,           0,         0,        0,         0,          0,                  0,         0,               0,           0,         0,            0,            0,            0,         0,         0,          1,          0,              0,           0,           0,               0,          0,                   0,                1,                        0,             0,             0,                0,                  0,                0,                  1,                0,
]


# 1) Load data

data = pd.read_csv("../data/online-adds-of-used-cars/data.csv",
                   sep=";",
                   usecols=["make",
                            "model",
                            "power",
                            "gear_type",
                            "fuel_type"
                            ],
                   index_col=False,
                   error_bad_lines=False,
                   )


# 2) Prepare the data

# Drop rows with NaN
data.dropna(inplace=True)

# One-hot encoding
X = pd.get_dummies(data, columns=["make", "gear_type", "fuel_type"])
# Note: We cannot convert categorical data (like gear_type or fuel_type) to numbers like:
# 1-gasoline, 2-electric, 3-diesel, 4-LPG, etc.
# because that way, the algorithm could assume that gasoline and electric are closer to each
# other than gasoline and others.

del X["model"]


# 3) Train

nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(X)


# 4) Recommend

distances, indices = nbrs.kneighbors([DREAM_CAR_SPECS])


# 5) Process the recommendation

names = data.loc[indices[0], ["make", "model", "power"]]
seen = set()
for recommended_car in names.itertuples():
    model = f"{recommended_car.make} {recommended_car.model} ({recommended_car.power} HP)"
    if model not in seen:
        print(model)
    seen.add(model)