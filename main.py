#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from scipy.stats import *


#%%
sns.set_theme(style="whitegrid")
#%% [markdown]
# Read in the data
#%%
df = pd.read_csv("./AB_NYC_2019.csv")
#%% md
# ## EDA
#%% [markdown]
# Checking data columns
#%%
df.info()
#%% [markdown]
# ### Room type
#%%
# Check the distribution of the room type
room_type = df["room_type"].value_counts()
#%%
# percentage of the room type
plt.figure(figsize=(5, 5))
plt.pie(
    room_type,
    labels=room_type.keys(),
    explode=[0.02] * len(room_type),
    autopct="%1.1f%%",
)
#%%
# drop shared room
df = df[df["room_type"] != "Shared room"]

#%% [markdown]
# The distribution after dropping shared room
#%%
room_type = df["room_type"].value_counts()
room_type / sum(room_type)

#%% [markdown]
# ### Neighborhood group

#%% [markdown]
# Check the counts of each neighborhood group
#%%
# neighborhood
sns.countplot(data=df, x="neighbourhood_group", hue="room_type")

#%% [markdown]
# Sample size for staten island is too small
#%%
df["neighbourhood_group"].value_counts()
#%% [markdown]
# Drop Staten Island
#%%
df = df[df["neighbourhood_group"] != "Staten Island"]


#%% [markdown]
# ### Rental Price
#%% [markdown]
# git rid of outlier using price
# only preserve 5% to 95% percentile data for each neighborhood
#%%
for neighbor in df["neighbourhood_group"].unique():
    five_per = np.percentile(df[df["neighbourhood_group"] == neighbor]["price"], 5)
    ninefive_per = np.percentile(df[df["neighbourhood_group"] == neighbor]["price"], 95)
    print(neighbor)
    print(f"5%: {five_per}")
    print(f"95%: {ninefive_per}")
    print()
    df = df[
        (df["neighbourhood_group"] != neighbor)
        | ((five_per <= df["price"]) & (df["price"] <= ninefive_per))
    ]

#%% [markdown]
# Generate statistics for each neighborhood group
#%%
statsArray = {}
for neighbor in df["neighbourhood_group"].unique():
    statsArray[neighbor] = df[df["neighbourhood_group"] == neighbor]["price"].describe()
#%%
statsDf = pd.DataFrame(statsArray)
statsDf = statsDf.round(2)

#%%
statsDf
#%%
statsDf.to_csv("./tables/neighbors_stats.csv")


#%% [markdown]
# Check the average price according to neighborhood group
# and room type
#%%
sns.barplot(
    data=df, y="price", hue="room_type", x="neighbourhood_group", orient="v", ci=None,
)
#%% [markdown]
# Generate the Manhattan price graph by longitude and latitude
#%%
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=df[
        (df["neighbourhood_group"] == "Manhattan")
        & (df["room_type"] == "Entire home/apt")
    ],
    x="longitude",
    y="latitude",
    hue="price",
    size="price",
    sizes=(10, 100),
    edgecolor=None,
    alpha=0.5,
    palette="RdYlGn_r",
)
plt.axis("equal")

#%% [markdown]
# Check the distribution of entire house price vs entire house log price
# Fit a gaussian curve on the log price
#%%
df["price_log"] = np.log(df["price"])
#%%

# more normally distributed
plt.figure(figsize=(5, 10))
plt.subplot(211)
# positively skew
sns.histplot(
    data=df[(df["room_type"] == "Entire home/apt")], x="price", bins=15, stat="density"
)
plt.title("Density of listing price")
plt.subplot(212)
sns.histplot(
    data=df[(df["room_type"] == "Entire home/apt")],
    x="price_log",
    bins=15,
    stat="density",
)
plt.title("Density of log of listing price")
mean = float(df["price_log"].mode()) + 0.05
std = float(df["price_log"].std()) - 0.15
x = np.linspace(mean - 4 * std, mean + 4 * std, num=100)
plt.plot(x, norm.pdf(x, loc=mean, scale=std), color="r")
print(f"mean: {(mean)}, std: {(std)}")

#%% md
# Acquire the 95% CI from Gaussian
#%%
x = norm.ppf(0.975)
interval = (mean - x * std, mean + x * std)
interval
#%% [markdown]
# Transform back from log interval
#%%
price_interval = np.exp(interval)
price_interval
#%% [markdown]
# Check the actual percentage of listings in between the interval
#%%
house = df[df["room_type"] == "Entire home/apt"]["price"]
((house >= price_interval[0]) & (house <= price_interval[1])).sum() / len(house)

#%%
house.describe()


#%% md
# ## Modeling
#%% [markdown]
# Do train test validation split
#%%
train, test = train_test_split(df, test_size=0.4)
valid, test = train_test_split(test, test_size=0.5)
#%%
# helper function to split by neighbor and room
def splitByNeighborAndRoom(df):
    data = {}
    for neighbor in df["neighbourhood_group"].unique():
        if type(neighbor) != str:
            continue
        for room_type in df["room_type"].unique():
            if type(room_type) != str:
                continue
            splitDf = df[
                (df["neighbourhood_group"] == neighbor) & (df["room_type"] == room_type)
            ]
            data[(neighbor, room_type)] = splitDf[
                ["latitude", "longitude", "price"]
            ].to_numpy()
    return data


#%%
trainData = splitByNeighborAndRoom(train)
validData = splitByNeighborAndRoom(valid)
testData = splitByNeighborAndRoom(test)
#%% md
# KNN parameters tuning
#%%
errors = {}
#%%
# Training classifier
for k in range(1, 60, 2):
    RMSE = 0
    for key, training in trainData.items():
        X_train = training[:, :2]
        y_train = training[:, -1]
        validation = validData[key]
        X_valid = validation[:, :2]
        y_valid = validation[:, -1]
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, (y_train))
        y_pred = model.predict(X_valid)
        RMSE += (((y_pred) - y_valid) ** 2).sum()
    RMSE /= len(valid)
    errors[k] = RMSE ** 0.5

#%% md
# Plot the RMSE according to number of neighbors
#%%
plt.figure(figsize=(10, 6))
plt.plot(
    (errors.keys()),
    (errors.values()),
    linestyle="--",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.xlabel("Number of neighbors")
plt.ylabel("Root Mean Sqaure Error")
plt.title("KNN validation error")
#%% md
# Find the optimal number of neighbors
#%%
min(errors.items(), key=lambda x: x[1])

#%% md
# Linear Regression parameters tuning
#%%
linear_errors = {}
#%%
# linear regression
for reg in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    RMSE = 0
    for key, training in trainData.items():
        X_train = training[:, :2]
        y_train = training[:, -1]
        validation = validData[key]
        X_valid = validation[:, :2]
        y_valid = validation[:, -1]
        model = Ridge(alpha=reg)
        model.fit(X_train, (y_train))
        y_pred = model.predict(X_valid)
        RMSE += (((y_pred) - y_valid) ** 2).sum()
    RMSE /= len(valid)
    RMSE = RMSE ** 0.5
    linear_errors[str(reg)] = RMSE


#%% [markdown]
# plot the RMSE according to regularization parameters

#%%
plt.plot(
    (linear_errors.keys()),
    (linear_errors.values()),
    linestyle="--",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.xlabel("lambda, lower value means less regularization")
plt.ylabel("Root Mean Sqaure Error")
plt.title("Linear Regression validation error")


#%% md
# ## Testing
#%%
test_errors = []
#%%
# baseline
RMSE = 0
for key, training in trainData.items():
    y_train = training[:, -1]
    testing = testData[key]
    y_test = testing[:, -1]
    y_pred = y_train.mean()
    local_RMSE = ((y_pred - y_test) ** 2).sum()
    RMSE += local_RMSE
    local_RMSE = (1 / len(y_test) * local_RMSE) ** 0.5
    test_errors.append({"model": "Baseline", "type": key, "RMSE": local_RMSE})

RMSE /= len(test)
RMSE **= 0.5
test_errors.append({"model": "Baseline", "type": "Overall", "RMSE": RMSE})

#%%
# KNN
RMSE = 0
for key, training in trainData.items():
    X_train = training[:, :2]
    y_train = training[:, -1]
    testing = testData[key]
    X_test = testing[:, :2]
    y_test = testing[:, -1]
    model = KNeighborsRegressor(n_neighbors=53)
    model.fit(X_train, (y_train))
    y_pred = model.predict(X_test)
    local_RMSE = ((y_pred - y_test) ** 2).sum()
    RMSE += local_RMSE
    local_RMSE = (1 / len(y_test) * local_RMSE) ** 0.5
    test_errors.append({"model": "KNN", "type": key, "RMSE": local_RMSE})
RMSE /= len(test)
RMSE **= 0.5
test_errors.append({"model": "KNN", "type": "Overall", "RMSE": RMSE})


#%%
# Linear Regression
RMSE = 0
for key, training in trainData.items():
    X_train = training[:, :2]
    y_train = training[:, -1]
    testing = testData[key]
    X_test = testing[:, :2]
    y_test = testing[:, -1]
    model = Ridge(alpha=0)
    model.fit(X_train, (y_train))
    y_pred = model.predict(X_test)
    local_RMSE = ((y_pred - y_test) ** 2).sum()
    RMSE += local_RMSE
    local_RMSE = (1 / len(y_test) * local_RMSE) ** 0.5
    test_errors.append({"model": "Ridge Regression", "type": key, "RMSE": local_RMSE})
RMSE /= len(test)
RMSE **= 0.5
test_errors.append({"model": "Ridge Regression", "type": "Overall", "RMSE": RMSE})

#%%
test_table = pd.DataFrame(test_errors)
test_table

#%%
heatmap = pd.pivot_table(test_table, values="RMSE", columns="model", index="type")
sns.heatmap(heatmap, annot=True, cmap="Blues_r", fmt=".2f")


#%%
