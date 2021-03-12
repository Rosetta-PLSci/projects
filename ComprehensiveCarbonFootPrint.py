import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import researchpy as rp
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import levene
from sklearn.neighbors import LocalOutlierFactor
import missingno as msno
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

CarbonData = pd.read_csv("WorldFootPr.csv")
data = CarbonData.copy()

# INFORMATIONS

print("INFORMATION")
print()
print(data.info())
print()
print(data.describe())
print()
print(data.shape)
print()
print(data.columns)
print()
print(data.corr())
print()
print(rp.summary_cont(data[["Year", "Built_up_Land", "Carbon"]]))
print()
print(rp.summary_cont(data[["Year", "Cropland", "Fishing_Grounds", "Forest_Products"]]))
print()
print(rp.summary_cont(data[["Year", "Grazing_Land", "Total"]]))
print()
print(data.groupby(["Built_up_Land", "Carbon", "Cropland"])["Year"].mean())
print()
print(data.groupby(["Fishing_Grounds", "Forest_Products"])["Year"].mean())
print()
print(data.groupby(["Grazing_Land", "Total"])["Year"].mean())
print()
print(data.isnull().sum())
print()

# CHECKING AGAINST VALUE

"""
df = data[["Built_up_Land", "Carbon", "Cropland", "Fishing_Grounds", "Forest_Products", "Grazing_Land", "Total"]]
print(df.head())

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df)

scores = clf.negative_outlier_factor_
sortscores = np.sort(scores)
print(sortscores[0:20])
print()

point = sortscores[7]
print(point)
print()
print(df[scores == point])
print()


againstvalue = scores < point
print(df[againstvalue])
print()

fitvalues = scores > point
print(df[fitvalues])
print()
"""

# VISUALIZATION

# sns
"""
sns.scatterplot(x="Year", y="Total", data=data)
plt.show()

sns.jointplot(x="Year", y="Total", data=data, kind="reg")
plt.show()

sns.pairplot(data, kind="reg")
plt.show()
"""

# plt
"""
plt.style.use("classic")
figure = plt.figure()

x1 = figure.add_subplot(2, 3, 1)
x2 = figure.add_subplot(2, 3, 2)
x3 = figure.add_subplot(2, 3, 3)
x4 = figure.add_subplot(2, 3, 4)
x5 = figure.add_subplot(2, 3, 5)
x6 = figure.add_subplot(2, 3, 6)
plt.legend(ncol=2, loc="best")

x1.plot(data["Total"], data["Grazing_Land"], "ro")
x2.plot(data["Carbon"], data["Grazing_Land"], "ko")
x3.plot(data["Built_up_Land"], data["Grazing_Land"], "bo")
x4.plot(data["Forest_Products"], data["Grazing_Land"], "go")
x5.plot(data["Cropland"], data["Grazing_Land"], "k-.")
x6.plot(data["Year"], data["Grazing_Land"], "k--")
plt.show()

plt.bar(data["Year"], data["Cropland"], label="Cropland")
plt.bar(data["Year"], data["Grazing_Land"], label="Grazing_Land")
plt.bar(data["Year"], data["Built_up_Land"], label="Built_up_Land")

plt.legend(loc="best")
plt.show()
"""

# MULTI STATSMODEL % SKLEARN MODEL TUNING
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

lm = sm.OLS(yTrain, xTrain)
model = lm.fit()

print(model.summary())
# R2 = 0.99
print()

reg = LinearRegression()
modelReg = reg.fit(xTrain, yTrain)

print(modelReg.intercept_)
print()
print(modelReg.coef_)
print()
print(modelReg.score(xTrain, yTrain))
# R2 = 0.99
print()
crossVR2 = cross_val_score(modelReg, x, y, cv=5, scoring="r2").mean()
print(crossVR2)
# VR2 = 0.84
print()

crossVError = -cross_val_score(modelReg, x, y, cv=5, scoring="neg_mean_squared_error").mean()
print(np.sqrt(crossVError))
# Verror = 1.28
print()

error = np.sqrt(mean_squared_error(yTrain, modelReg.predict(xTrain)))
print(error)
# error = 0.61
print()
"""

# PREDICT

x = data[["Total"]]
t = data[["Grazing_Land"]]
s = data[["Carbon"]]
g = data[["Cropland"]]
h = data[["Built_up_Land"]]
y = data["Year"]

lm = LinearRegression()
model = lm.fit(x, y)


print(model.score(x, y))
print()

predict = model.predict([[2]])
print(predict)
print()


model2 = lm.fit(t, y)
model3 = lm.fit(s, y)
model4 = lm.fit(g, y)
model5 = lm.fit(h, y)

predict2 = model2.predict([[0.06]])
print(predict2)
print()
predict3 = model3.predict([[1.1]])
print(predict3)
print()
predict4 = model4.predict([[0.4]])
print(predict4)
print()
predict5 = model5.predict([[0.05]])
print(predict5)
print()
