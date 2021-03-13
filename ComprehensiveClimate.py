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
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

Atmosphere = pd.read_csv("Global1751-2013.csv")
data = Atmosphere.copy()
# 'Year', 'Total', 'Gas', 'Liquids', 'Solids', 'Production', 'Flaring'
# 'Yıl' , 'Toplam', 'Gaz', 'Sıvılar', 'Katılar', 'Üretim', 'Parlama'

# INFORMATIONS

print("INFORMATION: ")
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
print(rp.summary_cont(data[["Total", "Gas", "Liquids"]]))
print()
print(rp.summary_cont(data[["Total", "Solids", "Production", "Flaring"]]))
print()
print(data.groupby(["Gas", "Liquids"])["Total"].mean())
print()
print(data.groupby(["Solids", "Production", "Flaring"])["Total"].mean())
print()
print(data.isnull().sum())
print()

# VISUALIZATION

# sns.scatterplot
"""
sns.scatterplot(x="Gas", y="Total", data=data)
plt.show()
# important

sns.scatterplot(x="Solids", y="Total" ,data=data)
plt.show()

sns.scatterplot(x="Liquids", y="Total", data=data)
plt.show()

sns.scatterplot(x="Production", y="Total", data=data)
plt.show()
# important

sns.scatterplot(x="Flaring", y="Total", data=data)
plt.show()

sns.scatterplot(x="Gas", y="Total", hue="Production", data=data)
plt.show()

sns.scatterplot(x="Production", y="Total", hue="Production", data=data)
plt.show()

sns.scatterplot(x="Production", y="Gas", data=data)
plt.show()
"""

# sns.jointplot
"""
sns.jointplot(x="Gas", y="Total", data=data, kind="reg")
plt.show()

sns.jointplot(x="Production", y="Total", data=data, kind="reg")
plt.show()

sns.jointplot(x="Production", y="Gas", data=data, kind="reg")
plt.show()

sns.set_theme(style="ticks")
sns.jointplot(x="Year", y="Total", data=data, kind="hex", color="#4CB391")
plt.show()

sns.jointplot(x="Year", y="Gas", data=data, kind="hex", color="#4CB391")
plt.show()

sns.jointplot(x="Year", y="Production", data=data, kind="hex", color="#4CB391")
plt.show()

sns.jointplot(x="Year", y="Flaring", data=data, kind="hex", color="#4CB391")
plt.show()

sns.jointplot(x="Year", y="Solids", data=data, kind="hex", color="#4CB391")
plt.show()

sns.jointplot(x="Year", y="Liquids", data=data, kind="hex", color="#4CB391")
plt.show()
"""

# sns.lineplot
"""
data1 = data[["Solids", "Production", "Gas","Liquids","Flaring"]]
sns.lineplot(data=data1, palette="tab10", linewidth=2.5)
plt.show()
"""

# plt
"""
figure = plt.figure()

x1 = figure.add_subplot(2, 3, 1)
x2 = figure.add_subplot(2, 3, 2)
x3 = figure.add_subplot(2, 3, 3)
x4 = figure.add_subplot(2, 3, 4)
x5 = figure.add_subplot(2, 3, 5)
x6 = figure.add_subplot(2, 3, 6)
plt.legend(ncol=5, loc="best")

x1.plot(data["Gas"], data["Year"], "ro")
x2.plot(data["Solids"], data["Year"], "ko")
x3.plot(data["Liquids"], data["Year"], "bo")
x4.plot(data["Production"], data["Year"], "go")
x5.plot(data["Flaring"], data["Year"], "k-.")
x6.plot(data["Total"], data["Year"], "k--")
plt.show()

plt.bar(data["Year"], data["Gas"], label="Gas")
plt.bar(data["Year"], data["Solids"], label="Solids")
plt.xlim(1751,2020)
plt.legend(loc="best")
plt.show()

plt.bar(data["Year"], data["Liquids"], label="Liquids")
plt.bar(data["Production"], data["Year"], label="Production")
plt.bar(data["Flaring"], data["Year"], label="Flaring")
plt.xlim(1751,2020)
plt.legend(loc="best")
plt.show(
)
"""

# CHECKING AGAINST VALUE
"""
df = data[['Gas', 'Liquids', 'Solids', 'Production', 'Flaring']]

clf = LocalOutlierFactor(n_neighbors=20,contamination=0.1)
clf.fit_predict(df)

scores = clf.negative_outlier_factor_
sortedscores = np.sort(scores)

print(sortedscores)
print()

point = sortedscores[4]
print(df[scores == point])
print()

against = scores < point
print(df[against])
print()

fitvalues = scores > point
print(df[fitvalues])
print()
"""


# r2 --> 0.74, error --> 38
# BASIC MODELS & ERROR & PREDICT
"""
# 'Year', 'Total', 'Gas', 'Liquids', 'Solids', 'Production', 'Flaring'

x = data[["Total"]]
t = data[["Gas"]]
s = data[["Liquids"]]
g = data[["Solids"]]
h = data[["Production"]]
p = data[["Flaring"]]

y = data["Year"]

lm = LinearRegression()

modelOne = lm.fit(x, y)
predictOne = modelOne.predict(x)
print(predictOne)

print(modelOne.score(x, y))
# 0.62
print()
print(np.sqrt(mean_squared_error(y,predictOne)))
# 46
print()

modelTwo = lm.fit(t, y)
predictTwo = modelTwo.predict(t)
print(modelTwo.score(t, y))
# 0.46
print()
print(np.sqrt(mean_squared_error(y,predictTwo)))
# 55
print()

modelThree = lm.fit(s, y)
predictThree = modelThree.predict(s)
print(modelThree.score(s, y))
# 0.55
print()
print(np.sqrt(mean_squared_error(y,predictThree)))
# 50
print()

modelFour = lm.fit(g, y)
predictFour = modelFour.predict(g)
print(modelFour.score(g, y))
# 0.74
print()
print(np.sqrt(mean_squared_error(y,predictFour)))
# 38
print()

modelFive = lm.fit(h, y)
predictFive = modelFive.predict(h)
print(modelFive.score(h, y))
# 0.38
print()
print(np.sqrt(mean_squared_error(y,predictFive)))
# 59
print()

modelSix = lm.fit(p, y)
predictSix = modelSix.predict(p)
print(modelSix.score(p, y))
# 0.46
print()
print(np.sqrt(mean_squared_error(y,predictSix)))
# 55
print()
"""

# r2 --> 0.88, error --> 25
# MULTI MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

lm = sm.OLS(yTrain, xTrain)
modellm = lm.fit()

print(modellm.summary())
# R2: 0.55
# FS: 44.67
print()

reg = LinearRegression()
modelreg = reg.fit(xTrain, yTrain)

print(modelreg.coef_)
print()

predicty = modelreg.predict(xTrain)
print(r2_score(yTrain, predicty))
# 0.88
print()

error = np.sqrt(mean_squared_error(yTrain, predicty))
print(error)
# 25
print()

crossVR2 = cross_val_score(modelreg, xTrain, yTrain, cv=10, scoring="r2").mean()
print(crossVR2)
# 0.87
print()

crossVE = -cross_val_score(modelreg, xTrain, yTrain, cv=10, scoring="neg_mean_squared_error").mean()
print(np.sqrt(crossVE))
# 25
print()
"""

# r2 --> 0.88, error --> 25
# PCR
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

pca = PCA()

xRTrain = pca.fit_transform(scale(xTrain))

lm = LinearRegression()
modellm = lm.fit(xRTrain, yTrain)
predicty = modellm.predict(xRTrain)

print(modellm.coef_)
print()

print(r2_score(yTrain, predicty))
# 0.88
print()

error = mean_squared_error(yTrain, predicty)
print(np.sqrt(error))
# 25
print()
"""

# r2 --> 0.83, error --> 918
# PLS
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

pls = PLSRegression().fit(xTrain, yTrain)

print(pls.coef_)
print()

predicty = pls.predict(xTrain)

print(r2_score(yTrain, predicty))
# 0.83
print()

error = mean_squared_error(yTrain, predicty)
print(error)
# 918
print()
"""

# r2 --> 0.88, error --> 25
# RIDGE
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

ridge = Ridge(alpha=0.1).fit(xTrain, yTrain)

print(ridge.coef_)
print()

predicty = ridge.predict(xTrain)

print(r2_score(yTrain, predicty))
# 0.88
print()

error = mean_squared_error(yTrain, predicty)
print(np.sqrt(error))
# 25
print()
"""

# r2 --> 0.88, error --> 25
# LASSO
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

lasso = Lasso(alpha=0.1).fit(xTrain, yTrain)
print(lasso.coef_)
print()

predict = lasso.predict(xTrain)
print(r2_score(yTrain, predict))
# 0.88
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# 25
print()
"""

# r2 --> 0.88, error --> 25
# ELASTICNET(ENET)
"""
x = data.drop("Year", axis=1)
y = data["Year"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

enet = ElasticNet().fit(xTrain, yTrain)

print(enet.coef_)
print()

predict = enet.predict(xTrain)

print(r2_score(yTrain,predict))
# 0.88
print()

error = mean_squared_error(yTrain,predict)
print(np.sqrt(error))
# 25
print()
"""
