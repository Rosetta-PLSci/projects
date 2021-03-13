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

SuicideData = pd.read_csv("Suicide.csv")
data = SuicideData.copy()

data["year"] = pd.Categorical(data["year"])
data["age"] = pd.Categorical(data["age"])

df = data.select_dtypes(["float64", "int64"])

# INFO GENERAL

print(data.info())
print()
print(data.describe())
print()
print(data.columns)
print()
print(rp.summary_cat(data[["year", "sex", "age"]]))
print()
print(data.groupby(["sex", "age", ])["suicidespercent"].mean())
print()
print(data.isnull().sum())
print()
print()

# INFO FLOAT & INT

print(df.info())
print()
print(df.describe())
print()
print(df.columns)
print()
print(rp.summary_cont(df[["gdppercapita", "gdpforyear", "suicidespercent"]]))
print()
print(df.corr())
print()
print()

# VISUALIZATION

# 'year', 'sex', 'age', 'population', 'suicidespercent', 'gdpforyear','gdppercapita'

# sns.barplot()
"""
sns.barplot(x="year", y="suicidespercent", orient="v", data=data)
plt.show()

sns.barplot(x="year", y="suicidespercent", hue="sex", orient="v", data=data)
plt.show()

sns.barplot(x="year", y="suicidespercent", hue="age", orient="v", data=data)
plt.show()
"""
# sns.lineplot()
"""
sns.lineplot(x="age", y="suicidespercent", hue="sex", data=data)
plt.show()

sns.lineplot(x="year", y="suicidespercent", hue="age", style="sex", data=data)
plt.show()

sns.lineplot(x="year", y="suicidespercent", data=data)
plt.show()
"""
# sns.joinplot()
"""
sns.set_theme(style="ticks")
sns.jointplot(x="gdppercapita", y="suicidespercent", data=data, kind="hex", color="#4CB391")
plt.show()

"""

# BASIC MODELS & ERROR & PREDICT
"""
x = data[["year"]]

y = data[["suicidespercent"]]

lm = sm.OLS(y, x)
modelOne = lm.fit()

print(modelOne.summary())
# 0.62
print(modelOne.mse_model)
# 472
print()

print(modelOne.fittedvalues)

predict = modelOne.predict(x)
"""

# PCR
"""
x = df.drop("suicidespercent", axis=1)

y = df["suicidespercent"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

pcrmodel = PCA()

xReduced = pcrmodel.fit_transform(scale(xTrain))

lm = LinearRegression()
modelpcr = lm.fit(xReduced, yTrain)
predict = modelpcr.predict(xReduced)

print(r2_score(yTrain, predict))
# 0.15
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# 1.6
print()
"""

# PLS
"""
x = df.drop("suicidespercent", axis=1)

y = df["suicidespercent"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

pls = PLSRegression().fit(xTrain,yTrain)
predict = pls.predict(xTrain)

print(r2_score(yTrain,predict))
# 0.14
print()

error = mean_squared_error(yTrain,predict)
print(np.sqrt(error))
# 1.6
print()
"""

# RIDGE
"""
x = df.drop("suicidespercent", axis=1)

y = df["suicidespercent"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=0.1).fit(xTrain, yTrain)
predict = ridge.predict(xTrain)

print(r2_score(yTrain, predict))
# 0.15
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# 1.6
print()
"""

# LASSO
"""
x = df.drop("suicidespercent", axis=1)

y = df["suicidespercent"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1).fit(xTrain, yTrain)
predict = lasso.predict(xTrain)

print(r2_score(yTrain, predict))
# 0.15
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# 1.6
print()
"""

# ENET
"""
x = df.drop("suicidespercent", axis=1)

y = df["suicidespercent"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

enet = ElasticNet().fit(xTrain, yTrain)
predict = enet.predict(xTrain)

print(r2_score(yTrain,predict))
# 0.15
print()

error = mean_squared_error(yTrain,predict)
print(np.sqrt(error))
# 1.6
print()
"""
