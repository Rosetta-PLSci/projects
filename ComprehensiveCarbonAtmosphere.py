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
from sklearn import model_selection
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
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit, GridSearchCV

CarbonData = pd.read_csv("globalatmospherevalues.csv")
data = CarbonData.copy()
data.drop("Month", axis=1, inplace=True)

df = data.drop("Year", axis=1)

data["Year"] = pd.Categorical(data["Year"])

# INFORMATIONS - GENERAL

print(data.shape)
print()
print(data.columns)
print()
print(data.info())
print()
print(data.describe())
print()
print(rp.summary_cont(data[["MEI", "CO2", "CH4", "N2O", "CFC-11", "CFC-12", "TSI", "Aerosols", "Temp"]]))
print()
print(rp.summary_cont(data[["Population", "CO2emissionspercapita"]]))
print()
print(data.groupby(["Year", "Population"])["Temp"].mean())
print()
print(data.groupby(["Year", "Population"])["CO2"].mean())
print()
print(data.groupby(["Year", "Population"])["Aerosols"].mean())
print()
print(data.groupby(["Year", "Population"])["CH4"].mean())
print()
print(df.corr())
print()
print(data.isnull().sum())
print()

# VISUALIZATION

# 'Year','MEI','CO2','CH4','N2O','CFC-11','CFC-12','TSI','Aerosols','Temp','Population','CO2emissionspercapita'

# BAR
"""
sns.barplot(x="Year", y="MEI", data=data)
plt.show()
sns.barplot(x="Year", y="CO2", data=data)
plt.show()
sns.barplot(x="Year", y="CH4", data=data)
plt.show()
sns.barplot(x="Year", y="N2O", data=data)
plt.show()
sns.barplot(x="Year", y="TSI", data=data)
plt.show()
sns.barplot(x="Year", y="CFC-11", data=data)
plt.show()
sns.barplot(x="Year", y="CFC-12", data=data)
plt.show()
sns.barplot(x="Year", y="Aerosols", data=data)
plt.show()
sns.barplot(x="Year", y="Temp", data=data)
plt.show()
sns.barplot(x="Year", y="CO2emissionspercapita", data=data)
plt.show()

"""

# LINE
"""
sns.lineplot(x="Year", y="MEI", data=data)
plt.show()
sns.lineplot(x="Year", y="CO2", data=data)
plt.show()
sns.lineplot(x="Year", y="CH4", data=data)
plt.show()
sns.lineplot(x="Year", y="N2O", data=data)
plt.show()
sns.lineplot(x="Year", y="CFC-11", data=data)
plt.show()
sns.lineplot(x="Year", y="CFC-12", data=data)
plt.show()
sns.lineplot(x="Year", y="TSI", data=data)
plt.show()
sns.lineplot(x="Year", y="Aerosols", data=data)
plt.show()
sns.lineplot(x="Year", y="Temp", data=data)
plt.show()
sns.lineplot(x="Year", y="CO2emissionspercapita", data=data)
plt.show()
"""

# JOINPLOT
"""
sns.jointplot(x="Temp", y="CO2", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="MEI", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="CH4", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="N2O", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="CFC-11", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="CFC-12", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="Aerosols", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="TSI", data=data, kind="reg")
plt.show()

sns.jointplot(x="Temp", y="CO2emissionspercapita", data=data, kind="reg")
plt.show()
"""

# ALL MODELS

# BASIC MODELS & ERROR & PREDICT
"""
x = data[["Year"]]
#t = data[["CO2"]]
#s = data[["CH4"]]
#p = data[["N2O"]]
#g = data[["Aerosols"]]
#k = data[["Population"]]
#r = data[["CO2emissionspercapita"]]

y = data["Temp"]

lm = LinearRegression()

model1 = lm.fit(x, y)
#model2 = lm.fit(t, y)
#model3 = lm.fit(s, y)
#model4 = lm.fit(p, y)
#model5 = lm.fit(g, y)
#model6 = lm.fit(k, y)
#model7 = lm.fit(r, y)

print(model1.score(x, y))
#print(model2.score(t, y))
#print(model3.score(s, y))
#print(model4.score(p, y))
#print(model5.score(g, y))
#print(model6.score(k, y))
#print(model7.score(r, y))
#print()

predict1 = model1.predict(x)
#predict2 = model2.predict(t)
#predict3 = model3.predict(s)
#predict4 = model4.predict(p)
#predict5 = model5.predict(g)
#predict6 = model6.predict(k)
#predict7 = model7.predict(r)

error1 = mean_squared_error(y, predict1)
print(np.sqrt(error1))
#error2 = mean_squared_error(y, predict2)
#print(np.sqrt(error2))
#error3 = mean_squared_error(y, predict3)
#print(np.sqrt(error3))
#error4 = mean_squared_error(y, predict4)
#print(np.sqrt(error4))
#error5 = mean_squared_error(y, predict5)
#print(np.sqrt(error5))
#error6 = mean_squared_error(y, predict6)
#print(np.sqrt(error6))
#print()
"""

# OLS MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

ols = sm.OLS(yTrain, xTrain)
modelols = ols.fit()

predictols = modelols.predict(xTest)

print(modelols.summary())
# R2 --> 0.77
# F --> 7.5
print()

error = mean_squared_error(yTest, predictols)
print(np.sqrt(error))
# E --> 0.07
print()
"""

# SKLEARN MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

lm = LinearRegression()
modellm = lm.fit(xTrain, yTrain)

predictlm = modellm.predict(xTest)

print(r2_score(yTest, predictlm))
# R2 --> 0.77
print()

error = mean_squared_error(yTest, predictlm)
print(np.sqrt(error))
# E --> 0.07
print()

r2cv = cross_val_score(modellm, x, y, cv=10, scoring="r2").mean()
print(r2cv)
# R2CV --> -2776
print()

errorcv = -cross_val_score(modellm, x, y, cv=10, scoring="neg_mean_squared_error").mean()
print(np.sqrt(errorcv))
# ECV --> 0.13
print()
"""

# PCR MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

pcr = PCA()

xRTrain = pcr.fit_transform(scale(xTrain))
xRTest = pcr.fit_transform(scale(xTest))

lm = LinearRegression()
modellm = lm.fit(xRTrain, yTrain)

predict = modellm.predict(xTest)

print(r2_score(yTest, predict))
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
print()

r2cv = cross_val_score(modellm, xRTrain, yTrain, cv=10, scoring="r2").mean()
print(r2cv)
print()

errorcv = -cross_val_score(modellm, xRTrain, yTrain, cv=10, scoring="neg_mean_squared_error").mean()
print(np.sqrt(errorcv))
print()
"""

# PLS MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

pls = PLSRegression()
modelpls = pls.fit(xTrain, yTrain)

predict = modelpls.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.68
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.08
print()

plscv = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

for i in range(2, 10):
    plsmodeltuned = PLSRegression(n_components=i).fit(xTrain, yTrain)
    print(f"{i} -->\n")
    predicttuned = plsmodeltuned.predict(xTest)

    print(r2_score(yTest, predicttuned))
    print()
    # best R2 --> n_components=9 / 0.77
    errortuned = mean_squared_error(yTest, predicttuned)
    print(np.sqrt(errortuned))
    # best Error --> n_components=9 / 0.07
    print()
"""

# RIDGE MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

ridge = Ridge()
modelridge = ridge.fit(xTrain, yTrain)

predict = modelridge.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.72
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.07
print()

alpha = np.arange(1, 50, 100)

ridgecv = RidgeCV(alphas=alpha, scoring="neg_mean_squared_error", normalize=True)
ridgecv.fit(xTrain, yTrain)

ridgemodeltuned = Ridge(alpha=ridgecv.alpha_).fit(xTrain, yTrain)

predicttuned = ridgemodeltuned.predict(xTest)

print(r2_score(yTest, predicttuned))
# R2 --> 0.72
print()

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(errortuned))
# E --> 0.07
print()
"""

# LASSO MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

lasso = Lasso()
modellasso = lasso.fit(xTrain, yTrain)

predict = modellasso.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.80
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.06
print()

lassocv = LassoCV(alphas=None, cv=10, max_iter=10000, normalize=True).fit(xTrain, yTrain)

print(lassocv.alpha_)
print()

lassomodeltuned = Lasso(alpha=lassocv.alpha_).fit(xTrain, yTrain)

predicttuned = lassomodeltuned.predict(xTest)

print(r2_score(yTest, predicttuned))
# R2 --> 0.45
print()

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(errortuned))
# E --> 0.11
print()
"""

# ENET MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

enet = ElasticNet()
modelenet = enet.fit(xTrain, yTrain)

predict = modelenet.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.76
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.07
print()

enetcv = ElasticNetCV(cv=10, random_state=0).fit(xTrain, yTrain)

print(enetcv.alpha_)
print()

enetmodeltuned = ElasticNet(alpha=enetcv.alpha_).fit(xTrain, yTrain)

predicttuned = enetmodeltuned.predict(xTest)

print(r2_score(yTest, predicttuned))
# R2 --> 0.80
print()

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(errortuned))
# E --> 0.06
print()
"""

# K MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

knn = KNeighborsRegressor()
knnmodel = knn.fit(xTrain, yTrain)

predict = knnmodel.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.72
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.07
print()

params = {"n_neighbors": np.arange(1, 20, 1)}

knncv = GridSearchCV(knn, params, cv=10)
knncv.fit(xTrain, yTrain)

print(knncv.best_params_)
print()

knnmodeltuned = KNeighborsRegressor(n_neighbors=15).fit(xTrain, yTrain)

predicttuned = knnmodeltuned.predict(xTest)

print(r2_score(yTest, predicttuned))
# R2 --> 0.44
print()

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(errortuned))
# E --> 0.11
print()
"""

# SVR MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

xTrain = pd.DataFrame(xTrain["CO2"])
xTest = pd.DataFrame(xTest["CO2"])

svr = SVR("linear")
svrmodel = svr.fit(xTrain, yTrain)

predict = svrmodel.predict(xTest)

print(r2_score(yTest, predict))
# R2 --> 0.77
print()

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --> 0.07
print()

params = {"C": np.arange(0.1, 3, 0.1)}

svrcv = GridSearchCV(svrmodel, params, cv=10).fit(xTrain, yTrain)

print(svrcv.best_params_)
best = pd.Series(svrcv.best_params_)[0]
print()

svrmodeltuned = SVR("linear", C=best).fit(xTrain, yTrain)

predicttuned = svrmodeltuned.predict(xTest)

print(r2_score(yTest, predicttuned))
# R2 --> 0.80
print()

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(errortuned))
# E --> 0.06
print()
"""

# NOT DIRECTLY SVR MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

svrrbf = SVR("rbf")
svrrbfmodel = svrrbf.fit(xTrain, yTrain)

predict = svrrbfmodel.predict(xTest)

error = mean_squared_error(yTest, predict)
print(np.sqrt(error))
# E --z 0.09
print()

params = {"C": [0.1, 0.4, 5, 6, 7, 8, 10, 20, 30]}

svrrbfcv = GridSearchCV(svrrbfmodel, params, cv=10).fit(xTrain, yTrain)

print(svrrbfcv.best_params_)
print()

svrrbfmodeltuned = SVR("rbf", C=0.1).fit(xTrain, yTrain)

predicttuned = svrrbfmodeltuned.predict(xTest)

errortuned = mean_squared_error(yTest, predicttuned)
print(np.sqrt(error))
# E --> 0.09
print()
"""

# ARTIFICIAL NEURAL NETWORKS MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

scaler = StandardScaler().fit(xTrain, yTrain)

xRTrain = scaler.transform(xTrain)
xRTest = scaler.transform(xTest)

mlp = MLPRegressor()
mlpmodel = mlp.fit(xRTrain, yTrain)

predict = mlpmodel.predict(xRTrain)

print(r2_score(yTrain, predict))
# R2 --> 0.88
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# E --> 0.06
print()

params = {"alpha": [0.01, 0.02, 0.03, 0.1, 0.2],
          "hidden_layer_sizes": [(20, 20), (100, 50, 100), (300, 200, 250)],
          "activation": ["relu", "logistic"]}

mlpcv = GridSearchCV(mlpmodel, params, cv=10).fit(xRTrain, yTrain)

print(mlpcv.best_params_)
print()

mlpmodeltuned = MLPRegressor(alpha=0.02, hidden_layer_sizes=(20, 20), activation="logistic").fit(xRTrain, yTrain)

predicttuned = mlpmodeltuned.predict(xRTrain)

print(r2_score(yTrain, predicttuned))
# R2 --> 0.08
print()

error = mean_squared_error(yTrain, predicttuned)
print(np.sqrt(error))
# E --> 0.18
print()
"""

# CLASSIFICATION AND REGRESSION TREES MODELS & ERROR & TUNING & PREDICT
"""
x = data.drop("Temp", axis=1)

y = data["Temp"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=123)

xTrain = pd.DataFrame(xTrain["CO2"])
xTest = pd.DataFrame(xTest["CO2"])

cart = DecisionTreeRegressor()
cartmodel = cart.fit(xTrain, yTrain)

predict = cartmodel.predict(xTrain)

print(r2_score(yTrain, predict))
# R2 --> 1
print()

error = mean_squared_error(yTrain, predict)
print(np.sqrt(error))
# E --> 0
print()
"""

# FUTURE PREDICT CHECKING

x = data[["Year"]]
t = data[["CO2"]]
s = data[["CH4"]]
p = data[["N2O"]]
g = data[["Aerosols"]]
k = data[["Population"]]
r = data[["CO2emissionspercapita"]]

y = data["Temp"]

l = LinearRegression()
modell = l.fit(x, y)

for i in range(2020, 2060):
    print(f"{i} -->")
    predict = modell.predict([[i]])
    print(predict)

print()
