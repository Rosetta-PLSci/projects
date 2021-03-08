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
import missingno as msno
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf

# HISTORY

# year --> year
# crop_land --> ecological foodprint of cropland products demanded
# grazing_land --> ecological foodprint of grazingland products demanded
# forest_land --> ecological foodprint of forestland products demanded
# fishing_ground --> ecological foodprint of fishingground products demanded
# built_up_land --> ecological foodprint of builtupland products demanded
# carbon --> ecological foodprint of carbon products demanded
# total --> total carbon

CarbonData = pd.read_csv("WorldFootPr.csv")
data = CarbonData.copy()

# INFO

print(data.info())
print()
print(data.describe())
print()
print(data.isnull().sum())
print()
print(data.corr())
print()
print(data[["Carbon", "Total"]].corr())
print()
print(data.columns)
print()
print(rp.summary_cont(data[['Year', 'Built_up_Land', 'Carbon', 'Total']]))
print()
print(stats.describe(data["Total"]))
print()
print(stats.describe(data["Carbon"]))
print()
print(data.groupby(["Year", "Carbon"])["Total"].mean())
print()

# CHECKING AGAINST VALUE

Total = data["Total"]

Q1 = Total.quantile(0.25)
Q3 = Total.quantile(0.75)
IQR = Q3 - Q1

MinLine = Q1 - 1.5 * IQR
print(MinLine)
MaxLine = Q3 + 1.5 * IQR
print(MaxLine)

MinValues = Total < MinLine
print(Total[MinValues])
print()
MaxValues = Total > MaxLine
print(Total[MaxValues])
print()
# there is no against value


# CHECKING NORMALIZATION

staticN, pvalueN = shapiro(Total)
print("%.4f" % pvalueN)
print()
# pvalue > 0.05 --> 0.0511


# CHECKING HOMOGENEITY

staticH, pvalueH = levene(data.Year, data.Total)
print("%.4f" % pvalueH)
print()
# pvalue < 0.05 --> 0.0000


# VISUALIZATION

sns.scatterplot(x="Year", y="Total", hue="Carbon", data=data)
plt.show()

sns.jointplot(x="Year", y="Total", kind="reg", data=data)
plt.show()

sns.jointplot(x="Year", y="Grazing_Land", kind="reg", data=data)
plt.show()

sns.lmplot(x="Year", y="Total", data=data)
plt.show()

sns.lmplot(x="Carbon", y="Grazing_Land", data=data)
plt.show()

sns.lmplot(x="Total", y="Grazing_Land", data=data)
plt.show()

sns.kdeplot(data.Grazing_Land, shade=True)
plt.show()


# SKLEARN

x = data[["Year"]]
y = data["Total"]

reg = LinearRegression()
modelT = reg.fit(x, y)

print(modelT.intercept_)
print()
print(modelT.coef_)
print()
print(modelT.score(x, y))
print()

predict = reg.predict([[2040]])
print(predict)
