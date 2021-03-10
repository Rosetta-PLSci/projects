import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import missingno as msno
from ycimpute.imputer import knnimput
from ycimpute.imputer import iterforest
from ycimpute.imputer import EM
import pandas_datareader as dtr
import researchpy as rp
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import shapiro
from scipy.stats import bernoulli
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import stats
from scipy.stats import levene
from scipy.stats import f_oneway
import statsmodels.stats.api as sms

IceMain = pd.read_csv("IceMean.csv")
data = IceMain.copy()


# INFORMATION
"""
print(data.shape)
print()
print(data.info())
print()
print(data.describe())
print()
print(rp.summary_cont(data[["MEANMASS"]]))
print()
print(data.columns)
print()
"""


# AGAINST VALUE
"""

MeanData = data["MEANMASS"]

Q1 = MeanData.quantile(0.25)
Q3 = MeanData.quantile(0.75)
IQR = Q3 - Q1

MinLine = Q1 - 1.5 * IQR
print(MinLine)
MaxLine = Q3 + 1.5 * IQR
print(MaxLine)
print()

minValues = MeanData < MinLine
print(MeanData[minValues])
print()
maxValues = MeanData > MaxLine
print(MeanData[maxValues])
print()

TotalAgainstValues = (MeanData < MinLine) | (MeanData > MaxLine)
print(MeanData[TotalAgainstValues])
print()
"""

# MATCHING THRESHOLD
"""
MeanData[maxValues] = MaxLine

# ADDING NEW DF

data["NMEAN"] = MeanData
data.drop("MEANMASS", axis=1, inplace=True)
print(data.head(10))
print()
"""

# CATEGORICAL FOR VISUAL
"""
data.YEAR = pd.Categorical(data.YEAR)
print(data.info())
print()
print(data.describe())
print()
"""

# VISUAL
"""
data["NMEAN"].plot.barh()
plt.show()

sns.distplot(data.NMEAN, bins=50)
plt.show()

sns.scatterplot(x="NMEAN",y="YEAR",data=data)
plt.show()

sns.lmplot(x="NMEAN",y="YEAR",data=data)
plt.show()

sns.pairplot(data,hue="NMEAN",x_vars="YEAR")
plt.show()

MeanData.plot()
plt.show()

data.plot()
plt.show()
"""

# DATA A. FOR TEMP AND MEAN
"""
print(data.columns)
print()
print(data.describe().T)
print()
print(stats.describe(data["MEANMASS"]))
print()
print(stats.describe(data["TEMP"]))
print()

print(data[["TEMP", "MEANMASS"]].cov())
print()
print(data[["TEMP", "MEANMASS"]].corr())
print()

LessTwenty = norm.cdf(25, 38.164737, 5.229237)
print(LessTwenty)

Between25and35 = norm.cdf(35, 38.164737, 5.229237) - norm.cdf(25, 38.164737, 5.229237)
print(Between25and35)
print()

statisticS, pvalueS = shapiro(data)
print(int(pvalueS))
statisticN, pvalueN = stats.ttest_1samp(data["MEANMASS"], popmean=25)
print(int(pvalueN))



sns.catplot(x="YEAR", y="MEANMASS", data=data)
plt.show()

sns.scatterplot(x="TEMP", y="MEANMASS", data=data)
plt.show()

sns.lmplot(x="TEMP", y="MEANMASS", data=data)
plt.show()

sns.boxplot(x="TEMP", y="MEANMASS", data=data)
plt.show()

sns.barplot(x="TEMP", y="MEANMASS", data=data)
plt.show()

sns.lineplot(x="MEANMASS", y="YEAR", hue="TEMP", data=data)
plt.show()

sns.kdeplot(data.MEANMASS, shade=True)
plt.show()

sns.distplot(data.MEANMASS, kde=False)
plt.show()

(sns.FacetGrid(data,
               hue="MEANMASS",
               height=10,
               xlim=(0, 2))
 .map(sns.kdeplot, "TEMP", shade=True)
 .add_legend()
)
plt.show()
"""
