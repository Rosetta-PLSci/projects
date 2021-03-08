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

Antartica = pd.read_csv("IceMean.csv")
df = Antartica.copy()

DataTEMP = df["TEMP"]
DataMASS = df["MEANMASS"]
DataPOP = df["POP"]

# INFO
"""
print(df.info())
print()
print(df.describe())
print()
print(df.corr())
# POP & YEAR --> 0.99
# POP & TEMP --> 0.84
# YEAR & TOTALMASS --> -0.64
# POP & TOTALMASS --z -0.64
print()
print(df.columns)
print()
print(df.value_counts())
print()
print(df.isnull().sum())
print()
print(rp.summary_cont(df[["YEAR", "MEANMASS", "TEMP", "POP"]]))
print()
print(stats.describe(DataMASS))
print()
print(stats.describe(DataTEMP))
print()
print(stats.describe(DataPOP))
print()

"""

# MODEL & PREDICT & HOMOGENEITY & NORMALIZATION
"""
# HOMOGENEITY

staticD, pvalueD = levene(df.MEANMASS, df.TEMP, df.POP)
print("%.4f" % pvalueD)
# pvalue < 0.05
print()

# NORMALIZATION

staticM, pvalueM = shapiro(DataMASS)
print("%.4f" % pvalueM)
staticT, pvalueT = shapiro(DataTEMP)
print("%.4f" % pvalueT)
# pvalue < 0.05
staticP, pvalueP = shapiro(DataPOP)
print("%.4f" % pvalueP)
print()

# MODEL WITH SKLEARN

x = df[["POP"]]
y = df["TEMP"]

t = df[["TEMP"]]
s = df["MEANMASS"]

reg = LinearRegression()
regTM = LinearRegression()
modelPT = reg.fit(x, y)
modelTM = regTM.fit(t, s)

print(modelPT.intercept_)
print()
print(modelPT.coef_)
print()
print(modelPT.score(x, y))
print()
print()


print(modelTM.intercept_)
print()
print(modelTM.coef_)
print()
print(modelTM.score(t, s))
print()
print()

# PREDICT WITH SKLEARN
print(modelPT.predict([[10000000000]]))
# The average temperature when there is 10 billion inhabitants --> 1.57 Global Temp
print()

print(modelTM.predict([[1.57]]))
print()
# The average amount of glacier when the earth's degree is 1.57 --> 36.15 GT
"""

# VISUALIZATION
"""
g = sns.regplot(df["MEANMASS"], df["TEMP"], ci=None, scatter_kws={"color": "k", "s": 8})
g.set_xlabel("MEANMASS")
g.set_ylabel("TEMP")
plt.show()


sns.pairplot(df, kind="reg")
plt.show()
sns.jointplot(x="TEMP", y="MEANMASS", kind="reg", data=df)
plt.show()


sns.scatterplot(x="TEMP", y="YEAR", hue="MEANMASS", data=df)
plt.show()

sns.boxplot(x="TEMP", y="MEANMASS", data=df)
plt.show()
"""
