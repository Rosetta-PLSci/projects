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

MainData = pd.read_csv("nsa.csv")
df = MainData.copy()

# GENERAL INFORMATION

print(df.info())
print()
print(df.describe())
print()
print(df.columns)
print()
print(df.shape)
print()
print(df.groupby("Absolute Magnitude")["Relative Velocity km per hr"].count())
print()
print(rp.summary_cat(df[["Hazardous", "Equinox", "Name"]]))
print()
print(rp.summary_cont(df[["Absolute Magnitude", "Relative Velocity km per hr", "Perihelion Distance"]]))
print()

# SEARCHING NaN VALUES

print(df.isnull().sum())
print()

# AGAINST VALUE / Absolute Magnitude

AMData = df["Absolute Magnitude"]

QM1 = AMData.quantile(0.25)
QM3 = AMData.quantile(0.75)
IQRM = QM3 - QM1

minLineM = QM1 - 1.5 * IQRM
print(minLineM)
print()
maxLineM = QM3 + 1.5 * IQRM
print(maxLineM)
print()

minValuesM = AMData < minLineM
maxValuesM = AMData > maxLineM

print(AMData[minValuesM])
print()
print(AMData[maxValuesM])
print()

# MATCHING AGAINST DATA TO THRESHOLD VALUES / Absolute Magnitude

AMData[minValuesM] = minLineM
AMData[maxValuesM] = maxLineM

print(AMData[minValuesM])
print()
print(AMData[maxValuesM])
print()

# ADDING NEW DATA

df["N Absolute Magnitude"] = AMData
df.drop("Absolute Magnitude", axis=1, inplace=True)
print(df.columns)
print()

# AGAINST VALUE / Relative Velocity km per sec

RVData = df["Relative Velocity km per sec"]

QR1 = RVData.quantile(0.25)
QR3 = RVData.quantile(0.75)
IQRR = QR3 - QR1

minLineR = QR1 - 1.5 * IQRR
print(minLineR)
print()
maxLineR = QR3 - 1.5 * IQRR
print(maxLineR)
print()

minValuesR = RVData < minLineR
maxValuesR = RVData > maxLineR
print(RVData[minValuesR])
print()
print(RVData[maxValuesR])
print()

# VISUALIZATION PROCESSES

print(df.columns)

sns.distplot(df["Relative Velocity km per sec"], bins=1000, kde=False).set_title("DISTPLOT")
plt.show()

sns.boxplot(df["Relative Velocity km per sec"], orient="v").set_title("BOX RV")
plt.show()

sns.boxplot(df["N Absolute Magnitude"]).set_title("BOX AM")
plt.show()

(sns.boxplot(x=df["N Absolute Magnitude"].head(10), y=df["Relative Velocity km per sec"].head(10), data=df)
 .set_title("BOX AM & RV"))
plt.show()

(sns.catplot(x=df["N Absolute Magnitude"].head(10),
             y=df["Relative Velocity km per sec"].head(10),
             kind="violin", data=df))
plt.show()

sns.scatterplot(x="Relative Velocity km per sec", y="N Absolute Magnitude", data=df).set_title("SCATTER AM & RV")
plt.show()

sns.lmplot(x="Relative Velocity km per sec", y="N Absolute Magnitude", data=df)
plt.show()

sns.kdeplot(df["Relative Velocity km per sec"], shade=True)
plt.show()
