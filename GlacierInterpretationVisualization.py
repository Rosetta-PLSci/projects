import cv2
import numpy as np
import glob
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# IMG COLOR RANGE --> 0,0,224,179,248,255

# total 42 main pictures

# TRANSFORM -->
"""
mainImagePath = glob.glob("ICE/*.jpg")

year = 0
for img in mainImagePath:
    image = cv2.imread(img)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    year += 1
    lower = np.array([0, 0, 224])
    upper = np.array([179, 248, 255])
    mask = cv2.inRange(imageHSV, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(f"Cut{year}.jpg", result)
"""

# CREATING DATA -->
"""
mainImagePath = glob.glob("IceYear/*.jpg")

MeanList = []
for img in mainImagePath:
    imgCut = cv2.imread(img)
    MeanList.append(int(np.mean(imgCut)))

print(MeanList)

years = list(range(1979, 2021))

Data = pd.DataFrame({"YEARS": years, "MEAN": MeanList})
Data.to_csv("ICEMEAN.csv", index=False)
"""

# GRAPHIC
"""
Data = pd.read_csv("ICEMEAN.csv")

print(Data.describe())
print()
print(Data.info())
print()
print(Data.corr())
# corr --> year & mean = -0.31
print()

sns.pairplot(Data)
plt.show()

sns.lmplot(x="YEARS",y="MEAN",data=Data)
plt.show()
"""
