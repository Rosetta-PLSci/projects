import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import seaborn as sns
import pandas as pd

# Based on photos taken by Nasa between 1980 and 2021
# The false-color view of the monthly-averaged total ozone over the Antarctic pole.
# The blue and purple colors are where there is the least ozone, and the yellows and reds are where there is more ozon.

"""
# DETECTION BLUE AREA

MainImgPath = glob.glob("ozone/*.png")

for image in MainImgPath:
    testImage = cv2.imread(image)
    years = random.randrange(1980,2022,1)
    imageHSV = cv2.cvtColor(testImage, cv2.COLOR_RGB2HSV)

    lower = np.array([6, 255, 42])
    upper = np.array([34, 255, 244])
    mask = cv2.inRange(imageHSV, lower, upper)
    result = cv2.bitwise_and(testImage, testImage, mask=mask)

    cv2.imwrite(f"New{years}.png", result)
"""

# Average values of the pictures containing the blue area
MainImgPathTwo = glob.glob("year/*.png")

OzoneMeanList = []

for mean in MainImgPathTwo:
    OzonePhoto = cv2.imread(mean)
    OzoneMeanList.append(np.mean(OzonePhoto))

years = list(range(1980, 2020))

MainImgPath = glob.glob("ozone/*.png")

OzoneTotalList = []
for image in MainImgPath:
    MainImage = cv2.imread(image)
    OzoneTotalList.append(int(np.mean(MainImage)))

"""
NewOzoneDic = {
    "YEARS": years,
    "OZONE BLUE MEAN": OzoneMeanList,
    "OZONE GENERAL MEAN": OzoneTotalList
}

NewPandas = pd.DataFrame(NewOzoneDic)
NewPandas.to_csv("OZONEPHOMEAN.csv", index=False)

"""

# Graphic
ozoneMain = pd.read_csv("OZONEPHOMEAN.csv")
ozoneMain = ozoneMain.pivot("YEARS", "OZONE BLUE MEAN", "OZONE GENERAL MEAN")
sns.heatmap(ozoneMain)
plt.show()
