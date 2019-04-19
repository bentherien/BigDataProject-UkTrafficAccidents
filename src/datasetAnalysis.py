import pandas as pd
from helpers import *



missing_values = ["n/a", "na", "--", "","NaN","Na","Nan","<"]
dataset2005_2007 = pd.read_csv("./data/accidents_2005_to_2007.csv", low_memory=False, na_values = missing_values)
dataset2009_2011 = pd.read_csv("./data/accidents_2009_to_2011.csv", low_memory=False, na_values = missing_values)
dataset2012_2014 = pd.read_csv("./data/accidents_2012_to_2014.csv", low_memory=False, na_values = missing_values)

dataset2005_2007.fillna(0, inplace=True)
dataset2009_2011.fillna(0, inplace=True)
dataset2012_2014.fillna(0, inplace=True)

dropHeavy_ = open("./augment/toDropHeavy.txt","r")
dropHeavy = dropHeavy_.readlines()

for c in dropHeavy:
    dataset2005_2007 = dataset2005_2007.drop(columns=c.strip())
    dataset2009_2011 = dataset2009_2011.drop(columns=c.strip())
    dataset2012_2014 = dataset2012_2014.drop(columns=c.strip())



dataset2005_2007= pd.concat([dataset2005_2007,dataset2009_2011,dataset2012_2014])

data = encode(formatTime(dataset2005_2007))

cols = list(dataset2005_2007.columns)
cols.pop(0)
cols.remove("Not_Fatal")
cols.remove("Fatal")

for c in cols:
    print(analyse(data,c))
