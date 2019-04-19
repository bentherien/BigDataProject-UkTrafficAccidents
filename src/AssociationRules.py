import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

'''Association Rules'''

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



datasetFull= pd.concat([dataset2005_2007,dataset2009_2011,dataset2012_2014])

datasetFull.dtypes
datasetFull.select_dtypes(include=['object'])
datasetFull[datasetFull.isnull().any(axis=1)]


OnehotEncoderSample =pd.get_dummies(datasetFull, columns=["Accident_Severity", "Number_of_Vehicles", "Day_of_Week",
                                     "Time", "Road_Type", "Speed_limit", "Light_Conditions",  "Weather_Conditions",
                                     "Road_Surface_Conditions", "Urban_or_Rural_Area", "Year"],
               prefix=["Accident_Severity", "Number_of_Vehicles",  "Day_of_Week",
                                     "Time", "Road_Type", "Speed_limit", "Light_Conditions",  "Weather_Conditions",
                                     "Road_Surface_Conditions", "Urban_or_Rural_Area", "Year"])

frequent_itemsets = apriori(OnehotEncoderSample, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


sed = rules[ (rules['lift'] >= 3) &
       (rules['confidence'] >= 0.8 ) ].sort_values(["confidence","lift"] ,ascending=[False,False])


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


export_csv = sed.to_csv(r'./CSV_Export/AssocitaionRules.csv', index = True, header=True)






