import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from helpers import *


missing_values = ["n/a", "na", "--", "","NaN","Na","Nan","<"]
dataset2005_2007 = pd.read_csv("./data/accidents_2005_to_2007.csv", low_memory=False, na_values = missing_values)
dataset2009_2011 = pd.read_csv("./data/accidents_2009_to_2011.csv", low_memory=False, na_values = missing_values)
dataset2012_2014 = pd.read_csv("./data/accidents_2012_to_2014.csv", low_memory=False, na_values = missing_values)

dataset2005_2007.fillna(0, inplace=True)
dataset2009_2011.fillna(0, inplace=True)
dataset2012_2014.fillna(0, inplace=True)

dropLight_ = open("./augment/toDropLight.txt","r")
dropHeavy_ = open("./augment/toDropHeavy.txt","r")

dropLight = dropLight_.readlines()
dropHeavy = dropHeavy_.readlines()

for c in dropHeavy:
    dataset2005_2007 = dataset2005_2007.drop(columns=c.strip())
    dataset2009_2011 = dataset2009_2011.drop(columns=c.strip())
    dataset2012_2014 = dataset2012_2014.drop(columns=c.strip())



dataset2005_2007= pd.concat([dataset2005_2007,dataset2009_2011,dataset2012_2014])
data = dataset2005_2007.copy()

data = formatDF(data)

#data = data.drop(columns="Time")
data = data.drop(columns="Year")
data = data.drop(columns="Urban_or_Rural_Area")


X2005 = data.iloc[:, 1:9]
y2005 = data.iloc[:, 0]
X2005_train, X2005_test, y2005_train, y2005_test = train_test_split(X2005, y2005, test_size=0.2, random_state=0)


#Train Model
rf = ExtraTreesClassifier(n_estimators=20, random_state=0)
rf.fit(X2005_train, y2005_train)
y_pred = rf.predict(X2005_test)
score = rf.score(X2005_test,y2005_test)

feature_importances = pd.DataFrame(
    rf.feature_importances_,
    index = X2005_train.columns,
    columns=['importance']).sort_values('importance',ascending=False)


print(feature_importances)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking:")
for f in range(X2005_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X2005_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X2005_train.shape[1]), indices)
plt.xlim([-1, X2005_train.shape[1]])

plt.savefig('withoutTime.png')
plt.show()





