from sklearn import preprocessing

#the following makes fatal accidents=1, and non fatal accidents = 3
def modSeverity(x):
    if (x < 2):
        return 1
    else:
        return 3

def mod(x):
    if x==0:
        return 0
    ph= str(x).split(":")
    if(int(ph[1])<30):
        return int(ph[0]+"00")
    else:
        return int(ph[0]+"30")

def modPretty(x):
    if x==0:
        return "00:00:00"
    ph= str(x).split(":")
    if(int(ph[1])<30):
        return ph[0]+":00:00"
    else:
        return ph[0]+":30:00"

def formatTime(data):
    data["Time"] = data["Time"].apply(modPretty)
    return data

#the followingmethod binarizes the accident severity column
def encode(c):
    def fatal(x):
        if x == 1:
            return 1
        else:
            return 0

    def nfatal(x):
        if x == 1:
            return 0
        else:
            return 1

    c.loc[:,"Not_Fatal"] = c["Accident_Severity"].copy()
    c["Fatal"] = c["Accident_Severity"].apply(fatal)
    c["Not_Fatal"] = c["Not_Fatal"].apply(nfatal)
    c = c.drop(columns="Accident_Severity")
    return c

def check():
    print("yes")

def analyse(df, col):
    temp = df[[col, "Fatal", "Not_Fatal"]]
    temp = temp.groupby(col).sum()
    temp["ratio"] = temp.Fatal/temp.Not_Fatal
    temp = temp.sort_values("ratio", ascending=False)
    temp.to_csv('./CSV_Export/'+col+'.csv', index=True, header=True)
    return temp


#The following method is responsible for encoding features for random forest
def formatDF(sample2005_2007):
    sample2005_2007["Accident_Severity"] = sample2005_2007["Accident_Severity"].apply(modSeverity)
    sample2005_2007["Time"] = sample2005_2007["Time"].apply(mod)

    ls = sample2005_2007["Weather_Conditions"].drop_duplicates().tolist()
    WCN = preprocessing.LabelEncoder().fit_transform(ls)
    WC = {}
    count = 0
    for z in ls:
        WC[z] = WCN[count]
        count = count + 1
    ls = sample2005_2007["Road_Type"].drop_duplicates().tolist()
    WCN1 = preprocessing.LabelEncoder().fit_transform(ls)
    WC1 = {}
    count = 0
    for z in ls:
        WC1[z] = WCN1[count]
        count = count + 1
    ls = sample2005_2007["Light_Conditions"].drop_duplicates().tolist()
    WCN3 = preprocessing.LabelEncoder().fit_transform(ls)
    WC3 = {}
    count = 0
    for z in ls:
        WC3[z] = WCN3[count]
        count = count + 1
    ls = sample2005_2007["Road_Surface_Conditions"].drop_duplicates().tolist()
    WCN4 = preprocessing.LabelEncoder().fit_transform(ls)
    WC4 = {}
    count = 0
    for z in ls:
        WC4[z] = WCN4[count]
        count = count + 1
    ls = sample2005_2007["Urban_or_Rural_Area"].drop_duplicates().tolist()
    WCN5 = preprocessing.LabelEncoder().fit_transform(ls)
    WC5 = {}
    count = 0
    for z in ls:
        WC5[z] = WCN5[count]
        count = count + 1

    sample2005_2007["Weather_Conditions"] = sample2005_2007["Weather_Conditions"].apply(lambda x: WC[x])
    sample2005_2007["Road_Type"] = sample2005_2007["Road_Type"].apply(lambda x: WC1[x])
    sample2005_2007["Light_Conditions"] = sample2005_2007["Light_Conditions"].apply(lambda x: WC3[x])
    sample2005_2007["Road_Surface_Conditions"] = sample2005_2007["Road_Surface_Conditions"].apply(lambda x: WC4[x])
    sample2005_2007["Urban_or_Rural_Area"] = sample2005_2007["Urban_or_Rural_Area"].apply(lambda x: WC5[x])

    return sample2005_2007



