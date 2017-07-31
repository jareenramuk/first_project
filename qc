import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve


input = pd.read_csv('/Users/sneha_neeraj/Documents/input.csv',low_memory=False)
input['wgt']=1
train, test = train_test_split(input, test_size = 0.3)
#test.info()
train_nomissing = train.replace('NaN',-999)
test_nomissing = test.replace('NaN',-999)
combined_set = pd.concat([train_nomissing, test_nomissing], axis = 0) # Stacks them vertically
#train_nomissing.info()
#test_nomissing.info()
#test_nomissing.info()
#combined_set.info()

for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer
        
#combined_set.info()
test_nomissing['product'].unique()
final_train = combined_set[:train_nomissing.shape[0]] # Up to the last initial training set row
final_test = combined_set[train_nomissing.shape[0]:] # Past the last initial training set row
y_train = final_train.pop('target')
y_test = final_test.pop('target')

combined_set1 = train.iloc[1:100,1:4]
stats = pd.DataFrame([])
for feature in train.columns: # Loop through all columns in the dataframe
    if train[feature].dtype != 'object': # Only apply for columns with categorical strings
        stats[feature] = train[feature].describe()
stats.to_csv('/Users/sneha_neeraj/Documents/numeric_train.csv',index=0)


stats = pd.DataFrame([])
for feature in test.columns: # Loop through all columns in the dataframe
    if test[feature].dtype != 'object': # Only apply for columns with categorical strings
        stats[feature] = test[feature].describe()
stats.to_csv('/Users/sneha_neeraj/Documents/numeric_test.csv',index=0)


from collections import Counter

trial = train[['product','PREMIUM','wgt','target','AGE8']]
print (trial.columns)

catDF = pd.DataFrame() #creates a new dataframe that's empty
categorical = pd.DataFrame()
charDF = pd.DataFrame() #creates a new dataframe that's empty
character = pd.DataFrame()
numDF = pd.DataFrame() #creates a new dataframe that's empty

for feature in trial.columns: # Loop through all columns in the dataframe
    if feature == 'wgt' or feature == 'target':
        continue
    print (feature)    
    categ = True if len(train_nomissing[feature].unique())<20 else False
    charac = True if (train_nomissing[feature].dtype == 'object') else False
    if charac:
        trialDF = train_nomissing.sort_values(by = [feature],ascending = False).reset_index(drop=True)
        character['target'] = trialDF['target'].groupby(trialDF[feature],as_index=True).sum()
        character['count'] = trialDF['wgt'].groupby(trialDF[feature],as_index=True).sum()
        character['varname'] = feature
        charDF = charDF.append(character, ignore_index = True)
        
    elif categ:
        trialDF = train_nomissing.sort_values(by = [feature],ascending = False).reset_index(drop=True)
        categorical['target'] = trialDF['target'].groupby(trial[feature],as_index=True).sum()
        categorical['count'] = trialDF['wgt'].groupby(trial[feature],as_index=True).sum()
        categorical['varname'] = feature
        catDF = catDF.append(categorical, ignore_index = True)   
        
    else:
        numDF=numDF.append(lorenz(train,'target',feature,"test",'wgt'),ignore_index=True)

charDF.to_csv("/Users/sneha_neeraj/Documents/char_bin.csv",index=True)
catDF.to_csv("/Users/sneha_neeraj/Documents/cat_bin.csv",index=True)
numDF.to_csv("/Users/sneha_neeraj/Documents/cat_bin.csv",index=True)    
    




depVar = 'target'
keyVar = 'index'
wgtVar = 'wgt'
def gini(inputDF,actualCol,scoreCol,lbl,wgtCol=None):
    if wgtCol is None:
        wgtCol=1
    binCount = 20
    binSize = np.ceil(inputDF.shape[0]/binCount)
    inputDF["wgtScore"] = inputDF[wgtVar]*inputDF[scoreCol]
    inputDF["wgtActual"] = inputDF[wgtVar]*inputDF[actualCol]
    
    inputDF = inputDF.sort_values(by = [actualCol],ascending = False).reset_index(drop=True)
    inputDF["actcumwgt"] = inputDF[wgtVar].cumsum()
    inputDF["actualBin"] = np.floor((inputDF.actcumwgt)/binSize)
    
    inputDF = inputDF.sort_values(by = [scoreCol],ascending = False).reset_index(drop=True)
    inputDF["scrcumwgt"] = inputDF[wgtVar].cumsum()
    inputDF["scoreBin"] = np.floor((inputDF.scrcumwgt)/binSize)
    
    mergedDF = inputDF[wgtCol].groupby(inputDF["actualBin"],as_index=True).sum().to_frame("count")
    
    mergedDF["bestMean"] = inputDF["wgtActual"].groupby(inputDF["actualBin"],as_index=True).sum()/mergedDF["count"]
    mergedDF["respMean"] = inputDF["wgtActual"].groupby(inputDF["scoreBin"],as_index=True).sum()/mergedDF["count"]
    mergedDF["predMean"] = inputDF["wgtScore"].groupby(inputDF["scoreBin"],as_index=True).sum()/mergedDF["count"]
    
    mergedDF["bestPct"] = mergedDF["bestMean"]/mergedDF["bestMean"].sum()
    mergedDF["respPct"] = mergedDF["respMean"]/mergedDF["respMean"].sum()  
    
    mergedDF["bestCumPct"] = mergedDF["bestPct"].cumsum()
    mergedDF["respCumPct"] = mergedDF["respPct"].cumsum()
    mergedDF["randomLine"] = mergedDF.index.get_values()/binCount
    
    mergedDF["SumAct"] = inputDF["wgtActual"].groupby(inputDF["scoreBin"],as_index=True).sum()
    mergedDF["SumScr"] = inputDF["wgtScore"].groupby(inputDF["scoreBin"],as_index=True).sum() 
    
    
    mergedDF.to_csv(lbl+".csv")
    
    respAUC = np.sum(mergedDF["respCumPct"] - mergedDF["randomLine"])/binCount
    bestAUC = np.sum(mergedDF["bestCumPct"] - mergedDF["randomLine"])/binCount
    gini = respAUC/bestAUC
    
    accuracy = 1 - np.abs(mergedDF["predMean"] - mergedDF["respMean"]).sum()/mergedDF["respMean"].sum()
    
     
    return ["{0:0.2f}%".format(gini*100),"{0:0.2f}%".format(accuracy*100)]
    
    depVar = 'target'
keyVar = 'index'
wgtVar = 'wgt'
def lorenz(inputDF,actualCol,scoreCol,lbl,wgtCol=None):
    if wgtCol is None:
        wgtCol=1
    binCount = 20
    binSize = np.ceil(inputDF.shape[0]/binCount)
    inputDF["wgtScore"] = inputDF[wgtVar]*inputDF[scoreCol]
    inputDF["wgtActual"] = inputDF[wgtVar]*inputDF[actualCol]
    
    inputDF = inputDF.sort_values(by = [actualCol],ascending = False).reset_index(drop=True)
    inputDF["actcumwgt"] = inputDF[wgtVar].cumsum()
    inputDF["actualBin"] = np.floor((inputDF.actcumwgt)/binSize)
    
    inputDF = inputDF.sort_values(by = [scoreCol],ascending = False).reset_index(drop=True)
    inputDF["scrcumwgt"] = inputDF[wgtVar].cumsum()
    inputDF["scoreBin"] = np.floor((inputDF.scrcumwgt)/binSize)
    
    mergedDF = inputDF[wgtCol].groupby(inputDF["actualBin"],as_index=True).sum().to_frame("count")
    
    mergedDF["bestMean"] = inputDF["wgtActual"].groupby(inputDF["actualBin"],as_index=True).sum()/mergedDF["count"]
    mergedDF["respMean"] = inputDF["wgtActual"].groupby(inputDF["scoreBin"],as_index=True).sum()/mergedDF["count"]
    mergedDF["predMean"] = inputDF["wgtScore"].groupby(inputDF["scoreBin"],as_index=True).sum()/mergedDF["count"]
    
    mergedDF["bestPct"] = mergedDF["bestMean"]/mergedDF["bestMean"].sum()
    mergedDF["respPct"] = mergedDF["respMean"]/mergedDF["respMean"].sum()  
    
    mergedDF["bestCumPct"] = mergedDF["bestPct"].cumsum()
    mergedDF["respCumPct"] = mergedDF["respPct"].cumsum()
    mergedDF["randomLine"] = mergedDF.index.get_values()/binCount
    
    mergedDF["SumAct"] = inputDF["wgtActual"].groupby(inputDF["scoreBin"],as_index=True).sum()
    mergedDF["SumScr"] = inputDF["wgtScore"].groupby(inputDF["scoreBin"],as_index=True).sum() 
    
    
    mergedDF.to_csv(lbl+".csv")
    
    respAUC = np.sum(mergedDF["respCumPct"] - mergedDF["randomLine"])/binCount
    bestAUC = np.sum(mergedDF["bestCumPct"] - mergedDF["randomLine"])/binCount
    gini = respAUC/bestAUC
    
    accuracy = 1 - np.abs(mergedDF["predMean"] - mergedDF["respMean"]).sum()/mergedDF["respMean"].sum()
    
    mergedDF['variable'] =  scoreCol
    return [mergedDF]
    
 
