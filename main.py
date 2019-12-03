import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import hashlib
import ctypes
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data_dirname = "./data/"
data_filename = "data.csv"

data = pd.read_csv(data_dirname+data_filename)

#data = data.apply(pd.to_numeric)
#data['facilityId'] = data['facilityId'].astype(int)
#print (data.dtypes)

#def hash(string):
#	return abs(hash(string)) % (10 ** 8)


for col in data:
	data[col] = abs(data[col].apply(hash))


#data["facilityId"] = data["facilityId"].apply(hash)


# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
target = 'numberOfFbiCritical'
X = data.loc[:, data.columns != target]  #independent columns
y = data[target]    # target

"""
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 10 best features
"""

# ----------------------------------
#model = tree.DecisionTreeClassifier(max_depth=200)
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances.nlargest(40))

feat_importances.nlargest(40).plot(kind='barh')
#tree.plot_tree(model)  
plt.show()




# -----------------------------------------------
# RESET
plt.clf()
del X['03e']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances.nlargest(40))

feat_importances.nlargest(40).plot(kind='barh')
#tree.plot_tree(model)  
plt.show()



#-----------------------------
X_new = pd.DataFrame(columns=['pe', 'peDesc', 'description'])
X_new["pe"] = data["pe"]
X_new["peDesc"] = data["peDesc"]
X_new["description"] = data["description"]

print(X_new)

#corrmat = X_new.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()




"""
# ---------------------------------------
ids = data['facilityId'][1:]
seen = set()
uniq = []
double = []
for x in ids:
    if x not in seen:
        uniq.append(x)
        seen.add(x)
    else:
    	double.append(x)

print(len(uniq))
print(double)
"""