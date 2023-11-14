# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# PROGRAM:
```
NAME : ASHWIN KUMAR
REG.NO : 212222100006
```
## CODE
### DATA PREPROCESSING BEFORE FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/a8581dea-565c-43e1-87d0-33a4dd34c2ec)

### CHECKING NULL VALUES:
```
df.isnull().sum()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/110eaeee-c22d-425e-b19d-98c4036572dd)

### DROPPING UNWANTED DATAS:
```
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/bff9649e-30fc-414d-8ac1-02d405a3d6cb)

### DATA CLEANING:
```
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/b0d7afa6-e49d-46a4-8b01-aaccac33dcda)

### REMOVING OUTLIERS:
**Before**
```
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/20074cd3-e12c-44e2-8416-70fcb4ecce0e)

**After**
```
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/685abd88-f933-4475-8a34-dfbffa54238c)

### FEATURE SELECTION:
```
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/8bd1de29-cfe1-4830-96e9-8f903e34d10e)
```
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/b00a63e8-451c-4d6f-9873-6b8e9549c06d)
```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/e9fad6f1-41cc-46a6-877c-7e45120f27f9)
```
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/eb977f77-79c6-41ba-b48c-3a8bf3bfe706)
```
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 
```
### FILTER METHOD:
```
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/3ecb7dd1-2e6b-462b-8680-654be138212d)

### HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/592a76a9-2b1a-4582-8d1e-fbf07ecac5a8)

### BACKWARD ELIMINATION:
```
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/ee297302-0338-4f77-a2fe-7f5e3a641705)

### OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/4f001dd0-b845-410f-8ae2-bf180515a5b5)

### FINAL SET OF FEATURE:
```
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/7061418c-774a-413f-a28f-80ee49330811)

### EMBEDDED METHOD:
```
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex-07/assets/119407186/101a3c10-981c-44a0-b450-d562c35bbfe1)

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
