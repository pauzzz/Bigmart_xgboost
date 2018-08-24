import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv('Train.csv')

test=pd.read_csv('Test.csv')

train['source']='train'
test['source']='test'

data=pd.concat([train,test], sort=True)

print (train.shape, test.shape, data.shape)

data.apply(lambda x: sum(x.isnull()))
pd.set_option('display.max_columns', 10)
data.describe()

data.apply(lambda x: len(x.unique()))

#Filtering categorical columns

categorical_columns=[x for x in data.dtypes.index if data.dtypes[x]=='object']

#Excluding ID columns and source columns:

categorical_columns=[x for x in categorical_columns if x not in ['Item Identifier', 'Outlet Identifier', 'source']]

#Printing frequency of categories

for col in categorical_columns:
  print ('Frequency of Categories for variable '+ col)
  print (data[col].value_counts())


#Get average weight for item  
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#get a bool value to show missing Item_Weight values
miss_bool=data['Item_Weight'].isnull()
sum_miss_bool=str(sum(miss_bool))
#print # missing from original data, imput missing values as mean weight, print final data


print('Original num missing: ' + sum_miss_bool)
#!!!requires .loc because item_avg_weight is a dataframe object.!!
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])

print('Final num missing: ' + str((sum(data['Item_Weight'].isnull()))))

#!!! does not require .loc
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x: mode(data['Outlet_Size'])))

print('Mode for each Outlet Type: ')
print(outlet_size_mode)

#do same as before to input mode as the missing values in the Outlet Size column

miss_bool1=data['Outlet_Size'].isnull()
sum_miss_bool1=str(sum(miss_bool1))


print('Original num missing store sizes: ' +sum_miss_bool1)

data.loc[miss_bool1,'Outlet_Size'] = data.loc[miss_bool1,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

print('Final num missing store sizes: ' +str(sum(data['Outlet_Size'].isnull())))


#now clean item visibility as well using the mean of item visibility

visibility_avg=data.pivot_table(values='Item_Visibility', index='Item_Identifier')
miss_bool2=data['Item_Visibility']==0
sum_miss_bool2=str(sum(miss_bool2))

print('Original num of 0% visibility data: '+sum_miss_bool2)

data.loc[miss_bool2,'Item_Visibility']=data.loc[miss_bool2,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])

print('Final num of 0% visibility data: '+ str(sum(data['Item_Visibility']==0)))

#determine ratio of visibility in particular store to visibility across all stores


data['Item_Visibility_Ratio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)






#do the same as before and create a dictionary to clean fat content
#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

data['Outlet_Years']=2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()
  
print('Original Categories')
print(data['Item_Fat_Content'].value_counts())


data.loc[data['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content']='Non-Edible'

data['Item_Type_Combined'].value_counts()
print(data['Item_Fat_Content'].value_counts())

print('Modified Categories')
data['Item_Fat_Content']= data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                              'reg':'Regular',
                                                              'low fat':'Low Fat'})
  
print(data['Item_Fat_Content'].value_counts())









#Numerical and One-hot coding of categorical variables #RUN EACH LINE INDIVIDUALLY

le1=LabelEncoder()
le2=LabelEncoder()
data['Outlet']= le1.fit_transform(data['Outlet_Identifier'])
data['Item']=le2.fit_transform(data['Item_Identifier'])
var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

le=LabelEncoder()

for i in var_mod:
  data[i]=le.fit_transform(data[i])

#One Hot Coding
  
datadummies=pd.get_dummies(data,columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type_Combined', 'Outlet', 'Item'])

data.dtypes


data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)

#Drop the columns which have been converted to different types:

data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1, inplace=True)

#Divide into test and train then delete source and item outlet sales columns
newTrain=data.loc[data['source']=='train']
newTest=data.loc[data['source']=='test']

newTest.drop(['source'],axis=1,inplace=True)
newTrain.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:

newTrain.to_csv('train_modified1.csv',index=False)
newTest.to_csv('test_modified1.csv',index=False)

### FINISHED CLEANING, NOW MODELING ####



#read in data
data1=pd.read_csv('train_modified1.csv')
dtest=pd.read_csv('test_modified1.csv')

#param={'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear' }


print(data1.columns)

data1.head()

data1.info()

data1.describe()

#we need to find sales price from Item_Outlet_Sales

target='Item_Outlet_Sales'
data1.drop(['Item_Fat_Content','Item_Identifier','Outlet_Identifier'], axis=1, inplace=True)
dtest.drop(['Item_Fat_Content','Item_Identifier','Outlet_Identifier'], axis=1, inplace=True)
y = data1.loc[:,target]

data1.head(5)
y.head(5)
#converting dataset into xgb DMatrix


data_dmatrix = xgb.DMatrix(data=data1,label=y)

#now we train test split using train_test_split which splits arrays or matrices into random train and test subsets. used for cross validation

X_train, X_test, y_train, y_test = train_test_split(data1,y, test_size=0.25, random_state=144)

#instantiate a XGBoost regressor object with hyper-params passed as args. Use XGBClassifier() for classification type problems

xg_reg=xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.3, max_depth=5, alpha=10, n_estimators=10)

#fit model to training data

xg_reg.fit(X_train,y_train)

#perform prediction on test data
print(X_test)
X_test=dtest
print(X_test)

preds= xg_reg.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,preds))

print(rmse)

#RMSE WITH K-FOLD CROSS VALIDATION

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5 , 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,             num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=144)

cv_results.head()

print(cv_results['test-rmse-mean'].tail(1))

#examine the steps the model took to arrive at conclusion

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

#examine the importance of each feature column in the original dataset within the model

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


X_test['Item_Outlet_Sales']=xg_reg.predict(X_test)

X_test['Item_Identifier']=le2.inverse_transform(X_test['Item'])

X_test['Outlet_Identifier']=le1.inverse_transform(X_test['Outlet'])

newPred=X_test

newPred.to_csv('test_sub.csv')
























#Baseline model : mean-based
#mean_sales=newTrain['Item_Outlet_Sales'].mean()

#define a dataframe with IDs for submission:
#base1=newTest[['Item_Identifier','Outlet_Identifier']]
#base1['Item_Outlet_Sales']=mean_sales

#Export submission file:

#base1.to_csv('algorithm1.csv',index=False)

#Baseline gives one value for all stores because it is mean-based. It's a baseline.





















