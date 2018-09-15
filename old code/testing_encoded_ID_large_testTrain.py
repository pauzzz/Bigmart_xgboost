import pandas as pd
import numpy as np
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

le=LabelEncoder()

data['Outlet']= le.fit_transform(data['Outlet_Identifier'])

var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet','Item_Identifier']

le=LabelEncoder()

for i in var_mod:
  data[i]=le.fit_transform(data[i])

#One Hot Coding
  
data=pd.get_dummies(data,columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type_Combined', 'Outlet','Item_Identifier'])

data.dtypes


data.head(10)

#Drop the columns which have been converted to different types:

data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1, inplace=True)

#Divide into test and train then delete source and item outlet sales columns
newTrain=data.loc[data['source']=='train']
newTest=data.loc[data['source']=='test']

newTest.drop(['Item_Outlet_Sales', 'source'],axis=1,inplace=True)
newTrain.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:

newTrain.to_csv('train_modified.csv',index=False)
newTest.to_csv('test_modified.csv',index=False)

### FINISHED CLEANING, NOW MODELING ####

#Baseline model : mean-based
mean_sales=newTrain['Item_Outlet_Sales'].mean()

#define a dataframe with IDs for submission:
base1=newTest[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales']=mean_sales

#Export submission file:

base1.to_csv('algorithm1.csv',index=False)

#Baseline gives one value for all stores because it is mean-based. It's a baseline.





















