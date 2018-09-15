import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

data=pd.concat([train,test], axis=0)

data.reset_index()

train.info()
test.info()
#fill nulls in item weight
data['Item_Weight']=data['Item_Weight'].fillna(data['Item_Weight'].mean())

data.info()
#Fill nulls in outlet size
#find outlet size median

data['Outlet_Size'].value_counts()
data['Outlet_Size']=data['Outlet_Size'].fillna('Medium')
data.info()

#Clean data in item fat content

for feat in data:
  if data[feat].dtype=='object':
    print(data[feat].unique())

#replace values for uniqueness
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'})

#now ordinal encode for XGB

for feat in data.columns:
  if data[feat].dtype =='object':
    data[feat]=pd.Categorical(data[feat]).codes
    
data.info()

clean_train=data[:train.shape[0]]
clean_test=data[train.shape[0]:]


train_targ=clean_train.pop('Item_Outlet_Sales')
test_targ=clean_test.pop('Item_Outlet_Sales')


#first optimization

cv_params1={'max_depth':[3,5,7], 'min_child_weight':[1,3,5]}

ind_params1={'learning_rate': 0.1 , 'n_estimators':1000, 'seed':123, 'subsample':0.8, 'colsample_bytree':0.8, 'objective':'reg:linear' }

optimized_GBM=GridSearchCV(xgb.XGBRegressor(**ind_params1), cv_params1, scoring='r2', cv=5, n_jobs=-1)

optimized_GBM.fit(clean_train, train_targ)

print("Best parameters found: ",optimized_GBM.best_params_)
print("Score found: ", optimized_GBM.best_score_)

#second optimization: subsampling

cv_params2={'learning_rate': [0.1,0.01], 'subsample': [0.7,0.8,0.9]}
ind_params2={'n_estimatiors':1000, 'seed':123, 'colsample_bytree':0.8, 'objective':'reg:linear', 'max_depth':3, 'min_child_weight':5}


optimized_GBM2=GridSearchCV(xgb.XGBRegressor(**ind_params2), cv_params2, scoring='r2', cv=5, n_jobs=-1)

optimized_GBM2.fit(clean_train, train_targ)


print("Best parameters found: ",optimized_GBM2.best_params_)
print("Score found: ", optimized_GBM2.best_score_)


#setup model

xgdmat=xgb.DMatrix(clean_train,train_targ)
cv_params={'eta':0.1, 'seed':123, 'subsample':0.8, 'colsample_bytree': 0.8, 'objective': 'reg:linear', 'max_depth':3, 'min_child_weight':5, 'learning_rate':0.1}

xgb_cv=xgb.cv(params=cv_params, dtrain=xgdmat, num_boost_round=3000, nfold=5, metrics=['error'], early_stopping_rounds=100)

print(xgb_cv.tail(5))

final_params={'eta':0.1, 'seed':123, 'subsample':0.8, 'colsample_bytree': 0.8, 'objective': 'reg:linear', 'max_depth':3, 'min_child_weight':5, 'learning_rate':0.1}

model=xgb.train(final_params, xgdmat)




#plot feature importance. 

xgb.plot_importance(model)

#make our own nice looking feature importance plot instead of using the builtin xgb.plot_importance

#use seaborn to plot

sns.set(font_scale=1.5)
importances = model.get_fscore()
importances

importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')



testdmat=xgb.DMatrix(clean_test)
y_pred=model.predict(testdmat)
y_pred

test['Item_Outlet_Sales']=y_pred

test=test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
test.set_index('Item_Identifier')
test.to_csv('Submission.csv', index='Item_Identifier')







