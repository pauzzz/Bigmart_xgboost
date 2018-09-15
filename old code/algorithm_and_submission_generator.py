import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#read in data
data=pd.read_csv('train_modified.csv')
dtest=pd.read_csv('test_modified.csv')

#param={'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear' }


print(data.columns)

data.head()

data.info()

data.describe()

#we need to find sales price from Item_Outlet_Sales

target='Item_Outlet_Sales'

X,y=data.loc[:,:target],data.loc[:,target]

print(X)
print(y)
#converting dataset into xgb DMatrix

data_dmatrix = xgb.DMatrix(data=X,label=y)

#now we train test split using train_test_split which splits arrays or matrices into random train and test subsets. used for cross validation

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=144)

#instantiate a XGBoost regressor object with hyper-params passed as args. Use XGBClassifier() for classification type problems

xg_reg=xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)

#fit model to training data

xg_reg.fit(X_train,y_train)


#perform prediction on test data


preds= xg_reg.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,preds))

print(rmse)

#LOWER OUR RMSE WITH K-FOLD CROSS VALIDATION

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

#calc predicteds after running data through xgboost model again
preds= xg_reg.predict(X_test)


