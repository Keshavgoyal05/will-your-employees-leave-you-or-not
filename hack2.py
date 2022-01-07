import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import utils
import xgboost as xgb
import lightgbm as lgb

test_data=pd.read_csv('Test.csv',na_values=" ")
training_data = pd.read_csv('Train.csv',na_values=' ')

dict={'Employee_ID':test_data['Employee_ID']}
#print(training_data.dtypes)
#print(test_data.dtypes)

############################# Handling categorial value###########################################################
training_data['Gender'] = training_data.Gender.map({'F': 0, 'M': 1})
test_data['Gender'] = test_data.Gender.map({'F': 0, 'M': 1})

training_data['Relationship_Status'] = training_data.Relationship_Status.map({'Single':0,'Married':1})
test_data['Relationship_Status'] = test_data.Relationship_Status.map({'Single':0,'Married':1})

mean_encode=training_data.groupby('Hometown')['Attrition_rate'].mean()
training_data.loc[:,'Hometown']=training_data['Hometown'].map(mean_encode)
test_data.loc[:,'Hometown']=test_data['Hometown'].map(mean_encode)

mean_encode=training_data.groupby('Unit')['Attrition_rate'].mean()
training_data.loc[:,'Unit']=training_data['Unit'].map(mean_encode)
test_data.loc[:,'Unit']=test_data['Unit'].map(mean_encode)

#print(training_data.Decision_skill_possess.value_counts())
training_data['Decision_skill_possess'] = training_data.Decision_skill_possess.map({'Conceptual':0,'Analytical':1,'Directive':2,'Behavioral':3})
test_data['Decision_skill_possess'] = test_data.Decision_skill_possess.map({'Conceptual':0,'Analytical':1,'Directive':2,'Behavioral':3})

mean_encode=training_data.groupby('Compensation_and_Benefits')['Attrition_rate'].mean()
training_data.loc[:,'Compensation_and_Benefits']=training_data['Compensation_and_Benefits'].map(mean_encode)
test_data.loc[:,'Compensation_and_Benefits']=test_data['Compensation_and_Benefits'].map(mean_encode)

#print(training_data.dtypes)
#print(test_data.dtypes)

########################## Handling null values ##################################################################
training_data['Age']=training_data['Age'].fillna(training_data['Age'].mean())
test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())

training_data['Time_of_service']=training_data['Time_of_service'].fillna(0)
test_data['Time_of_service']=test_data['Time_of_service'].fillna(0)

training_data['Pay_Scale']=training_data['Pay_Scale'].fillna(training_data['Pay_Scale'].mean())
test_data['Pay_Scale']=test_data['Pay_Scale'].fillna(test_data['Pay_Scale'].mean())

training_data['Work_Life_balance']=training_data['Work_Life_balance'].fillna(training_data['Work_Life_balance'].mean())
test_data['Work_Life_balance']=test_data['Work_Life_balance'].fillna(test_data['Work_Life_balance'].mean())

training_data['VAR2']=training_data['VAR2'].fillna(training_data['VAR2'].mean())
test_data['VAR2']=test_data['VAR2'].fillna(test_data['VAR2'].mean())

training_data['VAR4']=training_data['VAR4'].fillna(training_data['VAR4'].mean())
test_data['VAR4']=test_data['VAR4'].fillna(test_data['VAR4'].mean())

#print(training_data.isnull().sum())
#print(test_data.isnull().sum())

########### taking only required features into consideration for our training data ###############################
X_train = training_data.iloc[:, 1:-1].values
y_train = training_data.iloc[:, -1].values
#print(X_train)
#print(y_train)


#test_data = test_data.iloc[:, 1:].values
test_data=X_train
#print(test_data)

############## using different data models and finding the best one based on the Accuracy

#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(y_train)
#print(training_scores_encoded)
#print(utils.multiclass.type_of_target(y_train))
#print(utils.multiclass.type_of_target(y_train.astype('int')))
#print(utils.multiclass.type_of_target(training_scores_encoded))

clf=LinearRegression()
decision=clf.fit(X_train,y_train)


'''
clf=svm.SVR()
decision=clf.fit(X_train,y_train)
'''

'''
clf =xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,max_depth=5,alpha=10,n_estimators=10)
decision=clf.fit(X_train, y_train)
'''

'''
d_train=lgb.Dataset(X_train,label=y_train)
params={}
decision=lgb.train(params,d_train,100)
'''

prediction=decision.predict(test_data)
#print(prediction)

########## creating new dataframe for target result #############################################################
dict.update({'Attrition_rate':prediction})
target_data=pd.DataFrame(dict)
print(target_data)

#target_data.to_csv('target_svm.csv', index=False)
