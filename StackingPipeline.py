import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.externals import joblib

# sklearn, Catboost, LightGBM share the same method under the class
class SklearnMethod():
    def __init__(self, model, params=None):
        self.model=model(**params)
    def train(self, x_train, y_train):
        self.model.fit(x_train,y_train)
    def predict(self, x):
        return self.model.predict_proba(x_train)[:,1]
    
class XGboostMethod():
    def __init__(self, params=None):
        self.param= params
    def train(self, x_train, y_train):
        self.df_train= xgb.DMatrix(x_train, label=y_train)
        self.gdbt= xgb.train(self.param, dtrain)
        ###??? different
    def predict(self,x):
        return self.gdbt.predict(xgb.DMatrix(x))

def Stacking(model,train_x,train_y, test,n_splits=5, random_state=None, shuffle=False):
# only return the predictions of the model, no target
# average the prediction on test by every model

    df_kf_valid=np.zeros((train_x.shape[0],))
    df_kf_test =np.zeros((test.shape[0], n_splits))

    kf=KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x)):
        kf_train_x= train_x.reindex(train_index)
        kf_train_y=train_y.reindex(train_index)
        kf_valid_x=train_x.reindex(valid_index)

        model.train(kf_train_x, kf_train_y)

        df_kf_valid[train_index]= model.predict(kf_valid_x)
        df_kf_test[:,i]= model.predict(test)
    df_test=np.mean(df_kf_test ,axis=1)
    #samples x 1 ,
    return df_kf_valid.reshape(-1,1), df_kf_test.reshape(-1,1)

def PipeLine(model_dict,train_x,train_y,test,n_splits=5,random_state=None, shuffle=False):
    # model_dict is a dictionary key--->name, value--->dict{'model': model, 'params': params}
    for key, model_ in model_dict.items():
        if key =='xgboost':
            model_xgb=XGboostMethod(model_['params'])
            xg_train,xg_test=Stacking(model_xgb,train_x, train_y,test,n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        if key =='catboost':
            model_cat=SklearnMethod(CatBoostClassifier,model_['params'])
            cat_train,cat_test=Stacking(model_cat,train_x,train_y, test,n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        if key=='rft':
            model_rft=SklearnMethod(RandomForestClassifier,model_['params'])
            rft_train,rft_test=Stacking(model_rft,train_x,train_y, test,n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        if key=='et':
            model_extra=SklearnMethod(ExtraTreesClassifier, model_['params'])
            et_train,et_teest=Stacking(model_et,train_x,train_y, test,n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        if key=='lightgbm':
            model_light=SklearnMethod(LGBMClassifier,model_['params'])
            light_train,light_test=Stacking(model_light,train_x,train_y, test,n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    stacking_layer_1= pd.concat([xg_train, cat_train,rft_train,et_train, light_train], axis=1)
    stacking_test=pd.concat([xg_teest, cat_teest, rft_teest, et_test, light_test], axis=1)
    return stacking_layer_1, stacking_test


# Now we come to cherry. Use neural network, svm, generalized lienar model to learn function, boost the performance

#  we have stacking_layer_1 and train_y,

#  stacking_test

















