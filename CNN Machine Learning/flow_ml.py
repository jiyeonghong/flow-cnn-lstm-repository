
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns # visualization
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler



flow=pd.read_csv('reframed_so.csv')

X=flow[['inflow(t-2)','precipitation(t-2)','temp_1(t-2)','temp_2(t-2)','humidity(t-2)','wind(t-2)','solar(t-2)',	
       'inflow(t-1)','precipitation(t-1)','temp_1(t-1)','temp_2(t-1)','humidity(t-1)','wind(t-1)','solar(t-1)',	
       'precipitation(t)','temp_1(t)','temp_2(t)','humidity(t)','wind(t)','solar(t)']]


scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y=flow['inflow(t)']

n_train_hours = 13146
train_X = X[:n_train_hours, :]
test_X = X[n_train_hours:, :]
train_y = y[:n_train_hours]
test_y = y[n_train_hours:]

clf_tree=DecisionTreeRegressor(min_samples_split=2)               
clf_nn=MLPRegressor(hidden_layer_sizes=(50,50,50))
clf_rf=RandomForestRegressor(n_estimators=52)                       
clf_gb=GradientBoostingRegressor(max_depth=10)                       

clf_tree.fit(train_X,train_y)
clf_nn.fit(train_X,train_y)
clf_rf.fit(train_X,train_y)
clf_gb.fit(train_X,train_y)

pre_tree=clf_tree.predict(test_X)
pre_nn=clf_nn.predict(test_X)
pre_rf=clf_rf.predict(test_X)
pre_gb=clf_gb.predict(test_X)

score_t=clf_tree.score(test_X,test_y)
score_n=clf_nn.score(test_X,test_y)
score_r=clf_rf.score(test_X,test_y)
score_g=clf_gb.score(test_X,test_y)

def RMSE(y_pred,y_true):
    rmse = np.sqrt(mean_squared_error(y_pred,y_true))
    return rmse

def print_w(y_pred,y_true): 
    print("RMSE: {:.3f} \n MAE: {:.3f} \n  R: {:.3f} \n  R2: {:.3f}".
       format(RMSE(y_pred,y_true),mean_absolute_error(y_pred,y_true),
       (np.corrcoef(y_pred,y_true))[0,1],(np.corrcoef(y_pred,y_true)[0,1])**2))

    return 

print("\n결정트리\n NSE {:.3f}".format(score_t))
print_w(pre_tree,test_y)
print("\n신경망\n NSE {:.3f}".format(score_n))
print_w(pre_nn,test_y)
print("\n랜덤포레스트\n NSE {:.3f}".format(score_r))
print_w(pre_rf,test_y)
print("\n그래디언트부스트\n NSE {:.3f}".format(score_g))
print_w(pre_gb,test_y)

pre={'DT':pre_tree,'MLP':pre_nn,'RF':pre_rf,'GB':pre_gb,'inflow':test_y}
d_result=DataFrame(data=pre)
d_result.to_csv("result_ml.csv")

#머신러닝 비교그래프 그리기
plt.bar(('DT','MLP','RF','GB'),(score_t,score_n,score_r,score_g),width=0.5,color='G')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Machine Learning')
plt.ylabel('NSE')
plt.savefig('ML.png')
plt.show()

#관측값과 비교 시계열 그래프 그리기
a=np.arange(0,len(test_y),1)
plt.clf() 
ml= {'Decision Tree':pre_tree,'Multi Layer Perceptrons':pre_nn,
    'Random Forest':pre_rf,'Gradient Boosting':pre_gb}
for k,pre_ml in ml.items():
    plt.plot(a,test_y, label='Observation',color='orangered')
    plt.plot(a,pre_ml, label=k,color='darkblue', linestyle='--',marker='o',markersize=3)   
    plt.xlabel("Julian day")
    plt.ylabel("Dam Inflow (m$^3$/s)")
    plt.legend()    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.savefig(k)
    plt.show()
        

