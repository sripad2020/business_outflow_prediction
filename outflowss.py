import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data=pd.read_csv('outflowset.csv')
print(data.isna().sum())
print(data.describe())
print(data.info())
print(data.columns)
col=data.columns.values
for i in col:
    sn.boxplot(data[i])
    plt.show()
for i in col:
       for j in col:
              plt.plot(data[i].head(250),marker='o',label=f"{i}",color='red')
              plt.plot(data[j].head(250),marker="o",label=f'{j}',color='orange')
              plt.title(f'Its {i} vs {j}')
              plt.legend()
              plt.show()
plt.figure(figsize=(14,6))
corr = data.corr(method='kendall')
my_m=np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()


sn.relplot(data=data,x=data['No.of failedtransactions'],y=data['OverallPerspective'],hue=data['YearOutflow'])
plt.show()


sn.pairplot(data)
plt.show()

sn.displot(data=data,x=data['No.ofchequeBounces'],kind="ecdf")
plt.title('data[No.of chequebounces  with ecdf visualization')
plt.legend()
plt.show()
sn.displot(data=data,x=data['No.ofchequeBounces'],y=data['YearOutflow'],kind="kde")
plt.title('data[No.of chequebounces vs data[yearoutflow] with kde visualization')
plt.legend()
plt.show()
sn.displot(data=data,x=data['No.ofchequeBounces'],y=data['YearOutflow'],kind="hist")
plt.title('data[No.of chequebounces vs data[yearoutflow] with hist visualization')
plt.legend()
plt.show()


data['z-score']=(data.NoOfclosedAccounts-data.NoOfclosedAccounts.mean())/data.NoOfclosedAccounts.std()
df=data[(data['z-score'] >-3)&(data['z-score']<3)]
q1=df.NoOfclosedAccounts.quantile(0.25)
q3=df.NoOfclosedAccounts.quantile(0.75)
iqr=q3-q1
up=q3+1.5*iqr
lo=q1-1.5*iqr
df=df[(df.NoOfclosedAccounts < up)&(df.NoOfclosedAccounts >lo)]

q_1=df.NoOfclosedAccounts.quantile(0.25)
q_3=df.NoOfclosedAccounts.quantile(0.75)
iqR=q_3-q_1
upp=q_3+1.5*iqR
low=q_1-1.5*iqr
df=df[(df.NoOfclosedAccounts < upp)&(df.NoOfclosedAccounts >low)]

qu_1=df.NoOfclosedAccounts.quantile(0.25)
qu_3=df.NoOfclosedAccounts.quantile(0.75)
Iqr=qu_3-qu_1
upper=qu_3+1.5*Iqr
lowe=qu_1-1.5*Iqr
df=df[(df.NoOfclosedAccounts < upper)&(df.NoOfclosedAccounts >lowe)]

sn.countplot(df.OverallPerspective.value_counts().values)
plt.show()
x=df[['FMCG_CID', 'CurrentBalance', 'YearInflow', 'ManualTransactions',
       'onlineTransactions', 'TotalTransactions', 'NoOfclosedAccounts',
       'NewAccountsOpened', 'No.ofchequeBounces', 'No.of failedtransactions',
       'YearOutflow']]
y=df['OverallPerspective']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('This is logistic regression->',lr.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
tre=DecisionTreeClassifier()
tre.fit(x_train,y_train)
print('This is decision tree classification->',tre.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print('This is random forest classification-> ' ,rf.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print('This is kNN classification-> ',knn.score(x_test,y_test))
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print('This is XGB classification-> ',xgb.score(x_test,y_test))
x=df[['FMCG_CID', 'CurrentBalance', 'YearInflow', 'ManualTransactions',
       'onlineTransactions', 'TotalTransactions', 'NoOfclosedAccounts',
       'NewAccountsOpened', 'No.ofchequeBounces', 'No.of failedtransactions',
       'YearOutflow']]
y=df['OverallPerspective']
from sklearn.feature_selection import SelectKBest
skb=SelectKBest(k=7)
d=skb.fit(x,y)
print(skb.get_feature_names_out())
y=pd.get_dummies(y)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[skb.get_feature_names_out()],y)
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x_train.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=3,activation=keras.activations.softmax))
models.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics='accuracy')
models.fit(x_train,y_train,batch_size=20,epochs=30)