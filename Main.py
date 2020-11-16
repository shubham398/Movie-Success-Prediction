import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# In[6]:
from sklearn import tree
from pandas import *
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import linear_model

from sklearn.linear_model import ARDRegression, LinearRegression
# In[6]:
df = pd.read_csv("D:\\Backup\\.spyder-py3\\movie_metadata - Copy.csv")
df =df.dropna()

colu_names = ['color', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes',
       'actor_1_facebook_likes', 'gross', 'num_voted_users',
       'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'country', 'content_rating', 'budget',
       'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
       'movie_facebook_likes', 'label']

# In[6]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=1)

# In[6]:
from sklearn.preprocessing import LabelEncoder
X = pd.DataFrame()
df = pd.read_csv("D:\\Backup\\.spyder-py3\\movie_metadata - Copy.csv")
df = df.dropna()

# In[6]:

# In[6]:

columnsToEncode = list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
    try:
        df[feature] = le.fit_transform(df[feature])
    except:
        print('Error encoding ' + feature)
df.head()
df.dtypes
# In[6]:
import seaborn as sns
from IPython import get_ipython
get_ipython().magic('matplotlib inline')

corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, vmax=1, square=True)
# In[6]:
def func(x):
    if x >=8.5:
        return 1
    if x>6.9:
         if x<8.6:
             return 2
    if x<6.0:
        return 4
    if x < 7.0:
        if x>5.9:
            return 3
    else:
        return 4
df['label'] = df['imdb_score'].apply(func)

X=df
y=df['label']
# In[6]:

'''for col in colu_names:
    plt.scatter(X[col], y[0])
    plt.show()
'''
# In[6]:
#y.apply(np.round)
X = X.drop(['imdb_score'], axis = 1)
#X = X.drop(['label'], axis = 1)
# In[6]:
from sklearn.model_selection import train_test_split

y = np.array(y).astype(int)
X = np.append(arr=np.ones((3771, 1)).astype(float), values=X, axis=1)   
# In[6]:

                            #appending 1 in first column
import statsmodels.formula.api as sm
def backwardElimination(x, sl):                                                                     #backward elemination
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.1                                                                                         #significance level
X_opt = X
Z=X
X_Modeled = backwardElimination(X_opt, SL) 
      

# In[6]:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Modeled = sc_X.fit_transform(X_Modeled)
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=.2, random_state =0)
# In[6]:
#In[7]:
'''from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='minkowski')
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

from sklearn import metrics
metrics.confusion_matrix(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred)'''
    # In[6]:
'''from sklearn.svm import SVC
regressor = SVC(kernel='rbf')
regressor.fit(X_train, y_train)
yyyyyy = regressor.predict(X_test)
metrics.accuracy_score(y_test, yyyyyy)'''
#In[7]:
#In[7]:
from sklearn.svm import SVR
start_svr = time.time()
c2 = SVR(kernel='rbf')
c2.fit(X_train, y_train)
y2_pred = c2.predict(X_test)
from sklearn.metrics import r2_score
acc1 = r2_score(y_test, y2_pred)
acc1
end_svr = time.time()
elap_svr = end_svr-start_svr
print('SVR took {} seconds with an accuracy of {}'.format(elap_svr,acc1))

from sklearn.ensemble import RandomForestRegressor
start_rfr = time.time()
cl = RandomForestRegressor(n_estimators=100)
cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)
from sklearn.metrics import r2_score
acc2 = r2_score(y_test, y_pred)
acc2
end_rfr = time.time()
elap_rfr = end_rfr - start_rfr
print('Random Forest Regressor took {} seconds with an accuracy of {}'.format(elap_rfr,acc2))

start_lasso = time.time()
reg1 = linear_model.Lasso(alpha=0.1)
reg1.fit(X_train,y_train)
y4_pred = reg1.predict(X_test)
from sklearn.metrics import r2_score
acc3 = r2_score(y_test, y4_pred)
acc3
end_lasso = time.time()
elap_lasso = end_lasso - start_lasso
print('Lasso took {} seconds with an accuracy of {}'.format(elap_lasso,acc3))

from sklearn.tree import DecisionTreeRegressor
start_dt = time.time()
regggg = DecisionTreeRegressor()
regggg.fit(X_train, y_train)
acc4 = r2_score(y_test, regggg.predict(X_test))
acc4
end_dt = time.time()
elap_dt = end_dt - start_dt
print('Decission Tree took {} seconds with an accuracy of {}'.format(elap_dt,acc4))
acclist = [acc1, acc2, acc3, acc4]
time_list = [elap_svr, elap_rfr, elap_lasso, elap_dt]

plt.bar(['svr','rfr', 'lml', 'dtr'],acclist)
plt.title('accuracy')
plt.show()

plt.bar(['svr','rfr', 'lml', 'dtr'],time_list)
plt.title('time')
plt.show()


plt.scatter(['svr','rfr', 'lml', 'dtr'],acclist)
plt.title('accuracy')
plt.show()

plt.scatter(['svr','rfr', 'lml', 'dtr'],time_list)
plt.title('time')
plt.show()
