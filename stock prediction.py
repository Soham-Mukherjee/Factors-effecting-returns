#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


# In[2]:


os.chdir('D:/nse data')


# In[3]:


df=pd.read_excel('cluster_25pct.xlsx')


# In[4]:


x_features=['ROCE', 'EV/EBIDTA', 'DE', 'NPM-last year',
       'Sales growth last 5 years', 'Industry PE', 'Stock PE', 'ROE']


# In[5]:


scaler=StandardScaler()


# In[6]:


scaled_stocks=scaler.fit_transform(df[['ROCE', 'EV/EBIDTA', 'DE', 'NPM-last year',
       'Sales growth last 5 years', 'Industry PE', 'Stock PE', 'ROE']])


# In[7]:


cmap=sns.cubehelix_palette(as_cmap=True,rot=-.3,light=1)
sns.clustermap(scaled_stocks,cmap=cmap,linewidth=.5,figsize=(8,8))


# In[8]:


df.iloc[[4,6]]


# In[9]:


cluster_range=range(1,8)
cluster_errors=[]
for num_cluster in cluster_range:
    clusters=KMeans(num_cluster)
    clusters.fit(scaled_stocks)
    cluster_errors.append(clusters.inertia_) 


# In[10]:


plt.figure(figsize=(6,4))
plt.plot(cluster_range,cluster_errors,marker='o')


# In[11]:


k=2
clusters=KMeans(k,random_state=11)
clusters.fit(scaled_stocks)
df['cluster_id']=clusters.labels_


# In[12]:


df[df['cluster_id']==1].Return.value_counts()


# In[13]:


df[df['cluster_id']==0].Return.value_counts()


# # Hierarchical Clustering

# In[14]:


h_clusters=AgglomerativeClustering(2)
h_clusters.fit(scaled_stocks)
df['h_cluster']=h_clusters.labels_


# In[15]:


df[df['h_cluster']==0].Return.value_counts()


# In[16]:


df[df['h_cluster']==1].Return.value_counts()


# In[17]:


df['return_cat']=df.Return.map(lambda x: 1 if x=='High' else 0)


# In[18]:


def draw_roc(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label='ROC CURVE(area=%0.2f)'%auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend(loc='lower right')
    plt.show()
    return fpr,tpr,thresholds


# In[19]:


df.columns


# In[20]:


x_new=['ROCE', 'EV/EBIDTA', 'DE', 'NPM-last year',
       'Sales growth last 5 years', 'Industry PE', 'Stock PE', 'ROE']


# In[21]:


for i in x_new:
    df[i]=(df[i]-min(df[i]))/(max(df[i]-min(df[i])))


# In[22]:


encoded_df=pd.get_dummies(df['Value/Growth'],drop_first=True)


# In[23]:


df_new=encoded_df.join(df)


# In[24]:


df_new.columns


# In[25]:


features=['Value', 'ROCE', 'EV/EBIDTA', 'DE', 'NPM-last year',
       'Sales growth last 5 years', 'Industry PE', 'Stock PE', 'ROE']


# # LOGISTIC REGRESSION-80% ROC SCORE

# In[26]:


x=df_new[features]
y=df_new['return_cat']


# In[27]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=11)


# In[28]:


logit=sm.Logit(train_y,train_x)
logit_model=logit.fit()


# In[29]:


logit_model.summary2()


# In[30]:


y_pred=pd.DataFrame({'actual':test_y,'predicted_prob':logit_model.predict(test_x[features])})


# In[31]:


draw_roc(y_pred.actual,y_pred.predicted_prob)


# #  (KNN)ROC SCORE(TRAIN : 0.84 TEST :0.50)

# In[32]:


tuned_parameters=[{'n_neighbors':range(3,10),'metric':['canberra','euclidean','minkowski']}]
clf=GridSearchCV(KNeighborsClassifier(),tuned_parameters,cv=10,scoring='roc_auc')
clf.fit(train_x,train_y)


# In[33]:


clf.best_score_


# In[34]:


clf.best_params_


# In[35]:


knn_clf=KNeighborsClassifier(metric='canberra',n_neighbors=8)
knn_clf.fit(train_x,train_y)


# # ROC ON THE TEST DATA(ROC : 0.50)

# In[36]:


y_pred_knn=pd.DataFrame({'actual':test_y,'knn':knn_clf.predict(test_x)})


# In[37]:


draw_roc(y_pred_knn.actual,y_pred_knn.knn)


# # Random forest-ROC SCORE(TRAIN : 0.73 TEST : 0.60)

# In[38]:


tuned_parameters=[{'max_depth':[10,15],'n_estimators':[10,20],'max_features':['sqrt',0.2]}]
radm_clf=RandomForestClassifier()
clf=GridSearchCV(radm_clf,tuned_parameters,cv=5,scoring='roc_auc')


# In[39]:


clf.fit(train_x,train_y)


# In[40]:


clf.best_score_


# In[41]:


clf.best_params_


# In[42]:


radm_clf=RandomForestClassifier(criterion='gini',max_depth=2)


# In[43]:


radm_clf.fit(train_x,train_y)


# In[44]:


y_pred_radm=pd.DataFrame({'actual':test_y,"random":radm_clf.predict(test_x)})


# In[45]:


draw_roc(y_pred_radm.actual,y_pred_radm.random)


# # DECISION TREE - ROC SCORE(TRAIN  : 0.63 TEST: 0.73)

# In[46]:


tuned_parameter=[{'criterion':['gini','entropy'],'max_depth':range(2,10)}]
tree_clf=DecisionTreeClassifier()
clf=GridSearchCV(tree_clf,tuned_parameter,cv=10,scoring='roc_auc')
clf.fit(train_x,train_y)


# In[47]:


clf.best_score_


# In[48]:


clf.best_params_


# In[49]:


tree_clf=DecisionTreeClassifier(criterion='gini',max_depth=2)


# In[50]:


tree_model=tree_clf.fit(train_x,train_y)


# In[51]:


y_pred_tree=pd.DataFrame({'actual':test_y,'tree':tree_model.predict(test_x)})


# In[52]:


draw_roc(y_pred_tree.actual,y_pred_tree.tree)


# # BOOSTING 

# # 1. BOOSTING LOGISTIC REGRESSION(ROC SCORE TRAIN : 0.65 TEST :0.50)
# 

# In[53]:


tuned_parameter=[{'n_estimators':range(2,100)}]
ada_clf=AdaBoostClassifier()
clf=GridSearchCV(ada_clf,tuned_parameter,cv=10,scoring='roc_auc')   
clf.fit(train_x,train_y)


# In[54]:


clf.best_score_


# In[55]:


clf.best_params_


# In[56]:


logreg_clf=LogisticRegression()
ada_log_clf=AdaBoostClassifier(logreg_clf,n_estimators=2)
ada_log_clf.fit(train_x,train_y)


# In[57]:


y_pred_ada=pd.DataFrame({'actual':test_y,'ada':ada_log_clf.predict(test_x)})


# In[58]:


draw_roc(y_pred_ada.actual,y_pred_ada.ada)


# # EXTREME GRADIENT BOOSTING(ROC : 0.76)

# In[59]:


gboost_clf=GradientBoostingClassifier(n_estimators=500,max_depth=10)
gboost_clf.fit(train_x,train_y)


# In[60]:


y_pred_gboost=pd.DataFrame({'actual':test_y,'gboost':gboost_clf.predict(test_x)})


# In[61]:


draw_roc(y_pred_gboost.actual,y_pred_gboost.gboost)


# In[62]:


def draw_cm(actual,predicted):
    cm=metrics.confusion_matrix(actual,predicted,[1,0])
    sns.heatmap(cm,annot=True,fmt='.2f',
               xticklabels=['high return','Low Return'],
               yticklabels=['high return','Low Return'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale(enable=True,axis='y')
    plt.show()


# In[63]:


draw_cm(y_pred_gboost.actual,y_pred_gboost.gboost)


# In[64]:


print(metrics.classification_report(y_pred_gboost.actual,y_pred_gboost.gboost))


# In[65]:


feature_rank=pd.DataFrame({'feature':train_x.columns,'importance':gboost_clf.feature_importances_})
feature_rank=feature_rank.sort_values('importance',ascending=False)
sns.barplot(y='feature',x='importance',data=feature_rank)


# In[66]:


df.return_cat.value_counts()


# In[ ]:


cross_val_scores(gboost_clf,train_x,train_y,cv=10,scoring='roc_auc')
print(cv_scores)
print('Mean:',)

