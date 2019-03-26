#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd
import numpy as np
import nltk

#viz y plots bonitos
import matplotlib.pyplot as plt
import seaborn as sb
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 16, 8

#ML
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import (explained_variance_score, roc_auc_score,
                            classification_report, confusion_matrix,
                            roc_curve, accuracy_score)

from sklearn.ensemble import GradientBoostingClassifier
import scikitplot as skp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


data = open('text_classfier.txt').read()
etiquetas, texto = [], []
for i, line in enumerate(data.split("\n")):
    if i>0:
        content = line.split()
        etiquetas.append(content[0])
        texto.append(" ".join(content[1:]))

# create a dataframe using texts and lables
df = pd.DataFrame()
df['texto'] = texto
df['etiqueta'] = etiquetas


# In[126]:


df.etiqueta.value_counts().plot.bar(rot = 0, color='coral', title='Hay muchos documentos en la clase 1 y 2 hay que hacer un undersample')


# In[18]:


# en porcentaje hay casi 50% de clase 1
df.etiqueta.value_counts(normalize=True).plot.bar(rot = 0, color='coral')


# In[72]:


df.etiqueta.value_counts(normalize=True).values


# # Preprocesamiento de datos

# In[50]:


# revisar si todas estan en minusculas
len([j for i in df.texto for j in i if j.isupper()])
#no tiene mayusculas


# In[44]:


# revisar si tiene stopwords
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))


# In[47]:


#no tiene stopwords
print(len([i for i in df.texto]))
print(len([i for i in df.texto if i not in sw]))


# In[51]:


#Se parte el DF en train, test, valid (80%, 10%, 10%) con semilla 8\
rs = check_random_state(8)
Xtrain, Xtest_valid, Ytrain, Ytest_valid = train_test_split(
    df.texto,
    df.etiqueta,
    test_size=0.20,
    random_state=rs)

Xtest, Xvalid, Ytest, Yvalid = train_test_split(
    Xtest_valid, Ytest_valid, test_size=0.50, random_state=rs)


# In[52]:


# vectorizando palabras 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['texto'])

# transformacion a vecotres
xtrain_count =  count_vect.transform(Xtrain)
xtest_count =  count_vect.transform(Xtest)
xvalid_count =  count_vect.transform(Xvalid)


# In[116]:


#Tamaño grid depth
ND = 5
min_depth = 8
max_depth =12
grid_depth =  np.linspace(min_depth,max_depth,ND).astype('int')
# tamaño grid ntrees
NT = 2
min_trees = 300
max_trees = 500
grid_tree = np.linspace(min_trees, max_trees, NT).astype('int')

acc_error = np.zeros((ND,NT))

for i, max_depth in enumerate(grid_depth):
    for j,ntrees in enumerate(grid_tree):
        clf = GradientBoostingClassifier(n_estimators= ntrees, max_depth = max_depth,
                                                  subsample=0.8, max_features='sqrt')

# hacer unestimación sobre train
        clf.fit(xtrain_count, Ytrain)
    # hagamos un predict: tst
        acc_test = accuracy_score(Ytest, clf.predict(xtest_count))
    # y guardemos
        acc_error[i,j] = acc_test
        print("iteracion:",j+1,"numero de arboles:", ntrees,"profundidad maxima:", max_depth,"error:", acc_test)


# In[119]:


clf = GradientBoostingClassifier(max_depth=12, n_estimators=500, 
          subsample=0.8, max_features='sqrt')
# hacer unestimación sobre train
clf.fit(xtrain_count, Ytrain)


# In[120]:


#se hace una prediccion para el conjunto de validacion
yhat_valid = clf.predict(xvalid_count)


# In[121]:


print(classification_report(Yvalid, yhat_valid))


# In[122]:


skp.metrics.plot_confusion_matrix(Yvalid, yhat_valid, normalize=True)


# In[123]:


print('Acuracy: %1.3f' % accuracy_score(Yvalid, yhat_valid))


# In[ ]:


clf.predict()

