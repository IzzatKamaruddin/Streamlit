# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB

import pandas as pd

url = "https://raw.githubusercontent.com/IzzatKamaruddin/Streamlit/master/Covid19_Prediction - ClassifactionBasedSpecies.csv"

df = pd.read_csv(url)
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
df.head(500)


df.count()

import numpy as np
import matplotlib.pyplot as plt

df.Species.value_counts().plot(kind = 'barh', color = 'lightblue', figsize = (10,5))
plt.title ("Number of Species datasets");

features = df.drop('Species',axis='columns')
target = df.Species

features.head()

target.head()

import seaborn as sn


corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.33)

print("X train data count :\n", X_train.count())

print("Y test data count :\n", y_test.count())

X_test.head()


from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()

NB_model.fit(X_train, y_train)
NB_pred = NB_model.predict(X_test)

display(NB_pred)

NB_df=pd.DataFrame(NB_pred)
display(NB_df)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in NB_df.columns:
    NB_df[col] = labelencoder.fit_transform(NB_df[col])

y_test = pd.DataFrame(y_test)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in y_test.columns:
  y_test[col] = labelencoder.fit_transform(y_test[col])

display(y_test)

import numpy as np
from sklearn import metrics

tpr, fpr, thresholds = metrics.roc_curve(y_test,NB_df, pos_label=2)
auc = metrics.auc(tpr,fpr )
print(auc)

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('Accuracy of Naive Bayes classifier on training set: {:.3f}'
     .format(NB_model.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.3f}'
     .format(NB_model.score(X_test, y_test)))

NB_model.predict([[0.999,1,0.729,0.845,0.442,0.0]])
#Human coronvirus NL63

NB_model.predict([[0.996,1.000,0.616,0.7520,0.436,0.126]])
#SARS Cov2

NB_model.predict([[0.999,1,0.720,0.840,0.450,0.0]])
#Human coronvirus NL63

NB_model.predict([[0.973,1,0.000,0.746,0.226,0.0]])
#PorcineVirus



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
KNN_pred = knn.predict(X_test)

knn.score(X_train,y_train)

print('Accuracy of KNN classifier on training set: {:.3f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of KNN classifier on test set: {:.3f}'
     .format(knn.score(X_test, y_test)))

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in KNN_df.columns:
    KNN_df[col] = labelencoder.fit_transform(KNN_df[col])

display(KNN_df)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in y_test.columns:
    y_test[col] = labelencoder.fit_transform(y_test[col])

display(y_test)

import numpy as np
from sklearn import metrics

tpr, fpr, thresholds = metrics.roc_curve(y_test,KNN_df, pos_label=2)
auc = metrics.auc(tpr, fpr)
print(auc)

plt.plot(fpr,tpr)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

knn.predict([[0.999,1,0.729,0.845,0.442,0.0]])
#Human coronavirus NL63

knn.predict([[0.973,1,0.000,0.746,0.226,0.0]])
#SARSCov2

knn.predict([[0.54,1.000,0.000,0.619,0.246,0.000]])
#PorcineCov


import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Covid19_Prediction - ClassifactionBasedSpecies.csv")

st.header("Prediction Streamlit App")

option = st.sidebar.selectbox(
    'Select chart to display the data',
     ['bar chart', 'pie chart', 'donut chart'])

if option == 'bar chart':
    chart_data = df['Species'].value_counts()
    st.bar_chart(chart_data)

elif option == 'pie chart':
    chart_data = df['Species'].value_counts()
    st.pie_chart(chart_data)

elif option == 'donut chart':
    chart_data = df['Species'].value_counts()
    st.donut_chart(chart_data)


st.sidebar.header('User Input Parameters')

def user_input_features():
    Nucleus = st.sidebar.slider('Nucleus', 0.0, 1.0, 0.5)
    Exosome = st.sidebar.slider('Exosome', 0.0, 1.0, 0.5)
    Ribosome = st.sidebar.slider('Ribosome', 0.0, 1.0, 0.5)
    Membrane = st.sidebar.slider('Membrane', 0.0, 1.0, 0.5)
    Endoplasmic_Reticulum = st.sidebar.slider('Endoplasmic Reticulum', 0.0, 1.0, 0.5)
    Cytosol = st.sidebar.slider('Cytosol', 0.0, 1.0, 0.5)

    data = {
        'Nucleus': Nucleus,
        'Exosome': Exosome,
        'Ribosome': Ribosome,
        'Membrane': Membrane,
        'Endoplasmic_Reticulum': Endoplasmic_Reticulum,
        'Cytosol': Cytosol
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

st.subheader('User Input:')
st.write(user_input)

NB_prediction = NB_model.predict(user_input)

st.subheader('Prediction:')
st.write(NB_prediction)

KNN_prediction = knn.predict(user_input)

st.subheader('Prediction:')
st.write(KNN_prediction)







