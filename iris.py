import pandas as pd
import streamlit as st
import sklearn as datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
#Prévision des fleurs d'Iris''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longeur du sépal',4.3,5.3,7.9)
    sepal_width=st.sidebar.slider('La largeur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longeur du Pétal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Pétal',0.1,2.5,1.3)
    data={'sepal_length' :sepal_length,
          'sepal_width' :sepal_width,
          'petal_width' :petal_width,
          'petal_length' :petal_length
          }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

df=user_input()

st.subheader('on veut trouver la catégorie de cete fleur')
st.write(df)

iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)

st.subheader("La catégorie de la fleur d'Iris est : ")
st.write(iris.target_names[prediction])