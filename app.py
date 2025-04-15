import streamlit as st 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data#every time we load this data it shoud be save into cache rather than loading it again and againd
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names

df,target_names=load_data()

model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

st.sidebar.title("Input features")
sepal_length=st.slider("Select Sepal Length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width=st.slider("Select sepal width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length=st.slider("Select petal length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width=st.slider("Select petal width",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)', 
                                   'petal length (cm)', 'petal width (cm)'])
#prediction
prediction=model.predict(input_data)
predicted_species=target_names[prediction[0]]

st.write(prediction)
st.write(f"the predicted species is  : {predicted_species}")