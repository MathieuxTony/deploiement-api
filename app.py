import pandas as pd
import streamlit as st
from joblib import load


st.write("""
# Application de suggestion de tags

Cette application propose une classification de question sur stackoverflow
""")


def user_input_features():
	text = st.text_area('Entrez votre question', "text")
	return text

df = user_input_features()

clf = load('model/mon_model.joblib')

prediction = clf.predict(df)

st.subheader('Suggestion')
st.write(prediction)