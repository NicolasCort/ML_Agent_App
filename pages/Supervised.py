from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import os 

if os.path.exists('./dataset.csv'): 
      df = pd.read_csv('dataset.csv', index_col=None)



with st.sidebar: 
      st.image("https://miro.medium.com/v2/resize:fit:720/format:webp/1*DjUEt5--t6lCjYG_MuZlLg.png")
      st.title("Auto Machine Learning Agent for Supervised ML")
      choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
      #st.info("This project application helps you build and explore your data.")
  #choice_upload = st.radio("Navigation",["Regression","Classification"])
  #if choice_upload == "Regression":



if choice == "Upload":
      st.title("Upload Your Dataset (file_name.csv)")
      file = st.file_uploader("Upload Your Dataset")
      if file: 
          df = pd.read_csv(file, index_col=None)
          df.to_csv('dataset.csv', index=None)
          st.dataframe(df)

if choice == "Profiling":
      st.title("Exploratory Data Analysis")
      profile_df = df.profile_report()
      st_profile_report(profile_df)
     



if choice == "Modelling":
      
      st.title("""Choose the Supervised ML""")
      RegorClas = st.radio("",["Regresion","Clasification"])
      if st.button('What is Regresion and Clasification ?'):
    
        st.write("""
        Regression is like predicting the weather ☀️ - you're trying to forecast a continuous value, like temperature or humidity. It's like having a crystal ball 🔮 that tells you what the future holds, but for numbers! If you want to predict a numerical value, like stock prices 📈 or house prices 🏠, then Regression is the way to go.

        Classification, on the other hand, is like sorting laundry 👕👖 - you're trying to categorize data into different groups or classes. It's like playing a game of "Guess Who?" 🤔, but with data! If you want to classify data into different categories, like predicting if an email is spam 📧 or not spam 📩, then Classification is your go-to.

        So which one should you choose? Well, it all depends on what you're trying to predict! Are you interested in numerical values or categories? Once you figure that out, choosing between Regression and Classification will be as easy as pie 🥧!
    
        """)
      st.title("Choose the Target Column")
      chosen_target = st.selectbox('', df.columns)
      if RegorClas == "Regresion":
        from pycaret.regression import setup,pull,compare_models,save_model
        if st.button('Run Modelling'): 
            setup(df, target=chosen_target, silent=True)
            setup_df = pull()
            #st.dataframe(setup_df)
            with st.spinner('Loading... ⏳🤖 Our robots are working hard. Please wait'):
                best_model = compare_models()
                compare_df = pull()
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')   
      if RegorClas == "Clasification":
        from pycaret.classification import setup,pull,compare_models,save_model
        if st.button('Run Modelling'):
            setup(df, target=chosen_target, silent=True)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')


if choice == "Download": 
      st.title("💻 Download the best model now 👇👇👇")
      with open('best_model.pkl', 'rb') as f: 
          st.download_button('Download Model', f, file_name="best_model.pkl")


