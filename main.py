import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(
  base_url = "https://api.groq.com/openai/v1",
  api_key = os.environ.get("GROQ_API_KEY")
)

def load_model(filename):
    with open(filename, "rb") as file:
      return pickle.load(file)

xgboost_model = load_model("xgb_model.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
svm_model = load_model("svm_model.pkl")
decision_tree_model = load_model("dt_model.pkl")
voting_clf_model = load_model("voting_clf_model.pkl")
xgb_SMOTE_model = load_model("xgb_SMOTE_model.pkl")
xgb_FE_model = load_model("xgb_FE_model.pkl")
knn_model = load_model("knn_model.pkl")

def prepare_input(credit_score , location, gender, age, tenure , balance , num_products, has_credit_card , is_active_member , estimated_salary):
  
  input_dict = {
    'Creditscore': credit_score,
    'Age': age,
    'Tenures': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'GeographyFrance': 1 if location == "France" else 0,
    'GeographySpain': 1 if location == "Spain" else 0,
    'GeographyGermany': 1 if location == "Germany" else 0,
    'GenderMale': 1 if gender == "Male" else 0,
    'GenderFemale': 1 if gender == "Female" else 0
  }
  
  input_df = pd.DataFrame([input_dict])
  return input_df , input_dict

def make_predictions(input_df , input_dict):
  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random forest': random_forest_model.predict_proba(input_df)[0][1],
    'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
    #'SVM': svm_model.predict_proba(input_df)[0][1],
  #  'Decision Tree': decision_tree_model.predict_proba(input_df)[0][1],
    #'Voting Classifier': voting_clf_model.predict_proba(input_df)[0][1],
    #'XGBoost SMOTE': xgb_SMOTE_model.predict_proba(input_df)[0][1],
    'KNN': knn_model.predict_proba(input_df)[0][1],
    #'XGBoost FE': xgb_FE_model.predict_proba(input_df)[0][1]
  }
  average_probability = np.mean(list(probabilities.values()))
  col1,col2 = st.columns(2)
  with col1:
    fig = ut.create_gauge_chart(average_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has {average_probability:.2%} probability of churning.")

  with col2:
    fig_probs =   ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)


  return average_probability


def explain_prediction(probability , input_dict , surname):
  prompt = f"""You are an expert Data Scientist at a Bank, where you specialize in interpreting and explaining the predictions made by a machine learning model. You have access to the model's predictions and the input data used to train.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability*100,1)}% probability of churning based on the information provided below: 

  here is the customer's information:
  {input_dict}

  Here are the model's top features for predicting churn:

        Features           |   Importance 
        CreditScore        |  	0.034457
        Age                |	  0.114715
        Tenure	           |    0.032618
        Balance	           |    0.054220
        NumOfProducts	     |    0.285859
        HasCrCard	         |    0.032506
        IsActiveMember	   |    0.192457
        EstimatedSalary    |	  0.034179
        Geography_France	 |    0.034263
        Geography_Germany	 |    0.105095
        Geography_Spain	   |    0.036177
        Gender_Female      |  	0.043452
        Gender_Male	       |    0.000000
        
        {pd.set_option('display.max_columns', None)}

        Here are the summary statistics for the churned customers
        {df[df['Exited'] == 1].describe()}

        Here are the summary statistics for the non churned customers
        {df[df['Exited'] == 0].describe()}

        -If the customer has over 48% risk of churning, Explain a 3 sentence explaination of why they are churning.
        -If the customer has less than 40% risk of churning, Explain a 3 sentence explaination of why they might not be at  risk of churning.
        -Your explanation should be based on the customer's information, the summary statistics of the churned and non churned customers and feature importances provided.

        Dont mention based on the provided information.
        Dont mention the probabilty of churning or the machine learning model, or say anything like based on model's prediction and top 10 most important features, just explain the prediction.

  """
  print("Explaination prompt" , prompt)
  
  raw_response = client.chat.completions.create(
    model = "llama-3.2-3b-preview",
    messages = [{
      "role" : "user",
      "content" : prompt
    }],
  )
  return raw_response.choices[0].message.content 


def genrate_email(probability , input_dict ,explaination, surname):

  prompt = f""" You are a manager at HB bank. You are ensuring the customer stay with the bank and are incentivizing them to stay with the bank through various offers
  
  You notice that a customer named {surname} has a {round(probability*100,1)}% probability of churning
  
  here is the customer's information:
  {input_dict}

  Genrate an email to the customer based on their information asking them to stay with the bank with if they are at risk of churning, or offering them incentives to become more loyal to the bank.

  make sure list out the set of incentives to stay based on their information in bullet point format. Dont ever mention the probabiltu of churning or the machine learning model, or say anything like based on model's prediction  to the customer.
  
  
  """
  raw_response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [{
      "role" : "user",
      "content" : prompt
    }],
  )
  print("\n\n Email Prompt ", prompt)
  return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

  select_customer_id = int(selected_customer_option.split(" - ")[0])
  select_customer_surname = selected_customer_option.split(" - ")[1]

  selected_customer = df.loc[df["CustomerId"] == select_customer_id].iloc[0]

  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input(
      "Credit Score", 
      min_value=300,
      max_value=850,
      value=int(selected_customer["CreditScore"]))

    location =st.selectbox(
      "Location", ["Spain", "France", "Germany"],
      index = ["Spain", "France", "Germany"].index(
        selected_customer['Geography']))

    gender = st.radio( "Gender", ["Male", "Female"],
                      index = 0 if selected_customer["Gender"] == 'Male' else
                      1)
    
    age = st.number_input(
      "Age",
      min_value=18,
      max_value= 100,
      value=int(selected_customer["Age"]))
    
    tenure = st.number_input(
      "Tenure (Years",
      min_value=0,
      max_value= 50,
      value=int(selected_customer["Tenure"]))

  with col2:
    
    balance = st.number_input(
      "Balance",
      min_value=0.00,
      value = float(selected_customer["Balance"]))
    num_products = st.number_input(
      "Number of Products",
      min_value= 1,
      max_value= 10,
      value = int(selected_customer["NumOfProducts"]))
    has_credit_card = st.checkbox(
      "Has Credit Card",
      value = bool(selected_customer["HasCrCard"]))
    is_active_member = st.checkbox(
      "Is Active Member",
      value = bool(selected_customer["IsActiveMember"]))
    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value = float(selected_customer["EstimatedSalary"]))

  input_df , input_dict = prepare_input(credit_score , location, gender, age, tenure , balance , num_products, has_credit_card , is_active_member , estimated_salary)
  
  average_probability = make_predictions(input_df , input_dict)
  
  explaination = explain_prediction(average_probability , input_dict , selected_customer['Surname'])
  
  st.markdown('----------')
  st.subheader("Explaination of Prediction")
  st.markdown(explaination)

  email = genrate_email(average_probability , input_dict , explaination,selected_customer['Surname'])

  st.markdown('----------')
  st.subheader("Email Prompt")
  st.markdown(email)
