import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import matplotlib
import json

with open("config.json") as file:
   my_file = json.load(file)
openai = my_file["openai"]

matplotlib.use('TkAgg')

# Add your openai api key for use
os.environ["OPENAI_API_KEY"] = openai["api_key"]

# We pass the model name (3.5) and the temperature (Closer to 1 means creative response)
llm = ChatOpenAI(model_name = openai["model_id"], temperature = 0)
df = pd.read_csv('data/Mall_Customers.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose = True, max_iterations = 6)

# Setup streamlit app
# Display the page title and the text box for the user to ask questions
st.title('😺 Query data ')
placeholder = st.empty()
prompt = placeholder.text_input("Enter a question to query the PDF document", key = 1)

if prompt:
    # Get the resonse from LLM
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response =  agent.run(prompt)

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html = True)
    prompt = placeholder.text_input("Enter a question to query the PDF document", value = "", key = 2)