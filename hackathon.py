# Imports
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')

# Set API keys and the models to use
API_KEY = "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"
model_id = "gpt-3.5-turbo"

# Add your openai api key for use
os.environ["OPENAI_API_KEY"] = API_KEY

llm = ChatOpenAI(model_name = model_id, temperature=0)
df = pd.read_csv('C:\\Users\\willi\\Downloads\\Mall_Customers.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose=True, max_iterations=6)

# Setup streamlit app
# Display the page title and the text box for the user to ask the question
st.title('✨ Query your Data ')
prompt = st.text_input("Enter your question to query your PDF documents")

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response =  agent.run(prompt)

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )