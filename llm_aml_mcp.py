import os

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def load_data(file_path):
    return pd.read_csv(file_path)

def create_vector_store(data):
    embeddings = OpenAIEmbeddings()
    docs = [str(row) for _, row in data.iterrows()]
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

def create_llm_chain(llm, prompt_template):
    return LLMChain(llm=llm, prompt=prompt_template)

def generate_report(llm_chain, vector_store, transaction_details, news_context):
    # Retrieve relevant information from the knowledge base

    search_results = vector_store.similarity_search(transaction_details)

    context = "\n".join([result.page_content for result in search_results])

    # Prepare the prompt
    prompt_input = {
    "transaction_details": transaction_details,
    "news_context": news_context,
    "known_aml_typologies": context
}
    # Get the LLM response
    response = llm_chain.run(prompt_input)
    return response

if __name__ == '__main__':
    st.title("LLM-Powered AML Risk Management")
    #Dummy data for demonstration
    data_path = "aml_data.csv"
    # Create a dummy CSV file if it doesn't exist
    if not os.path.exists(data_path):
        dummy_data = {
            "transaction_id": [1, 2, 3],
            "customer_id": ["C101", "C102", "C103"],
            "amount": [9000, 15000, 8500],
            "destination": ["High-Risk Country", "Domestic", "High-Risk Country"],
            "description": ["Consulting fee", "Salary payment", "Invoice payment"],
            "risk_type": ["Structuring", "Low-Risk", "Structuring"]
        }
        pd.DataFrame(dummy_data).to_csv(data_path, index=False)

    # Load the data
    aml_data = load_data(data_path)

    # Create the vector store
    vector_store = create_vector_store(aml_data)

    # Define the prompt template for the LLM
    prompt_template = PromptTemplate(
    input_variables=["transaction_details", "news_context", "known_aml_typologies"],
    template="""
    You are an expert AML investigator. Analyze the following transaction details, news, and known AML typologies to provide a risk assessment.
    **Transaction Details**: {transaction_details}
    **News Context**: {news_context}
    **Relevant AML Typologies**: {known_aml_typologies}

    Based on this information, provide a detailed risk score (1-100), a summary of the potential money laundering activity, and a narrative for the AML analyst.
    """
)

    # Initialize the LLM
    llm = OpenAI(temperature=0)

    # Create the LLM chain
    llm_chain = create_llm_chain(llm, prompt_template)

    st.header("Generate AML Risk Report")

    transaction_input = st.text_area("Enter transaction details (e.g., 'Customer C101 received 5 transfers of $8500 from various accounts in High-Risk Country'):")

    news_input = st.text_area("Enter relevant news context (e.g., 'Reports of sanctions evasion involving High-Risk Country'):")

    if st.button("Generate Report"):
        if transaction_input:
            with st.spinner("Generating AML report..."):
             report = generate_report(llm_chain, vector_store, transaction_input, news_input)
             st.subheader("Generated AML Report:")
             st.markdown(report)
        else:
            st.error("Please enter transaction details.")