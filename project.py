import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import pickle



os.environ["GOOGLE_API_KEY"] ='AIzaSyC0J8ga-xjEc-4nwFuqFE3EwH0EPVZsZuM'
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index.pkl"

st.title("Get Your Answers!! 🌱")
btn = st.button("Create Knowledgebase")
vectordb = None
if btn:
   # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt",encoding='ISO-8859-1')
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    # Save vectordb as a pickle file
    with open(vectordb_file_path, 'wb') as f:
      pickle.dump(vectordb, f)

    

question = st.text_input("Question: ")

if question:
    with open(vectordb_file_path, 'rb') as f:
      vectordb = pickle.load(f)

   # vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    try:
       response = chain(question)
       st.header("Answer")
       answer = response["result"]
       st.write(answer)
    except:
      answer = "I don't know."
      st.header("Answer")
      st.write(answer)