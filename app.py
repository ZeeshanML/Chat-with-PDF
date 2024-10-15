import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache() 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG With PDF Uploads And Chat History")
st.write("Upload PDFs and chat with their content")

groq_api_key = st.text_input("Enter your GROQ API key", type="password")

# Create the chat model
if groq_api_key:
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)
    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_file = f"temp_{uploaded_file.name}"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            documents.extend(docs)
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(documents=splits,embedding=embeddings)

        retriever = vector_store.as_retriever()

        contextualized_q_system_prompt = (
            """
            Given a chat history and the latest user question,
            which might reference context from the chat history,
            formulate a standalone question which can be understood
            without the chat history. Do not answer the question,
            just re-formulate it if needed and otherwise return it as is.
            """
        )

        contextualized_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualized_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_q_prompt)

        system_prompt = (
            """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the user's question.
            If you don't know the answer, say I don't know. Use three sentences at most and be concise.
            \n\n
            {context}
            """
        )

        QA_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        QA_chain = create_stuff_documents_chain(llm, QA_prompt)
        RAG_chain = create_retrieval_chain(history_aware_retriever, QA_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_RAG_chain = RunnableWithMessageHistory(
            RAG_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_RAG_chain.invoke(
                {"input" : user_input},
                config={"configurable" : {"session_id" : session_id}}
            )

            st.write(st.session_state.store)
            st.write("Assistant: ", response["answer"])
            st.write("Chat History: ", session_history.messages)

else:
    st.warning("Please Enter Your GROQ API Key")


