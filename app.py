from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import SQLiteVec
from os import getenv
from dotenv import load_dotenv
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

load_dotenv()

# set nltk data path
nltk.data.path.append("./nltk_data")

st.title("❤ CardioRAG")

# load in PDF for RAG
txt = None
if "retriever" not in st.session_state:
    txt = st.text("Loading knowledge...")
    prog_bar = st.progress(0)
    hf = HuggingFaceEmbeddings(
        model_name="abhinand/MedEmbed-small-v0.1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False},
        cache_folder="./models"
    )
    prog_bar.progress(50)
    retriever = SQLiteVec(embedding=hf, table='langchain', db_file='./vector_sqlite.db').as_retriever(search_kwargs={"k": 5})
    st.session_state["retriever"] = retriever
    prog_bar.progress(100)
if txt:
    txt.text("Loaded knowledge")
else:
    st.text("Loaded knowledge")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who has read the Moss & Adams Cardiology textbook. How can I help you?"}
    ]

# set up a textbox to enter the password if not already set
if "password" not in st.session_state:
    with st.form("pw_input", clear_on_submit=True):
        password = st.text_input("Enter password", type="password")
        if st.form_submit_button("Submit"):
            if password == getenv("PASSWORD"):
                st.session_state["password"] = password
            else:
                st.error("Incorrect password")

with st.form("chat_input", clear_on_submit=True):
    a,b = st.columns([4,1])
    if "messages" in st.session_state and len(st.session_state['messages']) > 1:
        placeholder = "Ask a new question"
    else:
        placeholder = "What is the incidence of congenital heart disease?"
    user_input = a.text_input(
        label="Question:",
        placeholder=placeholder,
        label_visibility="collapsed",
        disabled="password" not in st.session_state,
    )
    print("user_input", user_input)

    submit_pressed = b.form_submit_button("Send", use_container_width=True)

if submit_pressed and user_input and st.session_state["password"]:
    if 'asked_message' in st.session_state:
        st.session_state['messages'] = [{"role": "user", "content": user_input}]
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg['role']):
            st.text(msg["content"])

    with st.chat_message("assistant"):
        assistant_message = st.empty()
    assistant_message.text('Loading...')

    st.session_state['asked_message'] = True

    llm = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.2-3b-instruct",
        streaming=True)
    
    retriever = st.session_state["retriever"]
    docs = retriever.invoke(user_input)
    DIVIDER = "-"*10
    context = DIVIDER.join([f"Page {d.metadata['page_num']}: {d.page_content}" for d in docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful AI assistant who has read the Moss & Adams Cardiology textbook. \
Use the following context to answer the question. If you don't know the answer, just say you don't know.

Context: {context}

Question: {question}

Answer:"""
    )

    response = assistant_message.write_stream(llm.stream(prompt.format(context=context, question=user_input)))
    st.session_state['messages'].append({"role": "assistant", "content": response})

    st.subheader('Sources', divider=True)
    for doc in docs:
        # adding 2 here to convert between zero-index and 1-index and also I messed up somewhere
        with st.expander(f"Page {doc.metadata['page_num']+2}"):
            st.text(str(doc.page_content))
else:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg['role']):
            st.text(msg["content"])