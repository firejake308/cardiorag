from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from os import getenv
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from nltk.tokenize import word_tokenize

load_dotenv()

st.title("♥ CardioRAG")

# load in PDF for RAG
if "retriever" not in st.session_state:
    st.text("Loading PDF...")
    prog_bar = st.progress(0)
    pdf_reader = PyPDF2.PdfReader(open("Moss and Adams 10e Vol 1 & 2.pdf", 'rb'))
    chunks = []
    for page_num in range(60, 600):
        chunks.append(pdf_reader.pages[page_num].extract_text())
        prog_bar.progress((page_num-60+1)/(600-60))
    # put chunks into vector store
    retriever = BM25Retriever.from_texts(chunks, metadatas=[{"page_num": p } for p in range(60, 600)], preprocess_func=word_tokenize)
    st.session_state["retriever"] = retriever
st.text("Loaded PDF")

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
    user_input = a.text_input(
        label="Question:",
        placeholder="What is the incidence of congenital heart disease?",
        label_visibility="collapsed",
        disabled="password" not in st.session_state,
    )
    b.form_submit_button("Send", use_container_width=True)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg['role']):
        st.text(msg["content"])

if user_input and st.session_state["password"]:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    llm = ChatOpenAI(
        api_key=getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.2-3b-instruct",
        streaming=True)
    
    retriever = st.session_state["retriever"]
    docs = retriever.get_relevant_documents(user_input)
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

    with st.chat_message("assistant"):
        response = st.write_stream(llm.stream(prompt.format(context=context, question=user_input)))
    st.session_state['messages'].append({"role": "assistant", "content": response})

    st.subheader('Sources', divider=True)
    for doc in docs:
        with st.expander(f"Page {doc.metadata['page_num']}"):
            st.text(str(doc.page_content))
    