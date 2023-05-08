import streamlit as st
from streamlit_chat import message
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key=str(os.environ['PINECONE_API_KEY']),  # find at app.pinecone.io
    environment=str(os.environ['PINECONE_ENV'])  # next to api key in console
)

index_name = str(os.environ['PINECONE_INDEX_NAME'])

# Backend / langchain
def load_chain():
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    docs = chain.similarity_search(user_input)
    output = docs[0].page_content

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
