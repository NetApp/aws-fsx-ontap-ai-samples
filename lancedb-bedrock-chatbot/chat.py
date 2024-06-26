import boto3
import os
from requests_aws4auth import AWS4Auth
import streamlit as st
import random
import lancedb
import numpy as np

from config import config
from typing import Tuple, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores import LanceDB
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


INIT_MESSAGE = {"role": "assistant",
                "content": "Hi! I'm Claude on Bedrock. I can help you with quries on your FSxN data. \n What would you like to know?",
                "documents": []}

def new_chat() -> None:
    st.session_state["sessionId"] = str(random.randint(1, 1000000))
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []

def set_page_config() -> None:
    st.set_page_config(page_title="🤖 Chat with your FSxN data", layout="wide")
    st.title("🤖 Chat with your FSxN data")

def render_sidebar() -> Tuple[Dict, int, str]:
    with st.sidebar:           
        # st.markdown("## Inference Parameters")
        chat_url = os.environ['CHAT_URL']
        model_name_select = st.selectbox(
            'Model',
            list(config["models"].keys()),
            key=f"{st.session_state['sessionId']}_Model_Id",
        )
        db = lancedb.connect(chat_url)
        tables = db.table_names()

        knowledge_base = st.selectbox(
            'Knowledge Base',
            tables,
            key=f"{st.session_state['sessionId']}_Knowledge_Base",
        )

        st.session_state["model_name"] = model_name_select

        model_config = config["models"][model_name_select]

        metadata = st.text_input(
                    'User (SID) filter search',
                    key=f"{st.session_state['sessionId']}_Metadata",
                )  
        with st.container():
            col1, col2 = st.columns(2)
            with col1:   
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("temperature", 1.0),
                    step=0.1,
                    key=f"{st.session_state['sessionId']}_Temperature",
                )   
            with col2:  
                max_tokens = st.slider(
                    "Max Token",
                    min_value=0,
                    max_value=4096,
                    value=model_config.get("max_tokens", 4096),
                    step=8,
                    key=f"{st.session_state['sessionId']}_Max_Token",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                top_p = st.slider(
                    "Top-P",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("top_p", 1.0),
                    step=0.01,
                    key=f"{st.session_state['sessionId']}_Top-P",
                )
            with col2:
                top_k = st.slider(
                    "Top-K",
                    min_value=1,
                    max_value=model_config.get("max_top_k", 500),
                    value=model_config.get("top_k", 500),
                    step=5,
                    key=f"{st.session_state['sessionId']}_Top-K",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                memory_window = st.slider(
                    "Memory Window",
                    min_value=0,
                    max_value=10,
                    value=model_config.get("memory_window", 10),
                    step=1,
                    key=f"{st.session_state['sessionId']}_Memory_Window",
                )
        with st.container():
            with st.expander("Chat URL"):
                form = st.form("chat_form")
                url = form.text_input("Chat Url",chat_url)
                form.form_submit_button("Submit")
                if not url:
                    st.error("Please enter a valid URL")
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")

    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }


    return model_config['model_id'],model_kwargs, memory_window, metadata, knowledge_base, url


def main():
    history = []

    set_page_config() 
    # Generate a unique widget key only once
    if "sessionId" not in st.session_state:
        st.session_state["sessionId"] = str(random.randint(1, 1000000))

    bedrock_model_id, model_kwargs, memory_window, metadata, knowledge_base, url = render_sidebar()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state["messages"] = [INIT_MESSAGE]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["documents"]:
                with st.expander("Sources"):
                    st.write(f"Source: {str(set(message['documents']))}")

    # User-provided prompt
    prompt = st.chat_input()
 
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "documents": []})
        
        # Add assistant message to chat history
        with st.chat_message("assistant"):
            bedrock = invoke_bedrock('1',st.session_state["sessionId"])
            conversation = bedrock.call_bedrock(url,knowledge_base,bedrock_model_id,metadata,model_kwargs)

            content = st.write_stream(bedrock.stream_chain(conversation,prompt))
            with st.expander("Sources"):
                st.write(f"Sources: {str(set(bedrock.doc_url))}")
        st.session_state.messages.append({"role": "assistant", "content": content, "documents": bedrock.doc_url})

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class invoke_bedrock:
    def __init__(self, connectionId,session_id):
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=self.region)
        self.session_id = session_id
        self.doc_url = []
        self.params = {
            "Data":"",
            "ConnectionId": connectionId
        }

    def stream_chain(self, chain,prompt):
         
        response = chain.stream(
            {"input": prompt},
            config={"configurable": {"session_id": self.session_id}},
            )
        self.doc_url = []
        for chunk in response:
            for key in chunk:
                if key == 'answer':
                    yield(chunk[key])
                if key == 'context':
                    self.doc_url.append(chunk[key][0].metadata['full_path'])
        return response
    def set_prompt(self):
        
        ### Contextualize question ###
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        ### Answer question ###
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return qa_prompt,contextualize_q_prompt
    def call_bedrock(self, os_host,knowledge_base,bedrock_model_id, metadata, model_kwargs):
        
        qa_prompt, contextualize_q_prompt = self.set_prompt()
        retriever = self.init_lancedb(os_host,knowledge_base, metadata)

        llm = BedrockChat(
            model_id=bedrock_model_id,
            model_kwargs=model_kwargs,
            streaming=True,
            client=self.bedrock_client,
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain
    
    def init_lancedb(self, host,knowledge_base, metadata):
        
        print(host)
        db = lancedb.connect(host)
        print(db)
        table = db.open_table(knowledge_base)
        print(table)
        dimentions = table.schema.field("vector").type.list_size
        df = table.search(np.random.random((dimentions))) \
            .limit(10) \
            .to_list()
        
        bedrock_embedding_model_id = df[0].get("embedding_model")
        print("bedrock_embedding_model_id")
        print(bedrock_embedding_model_id)
        
        model_kwargs = {}
        if bedrock_embedding_model_id == "amazon.titan-embed-text-v1":
            model_kwargs = {}
        elif bedrock_embedding_model_id == "amazon.titan-embed-text-v2:0":
            model_kwargs = {"dimensions": dimentions}
        else:
            print("Invalid bedrock_embeddings")

        bedrock_embeddings = BedrockEmbeddings(model_id=bedrock_embedding_model_id,
                                               client=self.bedrock_client,model_kwargs=model_kwargs)
        
        vector_store = LanceDB(
                uri=db.uri,
                region=self.region,
                embedding=bedrock_embeddings,
                text_key='document',
                table_name=knowledge_base   
            )
        
        print(metadata)
        if metadata == "":
            retriever = vector_store.as_retriever()
        else:
            sql_filter = f"array_has(acl,'{metadata}') OR array_has(acl,'*:ALLOWED')"
            print(sql_filter)
            retriever = vector_store.as_retriever(search_kwargs={"filter": {
                                                            'sql_filter': sql_filter,
                                                            'prefilter': True
                                                            }
                                                        })
        return retriever

if __name__ == "__main__":
    main()

