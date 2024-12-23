import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

# 벡터 스토어 로드 함수
def load_vectorstore(path):
    if os.path.exists(path):
        return FAISS.load_local(
            path,
            OpenAIEmbeddings(openai_api_key=st.session_state["api_key"]),
            allow_dangerous_deserialization=True  # 위험 역직렬화 허용
        )
    return None

# 벡터 스토어 저장 함수 (변경 없음)
def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)