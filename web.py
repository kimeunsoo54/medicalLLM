import os
import tempfile
import streamlit as st
from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai
from langchain_community.vectorstores import FAISS  # pip install langchain-community
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from src.vectorstore import load_vectorstore, save_vectorstore

# 파라미터
VECTORSTORE_PATH = ".vectorstore.faiss"

st.set_page_config(page_title="나만의 의료 챗봇 만들기", page_icon="🩺")

# 세션 상태 초기화
if "api_key" not in st.session_state:
    st.session_state["api_key"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "당신은 대한민국 의사입니다. 해당 문서를 기반으로 사용자의 질문에 대해 답변해주세요. 한국어로 답변해주세요."
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gpt-3.5-turbo"
if "chat_waiting" not in st.session_state:
    st.session_state["chat_waiting"] = False
if "use_rag" not in st.session_state:
    st.session_state["use_rag"] = False
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.0
if "uploaded_file_data" not in st.session_state:
    st.session_state["uploaded_file_data"] = {}  # 원본 파일명 -> bytes 데이터 매핑

# API Key 설정
if st.session_state["api_key"]:
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]
elif os.getenv("OPENAI_API_KEY"):
    st.session_state["api_key"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]
else:
    st.warning("API Key가 설정되지 않았습니다. 사이드바에서 API Key를 입력해주세요.")

# 사이드바: API Key 입력
st.sidebar.header("🔑 API Key 설정")
api_key_input = st.sidebar.text_input("OpenAI API Key를 입력하세요:", type="password", help="복사한 OpenAI API Key를 입력하세요.")
if st.sidebar.button("API Key 저장"):
    if api_key_input.strip():
        st.session_state["api_key"] = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.sidebar.success("API Key가 저장되었습니다.")
    else:
        st.sidebar.warning("유효한 API Key를 입력해주세요.")

# RAG 사용 여부
st.sidebar.header("🔧 RAG 모드")
rag_use = st.sidebar.checkbox("RAG 사용", value=st.session_state["use_rag"], help="PDF 벡터 DB를 이용해 문서 기반 답변을 수행합니다.")
st.session_state["use_rag"] = rag_use

# 사이드바: 초기 안내 (expander로 토글)
with st.sidebar.expander("초기 세팅 가이드", expanded=False):
    st.info(
        """
        **초기 세팅 가이드**
        1. **System Prompt**: 챗봇의 기본 역할 지시문을 수정할 수 있습니다.
        2. **모델 선택**: gpt-3.5-turbo, gpt-4, gpt-4o 중 선택 가능.
        3. **RAG 사용 여부**: 체크 해제 시 문서 없이 대화 가능.
        4. (RAG 활성화 시) PDF 업로드 후 '벡터 DB 생성' 버튼을 눌러 문서 임베딩을 완료한 후 질문을 시작할 수 있습니다.
        """
    )

# 사이드바: System Prompt 설정
st.sidebar.header("📝 System Prompt 설정")
new_system_prompt = st.sidebar.text_area("System Prompt:", value=st.session_state["system_prompt"], height=120, help="챗봇의 역할을 부여하고, 필요한 대답을 할 수 있도록 요청하세요.")
if st.sidebar.button("System Prompt 업데이트"):
    st.session_state["system_prompt"] = new_system_prompt
    st.toast("System Prompt 업데이트 완료!", icon="✅")


# 사이드바: 모델 설정
st.sidebar.header("🤖 모델 설정")
model_option = st.sidebar.selectbox("모델 선택:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], help="가장 싼 모델 : gpt-3.5-turbo, 가장 비싼 모델 : gpt-4o")
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state["temperature"], step=0.1, help="모델의 창의성(Temperature)를 조절합니다.")
if st.sidebar.button("모델 설정"):
    st.session_state["model_name"] = model_option
    st.session_state["temperature"] = temp
    st.toast(f"모델: {model_option}, Temperature: {temp}로 설정되었습니다.", icon="✅")


# RAG 모드일때만 PDF 업로드 섹션 표시
uploaded_files = None
if st.session_state["use_rag"]:
    # 사이드바: PDF 업로드
    st.sidebar.header("📄 PDF 문서 업로드", help="사용자가 챗봇에게 물어볼 파일을 넣어주세요.")
    uploaded_files = st.sidebar.file_uploader("여러 PDF 파일을 선택할 수 있습니다.", type=["pdf"], accept_multiple_files=True)

    # 업로드한 파일을 사이드바에서 바로 다운로드 가능하게 하기
    if uploaded_files:
        for uf in uploaded_files:
            # 원본 파일 데이터를 세션에 저장
            st.session_state["uploaded_file_data"][uf.name] = uf.read()
            # 다시 읽기 위해 seek(0)
            uf.seek(0)

        # 업로드한 파일들 다운로드 버튼 표시 (expander 사용)
        with st.sidebar.expander("업로드한 PDF 파일 보기/다운로드", expanded=False):
            st.markdown("#### 업로드한 PDF 다운로드")
            for i, (file_name, file_data) in enumerate(st.session_state["uploaded_file_data"].items()):
                st.download_button(
                    label=f"📥 {file_name} 다운로드",
                    data=file_data,
                    file_name=file_name,
                    mime="application/pdf",
                    key=f"download_orig_{file_name}_{i}"  # 고유한 키 사용
                )
    # 벡터 DB 생성 버튼
    if uploaded_files and st.sidebar.button("벡터 DB 생성"):
        all_docs = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            temp_dir = tempfile.gettempdir()
            tmp_file_path = os.path.join(temp_dir, file_name)

            with open(tmp_file_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.read())

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        existing_vectorstore = load_vectorstore(VECTORSTORE_PATH)
        if existing_vectorstore is not None:
            st.session_state["retriever"] = existing_vectorstore.as_retriever(search_kwargs={"k": 1})
            st.toast("기존 벡터 DB 로드 완료!", icon="✅")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 1})
            save_vectorstore(vectorstore, VECTORSTORE_PATH)
            st.toast("새 벡터 DB 생성 완료!", icon="✅")

    # RAG 사용인데 retriever가 없으면 경고
    if not st.session_state["retriever"]:
        st.toast("PDF 파일 업로드 후 '벡터 DB 생성' 버튼을 눌러주세요.", icon="⚠️")


# 메인 화면 타이틀
st.title("🩺 나만의 의료 챗봇 만들기")
st.markdown("""
**이 챗봇은 업로드한 의료 관련 문서를 기반으로(또는 RAG 비활성화 시 일반 대화) 질문에 대한 답변을 제공합니다.**  
""")

# 대화 초기화 버튼
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.toast("대화가 초기화되었습니다.", icon="🗑️")

# 메시지 표시용 컨테이너
chat_container = st.container()

#%% 질문 입력
user_query = st.chat_input("질문을 입력하세요...")

if user_query:
    st.session_state["chat_waiting"] = True

    placeholder = chat_container.chat_message("assistant")
    placeholder.markdown("생각중입니다...")  # 임시 표시

    # LLM 설정
    llm = ChatOpenAI(
        model_name=st.session_state["model_name"],
        temperature=st.session_state["temperature"],
        openai_api_key=st.session_state["api_key"]
    )
    system_message = SystemMessagePromptTemplate.from_template(
        st.session_state["system_prompt"]
    )

    if st.session_state["use_rag"]:
        # RAG 모드
        if st.session_state["retriever"]:
            human_message = HumanMessagePromptTemplate.from_template(
                "질문: {question}\n\n문맥: {context}"
            )
            chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            with st.spinner("답변을 생성하는 중입니다..."):
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state["retriever"],
                    chain_type_kwargs={"prompt": chat_prompt},
                    return_source_documents=True
                )
                result = chain({"query": user_query})
            answer = result["result"]
            source_docs = result["source_documents"]
        else:
            # RAG 모드인데 벡터 스토어 없음 → 문맥없이 일반 답변
            response = llm([
                system_message.format_messages()[0],
                HumanMessagePromptTemplate.from_template("{question}").format(question=user_query)
            ])
            answer = response.content
            source_docs = []
    else:
        # RAG 비활성화 모드: 벡터 DB 없이 LLM 단독 응답
        response = llm([
            system_message.format_messages()[0],
            HumanMessagePromptTemplate.from_template("{question}").format(question=user_query)
        ])
        answer = response.content
        source_docs = []

    st.session_state["messages"].append({
        "question": user_query,
        "answer": answer,
        "sources": source_docs
    })

    placeholder.markdown(f"**💬 답변:** {answer}")
    st.session_state["chat_waiting"] = False
    st.rerun()

with chat_container:
    for i, msg in enumerate(st.session_state["messages"]):
        # 사용자 메시지
        with st.chat_message("user"):
            st.markdown(f"**❓ 질문:** {msg['question']}")

        # 어시스턴트 메시지
        with st.chat_message("assistant"):
            st.markdown(f"**💬 답변:** {msg['answer']}")
            if st.session_state["use_rag"] and msg["sources"]:
                with st.expander("🔍 근거 문서 확인하기", expanded=False):
                    st.markdown("**참조한 문서(페이지 정보 및 내용):**")
                    for doc_idx, src_doc in enumerate(msg["sources"]):
                        src_path = src_doc.metadata.get("source", "unknown")
                        page_num = src_doc.metadata.get("page", "Unknown")
                        doc_content = src_doc.page_content
                        snippet = doc_content[:500] + ("..." if len(doc_content) > 500 else "")

                        st.markdown(f"**문서명:** {src_path}")
                        st.markdown(f"**페이지:** {page_num+1}page")
                        st.markdown("**내용 발췌:**\n" + snippet)

                        # PDF 다운로드 버튼
                        if os.path.exists(src_path):
                            import PyPDF2
                            page_number = int(page_num)
                            reader = PyPDF2.PdfReader(src_path)
                            if 0 <= page_number < len(reader.pages):
                                writer = PyPDF2.PdfWriter()
                                writer.add_page(reader.pages[page_number])
                                original_filename = os.path.basename(src_path)
                                base_name, ext = os.path.splitext(original_filename)

                                temp_dir = tempfile.gettempdir()
                                snippet_pdf_path = os.path.join(temp_dir, f"{base_name}-{page_number}p.pdf")
                                with open(snippet_pdf_path, "wb") as snippet_file:
                                    writer.write(snippet_file)

                                with open(snippet_pdf_path, "rb") as f:
                                    snippet_pdf_data = f.read()

                                st.download_button(
                                    label=f"📥 해당 페이지({page_number+1}p) 발췌 PDF 다운로드",
                                    data=snippet_pdf_data,
                                    file_name= f"{base_name}-{page_number+1}p.pdf",
                                    mime="application/pdf",
                                    key=f"snippet_download_btn_{i}_{doc_idx}_{page_number}"  # 고유한 키 사용
                                )
                            else:
                                st.markdown("_해당 페이지를 찾을 수 없습니다._")

                            with open(src_path, "rb") as f:
                                file_data = f.read()
                            file_name = os.path.basename(src_path)
                            st.download_button(
                                label="📥 원문 전체 파일 다운로드",
                                data=file_data,
                                file_name=file_name,
                                mime="application/pdf",
                                key=f"download_btn_{i}_{doc_idx}_{file_name}"  # 고유한 키 사용
                            )
                        else:
                            st.markdown("_이 파일은 더 이상 존재하지 않습니다._")
# 추가적인 CSS 커스터마이징
st.markdown("""
<style>
/* 전체 폰트 및 레이아웃 */
body {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 채팅 컨테이너를 감싸는 요소를 flex 컨테이너로 만들어,
   column-reverse로 최신 메시지가 아래쪽에 위치하도록 함 */
section.main > div.block-container {
    display: flex;
    flex-direction: column-reverse;
}

/* 채팅 메세지 컨테이너 스타일 */
.st-chat-message-user, .st-chat-message-assistant {
    border-radius: 10px; 
    padding: 15px; 
    margin: 10px 0; 
    line-height: 1.6;
    word-break: break-word;
    max-width: 80%;
}

/* 사용자 메시지 스타일 (오른쪽 정렬) */
.st-chat-message-user {
    background-color: #e6f7ff; 
    border: 1px solid #91d5ff; 
    align-self: flex-end; 
    margin-left: auto;
    text-align: right;
}

/* 어시스턴트 메시지 스타일 (왼쪽 정렬) */
.st-chat-message-assistant {
    background-color: #f6ffed; 
    border: 1px solid #b7eb8f; 
    align-self: flex-start; 
    margin-right: auto;
    text-align: left;
}

/* 확장 컨테이너(근거 확인하기) 스타일 */
.st-expander {
    background: #fafafa;
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
}

/* 토스트 스타일(선택사항) */
#toast-container > div {
    background-color: #1890ff !important;
    color: white !important;
}

/* 입력창 스타일 */
.st-chat-input .stTextInput input {
    border-radius: 10px;
    border: 1px solid #d9d9d9;
}
</style>
""", unsafe_allow_html=True)
