import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import hashlib

# 환경변수 로드
load_dotenv("env.txt")

# 문서 로드 및 분할
def load_and_split_docs(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name, encoding="utf-8")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# 벡터 저장소 생성 (FAISS 사용)
def get_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# RAG 체인 구성
def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        너는 반도체 기술 문서를 기반으로 답변하는 AI야.
        주어진 문서를 참고해 아래 질문에 정확하고 간결하게 답해:

        질문: {question}

        참고 문서:
        {context}
        """
    )
    llm = ChatOpenAI(model="openai/gpt-4.1-mini", temperature=0)
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
    )
    return rag_chain

# Streamlit UI
st.set_page_config(page_title="반도체 문서 RAG 챗봇")
st.title("📘 반도체 기술문서 요약 및 질의응답 챗봇")

# 세션 상태 초기화
if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 파일 업로드
uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF 또는 TXT)", type=["pdf", "txt"])

# 파일 해시 계산 함수
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

if uploaded_file:
    file_hash = get_file_hash(uploaded_file.getvalue())

    # 이전 파일과 다르면 초기화
    if st.session_state.uploaded_file_hash != file_hash:
        st.session_state.uploaded_file_hash = file_hash
        st.session_state.vectordb = None
        st.session_state.rag_chain = None

        with st.spinner("문서를 처리하고 임베딩 중입니다..."):
            split_docs = load_and_split_docs(uploaded_file)
            st.session_state.vectordb = get_vectorstore(split_docs)
            st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb)
            st.success("✅ 문서 처리 완료! 질문을 입력해보세요.")

    question = st.text_input("질문을 입력하세요:")
    if question and st.session_state.rag_chain:
        with st.spinner("답변 생성 중..."):
            result = st.session_state.rag_chain.invoke(
                {"question": question},
            )
            st.write("### 📎 답변:")
            st.write(result.content)
else:
    st.info("PDF 또는 텍스트 문서를 업로드해주세요.")
