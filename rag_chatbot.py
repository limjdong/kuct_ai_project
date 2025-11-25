import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv("env.txt")

# 벡터 저장소 로드 (사전 생성된 임베딩 사용)
def load_vectorstore(vectorstore_path="vectorstore"):
    """사전 생성된 FAISS 벡터 저장소 로드"""
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(
            f"벡터 저장소를 찾을 수 없습니다: {vectorstore_path}\n"
            "먼저 'python create_embeddings.py'를 실행하여 임베딩을 생성하세요."
        )
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

# RAG 체인 구성
def build_rag_chain(vectordb):
    # MMR 방식으로 검색 다양성 확보 + k=4로 증가
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={
            "k": 4,  # 최종 반환 문서 수
            "fetch_k": 10,  # 후보 문서 수
            "lambda_mult": 0.7  # 유사도(1.0)와 다양성(0.0) 균형
        }
    )

    # 개선된 프롬프트: 더 구체적인 답변 유도
    prompt = ChatPromptTemplate.from_template(
        """
        너는 2024년도 장기요양기관 재가급여 평가매뉴얼을 기반으로 답변하는 전문 AI 어시스턴트입니다.

        다음 지침을 따라 답변해주세요:
        1. 관련된 평가지표 번호가 있다면 명시해주세요 (예: "지표 1번 운영규정")
        2. 점수 및 평가기준을 구체적으로 제시해주세요
        3. 참고한 문서의 페이지나 섹션 정보를 표시해주세요
        4. 정확하고 객관적인 정보만 제공하고, 확실하지 않은 경우 "문서에서 명확한 정보를 찾을 수 없습니다"라고 말해주세요

        질문: {question}

        참고 문서:
        {context}

        답변:
        """
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,  # 약간의 창의성 허용
        max_tokens=3000  # 더 상세한 답변 가능
    )

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
st.set_page_config(page_title="장기요양 재가급여 평가 메뉴얼", page_icon="📘")
st.title("📘 장기요양 재가급여 평가 메뉴얼 챗봇")

# 챗봇 소개 및 설명
with st.expander("ℹ️ 챗봇 사용 안내", expanded=False):
    st.markdown("""
    ### 📋 이 챗봇에 대해

    이 챗봇은 **2024년도 장기요양기관 재가급여 평가매뉴얼 Ⅱ (주야간보호, 단기보호)**를 기반으로
    답변을 제공하는 AI 어시스턴트입니다.

    ### 💡 제공하는 정보

    - **평가지표 및 평가기준**: 각 지표별 상세 평가 기준
    - **점수 구성 및 배점**: 평가 척도 및 점수 산정 방법
    - **2020년 vs 2024년 변경사항**: 매뉴얼 개정 내용 비교
    - **평가자 준수사항**: 평가자가 지켜야 할 사항
    - **매뉴얼 일반사항**: 평가 방법, 적용 기간 등

    ### 🎯 활용 예시

    - "주야간보호 평가지표 1번에 대해 설명해주세요"
    - "경력직 평가기준은 무엇인가요?"
    - "2020년과 2024년 매뉴얼의 차이점은?"
    - "인력추가배치 가산점수 계산 방법을 알려주세요"
    - "건강검진 관련 규정이 무엇인가요?"

    ### ⚙️ 시스템 특징

    - **정확한 검색**: MMR(Maximal Marginal Relevance) 알고리즘으로 다양하고 관련성 높은 정보 검색
    - **메타데이터 활용**: 섹션, 페이지, 문서 타입 정보를 활용한 정밀 검색
    - **출처 표시**: 답변에 관련 페이지 및 지표 번호 포함
    - **객관적 답변**: 문서에 기반한 정확하고 객관적인 정보만 제공

    ### ⚠️ 주의사항

    - 이 챗봇은 매뉴얼 문서 내용만을 참조하여 답변합니다
    - 법적 자문이나 공식 결정을 대체할 수 없습니다
    - 정확한 정보는 공식 매뉴얼을 직접 확인해주세요
    """)

st.divider()

# 세션 상태 초기화
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 벡터 저장소 로드 (앱 시작 시 한 번만)
if st.session_state.vectordb is None:
    try:
        with st.spinner("벡터 저장소 로딩 중..."):
            st.session_state.vectordb = load_vectorstore()
            st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb)
            st.success("✅ 문서 준비 완료! 질문을 입력해보세요.")
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("💡 사용 방법:\n1. 터미널에서 `python create_embeddings.py` 실행\n2. 이 페이지 새로고침")
        st.stop()

# 질문 입력
st.subheader("💬 질문하기")

# 예시 질문 버튼
st.write("**빠른 질문 예시:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📋 평가지표 1번", use_container_width=True):
        st.session_state.example_question = "주야간보호 평가지표 1번 운영규정에 대해 설명해주세요"
with col2:
    if st.button("👥 경력직 기준", use_container_width=True):
        st.session_state.example_question = "경력직 평가기준은 무엇인가요?"
with col3:
    if st.button("📊 2020 vs 2024", use_container_width=True):
        st.session_state.example_question = "2020년과 2024년 매뉴얼의 주요 차이점은 무엇인가요?"

# 질문 입력창
if "example_question" in st.session_state:
    question = st.text_input("질문을 입력하세요:", value=st.session_state.example_question)
    del st.session_state.example_question
else:
    question = st.text_input("질문을 입력하세요:", placeholder="예: 건강검진 관련 규정이 무엇인가요?")

# 답변 생성
if question and st.session_state.rag_chain:
    with st.spinner("🔍 관련 문서를 검색하고 답변을 생성하는 중..."):
        result = st.session_state.rag_chain.invoke(
            {"question": question},
        )
        st.success("✅ 답변이 생성되었습니다!")
        st.write("### 📎 답변:")
        st.write(result.content)

        # 참고 정보
        with st.expander("📌 참고 정보"):
            st.info("""
            **답변 생성 방식**
            - MMR 알고리즘으로 4개의 관련 문서 검색
            - GPT-4o-mini 모델로 답변 생성
            - 메타데이터(섹션, 페이지)를 활용한 정확한 출처 제공
            """)
