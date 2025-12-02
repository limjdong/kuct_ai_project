import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re

# 환경변수 로드
load_dotenv("env.txt")

# 벡터 저장소 로드 (캐시 적용으로 성능 향상)
@st.cache_resource
def load_vectorstore(vectorstore_path="vectorstore"):
    """사전 생성된 FAISS 벡터 저장소 로드 (캐시됨)"""
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(
            f"벡터 저장소를 찾을 수 없습니다: {vectorstore_path}\n"
            "먼저 'python create_embeddings.py'를 실행하여 임베딩을 생성하세요."
        )
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

# 지표 번호 추출 함수
def extract_indicator_number(question):
    """질문에서 지표 번호를 추출 (예: '지표 1번', '평가지표 5번')"""
    patterns = [
        r'지표\s*(\d+)\s*번',
        r'평가지표\s*(\d+)\s*번',
        r'(\d+)\s*번\s*지표',
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    return None

# 향상된 문서 검색 함수
def search_documents(vectordb, question, k=5, search_type="mmr"):
    """
    질문에 맞는 문서 검색
    - 지표 번호가 있으면 우선 메타데이터 필터링
    - MMR/유사도 검색 사용
    """
    indicator_num = extract_indicator_number(question)

    # 지표 번호가 있는 경우 메타데이터 기반 검색
    if indicator_num:
        # 메타데이터 필터로 특정 지표 검색
        filter_dict = {"type": "평가지표"}
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": filter_dict
            }
        )
        docs = retriever.get_relevant_documents(f"지표 {indicator_num}번")
        # 추가로 지표 번호가 포함된 문서만 필터링
        filtered_docs = [doc for doc in docs if f"{indicator_num}번" in doc.page_content or f"지표{indicator_num}" in doc.page_content]
        if filtered_docs:
            return filtered_docs[:k]

    # 일반 검색 (MMR 또는 유사도)
    if search_type == "mmr":
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,
                "lambda_mult": 0.7
            }
        )
    else:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    return retriever.get_relevant_documents(question)

# RAG 체인은 사용하지 않고 직접 검색 + LLM 호출 방식으로 변경
def generate_answer(vectordb, question, k=5, search_type="mmr"):
    """
    질문에 대한 답변 생성
    Returns: (answer, retrieved_docs)
    """
    # 문서 검색
    docs = search_documents(vectordb, question, k=k, search_type=search_type)

    # 컨텍스트 구성
    context = "\n\n---\n\n".join([
        f"[문서 {i+1}] (페이지: {doc.metadata.get('page', 'N/A')}, 섹션: {doc.metadata.get('section', 'N/A')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # 프롬프트 구성
    prompt_template = """
    너는 2024년도 장기요양기관 재가급여 평가매뉴얼을 기반으로 답변하는 전문 AI 어시스턴트입니다.

    다음 지침을 따라 답변해주세요:
    1. 관련된 평가지표 번호가 있다면 명시해주세요 (예: "지표 1번 운영규정")
    2. 점수 및 평가기준을 구체적으로 제시해주세요
    3. 참고한 문서의 페이지나 섹션 정보를 표시해주세요
    4. 정확하고 객관적인 정보만 제공하고, 확실하지 않은 경우 "문서에서 명확한 정보를 찾을 수 없습니다"라고 말해주세요
    5. 표 형식의 정보가 있다면 표로 정리해서 보여주세요

    질문: {question}

    참고 문서:
    {context}

    답변:
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=3000
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)

    return response.content, docs

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

# 검색 옵션 설정
with st.sidebar:
    st.header("⚙️ 검색 설정")
    search_type = st.radio(
        "검색 방식",
        ["mmr", "similarity"],
        format_func=lambda x: "MMR (다양성 중심)" if x == "mmr" else "유사도 (정확성 중심)",
        help="MMR: 다양한 정보 검색, 유사도: 가장 관련성 높은 정보 검색"
    )
    k_docs = st.slider(
        "검색할 문서 개수",
        min_value=3,
        max_value=10,
        value=5,
        help="더 많은 문서를 검색하면 더 포괄적인 답변을 얻을 수 있습니다"
    )
    show_sources = st.checkbox("검색된 문서 표시", value=True, help="답변 생성에 사용된 원본 문서를 표시합니다")

# 벡터 저장소 로드 (캐시로 한 번만 로드)
try:
    vectordb = load_vectorstore()
    st.sidebar.success("✅ 문서 준비 완료!")
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
if question:
    # 지표 번호 감지 표시
    indicator_num = extract_indicator_number(question)
    if indicator_num:
        st.info(f"🎯 지표 {indicator_num}번에 대한 질문으로 감지되었습니다. 관련 문서를 우선 검색합니다.")

    with st.spinner("🔍 관련 문서를 검색하고 답변을 생성하는 중..."):
        answer, retrieved_docs = generate_answer(
            vectordb,
            question,
            k=k_docs,
            search_type=search_type
        )

        st.success("✅ 답변이 생성되었습니다!")
        st.write("### 📎 답변:")
        st.write(answer)

        # 검색된 문서 표시
        if show_sources and retrieved_docs:
            st.divider()
            st.subheader("📚 검색된 원본 문서")
            st.caption(f"총 {len(retrieved_docs)}개의 문서가 검색되었습니다.")

            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"📄 문서 {i+1} - 페이지 {doc.metadata.get('page', 'N/A')} ({doc.metadata.get('section', '섹션 미분류')})"):
                    st.markdown(f"**메타데이터:**")
                    st.json({
                        "페이지": doc.metadata.get('page', 'N/A'),
                        "섹션": doc.metadata.get('section', '미분류'),
                        "타입": doc.metadata.get('type', '미분류')
                    })
                    st.markdown(f"**내용:**")
                    st.text_area(
                        f"내용 {i+1}",
                        value=doc.page_content,
                        height=200,
                        key=f"doc_{i}",
                        label_visibility="collapsed"
                    )

        # 검색 방식 정보
        with st.expander("ℹ️ 검색 설정 정보"):
            st.info(f"""
            **현재 검색 설정**
            - 검색 방식: {'MMR (다양성 중심)' if search_type == 'mmr' else '유사도 (정확성 중심)'}
            - 검색 문서 개수: {k_docs}개
            - 지표 번호 감지: {'예 (지표 ' + str(indicator_num) + '번)' if indicator_num else '아니오'}
            - LLM 모델: GPT-4o-mini
            """)
