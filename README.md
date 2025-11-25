# 장기요양 재가급여 평가 매뉴얼 챗봇

**2024년도 장기요양기관 재가급여 평가매뉴얼 Ⅱ (주야간보호, 단기보호)**를 기반으로 답변을 제공하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 주요 기능

- **정확한 정보 검색**: FAISS 벡터 저장소와 MMR 알고리즘을 활용한 관련성 높은 정보 검색
- **메타데이터 활용**: 섹션, 페이지, 문서 타입 정보를 활용한 정밀 검색
- **출처 표시**: 답변에 관련 페이지 및 지표 번호 포함
- **객관적 답변**: 매뉴얼 문서에 기반한 정확하고 객관적인 정보만 제공

## 제공하는 정보

- 평가지표 및 평가기준
- 점수 구성 및 배점
- 2020년 vs 2024년 변경사항
- 평가자 준수사항
- 매뉴얼 일반사항

## 시스템 요구사항

- Python 3.8 이상
- OpenAI API 키

## 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd toy_project
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경변수 설정
   - `env.txt.example` 파일을 `env.txt`로 복사
   - OpenAI API 키를 설정
```bash
cp env.txt.example env.txt
# env.txt 파일을 열어 OPENAI_API_KEY를 입력하세요
```

4. PDF 문서 준비
   - `manual.pdf` 파일을 프로젝트 루트 디렉토리에 배치
   - 또는 `manual_short.pdf` 파일 사용 (테스트용)

## 사용 방법

### 1. 임베딩 생성

처음 실행하거나 PDF 문서가 업데이트된 경우, 먼저 임베딩을 생성해야 합니다:

```bash
python create_embeddings.py
```

이 과정은 다음을 수행합니다:
- PDF 파일 로드 및 텍스트 추출
- 문서를 최적화된 청크로 분할 (chunk_size=800, overlap=150)
- 메타데이터 보강 (섹션, 페이지, 타입 정보)
- OpenAI Embeddings를 사용한 벡터화
- FAISS 벡터 저장소 생성 및 저장 (`vectorstore/` 디렉토리)

**주의**: 이 과정은 문서 크기에 따라 시간이 걸릴 수 있으며, OpenAI API 사용료가 발생합니다.

### 2. 챗봇 실행

임베딩 생성이 완료되면 챗봇을 실행할 수 있습니다:

```bash
streamlit run rag_chatbot.py
```

브라우저에서 자동으로 열리거나, 터미널에 표시된 URL(보통 `http://localhost:8501`)로 접속하세요.

### 3. 질문 예시

- "주야간보호 평가지표 1번에 대해 설명해주세요"
- "경력직 평가기준은 무엇인가요?"
- "2020년과 2024년 매뉴얼의 차이점은?"
- "인력추가배치 가산점수 계산 방법을 알려주세요"
- "건강검진 관련 규정이 무엇인가요?"

## 프로젝트 구조

```
toy_project/
├── create_embeddings.py    # PDF 임베딩 생성 스크립트
├── rag_chatbot.py          # Streamlit 챗봇 애플리케이션
├── manual.pdf              # 평가 매뉴얼 원본 문서
├── manual_short.pdf        # 테스트용 축약 문서
├── env.txt                 # 환경변수 설정 (git에 포함되지 않음)
├── env.txt.example         # 환경변수 설정 예시
├── requirements.txt        # Python 패키지 의존성
├── vectorstore/            # FAISS 벡터 저장소 (생성됨)
└── README.md              # 프로젝트 문서
```

## 기술 스택

- **LangChain**: RAG 파이프라인 구성
- **OpenAI**: 임베딩 생성 및 LLM (GPT-4o-mini)
- **FAISS**: 벡터 저장소 및 유사도 검색
- **Streamlit**: 웹 인터페이스
- **PyPDF**: PDF 문서 처리

## 성능 최적화

이 프로젝트는 다음과 같은 최적화를 적용했습니다:

1. **청킹 최적화**: chunk_size 800, overlap 150으로 조정하여 검색 정확도 향상
2. **메타데이터 보강**: 섹션, 페이지, 타입 정보를 추가하여 관련성 높은 검색 결과 제공
3. **MMR 검색**: Maximal Marginal Relevance 알고리즘으로 검색 결과의 다양성 확보
4. **배치 처리**: 대용량 문서 처리를 위한 배치 임베딩 생성 (100개 청크씩)
5. **프롬프트 엔지니어링**: 구체적이고 명확한 답변을 유도하는 프롬프트 설계

## 주의사항

- 이 챗봇은 매뉴얼 문서 내용만을 참조하여 답변합니다
- 법적 자문이나 공식 결정을 대체할 수 없습니다
- 정확한 정보는 공식 매뉴얼을 직접 확인해주세요
- OpenAI API 사용료가 발생합니다 (임베딩 생성 및 질의응답)

## 라이선스

본 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.
