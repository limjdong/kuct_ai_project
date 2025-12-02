"""
PDF 문서를 임베딩하여 FAISS 벡터 저장소로 저장하는 스크립트
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import json
import re

# 환경변수 로드
load_dotenv("env.txt")

def extract_indicator_numbers(text):
    """텍스트에서 지표 번호를 추출"""
    patterns = [
        r'지표\s*(\d+)',
        r'평가지표\s*(\d+)',
    ]
    numbers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.update([int(m) for m in matches])
    return list(numbers)

def create_and_save_embeddings(pdf_path, save_dir="vectorstore"):
    """
    PDF 파일을 로드하여 임베딩을 생성하고 저장

    Args:
        pdf_path: PDF 파일 경로
        save_dir: 벡터 저장소를 저장할 디렉토리
    """
    print(f"📖 PDF 로드 중: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"📄 총 {len(documents)} 페이지 로드됨")

    # 메타데이터 보강 - 섹션 및 페이지 정보 추가
    print("📋 메타데이터 보강 중...")
    for i, doc in enumerate(documents):
        doc.metadata['page'] = i + 1

        # 섹션 정보 추출
        content = doc.page_content
        if "주야간보호" in content:
            doc.metadata['section'] = "주야간보호"
        elif "단기보호" in content:
            doc.metadata['section'] = "단기보호"
        elif "평가자 준수사항" in content or "평가자의 기본자세" in content:
            doc.metadata['section'] = "평가자준수사항"
        elif "매뉴얼 일반사항" in content:
            doc.metadata['section'] = "일반사항"

        # 문서 타입 정보 추출
        if "평가지표" in content:
            doc.metadata['type'] = "평가지표"
        elif "평가기준" in content:
            doc.metadata['type'] = "평가기준"
        elif "확인방법" in content:
            doc.metadata['type'] = "확인방법"

    print("✂️  문서 분할 중...")
    # 청킹 최적화: chunk_size 800, overlap 150으로 조정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", " ", ""],
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)

    print(f"📝 총 {len(split_docs)} 개의 청크로 분할됨 (최적화된 크기)")

    print("🔄 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
    embeddings = OpenAIEmbeddings()

    # 배치 처리로 토큰 제한 회피 (한 번에 100개씩 처리)
    batch_size = 100
    total_batches = (len(split_docs) - 1) // batch_size + 1

    # 첫 번째 배치로 벡터스토어 초기화
    first_batch = split_docs[0:batch_size]
    print(f"  → 배치 1/{total_batches} 처리 중... ({len(first_batch)}개 청크)")
    vectorstore = FAISS.from_documents(first_batch, embeddings)

    # 나머지 배치 처리
    for i in range(batch_size, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  → 배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 청크)")

        temp_vectorstore = FAISS.from_documents(batch, embeddings)
        vectorstore.merge_from(temp_vectorstore)

    print(f"💾 벡터 저장소 저장 중: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    vectorstore.save_local(save_dir)

    # 지표 번호별 매핑 생성
    print("🔢 지표 번호 매핑 생성 중...")
    indicator_mapping = {}

    for idx, doc in enumerate(split_docs):
        # 문서 내용에서 지표 번호 추출
        indicator_nums = extract_indicator_numbers(doc.page_content)

        for num in indicator_nums:
            if num not in indicator_mapping:
                indicator_mapping[num] = {
                    'chunk_indices': [],
                    'chunks': []
                }
            indicator_mapping[num]['chunk_indices'].append(idx)
            # 청크 정보 저장 (내용, 메타데이터)
            indicator_mapping[num]['chunks'].append({
                'content': doc.page_content,
                'metadata': {
                    'page': doc.metadata.get('page', 'N/A'),
                    'section': doc.metadata.get('section', 'N/A'),
                    'type': doc.metadata.get('type', 'N/A')
                }
            })

    # JSON으로 저장
    mapping_file = os.path.join(save_dir, "indicator_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(indicator_mapping, f, ensure_ascii=False, indent=2)

    print(f"📊 지표 매핑 저장 완료: {mapping_file}")
    print(f"   총 {len(indicator_mapping)}개의 지표 번호가 매핑되었습니다.")
    print(f"   매핑된 지표 번호: {sorted(indicator_mapping.keys())}")

    print("✅ 완료! 메타데이터가 포함된 최적화된 임베딩이 저장되었습니다.")
    print(f"   총 {len(split_docs)}개의 청크가 임베딩되었습니다.")

if __name__ == "__main__":
    # manual.pdf 파일 임베딩 생성 (테스트용)
    pdf_file = "manual.pdf"

    if not os.path.exists(pdf_file):
        print(f"❌ 오류: {pdf_file} 파일을 찾을 수 없습니다.")
        print("💡 manual.pdf를 사용합니다...")
        pdf_file = "manual.pdf"

    if not os.path.exists(pdf_file):
        print(f"❌ 오류: PDF 파일을 찾을 수 없습니다.")
        exit(1)

    create_and_save_embeddings(pdf_file)
