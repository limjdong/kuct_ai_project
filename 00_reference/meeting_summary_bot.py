import os
import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader

# 환경변수 로드
load_dotenv("env.txt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 초기화
llm = ChatOpenAI(model="openai/gpt-4.1-mini", temperature=0.3)

# 보고서 프롬프트
report_prompt = ChatPromptTemplate.from_template("""
다음은 내부 회의록입니다.

회의 내용을 바탕으로 다음 형식의 **회의 요약 보고서**를 작성해주세요:

---

 **1. 회의 제목 또는 주제**  
(핵심 주제를 명확하게 서술)

 **2. 주요 논의사항 요약**  
- 항목별로 핵심 논의 내용을 정리

---

회의록 전문:
===========
{meeting_text}
""")

rag_chain = (
    {"meeting_text": lambda x: x["text"]}
    | report_prompt
    | llm
)

# 텍스트 파일 로더
def load_txt_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

# 회의록 요약 함수
def summarize_report(file):
    # 파일 이름에서 순수 파일명만 추출
    file_path = file.name  # 경로 추출
    text = load_txt_file(file_path)  # TextLoader로 로드
    result = rag_chain.invoke({"text": text})  # LangChain 체인 실행
    return result.content

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## 🧾 회의록 요약 보고서 생성기 (.txt 전용)")
    gr.Markdown("사내 회의록(.txt)을 업로드하면, 일반적인 보고서 형식으로 자동 요약해드립니다.")

    file_input = gr.File(label="📂 회의록 업로드 (.txt 형식만 가능)", type="filepath", file_types=[".txt"])
    output = gr.Textbox(label="📋 생성된 회의 보고서", lines=25)

    submit_btn = gr.Button("📝 보고서 생성")

    submit_btn.click(fn=summarize_report, inputs=file_input, outputs=output)

demo.launch()
