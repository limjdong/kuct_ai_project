import os
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from datetime import date

# 환경변수 로드
load_dotenv('env.txt')
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#  뉴스 기사 수집 함수
def get_news_articles(keyword, start_date, end_date, page_size):
    """
    주어진 키워드와 날짜 범위에 해당하는 뉴스 기사를 NewsAPI에서 검색합니다.
    - keyword: 검색 키워드
    - start_date, end_date: YYYY-MM-DD 형식
    - page_size: 키워드당 가져올 뉴스 개수
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "from": start_date,
        "to": end_date,
        "sortBy": "relevancy",  # 관련도 높은 순
        "language": "en",       # 영어 뉴스만 수집
        "apiKey": NEWS_API_KEY,
        "pageSize": page_size
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    # 뉴스 제목, 설명, URL 정보 정리
    articles = response.json().get("articles", [])
    return [
        {
            "title": article["title"],
            "url": article["url"],
            "description": article.get("description", "")
        }
        for article in articles
    ]

# 전체 뉴스 요약 함수
def summarize_all_articles(all_articles):
    """
    수집한 전체 뉴스 기사들을 하나로 합쳐 OpenAI GPT를 이용해 요약합니다.
    """
    if not all_articles:
        return "입력한 키워드 및 날짜 범위에 해당하는 뉴스가 없습니다."

    # 기사들을 텍스트로 병합
    merged_text = "\n\n".join(
        f"제목: {item['title']}\n설명: {item['description']}" for item in all_articles
    )

    # GPT에게 전달할 요약 프롬프트 작성
    prompt = (
        "다음은 특정 키워드에 대한 뉴스 기사 목록입니다. 전체적인 흐름과 주요 내용을 간결하게 요약해주세요:\n\n"
        f"{merged_text}"
    )

    # GPT를 호출해 요약 생성
    response = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()

#  Gradio UI 실행 함수
def news_summary_main(keyword_input, start_date, end_date, articles_per_keyword):
    """
    Gradio 입력값을 받아 키워드별 뉴스 수집 → 전체 요약을 수행합니다.
    """
    # #AI #삼성전자 형식 입력을 → ["AI", "삼성전자"] 로 분리
    keywords = [kw.strip().lstrip('#') for kw in keyword_input.split('#') if kw.strip()]
    all_articles = []  # 전체 기사 저장 리스트

    news_display = ""
    for keyword in keywords:
        articles = get_news_articles(keyword, start_date, end_date, articles_per_keyword)
        if not articles:
            news_display += f"\n### ❌ '{keyword}'에 대한 뉴스가 없습니다.\n"
            continue

        news_display += f"\n### 📰 '{keyword}' 관련 뉴스 목록\n"
        for idx, article in enumerate(articles, 1):
            news_display += f"- [{article['title']}]({article['url']})\n"
            all_articles.append(article)

    # 전체 요약 생성
    summary = summarize_all_articles(all_articles)

    return news_display, summary

#  Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 산업 뉴스 수집 및 요약 챗봇\n키워드를 입력하고, 날짜 범위를 지정하면 관련 영어 뉴스 요약을 확인할 수 있습니다.")
    
    # 입력 행 1: 키워드 / 기사 수
    with gr.Row():
        keyword_input = gr.Textbox(label="키워드 입력 (예: #AI #semiconductor #NVIDIA)")
        articles_per_keyword = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="키워드당 뉴스 개수")

    # 입력 행 2: 시작일 / 종료일
    with gr.Row():
        start_date = gr.Textbox(label="시작 날짜 (예: 2025-05-01)", value=str(date.today()))
        end_date = gr.Textbox(label="종료 날짜 (예: 2025-05-15)", value=str(date.today()))

    # 실행 버튼
    submit_btn = gr.Button("🔍 뉴스 수집 및 요약")

    # 출력 영역: 뉴스 목록 / 요약 결과
    news_output = gr.Markdown(label="🗂️ 수집된 뉴스 목록")
    summary_output = gr.Textbox(label="📝 요약 결과", lines=10)

    # 버튼 클릭 시 처리 함수 연결
    submit_btn.click(
        fn=news_summary_main,
        inputs=[keyword_input, start_date, end_date, articles_per_keyword],
        outputs=[news_output, summary_output]
    )

# Gradio 앱 실행
demo.launch()