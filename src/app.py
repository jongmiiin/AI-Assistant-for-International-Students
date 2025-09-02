# -*- coding: utf-8 -*-

import os
import json
import faiss
import gradio as gr
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from langdetect import detect   
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from utils import load_jsonl, pick_language_tag, DISCLAIMER_KO, DISCLAIMER_EN, DISCLAIMER_ZH

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(ROOT, "data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")

# Embedding model
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
e5 = SentenceTransformer(E5_MODEL_NAME)

# Load index
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("인덱스가 없습니다. 먼저 scraper.py와 ingest.py를 실행하세요.")
index = faiss.read_index(INDEX_PATH)
metas = load_jsonl(META_PATH)

def embed_query(q: str):
    return e5.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True)[0]

def search(q: str, top_k: int = 5) -> List[Dict]:
    vec = embed_query(q)
    D, I = index.search(vec.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        results.append({
            "score": float(score),
            "source": m.get("source"),
            "title": m.get("title", ""),
            "text": m.get("text", "")[:1500]  # 길이 제한
        })
    return results


SYSTEM_PROMPT = (
    "You are an AI assistant for international students at Jeonbuk National University (JBNU), "
    "specialized in visa-related guidance. "
    "Answer ONLY based on the provided context (official notices). "
    "If the information is not present in the context, respond with: "
    "'The information is not found in the official notice.' "
    "Whenever possible, provide a simple step-by-step guide and cite the source URL. "
    "Always respond in the same language as the user's question. "
    "Supported languages: Korean, English, Simplified Chinese. "
    "For example, if the question is in Korean, answer in Korean; "
    "if the question is in English, answer in English; "
    "if the question is in Chinese, answer in Chinese."
)

def build_context(snippets: List[Dict]) -> Tuple[str, List[str]]:
    ctx = []
    urls = []
    for i, s in enumerate(snippets, start=1):
        ctx.append(f"[{i}] URL: {s['source']}\n{text_trim(s['text'], 1200)}")
        urls.append(s["source"])
    return "\n\n".join(ctx), urls

def text_trim(t: str, maxlen: int) -> str:
    return (t[: maxlen - 3] + "...") if len(t) > maxlen else t

def pick_disclaimer(lang_tag: str) -> str:
    if lang_tag == "ko":
        return DISCLAIMER_KO
    if lang_tag == "zh":
        return DISCLAIMER_ZH
    return DISCLAIMER_EN

def answer_fn(message: str, history: List[Dict], lang_mode: str, top_k: int, show_sources: bool):
    # 검색
    snippets = search(message, top_k=top_k)
    context, urls = build_context(snippets)

    # 언어 결정
    if lang_mode == "Auto":
        try:
            qlang = detect(message)
        except Exception:
            qlang = "en"
    else:
        qlang = {"한국어": "ko", "English": "en", "中文(简体)": "zh-cn"}.get(lang_mode, "en")


    LANG_LABEL = {"ko": "Korean", "en": "English", "zh-cn": "Simplified Chinese"}
    LANG_HARD_RULE = {
        "ko": "항상 한국어로만 답하세요. 다른 언어를 섞지 마세요.",
        "en": "Always answer in English only. Do not include any other language.",
        "zh-cn": "请始终只使用简体中文回答，不要使用其他语言。"
    }
    lang_system = (
        f"Output language: {LANG_LABEL[qlang]}. "
        f"{LANG_HARD_RULE[qlang]} "
        "If the sources are in a different language, translate faithfully into the output language."
    )

    # 모델 질의
    user_instruction = (
        f"[Answer Language: {LANG_LABEL[qlang]}]\n"
        f"Question: {message}\n\n"
        f"Context (use ONLY this):\n{context}\n\n"
        "Answer only based on the provided context. If the context does not contain the information, say 'I do not know.' "
        "Include the source(s) at the end of your answer."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": lang_system},
            {"role": "user", "content": user_instruction},
        ],
        # stream=True,
    )
    reply = completion.choices[0].message.content

    # 디스클레이머 및 출처 표시
    disclaimer = pick_disclaimer(pick_language_tag(qlang))
    if show_sources:
        src_block = "\n".join([f"[{i+1}] {u}" for i, u in enumerate(urls)])
        reply = f"{reply}\n\n{disclaimer}\n\n{src_block}"
    else:
        reply = f"{reply}\n\n{disclaimer}"
    return reply

def demo():
    # 고정 설정값(원하면 여기서만 숫자 바꿔 쓰면 됨)
    LANG = "Auto"   # "Auto" | "한국어" | "English" | "中文(简体)"
    TOPK = 5        # 검색 개수
    SHOW_SOURCES = False  # 출처/근거 표시 여부

    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")
    with gr.Blocks(
        theme=theme,
        fill_height=True,
        css="""
        .gradio-container { max-width: 980px !important; margin: 0 auto; }
        .hero { text-align:center; padding: 8px 0 4px; }
        .hero h1 { font-size: 28px; line-height: 1.2; margin: 0; }
        .hero p { color: #64748b; margin: .35rem 0 0; }
        """
    ) as app:
        # 히어로 헤더(브랜딩은 여기서만!)
        gr.Markdown("""
        <div class="hero">
          <h1>JBNU International AI Assistant</h1>
          <p>외국인 유학생 안내 AI Agent 챗봇 · 한국어 / English / 中文(简体)</p>
        </div>
        """)

        chat = gr.ChatInterface(
            fn=lambda msg, hist: answer_fn(msg, hist, LANG, TOPK, SHOW_SOURCES),
            title=None,
            chatbot=gr.Chatbot(height=450, show_label=False, bubble_full_width=False),
            undo_btn=None,
            retry_btn="Retry",
            clear_btn="Clear",
            examples=[
                "외국인등록증 분실 시 어떻게 해야 하나요?",
                "How to extend my student visa (D-2)?",
                "兼职打工的时间限制是多少？",
            ],
            cache_examples=False,
        )
    return app

if __name__ == "__main__":
    app = demo()
    app.queue().launch()
