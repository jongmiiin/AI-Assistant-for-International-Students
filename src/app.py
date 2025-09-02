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

SYSTEM_PROMPT = ("당신은 전북대학교(JBNU) 외국인 유학생 비자 어시스턴트입니다. ""오직 제공된 컨텍스트(공지 텍스트)만 사용해 답변하세요. ""컨텍스트에 없는 내용은 추측하지 말고 '해당 내용은 공식 공지에서 확인되지 않습니다'라고 말하세요. ""가능하면 간단한 단계별 안내와 함께 출처(URL)를 [1], [2] 형태로 명시하세요. ""사용자 질문의 언어로 답변하세요. 한국어/영어/중국어(간체)를 지원합니다.")

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
        qlang = {"한국어": "ko", "English": "en", "中文(简体)": "zh"}.get(lang_mode, "en")

    # 모델 질의
    user_instruction = (
        f"질문: {message}\n\n"
        f"컨텍스트:\n{context}\n\n"
        "위 컨텍스트로만 답변하세요. 근거가 없으면 모른다고 하세요. "
        "마지막에 출처 인덱스([1], [2]...)를 포함하세요."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction},
        ],
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
    with gr.Blocks(fill_height=True) as app:
        gr.Markdown("""
        # JBNU International AI Assistant
        외국인 유학생 비자 안내 RAG 챗봇 (한국어/English/中文(简体))
        """)
        with gr.Row():
            lang_mode = gr.Dropdown(
                choices=["Auto", "한국어", "English", "中文(简体)"],
                value="Auto",
                label="응답 언어 (Language)"
            )
            topk = gr.Slider(1, 8, value=5, step=1, label="검색 개수 (Top-K)")
            show_sources = gr.Checkbox(value=True, label="출처/근거 표시")
        topk = gr.Slider(1, 8, value=5, step=1, label="검색 개수 (Top-K)")
        show_sources = gr.Checkbox(value=True, label="출처/근거 표시")

        chat = gr.ChatInterface(
            fn=lambda msg, hist: answer_fn(msg, hist, lang_mode.value, int(topk.value), bool(show_sources.value)),
            title="JBNU Visa Assistant",
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
