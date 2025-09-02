import os
import json
import time
from typing import List, Dict, Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def now_ms() -> int:
    return int(time.time() * 1000)

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    max_chunks: int = 1000,
    max_chars: int = 200_000
) -> List[str]:
    # 텍스트가 너무 길면 잘라 메모리 폭주 방지
    if len(text) > max_chars:
        text = text[:max_chars]
    # overlap이 chunk_size 이상이면 무한루프 위험 → 자동 조정
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    n = len(text)
    count = 0
    while start < n and count < max_chunks:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        count += 1
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return [c for c in chunks if c]

def pick_language_tag(q_lang: str) -> str:
    # langdetect 결과를 간단한 3분류로 축약
    if q_lang.startswith("ko"):
        return "ko"
    if q_lang.startswith("zh"):
        return "zh"
    return "en"

DISCLAIMER_KO = "※ 본 답변은 공개 공지에 기반한 참고 정보입니다. 최종 확인은 공식 안내와 행정 창구를 통해 진행하세요."
DISCLAIMER_EN = "※ This answer is for reference based on public notices. Please confirm with official guidance and administrative desks."
DISCLAIMER_ZH = "※ 本回答基于公开公告，仅供参考。请以学校官方指南与行政窗口为准。"
