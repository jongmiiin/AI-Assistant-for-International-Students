import os
import json
import faiss
from typing import List, Dict, Iterable

from sentence_transformers import SentenceTransformer
from utils import load_jsonl, chunk_text, ensure_dir

# 임베딩 모델 (다국어)
MODEL_NAME = "intfloat/multilingual-e5-base"

def embed_passages(model: SentenceTransformer, passages: List[str]):
    # e5는 'passage: ' prefix를 권장
    inputs = [f"passage: {p}" for p in passages]
    return model.encode(
        inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False
    )

def iter_chunks_from_docs(docs: Iterable[Dict], chunk_size=1200, overlap=200):
    for d in docs:
        source = d.get("url")
        title = d.get("title", "")
        text = d.get("text", "") or ""
        for ch in chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=1000, max_chars=200_000):
            yield {"source": source, "title": title, "text": ch}

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    clean_path = os.path.join(root, "data", "clean", "docs.jsonl")
    index_dir = os.path.join(root, "data", "index")
    ensure_dir(index_dir)

    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"{clean_path} 가 없습니다. 먼저 scraper.py를 실행하세요.")

    docs = load_jsonl(clean_path)
    if not docs:
        raise RuntimeError("정제 문서가 없습니다. 스크래핑 결과를 확인하세요.")

    model = SentenceTransformer(MODEL_NAME)

    meta_path = os.path.join(index_dir, "meta.jsonl")
    # 덮어쓰기 시작
    with open(meta_path, "w", encoding="utf-8") as _:
        pass

    index = None
    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    BATCH_SIZE = 64
    total = 0
    for item in iter_chunks_from_docs(docs):
        batch_texts.append(item["text"])
        batch_meta.append(item)
        if len(batch_texts) >= BATCH_SIZE:
            embs = embed_passages(model, batch_texts)
            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)  # cosine (normalized)
            index.add(embs)
            # 메타는 바로 파일에 append (순서 보존)
            with open(meta_path, "a", encoding="utf-8") as f:
                for m in batch_meta:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            total += len(batch_texts)
            batch_texts.clear(); batch_meta.clear()

    # 잔여 배치 처리
    if batch_texts:
        embs = embed_passages(model, batch_texts)
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(embs)
        with open(meta_path, "a", encoding="utf-8") as f:
            for m in batch_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        total += len(batch_texts)

    if index is None or total == 0:
        raise RuntimeError("청킹 결과가 비어 있습니다. 스크래핑 데이터를 확인하세요.")

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    print(f"[DONE] indexed {total} chunks in streaming mode")

if __name__ == "__main__":
    main()
