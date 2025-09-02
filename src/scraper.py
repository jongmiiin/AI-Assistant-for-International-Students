# -*- coding: utf-8 -*-
import argparse
import os
import time
import re
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

from utils import ensure_dir, save_jsonl

HEADERS = {
    "User-Agent": "JBNU-VisaBot/1.0 (+https://example.com; contact: admin@example.com)"
}

def allowed_by_robots(base_url: str, target_url: str, ignore_robots: bool) -> bool:
    if ignore_robots:
        return True
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], target_url)
    except Exception:
        # 일부 서버는 robots.txt 접근이 차단되거나 비표준 응답을 줄 수 있음.
        # 접근 실패 시에는 일단 허용(=True)으로 처리하고, 실제 요청에서 403/401이면 스킵.
        return True

# 사이트별 본문 선택자 후보 (많이 쓰이는 K2Web 패턴 포함)
CONTENT_SELECTORS = [
    ".board_view", ".view_cont", ".viewCont", ".article", ".boardDetail",
    ".bo_view", "#content", ".content", ".sub_content", ".subContent"
]

# def extract_main_text(html: str) -> str:
#     soup = BeautifulSoup(html, "html.parser")
#     for s in soup(["script", "style", "noscript"]):
#         s.extract()

#     # 후보 선택자 중 텍스트가 가장 긴 컨테이너 선택
#     best = None
#     best_len = 0
#     for sel in CONTENT_SELECTORS:
#         for node in soup.select(sel):
#             txt = node.get_text(separator=" ", strip=True)
#             L = len(txt or "")
#             if L > best_len:
#                 best, best_len = txt, L

#     text = (best or soup.get_text(separator=" ", strip=True) or "").strip()
#     text = re.sub(r"\s+", " ", text)
#     return text

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()

    # K2Web 패턴 우선
    CONTENT_SELECTORS = [
        ".board_view", ".view_cont", ".viewCont", ".article", ".boardDetail",
        ".bo_view", "#content", ".content", ".sub_content", ".subContent"
    ]

    best = None
    best_len = 0
    for sel in CONTENT_SELECTORS:
        for node in soup.select(sel):
            txt = node.get_text(separator=" ", strip=True)
            if txt and len(txt) > best_len:
                best, best_len = txt, len(txt)

    text = (best or soup.get_text(separator=" ", strip=True) or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def scrape(start_url: str, out_dir_raw: str, out_dir_clean: str,
           max_pages: int = 200, delay_sec: float = 1.0, ignore_robots: bool = False):
    ensure_dir(out_dir_raw); ensure_dir(out_dir_clean)
    visited = set()
    to_visit = [start_url]
    records = []

    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url))

    pages = 0
    while to_visit and pages < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        # 같은 도메인만
        if not url.startswith(base):
            continue

        if not allowed_by_robots(base, url, ignore_robots=ignore_robots):
            print(f"[SKIP robots] {url}")
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200 or not resp.text:
                print(f"[WARN] status={resp.status_code} url={url}")
                continue
        except Exception as e:
            print(f"[ERROR] GET {url}: {e}")
            continue

        pages += 1
        html = resp.text

        # 저장(원문 HTML)
        fname = f"page_{pages:04d}.html"
        with open(os.path.join(out_dir_raw, fname), "w", encoding="utf-8") as f:
            f.write(html)

        text = extract_main_text(html)

        # 비자 관련 키워드가 포함된 페이지만 보존
        if any(k in text for k in ["비자", "VISA", "출입국", "체류", "외국인", "유학생", "외국인등록", "visa", "Visa"]):
            # 제목 추출 (일반적인 패턴)
            title = ""
            h = None
            for sel in ["h1", "h2", ".tit", ".title", ".subject"]:
                h = BeautifulSoup(html, "html.parser").select_one(sel)
                if h and h.get_text(strip=True):
                    title = h.get_text(strip=True)
                    break

            records.append({"url": url, "title": title, "text": text})

        # -------- 링크 수집 로직: 상세글/목록/페이지네이션 모두 추적 --------
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            next_url = urljoin(url, href)

            # 같은 도메인만
            if not next_url.startswith(base):
                continue

            # (1) 해당 게시판 메인/페이지네이션
            if ("/ioffice/22303/" in next_url) or ("subview.do" in next_url):
                if next_url not in visited:
                    to_visit.append(next_url)
                    continue

            # (2) 상세 글(핵심): /bbs/ioffice/.../artclView.do
            if ("/bbs/ioffice/" in next_url) or ("artclView.do" in next_url):
                if next_url not in visited:
                    to_visit.append(next_url)
                    continue

        print(f"[OK] {url}  (queued={len(to_visit)})")
        time.sleep(delay_sec)

    save_jsonl(os.path.join(out_dir_clean, "docs.jsonl"), records)
    print(f"[DONE] scraped_pages={pages}, saved_docs={len(records)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", type=str, required=True, help="시작 URL (예: https://ioffice.jbnu.ac.kr/ioffice/22303/subview.do)")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--ignore-robots", action="store_true", help="robots.txt 확인을 건너뜁니다(서버가 robots 접근을 막는 경우 유용)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    out_raw = os.path.join(root, "data", "raw")
    out_clean = os.path.join(root, "data", "clean")

    scrape(args.start_url, out_raw, out_clean, max_pages=args.max_pages, delay_sec=args.delay, ignore_robots=args.ignore_robots)
