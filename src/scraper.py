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


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()


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

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        "h1", "h2", ".tit", ".title", ".subject", ".board_view .title", ".view_tit"
    ]
    for sel in candidates:
        node = soup.select_one(sel)
        if node:
            t = node.get_text(strip=True)
            if t:
                return t
    return ""

def is_same_domain(base: str, url: str) -> bool:
    return url.startswith(base)

def is_list_page(url: str) -> bool:
    # 정확히 비자 목록(22303) 페이지만 허용
    return "/ioffice/22303/" in url and "subview.do" in url

def is_visa_detail(url: str) -> bool:
    # 비자 게시판 상세 패턴: /bbs/ioffice/3549/**/artclView.do
    if "/bbs/ioffice/3549/" in url and "artclView.do" in url:
        return True
    # 일부 상세는 subview.do?enc=... 로 열리기도 하므로 enc가 있으면 후보로 인정
    if is_list_page(url) and "enc=" in url:
        return True
    return False

def should_skip(url: str) -> bool:
    # 첨부/다운로드, 타 게시판은 스킵
    if "download.do" in url:
        return True
    # 타 보드(ex. 4303, 5310, 5154 등) 스킵
    if "/bbs/ioffice/" in url and "/bbs/ioffice/3549/" not in url:
        return True
    return False

def scrape(start_url: str, out_dir_raw: str, out_dir_clean: str,
           max_pages: int = 120, delay_sec: float = 1.0, ignore_robots: bool = False):
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

        if not is_same_domain(base, url):
            continue
        if not allowed_by_robots(base, url, ignore_robots=ignore_robots):
            print(f"[SKIP robots] {url}")
            continue
        if should_skip(url):
            # 첨부/타 보드 배제
            # print(f"[SKIP non-visa/attach] {url}")
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
            if resp.status_code != 200 or not resp.text:
                print(f"[WARN] status={resp.status_code} url={url}")
                continue
        except Exception as e:
            print(f"[ERROR] GET {url}: {e}")
            continue

        pages += 1
        html = resp.text

        # 저장(원문 HTML) — 필요 없다면 주석 처리 가능
        raw_name = f"page_{pages:04d}.html"
        try:
            with open(os.path.join(out_dir_raw, raw_name), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            # 경로/인코딩 문제 등은 무시하고 진행
            pass
        
        # 본문 텍스트/제목 추출
        title = extract_title(html)
        text = extract_main_text(html)

        # 상세 페이지만 저장(목록은 제외). 단, subview.do?enc=...은 상세로 취급
        if is_visa_detail(url):
            # 너무 짧은 텍스트(메뉴/빈 페이지)는 제외
            if len(text) >= 200:
                records.append({"url": url, "title": title, "text": text})

        # -------- 링크 수집 로직: 상세글/목록/페이지네이션 모두 추적 --------
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            next_url = urljoin(url, href)

            # 같은 도메인만
            if not is_same_domain(base, next_url):
                continue
            if should_skip(next_url):
                continue

            # (A) 정확히 이 목록(22303)만 계속 탐색
            if is_list_page(next_url):
                if next_url not in visited:
                    to_visit.append(next_url)
                continue

            # (B) 비자 상세만!
            if is_visa_detail(next_url):
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
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--ignore-robots", action="store_true", help="robots.txt 확인을 건너뜁니다(서버가 robots 접근을 막는 경우 유용)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    out_raw = os.path.join(root, "data", "raw")
    out_clean = os.path.join(root, "data", "clean")

    scrape(args.start_url, out_raw, out_clean, max_pages=args.max_pages, delay_sec=args.delay, ignore_robots=args.ignore_robots)
