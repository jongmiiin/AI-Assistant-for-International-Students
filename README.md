# JBNU International AI Assistant

외국인 유학생의 비자(VISA) 관련 업무를 다국어(한국어/영어/중국어)로 안내하는 AI Agent 챗봇입니다.
- 소스: 전북대학교 국제교류 관련 공지 (예: https://ioffice.jbnu.ac.kr/ioffice/22303/subview.do)
- UI: Gradio
- 모델: OpenAI GPT (기본 `gpt-4o-mini`), 멀티링궐 임베딩(`intfloat/multilingual-e5-base`)
- RAG: FAISS


## 빠른 시작

1) 압축 해제 후 가상환경 생성 및 설치
```bash
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```

2) 환경변수 설정 (`.env` 파일 생성)
```bash
OPENAI_API_KEY=YOUR_KEY_HERE
MODEL_NAME=gpt-4o-mini
```

3) 스크래핑: 비자 등 관련 공지 수집
```bash
python src/scraper.py   --start-url https://ioffice.jbnu.ac.kr/ioffice/22303/subview.do   --max-pages 150
```

4) 임베딩 구축/인덱싱
```bash
python src/ingest.py
```

5) 앱 실행
```bash
python src/app.py
```
앱이 실행되면 출력되는 로컬 URL을 브라우저에서 열어 사용하세요.

## 구조
```
jbnu_visa_bot/
├── data/
│   ├── raw/        # 원본 HTML 저장
│   ├── clean/      # 정제된 텍스트(.jsonl) 저장
│   └── index/      # FAISS 인덱스 및 메타데이터
├── src/
│   ├── app.py      # Gradio 앱 (질의→검색→생성)
│   ├── scraper.py  # 공지 스크래핑(robots.txt 준수)
│   ├── ingest.py   # 텍스트 → chunk → 임베딩 → 인덱스
│   └── utils.py    # 공용 함수
├── requirements.txt
└── README.md
```

## 주의/윤리
- 민감정보를 저장/처리하지 않습니다. 공개 공지 내용만을 다룹니다.
- 답변은 참고용이며, 최종 책임은 사용자에게 있음을 명시합니다.

## 향후 확장 아이디어
- 음성(STT/TTS) 및 모바일 UI
- 공지 변경 감지 및 자동 재색인 (GitHub Action/크론)
- 기숙사/보험/학사 캘린더 등 도메인 확장
