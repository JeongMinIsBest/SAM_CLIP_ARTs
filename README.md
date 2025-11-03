# 🎨 Emotion Segmentation in Art : SAM + CLIP 기반 예술 작품 감정 시각화 프로젝트

이 프로젝트는 Meta의 Segment Anything Model (SAM)과 OpenAI의 CLIP을 결합하여  
예술 작품의 영역별 감정을 자동으로 분석하고, 시각적으로 표현하는 Streamlit 앱입니다.
<br/>
<br/>


## 🧠 프로젝트 개요

업로드한 예술 작품을 자동으로 분할한 뒤, 각 영역의 감정(joy, sadness, fear, calm, mystery 등)을 추정하고 색상으로 시각화합니다.

| 모델 | 역할 |
|------|------|
| **SAM (Segment Anything)** | 이미지를 객체별로 세그멘테이션 |
| **CLIP (Contrastive Language–Image Pretraining)** | 각 세그먼트의 감정적 분위기 분석 |
| **Streamlit** | 웹 인터페이스 및 시각화 제공 |
<br/>
<br/>


## 💬 주요 기능

✅ 자동 감정 분석 – SAM으로 분할된 각 영역별로 CLIP을 통해 감정 유사도 계산  
✅ 시각적 감정 표현 – 감정별 색상 맵핑 및 투명도 조절  
✅ 상위 감정 그래프 – 작품 전체의 감정 분포를 Plotly 그래프로 시각화  
<br/>
<br/>


## 📂 폴더 구조 및 실행 방법

```
SAM_CLIP_Art_Emotion/
├── app.py                # Streamlit 메인 앱
├── requirements.txt      # 필요한 패키지 목록
└── .gitignore            # 불필요한 파일 제외
```

1️⃣ 가상환경 설정
```
python -m venv .venv
source .venv/bin/activate       # (Windows: .venv\Scripts\activate)
```

2️⃣ 필요한 패키지 설치
```
pip install -r requirements.txt
```

3️⃣ 실행
```
streamlit run app.py
```

브라우저에서 자동으로 열리는 URL (기본값 : ```http://localhost:8501```)로 접속합니다.
<br/>
<br/>


## 🧩 기술 스택

| 분류            | 사용 도구                        |
| ------------- | ---------------------------- |
| AI Model      | SAM (Meta AI), CLIP (OpenAI) |
| Backend       | PyTorch, NumPy, OpenCV       |
| Visualization | Matplotlib, Plotly           |
| Frontend      | Streamlit                    |
| Deployment    | Streamlit Cloud              |
<br/>
<br/>

