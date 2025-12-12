# Spending Pattern Analyzer

AI 기반 소비 패턴 분석 및 페르소나 추천 시스템

## 📊 프로젝트 소개

경기도민의 실제 카드 소비 데이터를 기반으로 사용자의 소비 패턴을 분석하고, K-Means 클러스터링을 통해 유사한 소비 그룹을 찾아주는 웹 애플리케이션입니다. 

**주요 기능:**
- 월별 소비 내역 입력 및 분석
- AI 기반 소비 페르소나 생성 (닉네임 + 해시태그)
- 그룹 평균과의 비교 분석 (레이더 차트)
- 개인화된 절약 포인트 및 소비 특성 제공
- 분석 정밀도 조절 (K=3~8)

## 🗂️ 데이터 출처

본 프로젝트는 **[경기데이터드림](https://data.gg.go.kr/portal/mainPage.do)** 에서 제공하는 [**카드 소비 데이터**](https://data.gg.go.kr/portal/data/service/selectServicePage.do?page=1&rows=10&sortColumn=&sortDirection=&infId=7Y02TF04H1WUB55Q4IZL35052374&infSeq=1&order=)를 활용합니다.

- 데이터셋: 경기도 시군구별 업종별 카드 소비 데이터
- 기간: 일별 집계 데이터
- 분류: 대분류/중분류 카테고리 체계

## 🏗️ 시스템 아키텍처

```
Frontend (React + Vite)
    ↓
Backend (Flask API)
    ↓
├─ K-Means Clustering (Scikit-learn)
├─ LLM (EXAONE 3.5 - llama.cpp)
└─ Data Processing (Pandas, NumPy)
```

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 이상
- Node.js 16 이상
- 8GB 이상의 RAM (LLM 모델 실행 시)

### 백엔드 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/hwanbit/spending-pattern-analyzer.git
cd spending-analyzer

# 2. Python 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 패키지 설치
pip install flask flask-cors scikit-learn pandas numpy joblib llama-cpp-python

# 또는 requirements.txt 파일을 생성하여 설치:
pip install -r requirements.txt

# 4. 데이터 파일 준비
# data/ 폴더에 tbsh_gyeonggi_day_*.csv 파일 배치

# 5. LLM 모델 다운로드
# model/ 폴더에 EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf 배치

# 6. Flask 서버 실행
python app.py
```

서버가 `http://localhost:5000` 에서 실행됩니다.

### 프론트엔드 설치 및 실행

```bash
# 1. 프론트엔드 디렉토리로 이동
cd my-persona-app  # 또는 프론트엔드 폴더명

# 2. 의존성 패키지 설치
npm install

# 3. Tailwind CSS 설정
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p

# 4. 개발 서버 실행
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

## 📁 프로젝트 구조

```
spending-pattern-analyzer/
├── backend/
│   ├── app.py                   # Flask 백엔드 서버
│   ├── requirements.txt         # Python 패키지 의존성
│   ├── data/                    # 경기도 카드 소비 데이터
│   │   └── tbsh_gyeonggi_day_*.csv
│   └── model/                   # 학습된 모델 및 LLM
│       ├── kmeans_k3.pkl ~ kmeans_k8.pkl
│       ├── feature_names.pkl
│       ├── category_map.pkl
│       └── EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf
├── my-persona-app/              # React 프론트엔드
│   ├── public/
│   ├── src/
│   │   ├── App.jsx              # 메인 컴포넌트
│   │   ├── App.css              # 컴포넌트 스타일
│   │   ├── main.jsx             # 엔트리 포인트
│   │   ├── index.css            # 전역 스타일 (Tailwind)
│   │   └── assets/
│   │       └── fonts/           # Nanum 폰트
│   └── index.html
├── .gitignore
└── README.md
```

## 🎨 기술 스택

### Frontend
- **React 18** - UI 프레임워크
- **Vite** - 빌드 도구
- **Tailwind CSS 3** - 스타일링
- **Recharts** - 데이터 시각화
- **Lucide React** - 아이콘

### Backend
- **Flask** - 웹 프레임워크
- **Scikit-learn** - K-Means 클러스터링
- **Pandas & NumPy** - 데이터 처리
- **llama.cpp** - 로컬 LLM 실행
- **Joblib** - 모델 직렬화

### AI Models
- **MiniBatch K-Means** - 대규모 데이터 클러스터링
- **EXAONE 3.5 (2.4B)** - 페르소나 생성

## 🔍 주요 기능 상세

### 1. 소비 내역 입력
- 대분류/중분류 카테고리 선택
- 금액 입력 및 리스트 관리
- 실시간 합계 계산

### 2. K-Means 클러스터링
- K값 조절 (3~8): 분석 정밀도 조정
- 사용자 소비 벡터 정규화
- 가장 유사한 그룹 자동 매칭

### 3. AI 페르소나 생성
- LLM 기반 창의적 닉네임 생성
- 소비 패턴 기반 해시태그 추천
- Few-shot 프롬프팅 기법 적용

### 4. Gap Analysis
- **Unique Trait**: 그룹 대비 많이 소비하는 카테고리
- **Saving Point**: 그룹 대비 적게 소비하는 카테고리

### 5. 시각화
- 레이더 차트: 나 vs 그룹 평균 비교
- 대분류 단위 집계 및 표시

## 📊 데이터 처리 흐름

```
1. CSV 데이터 로드 (Chunk 단위 처리)
   ↓
2. 카테고리별 집계 (연령/성별/중분류)
   ↓
3. 피벗 테이블 생성 및 정규화
   ↓
4. K-Means 모델 학습 (K=3~8)
   ↓
5. 모델 저장 (.pkl)
```

사용자 입력 시:
```
1. 소비 내역 → 벡터 변환
   ↓
2. 정규화 (비율 계산)
   ↓
3. 클러스터 예측
   ↓
4. LLM 페르소나 생성
   ↓
5. Gap 분석 및 시각화
```

## ⚙️ 설정 및 커스터마이징

### K값 범위 변경
`app.py`의 `LargeScaleClusterManager` 클래스:
```python
self.k_levels = [i for i in range(3, 9)]  # 원하는 범위로 수정
```

### LLM 모델 교체
`app.py`의 `MODEL_PATH` 변경:
```python
MODEL_PATH = "./model/your-model.gguf"
```

### 폰트 변경
`src/assets/fonts/`에 폰트 추가 후 `index.css`에서 설정

## 🙏 감사의 말

- **경기데이터드림** - 카드 소비 데이터 제공
- **LG AI Research** - EXAONE 모델
- **Naver** - Nanum 폰트

## 👤 개발자

© 2025 Elphie. All rights reserved.
