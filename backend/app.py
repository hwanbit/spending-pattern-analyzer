import os
import glob
import random
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import MiniBatchKMeans
from llama_cpp import Llama

app = Flask(__name__)
CORS(app) # 모든 도메인 허용 (개발용)

category_map = {}
major_categories = []
models = {}

# Kmeans 모델 저장 경로 확보
model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"📁 '{model_dir}' 폴더를 생성했습니다.")

# ==========================================
# 1. 데이터 로드 및 계층 구조(Hierarchy) 구축
# ==========================================
def load_and_train_efficiently():
    global category_map, major_categories, models

    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_pattern = os.path.join(base_dir, '../data', 'tbsh_gyeonggi_day_*.csv')

    if not glob.glob(file_pattern):
        file_pattern = './data/tbsh_gyeonggi_day_*.csv'

    all_files = glob.glob(file_pattern)
    print(f"🔍 발견된 파일: {len(all_files)}개")

    if not all_files:
        return None, [], {}

    # 중분류(nm_2)까지 읽어옴
    eng_cols = ['age', 'sex', 'card_tpbuz_nm_1', 'card_tpbuz_nm_2', 'amt']
    rename_map = {
        'age': '연령별', 'sex': '성별',
        'card_tpbuz_nm_1': '대분류', 'card_tpbuz_nm_2': '중분류',
        'amt': '매출금액'
    }

    structure_data = set()
    aggregated_chunks = []
    local_category_map = {}

    print("🚀 데이터 로딩 시작...")

    for f in all_files:
        try:
            # 인코딩 감지
            target_encoding = 'utf-8'
            try:
                # nrows=10을 줘서 실제 데이터 10줄을 utf-8로 읽어봄
                # 데이터에 한글이 있다면 여기서 에러가 발생하여 except로 넘어감
                pd.read_csv(f, usecols=eng_cols, nrows=10, encoding='utf-8')
            except UnicodeDecodeError:
                target_encoding = 'cp949' # utf-8 실패 시 cp949 확정

            print(f"   -> 인코딩 확정: {target_encoding} ({os.path.basename(f)})")

            # 확정된 인코딩으로 Chunk Iterator 생성
            chunk_iter = pd.read_csv(f, usecols=eng_cols, chunksize=100000, encoding=target_encoding)

            # 데이터 처리
            for chunk in chunk_iter:
                chunk.rename(columns=rename_map, inplace=True)

                # 카테고리 매핑 수집
                pairs = chunk[['대분류', '중분류']].drop_duplicates()
                for _, row in pairs.iterrows():
                    structure_data.add((row['대분류'], row['중분류']))

                # 학습 데이터 집계
                grouped = chunk.groupby(['연령별', '성별', '중분류'])['매출금액'].sum().reset_index()
                aggregated_chunks.append(grouped)

        except Exception as e:
            print(f"🚫 파일 읽기 실패 ({f}): {e}")
            continue

    if not aggregated_chunks:
        return None, [], {}

    # 카테고리 맵 구축
    for major, middle in structure_data:
        if major not in local_category_map:
            local_category_map[major] = []

        local_category_map[major].append(middle)

    # 정렬
    for k in local_category_map: local_category_map[k].sort()

    # 데이터 병합
    total_df = pd.concat(aggregated_chunks, axis=0)
    final_df = total_df.groupby(['연령별', '성별', '중분류'])['매출금액'].sum().reset_index()

    features_list = sorted(final_df['중분류'].unique().tolist())

    # 피벗 및 정규화
    pivot_df = final_df.pivot_table(index=['연령별', '성별'], columns='중분류', values='매출금액', fill_value=0)
    model_data = pivot_df.div(pivot_df.sum(axis=1), axis=0).fillna(0)

    category_map = local_category_map
    major_categories = features_list

    return model_data, features_list, local_category_map

# ==========================================
# 2. 다중 K 모델 관리자
# ==========================================
class LargeScaleClusterManager:
    def __init__(self):
        self.models = {}
        self.k_levels = [i for i in range(3, 9)]
        self.feature_names = []
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    def train(self):
        global major_categories, category_map

        # 이미 학습된 .pkl 파일이 있으면 로드
        feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
        map_path = os.path.join(self.model_dir, 'category_map.pkl')

        # 저장된 데이터(컬럼, 맵) 로드 시도
        is_data_ready = False
        if os.path.exists(feature_path) and os.path.exists(map_path):
            try:
                self.feature_names = joblib.load(feature_path)
                loaded_map = joblib.load(map_path)

                # 전역 변수 동기화
                major_categories = self.feature_names
                category_map.update(loaded_map) # 전역 변수 업데이트

                print(f"✅ 저장된 데이터 로드 완료 (컬럼: {len(major_categories)}개)")
                is_data_ready = True
            except:
                print("⚠️ 데이터 파일 손상, 다시 만듭니다.")

        # 모델 파일 로드 시도
        all_models_exist = True
        for k in self.k_levels:
            path = os.path.join(self.model_dir, f'kmeans_k{k}.pkl')
            if os.path.exists(path):
                try:
                    self.models[k] = joblib.load(path)
                except:
                    all_models_exist = False
            else:
                all_models_exist = False

        # 모델이나 컬럼 정보가 없으면 전체 재학습
        if not all_models_exist or not self.feature_names:
            print("⚠️ 모델 또는 데이터가 없어 새로 학습합니다...")
            data, features, cat_map = load_and_train_efficiently()

            if data is None:
                print("❌ 학습 실패: 데이터를 찾을 수 없습니다.")
                return

            self.feature_names = features
            major_categories = features
            category_map.update(cat_map)

            joblib.dump(features, feature_path)
            joblib.dump(cat_map, map_path)
            print(f"💾 컬럼 정보 저장 완료: {feature_path}")

            print("🚀 모델 학습 시작 (MiniBatchKMeans)...")
            for k in self.k_levels:
                # MiniBatchKMeans는 속도가 빠르고 메모리를 적게 씀
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    batch_size=2048,  # 한 번에 학습할 샘플 수
                    random_state=42,
                    n_init=10
                )
                kmeans.fit(data)
                self.models[k] = kmeans

                # 모델 저장 (서버 재시작 시 저장된 모델 사용)
                joblib.dump(kmeans, f'./model/kmeans_k{k}.pkl')

            print("🎉 모든 모델 학습 및 저장 완료!")

    def get_feature_names(self):
        if self.feature_names:
            return self.feature_names
        else:
            # 메모리에 없으면 파일에서 로드 시도
            feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(feature_path):
                self.feature_names = joblib.load(feature_path)
                return self.feature_names
            return [] # 학습된 게 없으면 빈 리스트 반환

    def predict(self, user_vector_norm, k):
        model = self.models.get(k)
        if not model:
            return None, None

        # Feature 개수 검증
        if user_vector_norm.shape[1] != model.n_features_in_:
            print(f"⚠️ 차원 불일치: 모델({model.n_features_in_}) vs 입력({user_vector_norm.shape[1]})")
            # 에러 방지, 강제로 맞춤
            if user_vector_norm.shape[1] < model.n_features_in_:
                user_vector_norm = np.pad(user_vector_norm, ((0, 0), (0, model.n_features_in_ - user_vector_norm.shape[1])))
            else:
                user_vector_norm = user_vector_norm[:, :model.n_features_in_]

        cid = model.predict(user_vector_norm)[0]
        centroid = model.cluster_centers_[cid]

        return cid, centroid, user_vector_norm

# 실행부 수정
cluster_manager = LargeScaleClusterManager()
# 서버 시작 시 학습 실행 (이미 학습된 모델 파일이 있다면 로드하고 패스)
cluster_manager.train()
feature_names = cluster_manager.feature_names

# ==========================================
# 3. 로컬 LLM 래퍼
# ==========================================
# 실제 GGUF 파일 경로로 변경 필요
MODEL_PATH = "./model/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"

try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
    has_llm = True
except:
    print("Warning: GGUF 모델을 찾을 수 없습니다. Mock 응답을 보냅니다.")
    has_llm = False

def generate_persona(features_desc, max_retries=2):
    # 소비 패턴 기반 닉네임과 해시태그 생성
    if not has_llm:
        return "분석 완료 (AI 미연동)", "#데이터 #분석 #준비중"

    # Few-shot 프롬프트
    prompt = f"""당신은 창의적인 닉네임 전문가입니다. 소비 패턴을 보고 재미있는 닉네임과 해시태그를 만드세요.

예시 1:
소비 패턴: 카페 65%, 베이커리 25%, 디저트 10%
닉네임 | #태그1 #태그2 #태그3
카페마법사 | #커피홀릭 #디저트왕 #카페순례

예시 2:
소비 패턴: 온라인쇼핑 70%, 배송비 20%, 반품 10%
닉네임 | #태그1 #태그2 #태그3
택배기다리미 | #쇼핑왕 #클릭중독 #배송추적

예시 3:
소비 패턴: 편의점 50%, 야식 30%, 택시 20%
닉네임 | #태그1 #태그2 #태그3
야행성인간 | #편의점러버 #야식파티 #택시왕

이제 아래 패턴으로 창의적으로 만드세요:
소비 패턴: {features_desc}
닉네임 | #태그1 #태그2 #태그3
"""

    for attempt in range(max_retries + 1):
        try:
            output = llm(
                prompt,
                max_tokens=60,
                temperature=0.8 + (attempt * 0.1),  # 재시도마다 창의성 증가로 동일 닉네임이 생성되지 않도록 함
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,  # 반복 억제
                stop=["\n", "\n\n", "예시"],
                echo=False
            )
            result = output['choices'][0]['text'].strip()

            # 닉네임에 따옴표와 별표 제거
            result = result.replace('"', '').replace("'", '').replace('"', '').replace('"', '')
            result = result.replace('*', '')

            # 파싱
            if "|" in result:
                parts = result.split("|", 1)
                nickname = parts[0].strip()
                tags = parts[1].strip()

                # 유효성 검사 강화
                if (nickname and tags and
                    tags.startswith("#") and
                    len(nickname) >= 2 and
                    nickname not in ["닉네임", "소비왕", "분석중, 다시 버튼을 눌러주세요."] and
                    "#분석중" not in tags):
                    return nickname, tags

            # 실패 시 재시도
            if attempt < max_retries:
                print(f"[Debug] 재시도 {attempt+1}/{max_retries}: {result}")
                continue

        except Exception as e:
            print(f"[Error] LLM 오류 (시도 {attempt+1}): {e}")
            if attempt < max_retries:
                continue

    # 최종 실패 시 패턴 기반 기본값
    fallback_nicknames = ["소비탐험가", "지갑지킴이", "알뜰왕", "씀씀이마스터"]
    return random.choice(fallback_nicknames), "#소비패턴 #데이터분석 #라이프스타일"

# ==========================================
# 4. Flask 라우팅 및 비즈니스 로직
# ==========================================
@app.route('/categories', methods=['GET'])
def get_categories():
    # 데이터 매니저에 카테고리 정보가 없으면 기본값 반환
    if not category_map:
            return jsonify({"Error": ["데이터 로딩 실패"]}), 500
    return jsonify(category_map)

@app.route('/analyze', methods=['POST'])
def analyze():
    # 모델 준비 확인
    if not feature_names:
        return jsonify({"error": "Server is initializing..."}), 503

    req = request.json
    k = int(req.get('k', 3))
    items = req.get('items', []) # items 리스트 가져오기

    # 프론트엔드에서 온 items 리스트를 {카테고리: 금액} 형태로 1차 가공
    user_sums = {}
    for item in items:
        middle = item.get('middle')
        amount = float(item.get('amount', 0))
        user_sums[middle] = user_sums.get(middle, 0) + amount

    # 벡터 생성 (모델 기준 순서 feature_names를 따름)
    user_vector = []
    total = 0

    matched_log = []

    for feat in feature_names:
        # 합산한 user_sums에서 값을 찾음
        val = user_sums.get(feat, 0)

        if val > 0:
            matched_log.append(f"{feat}: {val}")

        user_vector.append(val)
        total += val

    if total == 0:
        print("❌ 매칭 실패: 입력한 카테고리 이름이 백엔드 feature_names와 다릅니다.")
        print(f"   (힌트) 백엔드가 기대하는 이름 예시: {feature_names[:5]}...")
        return jsonify({
            "error": "데이터 매칭 실패. 입력한 카테고리가 학습 데이터에 없습니다.",
            "debug_info": {
                "input_keys": list(user_sums.keys()),
                "expected_sample": feature_names[:5]
            }
        }), 400

    # 정규화
    user_vector_norm = np.array([v/total for v in user_vector]).reshape(1, -1)

    # 클러스터 예측
    cluster_id, centroid, final_vec = cluster_manager.predict(user_vector_norm, k)
    if cluster_id is None:
        return jsonify({"error": "Model Error"}), 500

    # 카테고리 이름 확보
    cats = feature_names
    my_vec = user_vector_norm[0]

    # Gap Analysis - 나와 평균의 차이 분석
    # (내 비율 - 그룹 평균 비율)
    diff = user_vector_norm[0] - centroid
    group_max_idx = np.argmax(centroid)
    group_main_cat = cats[group_max_idx] if group_max_idx < len(cats) else "기타"

    # 나의 특징 찾기
    my_max_idx = np.argmax(user_vector_norm[0])
    my_max_val = user_vector_norm[0][my_max_idx]
    my_main_cat = cats[my_max_idx] if my_max_idx < len(cats) else "기타"

    # 내가 그룹보다 압도적으로 많이 쓰는 것 (Gap Max)
    gap_max_idx = np.argmax(diff)
    gap_max_cat = cats[gap_max_idx]

    # 내가 그룹보다 훨씬 적게 쓰는 것 (Gap Min)
    gap_min_idx = np.argmin(diff)
    gap_min_cat = cats[gap_min_idx]

    feature_desc = ""
    prompt_style = "normal" # normal | obsessed

    # 조건부 프롬프트 생성
    if my_max_val >= 0.9: # 한 카테고리에 90% 이상 썼다면?
        feature_desc = f"Obsessed with {my_main_cat} (Spending {int(my_max_val*100)}% of money only on {my_main_cat})"
        prompt_style = "obsessed"
    else:
        prompt_style = "normal"
        top_indicies = my_vec.argsort()[-2:][::-1]

    if prompt_style == "obsessed":
         # 한 우물 파는 경우 -> "장인", "매니아" 같은 단어 유도
        llm_input = f"{feature_desc}. This person loves {my_main_cat} too much."
    else:
        llm_input = feature_desc

        # 0보다 큰 항목만 설명에 포함
        desc = []
        for i in top_indicies:
            if my_vec[i] > 0.01: # 1% 이상 쓴 것만
                desc.append(f"{cats[i]}(많음)")

        if not desc:
            llm_input = "Characteristic: Normal with no particular place to spend money."
        else:
            llm_input = "money spending pattern: " + ", ".join(desc)

    print(f"📝 LLM 요청: {llm_input}")
    nickname, tags = generate_persona(llm_input)

    # 만약 LLM이 빈 값을 뱉으면, 룰베이스 백업 닉네임 제공
    if not nickname or "Explanation" in nickname:
        if prompt_style == "obsessed":
            nickname = f"{my_main_cat} 매니아"
            tags = f"#{my_main_cat} #{group_main_cat} #마스터"
        else:
            nickname = "합리적인 밸런스족"
            tags = f"#{my_main_cat} #{group_main_cat} #스마트 컨슈머"

    # Case A: 내가 그룹보다 적게 쓰는 부분 (Saving)
    if gap_min_cat == group_main_cat:
        # 그룹은 이걸 좋아하는데 나는 안 쓰는 경우
        saving_text = f"이 소비 그룹은 '{group_main_cat}' 소비가 핵심인데, 사용자님은 이 부분에서 돈을 아끼셨어요!"
    else:
        saving_text = f"보통 '{group_main_cat}'을 많이 소비하는 그룹이지만, 사용자님은 이 소비 그룹보다 '{gap_min_cat}' 소비가 적은 편이에요."

    # Case B: 내가 그룹보다 많이 쓰는 부분 (Unique)
    if my_main_cat == group_main_cat:
        unique_text = f"이 소비 그룹과 마찬가지로 사용자님도 '{my_main_cat}'에 진심이시군요! (평균보다 {int(diff[gap_max_idx] * 100)}%p 더 높음)"
    else:
        unique_text = f"대체로 '{group_main_cat}' 위주 소비 그룹에 속하시지만, 사용자님은 '{my_main_cat}' 취향이 확고하시네요!"

    # K값 변화에 따른 히스토리 (Lineage)
    # 사용자가 현재 선택한 k 외에 다른 k에서는 어디에 속하는지 계산해서 반환
    lineage = {}
    for k in cluster_manager.k_levels:
        cid, _, _ = cluster_manager.predict(user_vector_norm, k)
        lineage[f"k={k}"] = int(cid)

    return jsonify({
        "current_k": k,
        "cluster_id": int(cluster_id),
        "persona_nickname": nickname,
        "persona_tags": tags,
        "gap_analysis": {
                "unique_trait": unique_text,
                "saving_trait": saving_text
            },
        "cluster_lineage": lineage,
        "debug_vector": [user_vector_norm[0].tolist()],
        "group_vector": centroid.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)