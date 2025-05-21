#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv("./dataset/train.csv")
test_df = pd.read_csv("./dataset/test.csv")
submission = pd.read_csv("./dataset/sample_submission.csv", encoding="utf-8-sig")

# ✅ NaN 처리
for col in ["사고원인", "공종", "작업프로세스", "장소", "부위"]:
    train_df[col] = train_df[col].fillna("").astype(str)
    test_df[col] = test_df[col].fillna("").astype(str)

# Step 2: GPU에서 실행되도록 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("jhgan/ko-sbert-sts", device=device)
print(device)

# ✅ 벡터화 (사고원인 + 추가 메타데이터)
def generate_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), 64), desc="Encoding", leave=False):
        batch = texts[i : i + 64]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# ✅ 사고원인 벡터화 (GPU)
train_vectors = generate_embeddings(train_df["사고원인"].tolist())
test_vectors = generate_embeddings(test_df["사고원인"].tolist())

# ✅ TF-IDF 벡터화 (공종, 작업프로세스, 장소, 부위)
metadata_cols = ["공종", "작업프로세스", "장소", "부위"]
tfidf_vectorizer = TfidfVectorizer()
train_metadata_text = train_df[metadata_cols].agg(" ".join, axis=1)
test_metadata_text = test_df[metadata_cols].agg(" ".join, axis=1)

train_tfidf_vectors = tfidf_vectorizer.fit_transform(train_metadata_text)
test_tfidf_vectors = tfidf_vectorizer.transform(test_metadata_text)

# ✅ 정규화
scaler = StandardScaler()
train_vectors = scaler.fit_transform(train_vectors)
test_vectors = scaler.transform(test_vectors)
train_tfidf_vectors = scaler.fit_transform(train_tfidf_vectors.toarray())
test_tfidf_vectors = scaler.transform(test_tfidf_vectors.toarray())
train_actions = train_df.apply(lambda row: (row["사고원인"], row["재발방지대책 및 향후조치계획"]), axis=1).tolist()

# ✅ 재발방지대책 및 향후조치계획 벡터화
prevention_vectors = generate_embeddings(train_df["재발방지대책 및 향후조치계획"].tolist())

# 평균 벡터 계산
mean_vector = np.mean(prevention_vectors, axis=0)

# 가장 평균에 가까운 재발방지대책 찾기
similarities = cosine_similarity([mean_vector], prevention_vectors)[0]
most_average_idx = np.argmax(similarities)
most_average_plan = train_df["재발방지대책 및 향후조치계획"].iloc[most_average_idx]

# ✅ mean_vector를 사용한 제출 파일 생성
submission = test_df[["ID"]].copy()  # ID 컬럼 유지

# 모든 행에 동일한 평균적인 재발방지대책 추가
submission["재발방지대책 및 향후조치계획"] = most_average_plan

# 결과 저장
submission_path = "./results/importance_text_extraction.csv"
submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

# 어떤 재발방지대책이 선택되었는지 확인
print("선택된 평균적인 재발방지대책:")
print(most_average_plan)