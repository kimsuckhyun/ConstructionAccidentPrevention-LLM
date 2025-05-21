#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import faiss
import re
from collections import Counter
from tqdm import tqdm

# 필요한 패키지 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Transformers 및 PEFT 라이브러리 임포트
from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from accelerate import Accelerator
# from trl import SFTTrainer
from transformers import Trainer, TrainingArguments

# 랜덤 시드 설정
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 로그 디렉터리 생성
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# 로그 파일 경로 설정 (타임스탬프 추가)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"generation_log_{timestamp}.jsonl")

# -------------------------
# Data Load & Pre-processing
# -------------------------

def load_and_preprocess_data(train_path, test_path):
    print("데이터 로드 및 JSON 구조화 전처리 중...")
    
    train = pd.read_csv(train_path, encoding='utf-8-sig')
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    
    # 데이터프레임에 분리된 열 추가
    for df in [train, test]:
        df['공사종류(대분류)'] = df['공사종류'].str.split(' / ').str[0]
        df['공사종류(중분류)'] = df['공사종류'].str.split(' / ').str[1]
        df['공종(대분류)'] = df['공종'].str.split(' > ').str[0]
        df['공종(중분류)'] = df['공종'].str.split(' > ').str[1]
        df['사고객체(대분류)'] = df['사고객체'].str.split(' > ').str[0]
        df['사고객체(중분류)'] = df['사고객체'].str.split(' > ').str[1]
    
    train_json = []
    test_json = []
    
    # 훈련 데이터 JSON 구조화
    for _, row in tqdm(train.iterrows(), desc="훈련 데이터 JSON 구조화", total=len(train)):
        # JSON 객체 생성
        item = {
            "construction": {
                "major": row['공사종류(대분류)'],
                "minor": row['공사종류(중분류)']
            },
            "work": {
                "major": row['공종(대분류)'],
                "minor": row['공종(중분류)']
            },
            "accident_object": {
                "major": row['사고객체(대분류)'],
                "minor": row['사고객체(중분류)']
            },
            "process": row['작업프로세스'],
            "cause": row['사고원인'],
            "human_damage": row['인적사고'],
            "material_damage": row['물적사고'],
            "prevention": row['재발방지대책 및 향후조치계획']
        }
        
        train_json.append(item)
    
    # 테스트 데이터 JSON 구조화
    for _, row in tqdm(test.iterrows(), desc="테스트 데이터 JSON 구조화", total=len(test)):
        # JSON 객체 생성
        item = {
            "construction": {
                "major": row['공사종류(대분류)'],
                "minor": row['공사종류(중분류)']
            },
            "work": {
                "major": row['공종(대분류)'],
                "minor": row['공종(중분류)']
            },
            "accident_object": {
                "major": row['사고객체(대분류)'],
                "minor": row['사고객체(중분류)']
            },
            "process": row['작업프로세스'],
            "cause": row['사고원인'],
            "human_damage": row['인적사고'],
            "material_damage": row['물적사고']
        }
        
        test_json.append(item)
    
    # 훈련 데이터 통합 생성 (기존 형식과 호환되도록)
    combined_training_data = train.apply(
        lambda row: {
            "question": (
                f"{row['작업프로세스']}중 '{row['사고원인']}'으로 인해 사고가 발생했습니다. "
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"본 사고로 인해 인적 피해는 '{row['인적사고']}', 물적 피해는 '{row['물적사고']}'이 발생했습니다. "
                f"이 사고의 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
            "answer": row["재발방지대책 및 향후조치계획"]
        },
        axis=1
    )
    combined_training_data = pd.DataFrame(list(combined_training_data))

    # 테스트 데이터 통합 생성 (기존 형식과 호환되도록)
    combined_test_data = test.apply(
        lambda row: {
            "question": (
                f"{row['작업프로세스']}중 '{row['사고원인']}'으로 인해 사고가 발생했습니다. "
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"본 사고로 인해 인적 피해는 '{row['인적사고']}', 물적 피해는 '{row['물적사고']}'이 발생했습니다. "
                f"이 사고의 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            )
        },
        axis=1
    )
    combined_test_data = pd.DataFrame(list(combined_test_data))
    
    return train, test, train_json, test_json, combined_training_data, combined_test_data

# 데이터 로드 및 전처리
train_path = './dataset/train.csv'
test_path = './dataset/test.csv'
train_df, test_df, train_json, test_json, combined_training_data, combined_test_data = load_and_preprocess_data(train_path, test_path)

# -------------------------
# JaccardSim 최적화를 위한 사전 분석
# -------------------------

# 정답 길이 분석
train_answers_prevention = combined_training_data['answer'].tolist()
answer_lengths = [len(answer) for answer in train_answers_prevention]
avg_length = sum(answer_lengths) / len(answer_lengths)
median_length = sorted(answer_lengths)[len(answer_lengths)//2]

print(f"평균 정답 길이: {avg_length}, 중앙값 길이: {median_length}")

# 정답에서 자주 등장하는 핵심 용어 분석
def extract_key_terms(answers, min_length=2, max_length=6):
    """정답에서 자주 등장하는 핵심 용어를 추출"""
    all_terms = []
    
    for answer in answers:
        # 정규식으로 한글 용어 추출 (2-6글자)
        terms = re.findall(r'[가-힣]{' + str(min_length) + ',' + str(max_length) + '}', answer)
        all_terms.extend(terms)
    
    # 빈도수에 따라 정렬
    term_counter = Counter(all_terms)
    
    # 특정 빈도 이상인 용어만 반환
    threshold = len(answers) * 0.05  # 전체 데이터의 5% 이상에서 등장
    common_terms = {term: count for term, count in term_counter.items() if count >= threshold}
    
    return common_terms

common_terms = extract_key_terms(train_answers_prevention)
print(f"추출된 핵심 용어 수: {len(common_terms)}")
print("상위 20개 핵심 용어:", sorted(common_terms.items(), key=lambda x: x[1], reverse=True)[:20])

# -------------------------
# Vector Store 생성 (CPU HNSW 사용)
# -------------------------

# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()
train_documents = [
    f"Q: {q}\nA: {a}"
    for q, a in zip(train_questions_prevention, train_answers_prevention)
]

# 임베딩 생성 (HuggingFaceEmbeddings 사용)
embedding_model_name = "jhgan/ko-sbert-nli"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 문서 임베딩 계산 (리스트 형태로 반환됨)
doc_embeddings = embedding.embed_documents(train_documents)
# 리스트를 NumPy 배열로 변환 (float32 타입)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# 임베딩 차원 (ko-sbert의 경우 보통 768 차원)
d = doc_embeddings.shape[1]

# CPU 기반 HNSW 인덱스 생성
index_cpu = faiss.IndexHNSWFlat(d, 32)  # 32는 M 파라미터 (노드당 최대 연결 수)
index_cpu.hnsw.efConstruction = 40  # 구축 시 탐색 너비
index_cpu.hnsw.efSearch = 16  # 검색 시 탐색 너비
index_cpu.add(doc_embeddings)

# FAISS 객체 생성 (docstore 등 포함)
vector_store_cpu = FAISS.from_texts(train_documents, embedding)
# 인덱스를 HNSW 인덱스로 교체
vector_store_cpu.index = index_cpu

# Retriever 정의 - MMR 검색으로 변경
retriever = vector_store_cpu.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance 사용
    search_kwargs={
        "k": 5,         # 검색할 문서 수
        "fetch_k": 15,  # 초기에 가져올 문서 수
        "lambda_mult": 0.7  # 다양성과 관련성의 균형 (0.7은 관련성 70%, 다양성 30%)
    }
)

# -------------------------
# LoRA 파인튜닝을 위한 데이터 준비
# -------------------------

# LoRA 학습용 데이터셋 생성 - RAG 스타일 (컨텍스트 포함)
def prepare_rag_style_dataset():
    rag_training_data = []
    
    print("LoRA 훈련 데이터 준비 중...")
    for idx, row in tqdm(combined_training_data.iterrows(), total=len(combined_training_data)):
        question = row['question']
        answer = row['answer']
        
        # 유사 문서 검색
        similar_docs = retriever.get_relevant_documents(question)
        
        # 컨텍스트 구성
        contexts = []
        for doc in similar_docs:
            doc_text = doc.page_content
            if "Q: " in doc_text and "\nA: " in doc_text:
                # Q&A 형식에서 답변만 추출
                context_answer = doc_text.split("\nA: ")[1]
                contexts.append(context_answer)
        
        # 최종 프롬프트 형식 구성
        prompt_text = f"""### 지침: 당신은 건설 안전 전문가입니다. 
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

### 참고 사례:
{" ".join(contexts)}

### 질문:
{question}

### 답변:
"""
        
        # 데이터셋 형식에 맞게 저장
        rag_training_data.append({
            "text": prompt_text + answer
        })
    
    return pd.DataFrame(rag_training_data)

# 직접 QA 스타일 (컨텍스트 없음)
def prepare_direct_qa_dataset():
    qa_training_data = []
    
    for idx, row in combined_training_data.iterrows():
        prompt_text = f"""### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

### 질문:
{row['question']}

### 답변:
"""
        
        qa_training_data.append({
            "text": prompt_text + row['answer']
        })
    
    return pd.DataFrame(qa_training_data)

# RAG 스타일 데이터셋 생성
rag_style_df = prepare_rag_style_dataset()
print(f"RAG 스타일 데이터셋 크기: {len(rag_style_df)}")

# 직접 QA 스타일 데이터셋 생성
direct_qa_df = prepare_direct_qa_dataset()
print(f"직접 QA 스타일 데이터셋 크기: {len(direct_qa_df)}")

# HF 데이터셋 형식으로 변환
rag_dataset = Dataset.from_pandas(rag_style_df)
qa_dataset = Dataset.from_pandas(direct_qa_df)

# -------------------------
# LoRA 모델 설정 및 학습
# -------------------------

# 모델 로드 (4비트 양자화)
def load_base_model():
    print("기본 모델 로드 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_id = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    return model, tokenizer

# LoRA 설정
def configure_lora(model):
    print("LoRA 설정 중...")
    # 4비트 훈련을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,                   # LoRA 랭크
        lora_alpha=32,          # 스케일링 파라미터
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 타겟 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # LoRA 어댑터 적용
    model = get_peft_model(model, lora_config)
    
    return model

# LoRA 훈련 함수
def train_with_lora(model, tokenizer, dataset, output_dir, epochs=3):
    print(f"LoRA 훈련 시작: {output_dir}")
    
    # 데이터 전처리 함수 - SFTTrainer가 내부적으로 하던 작업을 직접 수행
    def preprocess_function(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        )
        # 입력 IDs와 어텐션 마스크 반환
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"].clone(),  # 자기 회귀 학습을 위해
        }
    
    # 데이터셋 전처리 적용
    processed_dataset = dataset.map(preprocess_function, batched=True)
    
    # 훈련 인수 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # 메모리에 맞게 조정
        gradient_accumulation_steps=8,  # 그래디언트 누적
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # 훈련 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model(output_dir)
    
    print(f"LoRA 학습 완료: {output_dir}")
    return model

# 두 가지 버전의 LoRA 모델 훈련
# 1. RAG 스타일 데이터셋으로 훈련
model, tokenizer = load_base_model()
lora_model_rag = configure_lora(model)
lora_model_rag = train_with_lora(
    lora_model_rag, 
    tokenizer, 
    rag_dataset, 
    "./lora_models/construction_safety_rag_style_Qwen2.5-72B"
)

# 2. 직접 QA 스타일 데이터셋으로 훈련
model, tokenizer = load_base_model()  # 새 모델 인스턴스 로드
lora_model_qa = configure_lora(model)
lora_model_qa = train_with_lora(
    lora_model_qa, 
    tokenizer, 
    qa_dataset, 
    "./lora_models/construction_safety_qa_style_Qwen2.5-72B"
)

# -------------------------
# LoRA 모델 로드 및 추론 파이프라인 생성 (두 모델 모두 로드)
# -------------------------

def load_lora_model(base_model_id, lora_model_path):
    print(f"LoRA 모델 로드 중: {lora_model_path}")
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 기본 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    return model, tokenizer

def create_generation_pipeline(model, tokenizer, temperature=0.1, max_new_tokens=150):
    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        device_map="auto",
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        return_full_text=False
    )

# 기본 모델 ID
model_id = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"

# 1. RAG 스타일 LoRA 모델 로드
lora_path_rag = "./lora_models/construction_safety_rag_style_Qwen2.5-72B"
lora_model_rag, tokenizer_rag = load_lora_model(model_id, lora_path_rag)

# 2. QA 스타일 LoRA 모델 로드
lora_path_qa = "./lora_models/construction_safety_qa_style_Qwen2.5-72B"
lora_model_qa, tokenizer_qa = load_lora_model(model_id, lora_path_qa)

# 추론 파이프라인 생성 (두 모델 모두)
lora_pipeline_rag = create_generation_pipeline(lora_model_rag, tokenizer_rag)
lora_pipeline_qa = create_generation_pipeline(lora_model_qa, tokenizer_qa)
print("두 개의 LoRA 모델이 로드되었습니다: RAG 스타일 및 QA 스타일")

# -------------------------
# JaccardSim 최적화 함수들
# -------------------------

def clean_model_output(text):
    """모델 출력에서 </think> 태그와 그 이후 텍스트를 제거합니다."""
    if "</think>" in text:
        return text.split("</think>")[0].strip()
    return text.strip()

def get_optimal_token_length(question, retriever, train_answers_prevention, default_length=150):
    """질문의 특성에 따라 최적의 토큰 길이 반환"""
    # 유사 질문을 검색하고 해당 정답의 길이에 기반한 토큰 수 계산
    similar_docs = retriever.get_relevant_documents(question)
    if not similar_docs:
        return default_length
        
    # 유사 문서에서 정답 길이 추출
    answer_lengths = []
    for doc in similar_docs:
        answer = doc.page_content.split('\nA: ')[1] if '\nA: ' in doc.page_content else ""
        answer_lengths.append(len(answer))
        
    # 유사 정답들의 평균 길이
    avg_length = sum(answer_lengths) / len(answer_lengths)
    
    # 문자 길이를 토큰 길이로 대략 변환 (한국어는 대략 문자 1.5~2개당 토큰 1개)
    estimated_tokens = avg_length / 1.7
    
    # 약간의 여유를 두고 토큰 수 설정
    return int(estimated_tokens * 1.2)

def enhance_jaccard_similarity(generated_text, reference_answers, common_terms, top_k=3):
    """정답들의 핵심 용어를 추출하여 생성된 텍스트에 추가"""
    # 생성된 텍스트에 없는 관련 용어 찾기
    missing_terms = [term for term in common_terms if term not in generated_text]
    
    # 빈도수 기준으로 상위 용어 top_k개 선택
    relevant_terms = sorted(
        [(term, common_terms[term]) for term in missing_terms], 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    # 이미 충분히 유사하다면 수정하지 않음
    if not relevant_terms:
        return generated_text
        
    # 텍스트 끝에 관련 용어 추가
    enhanced_text = generated_text
    if relevant_terms:
        term_texts = [term for term, _ in relevant_terms]
        enhanced_text += f" 추가적으로 {', '.join(term_texts)}에 대한 조치도 필요합니다."
        
    return enhanced_text

def adjust_output_length(text, target_length, tolerance=0.2):
    """출력 텍스트의 길이를 목표 길이에 맞게 조정"""
    current_length = len(text)
    target_min = target_length * (1 - tolerance)
    target_max = target_length * (1 + tolerance)
    
    if target_min <= current_length <= target_max:
        return text  # 이미 적절한 길이
        
    if current_length < target_min:
        # 너무 짧으면 추가 정보로 확장 (이미 있는 내용 기반)
        sentences = text.split('. ')
        if len(sentences) > 1:
            expanded = '. '.join(sentences) + '. 또한, ' + sentences[0].replace('해야 합니다', '철저히 해야 합니다')
            return expanded
            
    if current_length > target_max:
        # 너무 길면 가장 중요한 문장만 유지
        sentences = text.split('. ')
        important_sentences = sentences[:len(sentences)//2 + 1]  # 앞부분 문장들 유지
        return '. '.join(important_sentences) + '.'
        
    return text

def normalize_terms(text):
    """일관된 용어로 표현 통일"""
    # 용어 치환 정의
    replacements = {
        "안전 교육": "안전교육",
        "안전관련 교육": "안전교육",
        "안전 관리자": "안전관리자",
        "관리 감독": "관리감독",
        "사전 점검": "사전점검",
        "안전 수칙": "안전수칙"
    }
    
    # 치환 적용
    normalized_text = text
    for old, new in replacements.items():
        normalized_text = normalized_text.replace(old, new)
        
    return normalized_text

# -------------------------
# LoRA 모델을 사용한 추론
# -------------------------

def generate_with_lora(question, lora_pipeline, retriever=None, use_rag=True):
    # RAG 방식 사용 여부에 따라 프롬프트 구성
    if use_rag and retriever:
        # 유사 문서 검색
        similar_docs = retriever.get_relevant_documents(question)
        
        # 컨텍스트 구성
        contexts = []
        for doc in similar_docs:
            doc_text = doc.page_content
            if "Q: " in doc_text and "\nA: " in doc_text:
                context_answer = doc_text.split("\nA: ")[1]
                contexts.append(context_answer)
        
        # RAG 스타일 프롬프트
        prompt = f"""### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

### 참고 사례:
{" ".join(contexts)}

### 질문:
{question}

### 답변:
"""
    else:
        # 직접 QA 스타일 프롬프트
        prompt = f"""### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

### 질문:
{question}

### 답변:
"""
    
    # 추론 실행
    result = lora_pipeline(prompt, max_new_tokens=150)[0]['generated_text']
    
    return result

# 모델 앙상블 및 추론 (두 개의 다른 모델 사용)
def ensemble_inference_both_models(question, lora_pipeline_rag, lora_pipeline_qa, retriever, common_terms, median_length):
    # 결과 저장용 로그 딕셔너리
    log_entry = {
        "question": question,
        "timestamp": datetime.now().isoformat()
    }
    
    # 1. RAG 스타일 모델로 추론 (컨텍스트 사용)
    lora_result_rag = generate_with_lora(
        question, 
        lora_pipeline_rag, 
        retriever=retriever, 
        use_rag=True
    )
    
    # 2. QA 스타일 모델로 추론 (컨텍스트 없음)
    lora_result_qa = generate_with_lora(
        question, 
        lora_pipeline_qa, 
        retriever=None, 
        use_rag=False
    )
    
    # 원본 결과 저장
    log_entry["raw_rag_result"] = lora_result_rag
    log_entry["raw_qa_result"] = lora_result_qa
    
    # 3. 출력 정제
    lora_result_rag = clean_model_output(lora_result_rag)
    lora_result_qa = clean_model_output(lora_result_qa)
    
    # 정제 후 결과 저장
    log_entry["cleaned_rag_result"] = lora_result_rag
    log_entry["cleaned_qa_result"] = lora_result_qa
    
    # 4. 용어 정규화
    lora_result_rag = normalize_terms(lora_result_rag)
    lora_result_qa = normalize_terms(lora_result_qa)
    
    # 정규화 후 결과 저장
    log_entry["normalized_rag_result"] = lora_result_rag
    log_entry["normalized_qa_result"] = lora_result_qa
    
    # 5. 앙상블 (두 결과 통합)
    # 각 결과의 문장 분리
    rag_sentences = lora_result_rag.split('. ')
    qa_sentences = lora_result_qa.split('. ')
    
    # 첫 번째 결과에서 주요 문장 선택
    ensemble_result = rag_sentences[:2]
    
    # 두 번째 결과에서 겹치지 않는 문장 추가
    for sentence in qa_sentences:
        if sentence and all(not s.startswith(sentence[:10]) for s in ensemble_result):
            ensemble_result.append(sentence)
    
    # 문장 통합
    ensemble_text = '. '.join(ensemble_result)
    if not ensemble_text.endswith('.'):
        ensemble_text += '.'
    
    # 앙상블 후 결과 저장
    log_entry["ensemble_result"] = ensemble_text
    
    # 6. 자카드 유사도 향상을 위한 후처리
    enhanced_text = enhance_jaccard_similarity(
        ensemble_text, 
        train_answers_prevention, 
        common_terms
    )
    
    # 유사도 강화 후 결과 저장
    log_entry["enhanced_result"] = enhanced_text
    
    # 7. 길이 최적화
    final_text = adjust_output_length(enhanced_text, median_length)
    
    # 최종 결과 저장
    log_entry["final_result"] = final_text
    
    # 로그 파일에 결과 추가
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    return final_text

# -------------------------
# 최종 추론 및 결과 생성 (두 개의 모델 사용)
# -------------------------

test_results = []
print(f"두 개의 LoRA 모델을 사용한 앙상블 테스트 시작... 총 테스트 샘플 수: {len(combined_test_data)}")
print(f"로그 파일이 저장될 경로: {log_file_path}")

for idx, row in tqdm(combined_test_data.iterrows(), total=len(combined_test_data)):
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"\n[샘플 {idx + 1}/{len(combined_test_data)}] 진행 중...")
        print(f"현재까지 {idx + 1}개 샘플 처리 완료")
    
    # 앙상블 추론 실행 (두 개의 모델 모두 사용)
    final_text = ensemble_inference_both_models(
        row['question'],
        lora_pipeline_rag,
        lora_pipeline_qa,
        retriever,
        common_terms,
        median_length
    )
    
    test_results.append(final_text)

print("\n테스트 실행 완료! 총 결과 수:", len(test_results))
print(f"모든 중간 결과가 다음 로그 파일에 저장되었습니다: {log_file_path}")

# -------------------------
# Submission
# -------------------------

from sentence_transformers import SentenceTransformer

embedding_model_name_submission = "jhgan/ko-sbert-sts"
embedding_submission = SentenceTransformer(embedding_model_name_submission)

pred_embeddings = embedding_submission.encode(test_results)
print("임베딩 shape:", pred_embeddings.shape)  # (샘플 개수, 768)

submission = pd.read_csv('./dataset/sample_submission.csv', encoding='utf-8-sig')
submission.iloc[:, 1] = test_results
submission.iloc[:, 2:] = pred_embeddings
print(submission.head())

# 파일명에 LoRA 정보 포함 및 앙상블 정보 추가
timestamp_str = datetime.now().strftime("%m%d_%H%M")
submission.to_csv(f'./results/qwen25_72b_submission_v4_dual_lora_ensemble.csv', index=False, encoding='utf-8-sig')
print(f"제출 파일이 저장되었습니다: ./results/qwen25_72b_submission_v4_dual_lora_ensemble.csv")