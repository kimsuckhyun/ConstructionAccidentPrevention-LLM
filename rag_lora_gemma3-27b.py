#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# CUDA 메모리 분할 방지
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 6번 GPU만 보이도록 설정
import json
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import faiss
import re
from collections import Counter
from tqdm import tqdm

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
# Vector Store 생성 (CPU HNSW 사용)
# -------------------------

# 필요한 패키지 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
# JaccardSim 최적화를 위한 사전 분석
# -------------------------

# 정답 길이 분석
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
# unsloth를 사용한 LoRA 파인튜닝
# -------------------------

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# RAG 스타일 데이터셋 생성
# 다음과 같이 데이터셋 형식을 변경합니다
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
                context_answer = doc_text.split("\nA: ")[1]
                contexts.append(context_answer)
        
        # unsloth 형식에 맞는 텍스트 구성
        # 문자열 포맷으로 변경
        text = f"""<start_of_turn>user
당신은 건설 안전 전문가입니다. 질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 불필요한 문구를 포함하지 마세요.

참고 사례:
{' '.join(contexts)}

질문:
{question}
<end_of_turn>

<start_of_turn>model
{answer}<end_of_turn>
"""
        
        # 데이터셋 형식에 맞게 저장 (text 필드로 직접)
        rag_training_data.append({"text": text})
    
    return rag_training_data

# 직접 QA 스타일 데이터셋 생성
def prepare_direct_qa_dataset():
    qa_training_data = []
    
    for idx, row in tqdm(combined_training_data.iterrows(), total=len(combined_training_data)):
        # unsloth 형식에 맞는 텍스트 구성
        text = f"""<start_of_turn>user
당신은 건설 안전 전문가입니다. 질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 불필요한 문구를 포함하지 마세요.

질문:
{row['question']}
<end_of_turn>

<start_of_turn>model
{row['answer']}<end_of_turn>
"""
        
        qa_training_data.append({"text": text})
    
    return qa_training_data

# 데이터셋 생성
rag_data = prepare_rag_style_dataset()
rag_dataset = Dataset.from_list(rag_data)
print(f"RAG 스타일 데이터셋 크기: {len(rag_dataset)}")

qa_data = prepare_direct_qa_dataset()
qa_dataset = Dataset.from_list(qa_data)
print(f"직접 QA 스타일 데이터셋 크기: {len(qa_dataset)}")

# -------------------------
# RAG 스타일 모델 파인튜닝
# -------------------------

def train_rag_model():
    print("RAG 스타일 모델 훈련 시작...")
    
    # 모델 로드
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",  # 27B 모델 사용
        max_seq_length = 2048,
        load_in_4bit = True,  # 4비트 양자화
        full_finetuning = False,  # LoRA 파인튜닝만 수행
        # device = "auto",
    )
    
    # LoRA 설정
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,  # 텍스트만 파인튜닝
        finetune_language_layers = True,  # 언어 레이어 파인튜닝
        finetune_attention_modules = True,  # 어텐션 모듈 파인튜닝
        finetune_mlp_modules = True,  # MLP 모듈 파인튜닝
        r = 16,  # LoRA 랭크 (정확도 향상)
        lora_alpha = 32,  # LoRA 알파 (r의 두배로 설정)
        lora_dropout = 0.05,  # 드롭아웃 추가
        bias = "none",
        random_state = 42,  # 랜덤 시드
    )
    
    # 채팅 템플릿 적용
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",  # Gemma-3 채팅 템플릿 사용
    )
    
    # standardize_data_formats 사용하지 않음
    # 데이터셋은 이미 "text" 필드를 포함하고 있음
    
    # 훈련 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=rag_dataset,  # 직접 데이터셋 전달
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field = "text",  # 이미 text 필드가 있음
            per_device_train_batch_size = 1,  # 배치 사이즈 (메모리에 맞게 조정)
            gradient_accumulation_steps = 8,  # 그래디언트 누적
            warmup_steps = 10,
            max_steps = 300,  # 학습 스텝 수
            learning_rate = 1e-4,  # 학습률
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",  # 학습률 스케줄러
            max_grad_norm = 1.0,  # 그래디언트 클리핑
            seed = 42,
            report_to = "none",
            output_dir = "./lora_models/unsloth_construction_safety_rag_style_gemma-3-27b",
        ),
    )
    
    # 모델 응답 부분만 훈련
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = '<start_of_turn>user\n',
        response_part = '<start_of_turn>model\n',
    )
    
    # 훈련 시작
    trainer_stats = trainer.train()
    
    # 모델 저장
    model.save_pretrained("./lora_models/unsloth_construction_safety_rag_style_gemma-3-27b")
    tokenizer.save_pretrained("./lora_models/unsloth_construction_safety_rag_style_gemma-3-27b")
    
    print("RAG 스타일 모델 학습 완료!")
    return model, tokenizer

# -------------------------
# QA 스타일 모델 파인튜닝
# -------------------------

def train_qa_model():
    print("QA 스타일 모델 훈련 시작...")
    
    # 모델 로드
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",  # 27B 모델 사용
        max_seq_length = 2048,
        load_in_4bit = True,  # 4비트 양자화
        full_finetuning = False,  # LoRA 파인튜닝만 수행
        # device = "auto",
    )
    
    # LoRA 설정
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,  # 텍스트만 파인튜닝
        finetune_language_layers = True,  # 언어 레이어 파인튜닝
        finetune_attention_modules = True,  # 어텐션 모듈 파인튜닝
        finetune_mlp_modules = True,  # MLP 모듈 파인튜닝
        r = 16,  # LoRA 랭크
        lora_alpha = 32,  # LoRA 알파
        lora_dropout = 0.05,  # 드롭아웃 추가
        bias = "none",
        random_state = 42,  # 랜덤 시드
    )
    
    # 채팅 템플릿 적용
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",  # Gemma-3 채팅 템플릿 사용
    )
    

    # 훈련 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=qa_dataset,  # 직접 qa_dataset 사용
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field = "text",  # 이미 존재하는 text 필드 사용
            per_device_train_batch_size = 1,  # 배치 사이즈 (메모리에 맞게 조정)
            gradient_accumulation_steps = 8,  # 그래디언트 누적
            warmup_steps = 10,
            max_steps = 300,  # 학습 스텝 수 증가
            learning_rate = 1e-4,  # 학습률
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",  # 학습률 스케줄러
            max_grad_norm = 1.0,  # 그래디언트 클리핑
            seed = 42,
            report_to = "none",
            output_dir = "./lora_models/unsloth_construction_safety_qa_style_gemma-3-27b",
        ),
    )
    
    # 모델 응답 부분만 훈련
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = '<start_of_turn>user\n',
        response_part = '<start_of_turn>model\n',
    )
    
    # 훈련 시작
    trainer_stats = trainer.train()
    
    # 모델 저장
    model.save_pretrained("./lora_models/unsloth_construction_safety_qa_style_gemma-3-27b")
    tokenizer.save_pretrained("./lora_models/unsloth_construction_safety_qa_style_gemma-3-27b")
    
    print("QA 스타일 모델 학습 완료!")
    return model, tokenizer

# -------------------------
# JaccardSim 최적화 함수들
# -------------------------

def clean_model_output(text):
    """모델 출력에서 특수 태그와 그 이후 텍스트를 제거합니다."""
    if "<end_of_turn>" in text:
        return text.split("<end_of_turn>")[0].strip()
    return text.strip()

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
# unsloth 모델 로드 및 추론
# -------------------------

def generate_with_unsloth(model, tokenizer, question, retriever=None, use_rag=True):
    """unsloth 모델을 사용하여 텍스트 생성"""
    
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
        
        # 메시지 구성
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"당신은 건설 안전 전문가입니다. 질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.\n- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.\n- 불필요한 문구를 포함하지 마세요.\n\n참고 사례:\n{' '.join(contexts)}\n\n질문:\n{question}"
            }]
        }]
    else:
        # 메시지 구성 (컨텍스트 없음)
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"당신은 건설 안전 전문가입니다. 질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.\n- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.\n- 불필요한 문구를 포함하지 마세요.\n\n질문:\n{question}"
            }]
        }]
    
    # 채팅 템플릿 적용
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    
    try:
        # 메모리 정리
        torch.cuda.empty_cache()
        
        # 추론 실행
        with torch.no_grad():
            outputs = model.generate(
                **tokenizer([text], return_tensors="pt").to("cuda"),
                max_new_tokens=150,
                temperature=0.1,  # 낮은 온도 값 (결정적 출력)
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,  # 반복 방지
                no_repeat_ngram_size=2,
            )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return result
        
    except Exception as e:
        print(f"생성 중 오류 발생: {str(e)}")
        # 기본 응답 반환
        return "안전관리자를 배치하고, 작업 전 안전교육을 실시해야 합니다. 위험요소에 대한 사전점검을 철저히 하고 안전장비를 착용해야 합니다."

# 앙상블 추론 (두 개의 다른 모델 사용)
def ensemble_inference(question, rag_model, rag_tokenizer, qa_model, qa_tokenizer, retriever, common_terms, median_length):
    # 결과 저장용 로그 딕셔너리
    log_entry = {
        "question": question,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # 메모리 정리
        torch.cuda.empty_cache()
        
        # 1. RAG 스타일 모델로 추론
        try:
            rag_result = generate_with_unsloth(
                rag_model, 
                rag_tokenizer,
                question, 
                retriever=retriever, 
                use_rag=True
            )
        except Exception as e:
            print(f"RAG 모델 추론 오류: {str(e)}")
            rag_result = "안전관리자를 배치하고, 작업 전 안전교육을 실시해야 합니다. 위험요소에 대한 사전점검을 철저히 하고 안전장비를 착용해야 합니다."
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        # 2. QA 스타일 모델로 추론
        try:
            qa_result = generate_with_unsloth(
                qa_model, 
                qa_tokenizer,
                question, 
                retriever=None, 
                use_rag=False
            )
        except Exception as e:
            print(f"QA 모델 추론 오류: {str(e)}")
            qa_result = "작업 관련 위험요소에 대한 사전교육을 철저히 하고, 안전장비와 보호구를 반드시 착용해야 합니다. 관리감독자는 작업 전 점검을 실시하고 안전수칙을 준수해야 합니다."
        
        # 원본 결과 저장
        log_entry["raw_rag_result"] = rag_result
        log_entry["raw_qa_result"] = qa_result
        
        # 3. 출력 정제
        rag_result = clean_model_output(rag_result)
        qa_result = clean_model_output(qa_result)
        
        # 정제 후 결과 저장
        log_entry["cleaned_rag_result"] = rag_result
        log_entry["cleaned_qa_result"] = qa_result
        
        # 4. 용어 정규화
        rag_result = normalize_terms(rag_result)
        qa_result = normalize_terms(qa_result)
        
        # 정규화 후 결과 저장
        log_entry["normalized_rag_result"] = rag_result
        log_entry["normalized_qa_result"] = qa_result
        
        # 5. 앙상블 (두 결과 통합)
        # 각 결과의 문장 분리
        rag_sentences = rag_result.split('. ')
        qa_sentences = qa_result.split('. ')
        
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
        
    except Exception as e:
        print(f"앙상블 추론 중 오류 발생: {str(e)}")
        # 기본 응답 반환
        default_response = "안전관리자를 배치하고, 작업 전 안전교육을 실시해야 합니다. 위험요소에 대한 사전점검을 철저히 하고 안전장비를 착용해야 합니다. 작업 절차와 안전수칙을 준수하고 주기적인 안전점검을 실시해야 합니다."
        
        # 오류 정보를 로그에 기록
        error_log = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "default_response": default_response
        }
        
        # 로그 파일에 오류 정보 추가
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_log, ensure_ascii=False) + '\n')
            
        return default_response

# -------------------------
# 메인 실행 함수
# -------------------------

def main():
    print("건설 안전 전문가 모델 실행 시작...")
    
    # 1. 모델 학습 (RAG 스타일)
    rag_model, rag_tokenizer = train_rag_model()
    
# 메인 함수 실행
if __name__ == "__main__":
    main()


