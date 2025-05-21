#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime
import re
from unsloth import FastModel

# 메모리 관리 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 3번 GPU 사용

# 로그 디렉터리 설정
log_dir = "./log/ensemble_selection"
os.makedirs(log_dir, exist_ok=True)

# 로그 파일 경로 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"ensemble_selection_log_{timestamp}.jsonl")

# 앙상블에 사용할 CSV 파일 경로 (2개 파일)
file_paths = [
    "./results/importance_text_extraction.csv",
    "./results/qwen25_72b_submission_v4_dual_lora_ensemble.csv",
]

def load_submission_files():
    """
    모든 제출 파일을 로드하고 ID와 재발방지대책 컬럼만 추출
    """
    print("파일 로딩 시작...")
    
    # 각 파일을 로드하고 필요한 컬럼만 추출
    candidate_solutions = {}
    
    for idx, path in enumerate(file_paths):
        print(f"파일 {idx+1}/{len(file_paths)} 로딩 중: {path}")
        df = pd.read_csv(path)
        
        # 첫 번째 파일에서 ID 컬럼 가져오기
        if idx == 0:
            # 첫 번째 열을 ID로 가정
            id_column = df.columns[0]
            ids = df[id_column].tolist()
            
        # 재발방지대책 컬럼 가져오기 (두 번째 열로 가정)
        prevention_column = df.columns[1]
        preventions = df[prevention_column].tolist()
        
        # 각 ID별로 재발방지대책 저장
        if idx == 0:
            # 처음 딕셔너리 초기화
            for i, id_val in enumerate(ids):
                candidate_solutions[id_val] = [preventions[i]]
        else:
            # 기존 딕셔너리에 추가
            for i, id_val in enumerate(ids):
                if id_val in candidate_solutions:
                    candidate_solutions[id_val].append(preventions[i])
    
    print(f"총 {len(candidate_solutions)} 개의 ID에 대해 데이터 로딩 완료")
    return candidate_solutions, ids

def load_original_test_data():
    """
    원본 테스트 데이터 로드 (앙상블 점수 계산을 위한 문맥 정보)
    """
    test_path = './dataset/test.csv'
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    
    # 데이터프레임에 분리된 열 추가
    test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
    test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
    test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
    test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
    test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
    test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]
    
    # 통합 질문 생성
    test_questions = test.apply(
        lambda row: (
            f"{row['작업프로세스']}중 '{row['사고원인']}'으로 인해 사고가 발생했습니다. "
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"본 사고로 인해 인적 피해는 '{row['인적사고']}', 물적 피해는 '{row['물적사고']}'이 발생했습니다. "
            f"이 사고의 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        axis=1
    ).tolist()
    
    # ID 컬럼 가져오기
    ids = test.iloc[:, 0].tolist()
    
    # ID를 키로 하는 질문 딕셔너리 생성
    questions_dict = {id_val: question for id_val, question in zip(ids, test_questions)}
    
    return questions_dict

def clean_text(text):
    """텍스트 정제 함수"""
    # null 값 처리
    if pd.isna(text):
        return ""
    
    # 중복된 마침표 처리
    text = re.sub(r'\.{2,}', '.', text)
    
    # 마침표 뒤에 공백 추가 (없는 경우)
    text = re.sub(r'\.([^\s])', '. \\1', text)
    
    # 문장 끝에 마침표 없는 경우 추가
    if not text.endswith('.'):
        text = text + '.'
    
    return text.strip()

def generate_comparison_prompt(question, candidates):
    """
    후보 솔루션을 비교하기 위한 프롬프트 생성
    """
    # 각 후보 솔루션 정제
    cleaned_candidates = [clean_text(candidate) for candidate in candidates]
    
    # 프롬프트 구성
    prompt = f"""
당신은 건설 안전 전문가입니다. 아래 질문과 관련된 사고 방지 대책 후보들 중 가장 적절한 것을 선택해주세요.

질문:
{question}

후보 대책들:
"""

    # 각 후보 대책 추가
    for idx, candidate in enumerate(cleaned_candidates):
        prompt += f"대책 {idx+1}: {candidate}\n\n"
    
    prompt += """
평가 기준:
1. 사고 원인과의 관련성: 제시된 사고 원인에 직접적으로 대응하는 대책인가?
2. 구체성: 대책이 구체적이고 실행 가능한가?
3. 포괄성: 인적, 물적 피해를 모두 고려한 종합적인 대책인가?
4. 효과성: 제안된 대책이 실제로 사고 재발을 방지할 수 있는가?

위 기준에 따라 가장 적합한 대책의 번호를 선택하고, 그 이유를 간략히 설명해주세요.
답변 형식: "선택: [번호], 이유: [간략한 설명]"
"""
    
    return prompt

def extract_selection(response):
    """모델 응답에서 선택된 번호 추출"""
    try:
        # "선택: 숫자" 패턴 찾기
        match = re.search(r'선택:\s*(\d+)', response)
        if match:
            selection = int(match.group(1))
            return selection
        
        # "대책 숫자" 패턴 찾기
        match = re.search(r'대책\s*(\d+)', response)
        if match:
            selection = int(match.group(1))
            return selection
            
        # 단순히 숫자만 있는 경우
        match = re.search(r'(\d+)', response)
        if match:
            selection = int(match.group(1))
            return selection
    except:
        pass
    
    # 기본값: 첫 번째 후보 선택
    return 1

def select_best_solution(model, tokenizer, question, candidates):
    """
    LLM 모델을 사용하여 최적의 솔루션 선택
    """
    # 비교 프롬프트 생성
    prompt = generate_comparison_prompt(question, candidates)
    
    # 메시지 형식으로 변환
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
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
                max_new_tokens=100,
                temperature=0.1,  # 낮은 온도 값 (결정적 출력)
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,  # 반복 방지
            )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 모델 응답만 추출
        if "\nmodel\n" in result:
            result = result.split("\nmodel\n")[-1].strip()
        
        # 응답에서 선택된 번호 추출
        selection = extract_selection(result)
        
        # 인덱스가 범위를 벗어나는 경우 처리
        if selection < 1 or selection > len(candidates):
            selection = 1
        
        # 선택된 솔루션 반환 (0-인덱스로 변환)
        return selection - 1, result
        
    except Exception as e:
        print(f"생성 중 오류 발생: {str(e)}")
        # 기본값: 첫 번째 후보 선택
        return 0, f"오류 발생: {str(e)}"

def main():
    print("앙상블 선택 프로세스 시작...")
    
    # LoRA 모델 로드
    print("LoRA 모델 로딩 중...")
    model, tokenizer = FastModel.from_pretrained(
        "./lora_models/unsloth_construction_safety_rag_style_gemma-3-27b"
    )
    
    # 제출 파일 로드
    candidate_solutions, ids = load_submission_files()
    
    # 원본 테스트 데이터 로드 (질문 컨텍스트)
    questions_dict = load_original_test_data()
    
    # 최종 결과 저장용 리스트
    final_selections = []
    
    print(f"총 {len(ids)}개의 샘플에 대해 최적의 솔루션 선택 시작...")
    
    # 각 ID에 대해 최적의 솔루션 선택
    for idx, id_val in enumerate(tqdm(ids)):
        # 진행 상황 출력
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"\n[샘플 {idx + 1}/{len(ids)}] 진행 중...")
        
        # 해당 ID의 질문과 후보 솔루션 가져오기
        question = questions_dict.get(id_val, "해당 ID의 질문 정보가 없습니다.")
        candidates = candidate_solutions.get(id_val, ["기본 솔루션"])
        
        # 로그 항목 초기화
        log_entry = {
            "id": id_val,
            "question": question,
            "candidates": candidates,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 최적의 솔루션 선택
            best_idx, model_response = select_best_solution(model, tokenizer, question, candidates)
            best_solution = candidates[best_idx]
            
            # 로그 항목 업데이트
            log_entry["selected_index"] = best_idx
            log_entry["selected_solution"] = best_solution
            log_entry["model_response"] = model_response
            
            # 후처리
            best_solution = str(best_solution)
            best_solution = best_solution.split('###')[0].strip()

            # 결과 저장
            final_selections.append(best_solution)
            
        except Exception as e:
            print(f"샘플 {idx+1} 처리 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 첫 번째 후보 사용
            best_solution = candidates[0] if candidates else "안전관리자를 배치하고, 작업 전 안전교육을 실시해야 합니다. 위험요소에 대한 사전점검을 철저히 하고 안전장비를 착용해야 합니다."
            final_selections.append(best_solution)
            
            # 로그 항목 업데이트
            log_entry["error"] = str(e)
            log_entry["selected_index"] = 0
            log_entry["selected_solution"] = best_solution
        
        # 로그 파일에 결과 기록
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 주기적으로 PyTorch 캐시와 CUDA 캐시 정리
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # 결과를 CSV 파일로 저장
    print("최종 결과 저장 중...")
    
    # 기존 샘플 제출 파일 로드
    sample_submission = pd.read_csv('./dataset/sample_submission.csv', encoding='utf-8-sig')
    
    # 선택된 솔루션으로 재발방지대책 컬럼 업데이트
    sample_submission.iloc[:, 1] = final_selections
    
    # 임베딩 생성 (원본 코드와 동일한 방식)
    from sentence_transformers import SentenceTransformer
    
    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding_model = SentenceTransformer(embedding_model_name)
    
    pred_embeddings = embedding_model.encode(final_selections)
    print("임베딩 shape:", pred_embeddings.shape)
    
    # 임베딩 값 업데이트
    sample_submission.iloc[:, 2:] = pred_embeddings
    
    # 파일 저장
    output_path = f'./results/ensemble_selection_submission.csv'
    sample_submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"제출 파일이 저장되었습니다: {output_path}")
    
    print("앙상블 선택 프로세스 완료!")

if __name__ == "__main__":
    main()