# 건설 안전 AI 경진대회 - 🏆 최종 순위 6위 (1,342명 참여 중)

<div align="center">
  <img src="https://img.shields.io/badge/Competition-6th_Place-red" alt="Competition Rank"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework"/>
  <img src="https://img.shields.io/badge/Models-Qwen2.5_Gemma3-blue" alt="Models"/>
  <img src="https://img.shields.io/badge/Technique-LoRA_RAG-green" alt="Technique"/>
</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b0b0c02-7f99-4905-a5b6-febe8147d3ad" alt="Competition Banner" width="800"/>
</p>

이 저장소는 건설 안전 AI 경진대회에 참가하여 **최종 6위**를 달성한 솔루션을 담고 있습니다. 건설 현장 안전사고 데이터 기반으로 최적의 재발방지대책을 제안하는 AI 모델을 개발했습니다. 대규모 언어 모델(LLM)과 검색 증강 생성(RAG)을 결합하여 건설 현장의 안전성을 향상시키는 솔루션을 제공합니다.
## 📊 프로젝트 개요
<div align="center">
  <img src="https://github.com/user-attachments/assets/17ea70b5-c666-4135-884d-75b467c99e84" alt="Project Architecture" width="800"/>
</div>

본 프로젝트는 건설 현장에서 발생하는 사고의 재발 방지를 위해 최적화된 AI 솔루션을 제공합니다. 주요 특징은 다음과 같습니다:

- **데이터 기반 접근**: 건설 사고 데이터를 활용한 최적의 재발방지대책 생성
- **하이브리드 모델링**: 대규모 언어 모델과 RAG(검색 증강 생성) 결합
- **앙상블 기법**: 다양한 모델의 장점을 결합하여 성능 향상
- **자동화된 서빙**: Streamlit과 Airflow를 활용한 지속적 모델 개선

## 💻 모델 파트

### 1. 환경 설정 및 패키지 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# Unsloth 라이브러리 설치
pip install git+https://github.com/unsloth/unsloth.git
```

### 2. 실행 순서


모델을 순차적으로 실행하여 최종 결과를 생성합니다:

```bash
# QA RAG LoRA 모델 (Qwen2.5-72B 기반)
python /project/ahnailab/jys0207/CP/aa/qa_rag_lora_qwen2.5-72b.py

# Gemma3 LoRA 모델 (이미 있을 경우 스킵 가능)
# python /project/ahnailab/jys0207/CP/aa/rag_lora_gemma3-27b.py

# 텍스트 중요도 추출
python /project/ahnailab/jys0207/CP/aa/importance_text_extraction.py

# 앙상블 모델 실행
python /project/ahnailab/jys0207/CP/aa/emsemble.py
```

### 3. 최종 결과 파일

```
./results/ensemble_selection_submission.csv
```

## 🛠️ 주요 기술 스택


<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/2a115bc7-b574-46bf-a6d1-18d24d817228" width="60"/><br/>Qwen2.5-72B</td>
      <td align="center"><img src="https://github.com/user-attachments/assets/a1429b07-792b-4fe4-9fda-143d583142ef"/><br/>Gemma3-27B</td>
    </tr>
  </table>
</div>

- **모델**: 
  - Qwen2.5-72B - 최신 대규모 언어 모델을 활용한 정교한 답변 생성
  - Gemma3-27B - Google의 최신 오픈소스 LLM을 활용한 보완적 답변 생성
- **RAG**: langchain + FAISS 벡터 스토어를 활용한 검색 증강 생성
  - ko-sbert 임베딩 모델을 사용한 의미적 검색
  - HNSW 알고리즘 기반의 효율적인 인덱싱 및 검색
- **앙상블**: 여러 모델의 예측을 결합하여 최적의 재발방지대책 제시
  - 문장 수준 앙상블로 각 모델의 강점 활용
  - 상호 보완적 정보 통합을 통한 품질 향상
- **최적화**: JaccardSim, 핵심 용어 추출 및 추가를 통한 유사도 향상
  - 도메인 특화 용어 정규화
  - 출력 길이 최적화를 통한 일관성 확보

## 🔄 모델 파이프라인

<div align="center">
  <img src="https://github.com/user-attachments/assets/828ddfca-e911-4580-bfe4-2996ff930b5f" alt="Detailed Model Pipeline" width="750"/>
</div>

1. **데이터 전처리**:
   - 건설 안전 사고 데이터 구조화 (공사종류, 공종, 사고객체 등 카테고리 분류)
   - 훈련 데이터셋 생성 (RAG 스타일, QA 스타일)
   - 데이터 정제 및 증강

2. **벡터 저장소 구축**:
   - ko-sbert 임베딩을 활용한 문서 인덱싱
   - HNSW(Hierarchical Navigable Small World) 인덱스 생성으로 검색 속도 최적화
   - MMR(Maximum Marginal Relevance) 검색으로 다양하고 관련성 높은 문서 검색

3. **LoRA 파인튜닝**:
   - RAG 스타일 (컨텍스트 포함) 모델 학습
   - QA 스타일 (컨텍스트 없음) 모델 학습
   - 최적화된 하이퍼파라미터 (r=16, lora_alpha=32)

4. **앙상블 추론**:
   - 두 모델의 결과를 문장 수준에서 결합
   - 겹치지 않는 정보 추출 및 통합
   - 핵심 정보 우선 배치를 통한 품질 향상

5. **후처리 최적화**:
   - 자카드 유사도 향상을 위한 핵심 용어 추가
   - 일관된 용어 사용을 위한 정규화
   - 최적 길이 조정으로 평가 메트릭 향상



## 🏆 대회 성과


본 프로젝트는 건설 안전 AI 경진대회에서 **최종 순위 6위**의 성적을 거두었습니다. 이는 다음과 같은 차별화된 접근 방식을 통해 달성했습니다:

- **앙상블 기법**: 단일 모델보다 우수한 성능을 발휘한 다중 모델 접근법
- **RAG + LoRA의 결합**: 검색 증강 생성과 파라미터 효율적 파인튜닝의 시너지
- **최적화된 후처리**: 도메인 지식을 활용한 출력 개선
- **효율적인 리소스 활용**: 제한된 컴퓨팅 리소스에서 최대 성능 달성


모델의 정확도, 관련성, 실용성 측면에서 뛰어난 평가를 받았으며, 건설 현장 안전을 위한 효과적인 재발방지대책을 제시하는 데 성공했습니다.

## 📈 결과 분석


### 모델별 성능 비교

| 모델 | 스코어 |
|------|-------|
| Qwen2.5-72B (RAG) | 0.4662 | 
| Gemma3-27B (LoRA) | 0.4885 | 
| 앙상블 모델 | *0.5069** |

### 결과 샘플

<div style="background-color: #f6f8fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
<strong>질문:</strong> 철골구조물공사 중 '작업자 부주의'로 인해 사고가 발생했습니다. 공사종류 대분류 '건축', 중분류 '골조공사' 공사 중 공종 대분류 '철골공사', 중분류 '설치작업' 작업에서 사고객체 '강재'(중분류: '앵글')와 관련된 사고가 발생했습니다. 본 사고로 인해 인적 피해는 '골절상', 물적 피해는 '자재파손'이 발생했습니다. 이 사고의 재발 방지 대책 및 향후 조치 계획은 무엇인가요?
<br><br>
<strong>앙상블 모델 답변:</strong> 철골 설치작업 전 작업자 안전교육을 철저히 실시하고, 작업 중 안전모, 안전화, 안전대 등 적절한 개인보호구 착용을 의무화해야 합니다. 작업 전 강재 취급 주의사항과 작업절차에 대한 작업지시를 명확히 하고, 중량물 취급 시 신호수를 배치하여 안전한 작업을 유도해야 합니다. 정기적인 안전점검과 작업자 피로도 관리를 통해 부주의를 방지하고, 작업구역 접근통제와 안전관리자 상주 감독을 시행해야 합니다.
</div>



## 📚 참고 자료

- [건설 안전 AI 경진대회 공식 페이지](https://dacon.io/competitions/official/236455/leaderboard)
- [Unsloth 라이브러리](https://github.com/unsloth/unsloth)


## 👥 팀 구성

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/username/repo/main/assets/team_member1.png" width="100"/><br/>홍길동<br/>모델링 & 앙상블</td>
      <td align="center"><img src="https://raw.githubusercontent.com/username/repo/main/assets/team_member2.png" width="100"/><br/>김철수<br/>데이터 & RAG</td>
      <td align="center"><img src="https://raw.githubusercontent.com/username/repo/main/assets/team_member3.png" width="100"/><br/>이영희<br/>서빙 & UI</td>
    </tr>
  </table>
</div>

## 📄 라이센스

본 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
