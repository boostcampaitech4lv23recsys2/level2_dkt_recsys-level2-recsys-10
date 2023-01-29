# 📚 지식 상태 추론(Deep Knowledge Tracing)

## 1. 프로젝트 개요

### 1-1. 프로젝트 주제
지식 구성 요소와 지식 상태를 이용하여, 변화하는 지식 상태를 지속적으로 추적하는 task 이다. 사용자가 푼 일련의 문제를 통해 다음 문항에 낸 답이 정답일지 오답일지 맞추는 것을 목표로 한다. 

### 1-2. 프로젝트 기간
2022.11.14 ~ 2022.12.08(4주)

### 1-3. 활용 장비 및 재료
- 개발환경 : VScode, PyTorch, Jupyter, Ubuntu 18.04.5 LTS, GPU Tesla V100-PCIE-32GB
- 협업 Tool : GitHub, Notion
- 시각화 : WandB

### 1-4. 프로젝트 구조도
```
|-- boosting
|   |-- XGBoptuna.ipynb
|   |-- boosting_baseline.py
|   |-- src
|   |-- train.py
|-- dkt
|   |-- README.md
|   |-- args.py
|   |-- inference.py
|   |-- requirements.txt
|   |-- src
|   |-- sweep.yaml
|   |-- train.py
|   |-- tuning.py
|   |-- wandb_train.py
|-- ensembles
|   |-- ensembles.py
|-- lgbm
|   |-- lgbm.ipynb
|   |-- lgbm_baseline.py
|   |-- lgbm_group_kfold.ipynb
|-- lightgcn
|   |-- README.md
|   |-- config.py
|   |-- inference.py
|   |-- install.sh
|   |-- lightgcn
|   |-- train.py
|-- lightgcn_custom
|   |-- README.md
|   |-- config.py
|   |-- inference.py
|   |-- install.sh
|   |-- lightgcn
|   |-- requirements_lightgcn_custom.txt
|   |-- train.py
```
- (1) boosting folder
	- LGBM, XGBoost, CatBoost baseline code
- (2) dkt folder
	- LSTM 계열 모델의 baseline code
- (3) ensembles
	- Weighted, voting, mix 방식의 ensemble code
- (4) lgbm
	- LGBM baseline code
- (5) lightgcn
	- lightgcn baseline code
- (6) lightgcn_custom
	- lightgcn + BERT , lightgcn + feature representation code

### 1-5. 데이터 구조
- `userID` 사용자의 고유번호
- `testId` 시험지의 고유번호
- `assessmentItemID` 문항의 고유번호
- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터
- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점
- `KnowledgeTag` 문항 당 하나씩 배정되는 태그

### 1-6. Metric
- **AUROC**(Area Under the ROC curve)와 **Accuracy**

<br>

## 2. 프로젝트 팀 구성 및 역할
|[구혜인](https://github.com/hyein99?tab=repositories)|[권은채](https://github.com/dmscornjs)|[박건영](https://github.com/kuuneeee)|[장현우](https://github.com/jhu8802)|[정현호](https://github.com/Heiness)|[허유진](https://github.com/hobbang2)|
|----|----|----|----|----|----|
|* 데이터 EDA<br>* BERT 모델 진행|* 데이터 EDA<br>* XGB 모델 진행|* 데이터 EDA<br>* Last Query 모델 진행|* 데이터 EDA<br>* LSTM+Attention 모델 진행|* 데이터 EDA<br>* LightGBM 모델 진행|* 데이터 EDA<br>* LightGCN 모델 진행|

<br>

## 3. 프로젝트 진행

### 3-1. 사전 기획
- 22.11.10(목): DKT 프로젝트 전 오프라인 미팅
- 22.11.14(월): 모델 세미나
- 일정 수립
    - 22.11.14(월) ~ 22.11.20(일) : EDA
    - 22.11.14(월) ~ 22.12.02(금) : Feature Engineering
    - 22.11.23(수) ~ 22.12.02(금) : Modeling
    - 22.12.03(토) ~ 22.12.09(금) : 최적화
    
### 3-2. 프로젝트 수행
![DKT drawio](https://user-images.githubusercontent.com/49949138/215059582-768852d3-16d1-4dec-9e28-e8ae537e9f39.png)

<br>

## 4. 프로젝트 수행 결과

### 4-1. 모델 성능 및 결과
**■ 결과 ( AUROC Score 상위 4 개) : Private 7위**
| LSTMAttention | BERT | LastQuery | XGBoost | LightGBM | LightGCN |
| --- | --- | --- | --- | --- | --- |
| 0.7594 | 0.7791 | 0.8063 | 0.8114 | 0.8210 | 0.7823 |

| 최종 선택 여부 | 모델 (Ensemble 비율) | public auroc | private auroc |
| --- | --- | --- | --- |
| O | LightGBM LightGCN LastQuery (0.65, 0.1, 0.25) | 0.8253 | 0.8479|
| O | LightGBM LightGCN LastQuery (0.7, 0.1, 0.2) | 0.8252 | 0.8476|
| X | LightGBM LastQuery XGBoost LightGCNx3 (hard voting) <br> - LightGCN , LightGCN + feature representation , LightGCN + Bert | 0.8094 | 0.8531|
| X | LightGBM LightGCN LastQuery  (0.65, 0.15, 0.2) | 0.8232 |	0.8506|

### 4-2. 모델 개요
- 1. Transformer 계열 모델
	- 1) LSTM + Attention
	- 2) BERT
	- 3) LastQuery
- 2. Boosting 계열 모델
	- 1) LightGBM
	- 2) XGBoost
- 3. Graph 모델
	- 1) LightGBM

### 4-3. 모델 선정
- 베이스라인 코드
    - LightGBM
        - 기본적으로 주어진 컬럼이 굉장히 적고 만들어내야 하는 상황이다. 따라서assessmentItemID, testId, KnowledgeTag 등 대부분이 범주형으로 주어졌지만 Feature로 통계값을 많이 사용할거라 예상하여 CatBoost 사용을 미루기로 했다. 또한 주어진 데이터 양이 적지 않으므로 XGBoost보다 LGBM이 효율적이라 생각했다.
- 추가적인 모델 선택
    - LastQuery
        - Riid 대회에서 1등을 기록한 모델로, sequence 길이에 따라 향상된 성능을 보였으며 다른 transformer 계열 모델에 비해 feature engineering이 적게 필요하여 모델링 소요시간과 성능 측면에서 장점을 보였기 때문에 선택했다.
    - XGBoost
        - LightGBM 모델이 성능이 잘 나와 비슷한 CART(Classification and regression tree) 앙상블 모델이면서 다양한 하이퍼 파라미터를 조절해 볼 수 있어 LightGBM과 비교를 위해 추가적으로 사용하게 되었다.

### 4-4. 모델 성능 개선 방법
- Hyperparameter tuning(Wandb, Sweep, Optuna)
- K-fold
- Ensemble

<br>

## 5. WrapUp Report
[Level_2_DKT_랩업리포트](https://www.notion.so/Level_2_DKT_-d6f19e429a6744369c121ac9d17e7f4b)

