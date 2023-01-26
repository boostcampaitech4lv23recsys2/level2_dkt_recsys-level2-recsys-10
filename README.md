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
### 1-5. 프로젝트 구조도
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
### 1-6. 데이터 구조
!!!

## 2. 프로젝트 팀 구성 및 역할
|[구혜인](https://github.com/hyein99?tab=repositories)|[권은채](https://github.com/dmscornjs)|[박건영](https://github.com/kuuneeee)|[장현우](https://github.com/jhu8802)|[정현호](https://github.com/Heiness)|[허유진](https://github.com/hobbang2)|
|----|----|----|----|----|----|
|* 데이터 EDA<br>* BERT 모델 진행|* 데이터 EDA<br>* XGB 모델 진행|* 데이터 EDA<br>* Last Query 모델 진행|* 데이터 EDA<br>* LSTM+Attention 모델 진행|* 데이터 EDA<br>* LightGBM 모델 진행|* 데이터 EDA<br>* LightGCN 모델 진행|

## 3. 프로젝트 진행
### 3-1. 사전 기획
### 3-2. 프로젝트 수행

## 4. 프로젝트 수행 결과
### 4-1. 모델 성능 및 결과
**■ 결과 (상위 4 개) : Private 7위 **
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
### 4-3. 모델 선정
### 4-4. 모델 성능 개선 방법

## 5. WrapUp Report
[Level_2_DKT_랩업리포트](https://www.notion.so/Level_2_DKT_-d6f19e429a6744369c121ac9d17e7f4b)






<br><br>
***Deep Knowledge Tracing(DKT) description*** :

> **GOAL** : 사용자가 푼 일련의 문제를 통해 다음 문항에 낸 답이 정답일지 오답일지 맞추는 것

- 주요 데이터는 `.csv` 형태로 제공되며, train/test 합쳐서 총 7,442명의 사용자가 존재합니다. 이 때 이 사용자가 푼 마지막 문항의 정답을 맞출 것인지 예측하는 것이 최종 목표입니다.

- `userID` 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 `userID`를 기준으로 9 : 1의 비율로 나누어졌습니다.

- `testId` 시험지의 고유번호입니다. 문항과 시험지의 관계는 아래 그림을 참고하여 이해하시면 됩니다. 총 1,537개의 고유한 시험지가 있습니다.

- `assessmentItemID` 문항의 고유번호입니다. 총 9,454개의 고유 문항이 있습니다. "A+앞 6자리"는 `testId`의 정보를 나타내고 있으며, 뒤 3자리는 문제의 번호를 의미합니다.

	![img](https://user-images.githubusercontent.com/38639633/123995680-8d975c80-da09-11eb-887b-5946aa82df37.png)

- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터이며 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.

- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.

- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 태그 자체의 정보는 비식별화 되어있지만, 문항을 군집화하는데 사용할 수 있습니다. 912개의 고유 태그가 존재합니다.

![image](https://user-images.githubusercontent.com/46401358/202663480-6296894d-c08a-4980-bb58-c8f00c4ee885.png)
		

- ***Metric*** : 

	- DKT는 주어진 마지막 문제를 맞았는지 틀렸는지로 분류하는 이진 분류 문제입니다. 

	- 평가를 위해 **AUROC**(Area Under the ROC curve)와 **Accuracy**를 사용합니다. 

	- 리더보드에 두 지표가 모두 표시되지만, **최종 평가는 AUROC 로만** 이루어집니다.

## 📁프로젝트 구조

```
code  
├── README.md  
├── args.py  
├── baseline.ipynb  
├── dkt  
│   ├── criterion.py  
│   ├── dataloader.py  
│   ├── metric.py  
│   ├── model.py  
│   ├── scheduler.py  
│   ├── trainer.py  
│   └── utils.py  
├── evaluation.py  
├── inference.py  
├── requirements.txt  
└── train.py
```

## :man_technologist: Members
구혜인 권은채 박건영 장현우 정현호 허유진


