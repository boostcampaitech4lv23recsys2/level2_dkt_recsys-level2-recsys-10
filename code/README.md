# Deep Knowledge Tracing Baseline Code

Boostcamp A.I. Tech 4기 DKT 트랙 베이스라인 코드입니다.
현재 DKT 대회는 두 종류의 베이스라인이 제공됩니다.
+ `dkt/` 이 폴더 내에는 **Sequential Model**로 풀어나가는 베이스라인이 담겨져있습니다.
+ `lightgcn/` 이 폴더 내에는 Graph 기법으로 풀어나가는 베이스라인이 담겨져있습니다.

## Features Engineering

train : 1 = train, 0 = test
same_item_cnt : 한 유저가 같은 문제를 푼 횟수
elapsed : 한 유저가 각각의 문제를 푸는데 걸린 시간
Bigcat : assessmentItemID의 index 2 -> 시험지의 카테고리 (!= KnowledgeTag)
solved_time_shift : 유저, assessmentItemID, same_item_cnt 구분했을 때 문제 푸는데 걸린 시간
Item_mean_solved_time : 문제를 맞춘 유저의 평균 풀이시간


> **유저 별**
- `user_avg` : 유저별 평균 정답률
- `user_time_avg` : 유저별 평균 풀이시간
- `user_std` : 유저별 표준편차
- `user_current_avg` : 유저별 해당 문제지에 대한 평균 정답률
- `user_current_time_avg` : 유저별 해당 문제지에 대한 평균 풀이시간
- `user_correct_answer` : 유저별 누적 정답 횟수
- `user_cumacc` : 유저별 누적정답률
- `user_Bigcat_correct_answer` : 유저의 카테고리별 누적 정답횟수
- `user_Bigcat_cumacc` : 유저의 카테고리별 누적정답률
- `user_retCount_correct_answer` : 유저별 최근 5개 정답횟수
- `user_retCumacc` : 유저별 최근 5개 정답률


> **문제 별**
- `item_avg` : 문제별 평균 정답률
- `item_time_avg` : 문제별 평균 풀이시간
- `item_std` : 문제별 표준편차 
- `item_seq` : 문제 푼 순서
- `item_correct_answer` : 문제별 누적 정답횟수
- `item_retCount_correct_answer` : 문제별 최근 5개 정답횟수
- `item_retCumacc` : 문제별 최근 5개 정답률


- > **카테고리(대분류) 별**
- `Bigcat_avg` : 카테고리별 평균 정답률
- `Bigcat_time_avg` : 카테고리별 평균 풀이시간
- `Bigcat_std` : 카테고리별 표준편차


> **태그 별**
- `tag_avg` : 태그별 평균 정답률
- `tag_time_avg` : 태그별 평균 풀이시간
- `tag_std` : 태그별 표준편차




