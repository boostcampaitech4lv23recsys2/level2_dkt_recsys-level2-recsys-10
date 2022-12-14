# π μ§μ μν μΆλ‘ (Deep Knowledge Tracing)

`DKT`  
- μ§μ κ΅¬μ± μμ ( νμμκ²μ μκ³  μΆμ μμ ) μ μ§μ μν ( κ° μ§μμ λν νμμ μ΄ν΄λ ) λ₯Ό μ΄μ©νμ¬,  **λ³ννλ μ§μ μνλ₯Ό μ§μμ μΌλ‘ μΆμ **

***κΈ°κ°*** : 2022.11.14 ~ 2022.12.08(4μ£Ό)

***Deep Knowledge Tracing(DKT) description*** :

> **GOAL** : μ¬μ©μκ° νΌ μΌλ ¨μ λ¬Έμ λ₯Ό ν΅ν΄ λ€μ λ¬Έν­μ λΈ λ΅μ΄ μ λ΅μΌμ§ μ€λ΅μΌμ§ λ§μΆλ κ²

- μ£Όμ λ°μ΄ν°λ `.csv` ννλ‘ μ κ³΅λλ©°, train/test ν©μ³μ μ΄ 7,442λͺμ μ¬μ©μκ° μ‘΄μ¬ν©λλ€. μ΄ λ μ΄ μ¬μ©μκ° νΌ λ§μ§λ§ λ¬Έν­μ μ λ΅μ λ§μΆ κ²μΈμ§ μμΈ‘νλ κ²μ΄ μ΅μ’ λͺ©νμλλ€.

- `userID` μ¬μ©μμ κ³ μ λ²νΈμλλ€. μ΄ 7,442λͺμ κ³ μ  μ¬μ©μκ° μμΌλ©°, train/testμμ μ΄ `userID`λ₯Ό κΈ°μ€μΌλ‘ 9 : 1μ λΉμ¨λ‘ λλμ΄μ‘μ΅λλ€.

- `testId` μνμ§μ κ³ μ λ²νΈμλλ€. λ¬Έν­κ³Ό μνμ§μ κ΄κ³λ μλ κ·Έλ¦Όμ μ°Έκ³ νμ¬ μ΄ν΄νμλ©΄ λ©λλ€. μ΄ 1,537κ°μ κ³ μ ν μνμ§κ° μμ΅λλ€.

- `assessmentItemID` λ¬Έν­μ κ³ μ λ²νΈμλλ€. μ΄ 9,454κ°μ κ³ μ  λ¬Έν­μ΄ μμ΅λλ€. "A+μ 6μλ¦¬"λ `testId`μ μ λ³΄λ₯Ό λνλ΄κ³  μμΌλ©°, λ€ 3μλ¦¬λ λ¬Έμ μ λ²νΈλ₯Ό μλ―Έν©λλ€.

	![img](https://user-images.githubusercontent.com/38639633/123995680-8d975c80-da09-11eb-887b-5946aa82df37.png)

- `answerCode` μ¬μ©μκ° ν΄λΉ λ¬Έν­μ λ§μ·λμ§ μ¬λΆμ λν μ΄μ§ λ°μ΄ν°μ΄λ©° 0μ μ¬μ©μκ° ν΄λΉ λ¬Έν­μ νλ¦° κ², 1μ μ¬μ©μκ° ν΄λΉ λ¬Έν­μ λ§μΆ κ²μλλ€.

- `Timestamp` μ¬μ©μκ° ν΄λΉλ¬Έν­μ νκΈ° μμν μμ μ λ°μ΄ν°μλλ€.

- `KnowledgeTag` λ¬Έν­ λΉ νλμ© λ°°μ λλ νκ·Έλ‘, μΌμ’μ μ€λΆλ₯ μ­ν μ ν©λλ€. νκ·Έ μμ²΄μ μ λ³΄λ λΉμλ³ν λμ΄μμ§λ§, λ¬Έν­μ κ΅°μ§ννλλ° μ¬μ©ν  μ μμ΅λλ€. 912κ°μ κ³ μ  νκ·Έκ° μ‘΄μ¬ν©λλ€.

![image](https://user-images.githubusercontent.com/46401358/202663480-6296894d-c08a-4980-bb58-c8f00c4ee885.png)
		

- ***Metric*** : 

	- DKTλ μ£Όμ΄μ§ λ§μ§λ§ λ¬Έμ λ₯Ό λ§μλμ§ νλ Έλμ§λ‘ λΆλ₯νλ μ΄μ§ λΆλ₯ λ¬Έμ μλλ€. 

	- νκ°λ₯Ό μν΄ **AUROC**(Area Under the ROC curve)μ **Accuracy**λ₯Ό μ¬μ©ν©λλ€. 

	- λ¦¬λλ³΄λμ λ μ§νκ° λͺ¨λ νμλμ§λ§, **μ΅μ’ νκ°λ AUROC λ‘λ§** μ΄λ£¨μ΄μ§λλ€.

## πνλ‘μ νΈ κ΅¬μ‘°

```
code  
βββ README.md  
βββ args.py  
βββ baseline.ipynb  
βββ dkt  
β   βββ criterion.py  
β   βββ dataloader.py  
β   βββ metric.py  
β   βββ model.py  
β   βββ scheduler.py  
β   βββ trainer.py  
β   βββ utils.py  
βββ evaluation.py  
βββ inference.py  
βββ requirements.txt  
βββ train.py
```

## :man_technologist: Members
κ΅¬νμΈ κΆμμ± λ°κ±΄μ μ₯νμ° μ ννΈ νμ μ§


