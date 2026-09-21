# K-RegGen 유효성 검사 보고서

검사일 **2026-09-19** · 국가법령정보 OPEN API 대조

> **이 검사는 법률 전문가 검수를 대체하지 않는다.** 확인한 것은 (1) 인용 조문이 실재하고 원문과 일치하는가, (2) 인용 판례·해석례가 실재하고 사건번호·일자가 맞는가, (3) gold의 주장이 조문 문언에서 도출되는가 세 가지다. **그 조문이 해당 사안에 적용되는지에 대한 법률 판단은 검증 범위 밖이다.**

## 근거 유형 분포

| 근거 유형 | 유형 수 |
|---|--:|
| `precedent` | 1 |
| `precedent+inference` | 1 |
| `text+precedent+interpretation` | 1 |
| `text+precedent` | 1 |
| `text+interpretation` | 2 |
| `text` | 44 |
| `text+inference` | 1 |

## 주의가 필요한 항목 — 조문 문언을 넘어선 주장

| 유형 | 문언을 넘어선 주장 |
|---|---|
| `MA-BEFORE-AFTER` | '전후 비교사진'은 법과 시행령 어디에도 금지 유형으로 열거되어 있지 않다 |
| `UP-MEDICAL-INVASIVE` | 약물 주입이 2026 전합 법리에 포섭되지 않는다는 점은 판결이 제시한 기준('염료 주입에 의한 색소 유지')의 반대해석이다 |

**2건.** 이 주장들은 조문 문언만으로는 확정되지 않는다. 전문가 검수 우선순위이며, 그때까지 해당 문항의 채점 결과는 보수적으로 해석해야 한다.

## 전체 유형별 판정

| 도메인 | 구분 | 유형 | 근거 | 확인된 권위 |
|---|---|---|---|---|
| 의료광고 | 위반 | `MA-TESTIMONIAL` | `text` | — |
| 의료광고 | 위반 | `MA-BEFORE-AFTER` | `text+inference` | — |
| 의료광고 | 위반 | `MA-SUPERLATIVE` | `text` | — |
| 의료광고 | 위반 | `MA-COMPARE` | `text` | — |
| 의료광고 | 위반 | `MA-NONCOVERED-DISCOUNT` | `text+precedent` | 2016헌바311(detc/144190) |
| 의료광고 | 위반 | `MA-INDUCEMENT-GIFT` | `text+interpretation` | 해석례 326841 |
| 의료광고 | 위반 | `MA-UNEVALUATED-TECH` | `text` | — |
| 의료광고 | 위반 | `MA-ARTICLE-FORM` | `text` | — |
| 의료광고 | 위반 | `MA-NO-REVIEW` | `text` | — |
| 의료광고 | 위반 | `MA-CERT-BADGE` | `text` | — |
| 의료광고 | 위반 | `MA-FALSE-CONTENT` | `text` | — |
| 의료광고 | 위반 | `MA-SURGERY-FOOTAGE` | `text` | — |
| 의료광고 | 위반 | `MA-FOREIGN-PATIENT` | `text` | — |
| 의료광고 | 위반 | `MA-SHORT-CAREER` | `text` | — |
| 의료광고 | 대조군 | `C-MA-HOURS` | `text` | — |
| 의료광고 | 대조군 | `C-MA-FEE-DISCLOSE` | `text` | — |
| 의료광고 | 대조군 | `C-MA-CAREER` | `text` | — |
| 의료광고 | 대조군 | `C-MA-INTERNAL` | `text` | — |
| 의료광고 | 대조군 | `C-MA-PROCEDURE-ILLUST` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-RRN-PARSE` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-RRN-PLAINTEXT` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-THIRD-PARTY` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-PURPOSE-BEYOND` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-MARKETING-CONSENT-BUNDLE` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-CONSENT-REFUSAL-BLOCK` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-SENSITIVE-HEALTH` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-CROSSBORDER` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-NO-DESTRUCTION` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-PROCESSOR-NODOC` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-CCTV` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-COLLECT-NOBASIS` | `text` | — |
| 개인정보 처리 코드 | 위반 | `PC-STOP-IGNORE` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-PSEUDONYM` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-ENCRYPT` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-WITHDRAW` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-CONSENT-NOTICE` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-DOMESTIC-REGION` | `text` | — |
| 개인정보 처리 코드 | 대조군 | `C-PC-DESTRUCTION` | `text` | — |
| 온라인 판매 금지 품목 | 위반 | `RS-OPTICAL-ONLINE` | `text+precedent+interpretation` | 2020헌가10(detc/189573), 해석례 315379 |
| 온라인 판매 금지 품목 | 위반 | `RS-DRUG-ONLINE` | `text+interpretation` | 해석례 315129 |
| 온라인 판매 금지 품목 | 대조군 | `C-RS-OFFLINE-PROMO` | `text` | — |
| 온라인 판매 금지 품목 | 대조군 | `C-RS-PHARMACY-INFO` | `text` | — |
| 온라인 판매 금지 품목 | 대조군 | `C-RS-COSMETIC` | `text` | — |
| 무자격 영업 | 위반 | `UP-LEGAL` | `text` | — |
| 무자격 영업 | 위반 | `UP-TAX` | `text` | — |
| 무자격 영업 | 위반 | `UP-REALESTATE` | `text` | — |
| 무자격 영업 | 위반 | `UP-MEDICAL-INVASIVE` | `precedent+inference` | 2021도15611(prec/622115) |
| 무자격 영업 | 대조군 | `C-UP-TATTOO` | `precedent` | 2021도15611(prec/622115), 2022도13370(prec/622263) |
| 무자격 영업 | 대조군 | `C-UP-SELF` | `text` | — |
| 무자격 영업 | 대조군 | `C-UP-INFO` | `text` | — |
| 무자격 영업 | 대조군 | `C-UP-BOOKKEEPING` | `text` | — |

## 검사 중 발견해 수정한 오류

| 대상 | 오류 | 수정 |
|---|---|---|
| 법령ID 5건 | 약사법=001760(실제 산업재해보상보험법), 변호사법=001729(실제 교육세법), 나머지 3건 미존재 | API 조회로 교체 |
| 의료기사법 §12⑤ | 각 호 열거 방식 개정을 놓침 | 제5항 제1호(전자상거래)·제2호(구매·배송대행)로 분리 |
| 개인정보보호법 §24조의2 | ①처리제한과 ②암호화를 뭉뚱그림 | 항 단위로 분리 |
| 헌재 2020헌가10 | 종국일자를 2024-04-25로 기재(뉴스 기사) | 실제 **2024-03-28** |
| 전합 판례 대응 | 2021도15611이 두 판례를 모두 변경한 것으로 기재 | **2021도15611→91도3219**, **2022도13370→2004도673** |
| 의료법 §56②13호 | 비급여 할인 광고 근거 조문 누락 | 제13호로 확정(gold 3차 수정) |
| RS-OPTICAL-ONLINE | 전면 금지로 서술 | 해석례 315379로 **외국 소비자 대상은 비금지** 반영 |
| MA-INDUCEMENT-GIFT | 교통편의 제공을 전면 금지로 서술 | 해석례 326841로 **환자 전용 직영 순환버스는 비해당** 반영 |

## 남은 공백

- **보건복지부 고시 미확보.** 의료법 시행령 제23조 제3항이 인터넷 홈페이지 의료광고의 세부기준을 고시에 위임하나, 행정규칙 API 검색으로 해당 고시를 찾지 못했다. 전후사진 기준이 여기 있을 수 있다.
- **판례 DB 수록 범위가 좁다.** 국가법령정보 판례 API는 주로 판례공보 수록분이라, 개인정보·의료광고 하급심 사례는 거의 조회되지 않았다.
- **개인정보위 처분 의결서 미수집.** 학습셋 출처로 계획했으나 이번 검사에는 포함하지 않았다.
- **`gloss`와 `gold_points`는 `lawsync --check`의 대조 대상이 아니다.** 틀린 해석이 남아 있어도 CI는 통과한다.
