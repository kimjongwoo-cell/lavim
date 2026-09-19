# LaViM 현안 분석 — 가장 큰 문제와 해결 과제 (2026-09-09 밤)

정본 위치: 로컬 `/home/super/hj/0902_2155/sender_relay_exp/ISSUES_0909.md` = 원격 `wsi_latent_0915_decode_hj/sender_relay_exp/ISSUES_0909.md`
근거: PROGRESS_0907.md + 실험 로그(§1~§28.10) + 대시보드(0909 stamp) + 0909 원격 실측 종합.

---

## 진단 요약 — 3문장

1. **성능은 있는데 "이야기"가 비어가고 있다**: 합격 구성(LaViM k75, 2/5 셋)의 방법론적 주장 성분이 통제 실험에서 연쇄 기각됐다(계층 재조립·graph·AAVM·event 쿼리·2-hop 전부 폐기). 살아남은 주장은 "층화 압축 + 원순서 recency 재배치 + PLIP 쿼리 검색" 세 조각뿐이다.
2. **프로그램의 기저 서사가 지금 뒤집히는 중이다**: §28.10 probe 결함(grown-cache) 발견으로 "mid-cache visual 불감"(§10, 실패법칙①)이 소급 의심되고, 수정판 스모크는 정반대(V→A 직접 읽기 2/2 → gtex 7/7) 신호를 냈다 [GPU실측]. 이 판정이 restage 기제 해석("recency-게이트")까지 물고 들어간다.
3. **EVQA 이득의 귀속이 오염 위험**: k75 EVQA +3.1은 메인테이블 합격 성분인데, EVQA에서는 내용-무관 섭동도 점수를 올린 전례가 둘(restage wrong +8 [0905 확정], AAVM v2 +10.9)이라 내용-특이성 통제 없이는 방법 성과로 못 세운다.

---

## P0 — 논문의 "왜"를 결정하는 것 (지금 진행 중인 것을 완결)

### P0-1. Pathways V→A 경로 확정 + 과거 판정 소급 정정
- **문제**: `generate_terminal_json(cache=cache)`가 캐시를 in-place로 키운 상태에서 probe가 돌아 §10(mid-cache 무반응 2/29)·1차 pass T(flip 0%) 전부 무효 의심 [코드읽기, §28.10]. 수정판 스모크에서 **vis 단독 이식만으로 wrong 답 재현 2/2, R1:10 단독 0/2** — "시각→답 경로 = mid-cache visual KV를 Answerer가 직접 읽는 V→A 단일 경로, latent 경유 아님" [GPU실측, gtex 7/7 진행].
- **할 일**:
  1. 5셋 T3 재실행 완주(진행 중, gtex close 모드 포함) + `none` 무이식 대조군·`vis_drop` knockout·지오메트리 가드 반영분 확인.
  2. 확정 시 **실패법칙① "mid-cache 불감" 공식 철회**, §10 판정 인용 전면 정정(로그·메모리·PROGRESS).
  3. 같은 결함 위에 있던 것 재검: §20 arbitration 채점, 0901 "Answerer 시각열 인과 무력" 메모리 [미검증-가설, 재검 대상].
  4. **restage 기제 서사 재작성**: "recency만 읽는다" → V→A 직접 읽기와 양립하는 해석(재배치가 왜 +3.5인가 — 위치 근접성? RoPE 거리? MRoPE 판정 +0.36은 소폭)으로 통일. §15·§18 "통일 법칙" 문구 수정 필요.
- **판정 기준**: 5셋에서 vis 단독 재현율이 close/latent 모드를 지배하면 V→A 확정.

### P0-2. EVQA 이득의 내용-특이성 통제 (메인테이블 방어)
- **문제**: 메인테이블 EVQA 48.44 ✓는 k75 성분인데, EVQA는 wrong-slide restage도 +8이었고 [0905 확정, 그때 "method 주장 제외" 판정] AAVM v2(내용-무관 섭동 확정)도 48.44와 **동수**를 냈다. 지금 상태로는 "EVQA 열은 내용-무관 효과"라는 반론을 못 막는다.
- **할 일**: `k75 + wrong-slide` EVQA arm 1개 (통제 1런). matched−wrong 격차가 서면 방법행 유지, 안 서면 **메인테이블 EVQA를 '효과는 있으나 내용-무관' 각주로 강등**하고 GTEx 중심으로 범위 재확정.
- 부속: AAVM v2 EVQA +10.9 현상 자체는 shuf_evqa 1런으로 귀속 정리 가능(방법 부활용 아님, 현상 기록용 — 선택).

### P0-3. 살아남은 방법 성분으로 승자 스택 재조립
- **현황**: 검증 양성 성분이 서로 다른 축에 흩어져 있다 — ① PLIP 쿼리 검색 acquisition(gtex +2.3 base 초과, 쿼리 기여 tissue 대비 +3.3/+1.6 [GPU실측]) ② 층화 압축 k75(strat>flat, EVQA B-문턱) ③ 원순서 recency 재배치(+park). **이 셋을 한 파이프라인에 합친 구성은 아직 한 번도 안 돌았다.**
- **할 일**: `PLIP검색 + strat k75 + original restage+park` 합성 arm — gtex·evqa 먼저, 합격선 기준으로 판정. 이것이 신규 메인 구성 후보(현 LaViM k75는 grid Navigator 기반).
- 주의: 검색 경로는 baseline 자체가 다르다(navsearchev gtex 20.49 vs grid base 21.41) — **합격선을 "grid base 대비"로 유지할지 "동일 acquisition 대비"로 이원화할지 사전 등록** 필요. 안 하면 §28.1 오독(24.21) 같은 기준 혼선 재발.

---

## P1 — 범위·완결성

### P1-1. 논문 주장 범위 확정 (기각 목록 정리)
- 남길 주장 후보: (a) 관찰-층화 압축 = 무손실 압축 + "창 구성" 순기여(c2orig k50 22.97 > 비압축 park 22.51) (b) recency 재배치 + 기제(P0-1 결과에 따름) (c) 쿼리-조건 frozen 검색 Navigator(0717 PLIP 실패 판정 뒤집은 서술구 쿼리).
- 기각 확정 → negative/ablation 절로 이동: 계층 재조립(3-way), graph/projective, AAVM(진단 3종 완비 — 오히려 좋은 부검 재료), event 쿼리, 2-hop, bind/read/VG 3형, 심판 3형, 재분배 계열.
- **경계 명시**: SB=개입 무익(base 최강), TCGA/PANDA=backbone-bound(45셀 실증), 8B=역효과(판독<prior), 범위=4B 중심 — 이 경계 자체를 결과로 서술.

### P1-2. PLIP 검색 잔여 판정 (진행 중 큐)
- navsearchplip SB/TCGA/PANDA 완주 대기 + plipwide(PER_ROOT=4) gtex/evqa. EVQA 절대값 미달(39.06 vs base 45.31)이라 "검색이 grid를 대체 가능"까지는 아직 못 감 — **acquisition 격차(navsearchev 20.49 vs 21.41)의 원인 분해**(후보 풀 크기? tissue 필터? 픽 다양성?)가 필요.

### P1-3. Navigator 픽 prior 문제 — 방향 전환 확정
- 2-hop까지 실패로 "번호 그리드에서 VLM이 고르게 하는 방식" 계열은 폐기 근거 완비(픽 상위 셀 59.5% 좌상단 편중 [GPU실측]). **선택은 frozen tool(argmax/PLIP)로, VLM은 쿼리 생성만** — 이 역할 분리를 방법 서술의 축으로 승격.

---

## P2 — 기회·효율·위생

### P2-1. Reasoner 프롬프트 다이어트 [기회, 미검증]
- 캐시 실측: Reasoner 프롬프트 1403토큰 중 56%가 빈 JSON 템플릿, 27%가 좌표 목록 — 마지막 크롭과 latent 사이 "벽"의 83%가 정보량 0 [코드읽기+실측, §28.5]. ~330토큰(−76%)으로 축약 가능. V→A 확정 시 이 벽이 visual→Answerer 거리를 늘리는 유해 요인일 가설도 검정 가능(일석이조). JSON 파싱 성공률 확인 선행.

### P2-2. efficiency 표 완결
- VLMAS_EFF_DUMP 가동 중. 시각 KV 25%↓(k75)·park 캐시 33%↓·latent 15초 vs VLMAS 90~111초 재료는 있음 — 메인 구성 확정(P0-3) 후 그 구성으로 FLOPs/TTFT/peak-mem 열 채우기. AIVA는 B_card −0.28 실측으로 지렛대 없음 확정 — 발화 버그 수정은 후순위(폐기 후보).

### P2-3. 운영 위생 (반복 사고 4종)
1. **큐 append 개행 접합** 2회 발생 → jobs.txt에 줄 추가하는 헬퍼 스크립트(마지막 개행 보장 + 중복 검사) 하나 두고 전 세션이 그것만 사용.
2. **동시편집**: 0909 00:15 같은 파일(aavm_runtime/qwen3vl)을 두 세션이 10초 간격 편집 — 유실 위험 실재. 파일 단위 소유 선언(PROGRESS에 "지금 만지는 파일" 줄) 또는 편집 전 mtime 확인 습관화.
3. **기본값 함정**: CONSOL_KEEP=0.25 오염으로 §25 order 비교 전체 폐기 전례 — env 기본값 의존 금지, 잡 라인에 전 노브 명시(현행 유지).
4. **채점 오독**(24.21) 전례 → 보고 숫자는 반드시 채점 스크립트 출력 복사, 대시보드 수기 전사 금지. 완료→채점→보고 워처(reported_runs.txt)가 현 워커 세대에 안 붙어 있음 — 재가동.

---

## 한 줄 우선순위

**T3 5셋 완주(V→A 확정)** > **k75-wrong EVQA 통제** > **PLIP+k75+restage 합성 arm** > 범위 확정·기각 정리 > PLIP 잔여 셋 > 프롬프트 다이어트 > efficiency 표 > 운영 위생.

*근거 태그: 별도 표기 없는 수치는 실험 로그·대시보드의 [GPU실측] 값. 0909 밤 시점, T3 재실행·navsearchplip 잔여 셋은 큐 진행 중.*
