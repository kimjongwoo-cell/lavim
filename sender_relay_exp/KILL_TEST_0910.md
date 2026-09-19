# Kill test (gtex40, 4B·s10) — 최종표 [GPU실측, 0910 21:30]

12런 전부 40/40 완주. 지표 정의: 정답=0806 채점기(expand_letter→acc_of_seq), BACC=gold 클래스별 macro; M_V=Answerer 첫 답 토큰 위치의 visual attention mass(median); margin=VER score-only teacher-forced logP(정답)−max logP(오답)(median, 40케이스); flip T/W=같은 케이스 true vs wrong-slide 답 상이 수; offset variance=base 런 해석 계산(δ∈{0,256,1024,4096}) median.

| arm | 정답/40 (true) | BACC | 정답/40 (wrong) | flip T/W | flip vs base | M_V % | margin | offset var |
|---|---|---|---|---|---|---|---|---|
| Base | 13 | .206 | 8 | 20/40 | — | 0.45 | −3.94 | 0.147 |
| Restage | 14 | .222 | 10 | 25/40 | 11/40 | 3.47 | −4.16 | 1.801 |
| Full-deRoPE (통제, u=0) | 8 | .134 | 8 | 31/40 | 22/40 | 28.71 | −8.91 | 0 |
| Canonical (u=(0,h,w)) | 11 | .225 | 7 | 23/40 | 14/40 | 11.23 | −1.91 | 0 |
| Canonical-crop t=p (u=(p,h,w)) | 12 | .237 | 8 | 23/40 | 14/40 | 13.52 | −1.62 | 0 |
| Canonical-crop t=π(p) (순열 통제) | 11 | .225 | 8 | 26/40 | 12/40 | 13.68 | −1.58 | — |

수치만(판정은 보고):
- 사용자 규칙 대입: canonical 11 ≈ crop-canonical 12 ≈ π(p) 통제 11, restage 14 — "canonical≈crop-canonical, restage가 위"(차이 2~3문항, n=40) → B 쪽. crop index t=p와 순열 t=π(p)가 동수 → crop aliasing 신호 없음.
- canonical 계열은 M_V 25~30배·margin −3.9→−1.6~−1.9로 "접근"은 열리지만 정답 수는 base 이하~동률; deRoPE는 접근 최고(28.7%)에 정답 최저(8) → (h,w) 위상은 필요.
- restage만 offset variance가 커지는데(1.80) 정답도 유일하게 +1 — 위치 편차 소거 자체는 정답과 무관.
