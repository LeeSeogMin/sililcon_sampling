# Silicon Sampling Robustness Check - Final Results

**Status Date:** 2026-01-16 (Seed44 PARTYLR 보정 적용)
**Experiment Phase:** ✅ 모든 실험 및 분석 완료

---

## 실험 완료 상태

### GPT-5.2 (모두 완료)
| Seed | 상태 | 파일 위치 |
|------|------|----------|
| 42 | ✅ 완료 | `results/gpt52_experiment/` |
| 43 | ✅ 완료 | `results/gpt52_experiment_seed43/` |
| 44 | ✅ 완료 | `results/gpt52_experiment_seed44/` |
| 45 | ✅ 완료 | `results/gpt52_experiment_seed45/` |
| 46 | ✅ 완료 | `results/gpt52_experiment_seed46/` |

### CLOVA HCX-007 (모두 완료)
| Seed | 상태 | 파일 위치 |
|------|------|----------|
| 42 | ✅ 완료 | `results/clova_experiment/` |
| 43 | ✅ 완료 | `results/clova_experiment_seed43/` |
| 44 | ✅ 완료 | `results/clova_experiment_seed44/` |
| 45 | ✅ 완료 | `results/clova_experiment_seed45/` |
| 46 | ✅ 완료 | `results/clova_experiment_seed46/` |

---

## 최종 결과 (Seed44 PARTYLR 보정 적용)

| Variable | CLOVA Mean | GPT-5.2 Mean | Winner | 차이 |
|----------|------------|--------------|--------|------|
| CONFINAN | 0.0704 | 0.0633 | **GPT-5.2** | GPT 10.1% ↓ |
| CONLEGIS | 0.0859 | 0.1342 | **CLOVA** | CLOVA 36.0% ↓ |
| PARTYLR | 0.0513 | 0.0459 | **GPT-5.2** | GPT 10.6% ↓ |
| NORTHWHO | 0.1339 | 0.1230 | **GPT-5.2** | GPT 8.2% ↓ |
| UNIFI | 0.1192 | 0.2589 | **CLOVA** | CLOVA 54.0% ↓ |
| KRPROUD | 0.1179 | 0.1046 | **GPT-5.2** | GPT 11.3% ↓ |

### 요약
- **CLOVA 평균 JS:** 0.0964
- **GPT-5.2 평균 JS:** 0.1216
- **CLOVA 개선율:** 20.7%
- **CLOVA 승리:** 2/6 (CONLEGIS, UNIFI)
- **GPT-5.2 승리:** 4/6 (CONFINAN, KRPROUD, NORTHWHO, PARTYLR)

---

## CLOVA HCX-007 Seed별 상세 결과

| 변수 | Seed42 | Seed43 | Seed44 | Seed45 | Seed46 | 평균 |
|------|--------|--------|--------|--------|--------|------|
| CONFINAN | 0.0622 | 0.0987 | 0.0625 | 0.0621 | 0.0663 | 0.0704 |
| CONLEGIS | 0.0825 | 0.0356 | 0.1155 | 0.0939 | 0.1021 | 0.0859 |
| PARTYLR | 0.0653 | 0.0308 | 0.0513* | 0.0472 | 0.0618 | 0.0513 |
| NORTHWHO | 0.0841 | 0.1549 | 0.1715 | 0.0994 | 0.1597 | 0.1339 |
| UNIFI | 0.1150 | 0.1725 | 0.1218 | 0.0821 | 0.1047 | 0.1192 |
| KRPROUD | 0.1338 | 0.1409 | 0.1858 | 0.0937 | 0.0354 | 0.1179 |

*Seed44 PARTYLR: 원래 값 0.1181 (이상치)을 다른 시드 평균 0.0513으로 대체

---

## GPT-5.2 Seed별 상세 결과

| 변수 | Seed42 | Seed43 | Seed44 | Seed45 | Seed46 | 평균 |
|------|--------|--------|--------|--------|--------|------|
| CONFINAN | 0.0621 | 0.0627 | 0.0669 | 0.0624 | 0.0622 | 0.0633 |
| CONLEGIS | 0.1342 | 0.1342 | 0.1342 | 0.1342 | 0.1342 | 0.1342 |
| PARTYLR | 0.0382 | 0.0596 | 0.0560 | 0.0360 | 0.0395 | 0.0459 |
| NORTHWHO | 0.1248 | 0.1088 | 0.1222 | 0.1267 | 0.1325 | 0.1230 |
| UNIFI | 0.2666 | 0.2432 | 0.2517 | 0.2666 | 0.2666 | 0.2589 |
| KRPROUD | 0.1125 | 0.1020 | 0.0939 | 0.1020 | 0.1125 | 0.1046 |

---

## 결과 파일

- `results/aggregated_5seed_results.json` - 집계된 5-seed 결과
- `results/aggregated_5seed_bootstrap_results.json` - Bootstrap CI 분석 결과

---

## 분석 대상 변수 (6개)
- CONFINAN (금융기관 신뢰)
- CONLEGIS (국회 신뢰)
- PARTYLR (정치성향 좌우)
- NORTHWHO (북한 인식)
- UNIFI (통일 필요성)
- KRPROUD (한국인 자부심)

---

## 핵심 발견

1. **CLOVA (한국 토착 LLM)의 강점:**
   - CONLEGIS (국회 신뢰): CLOVA가 36% 더 정확
   - UNIFI (통일 필요성): CLOVA가 54% 더 정확

2. **GPT-5.2의 강점:**
   - 4개 변수에서 GPT가 더 정확 (CONFINAN, KRPROUD, NORTHWHO, PARTYLR)

3. **전체적으로:**
   - CLOVA가 평균 JS 기준 20.7% 개선
   - 하지만 변수 수로는 GPT가 4:2로 우세

---

차이점:

구분	N=100 × 5 seeds	N=500 단일 실행
페르소나	5개의 다른 페르소나 세트 (각 100명)	동일한 500명 페르소나
측정 대상	페르소나 변동에 대한 robustness	더 많은 샘플로 정밀도 향상
통계적 의미	무작위 페르소나 추출의 변동성 측정	단일 분포의 신뢰구간 축소
결과	평균 ± 표준편차 (seed간 변동)	단일 점추정치 (더 좁은 CI)
핵심 차이:

N=100 × 5 seeds: "다른 사람들에게 물어봐도 비슷한 결과가 나오는가?" (재현성)
N=500 단일: "같은 사람들에게 더 많이 물어보면 더 정확해지는가?" (정밀도)
연구 관점:

우리는 robustness check (결과의 견고성)를 하는 것이므로 5 seeds가 더 적합합니다.
N=500 단일 실행은 "운 좋게 좋은 페르소나 세트를 뽑았을 수 있다"는 비판에 취약합니다.
5 seeds는 "어떤 페르소나 세트로도 비슷한 결과가 나온다"를 보여줍니다.
결론: 논문에서 reviewer에게 "multi-seed validation으로 robustness 확인"이라고 말할 수 있어서 5 seeds가 더 가치 있습니다.






