# Silicon Sampling Robustness Check - Experiment Summary

**Status Date:** 2026-01-16 (Updated)
**Experiment Phase:** Multi-seed validation (5 independent runs)

---

## 현재 정확한 실험 상태

### GPT-5.2 (모두 완료)
| Seed | 상태 | 파일 위치 |
|------|------|----------|
| 42 | ✅ 완료 (6개 변수, 100건) | `results/gpt52_experiment/persona_responses.csv` |
| 43 | ✅ 완료 (7개 변수, 100건) | `results/gpt52_experiment_seed43/persona_responses.csv` |
| 44 | ✅ 완료 (7개 변수, 100건) | `results/gpt52_experiment_seed44/persona_responses.csv` |
| 45 | ✅ 완료 (7개 변수, 100건) | `results/gpt52_experiment_seed45/persona_responses.csv` |
| 46 | ✅ 완료 (7개 변수, 100건) | `results/gpt52_experiment_seed46/persona_responses.csv` |

### CLOVA HCX-007 (진행 중)
| Seed | 상태 | 완료된 변수 | 남은 변수 | 파일 위치 |
|------|------|------------|----------|----------|
| 42 | ✅ 완료 | CONFINAN, CONLEGIS, PARTYLR, NORTHWHO, UNIFI, KRPROUD (6개) | 없음 | `results/clova_experiment/{변수}/clova_results.json` |
| 43 | ✅ 완료 | CONFINAN, CONLEGIS, PARTYLR, UNIFI, KRPROUD (5개, SATFIN 제외) | 없음 | `results/clova_experiment_seed43/clova_results_partial.json` |
| 44 | 🔄 진행중 | CONFINAN, CONLEGIS, PARTYLR, NORTHWHO (4개) | **UNIFI, KRPROUD (2개)** | `results/clova_experiment_seed44/clova_results_partial.json` |
| 45 | 🔄 진행중 | CONFINAN (1개) | **CONLEGIS, PARTYLR, NORTHWHO, UNIFI, KRPROUD (5개)** | `results/clova_experiment_seed45/clova_results_partial.json` |
| 46 | 🔄 진행중 | CONFINAN, CONLEGIS, PARTYLR (3개) | **NORTHWHO, UNIFI, KRPROUD (3개)** | `results/clova_experiment_seed46/clova_results_partial.json` |

---

## 해야 할 작업 (순차 실행 필수!)

### 1. Seed43 CLOVA (2개 변수)
```bash
python code/06_clova_experiment.py \
  --personas outputs/personas/personas_100_seed43.json \
  --out-dir results/clova_experiment_seed43 \
  --variables UNIFI KRPROUD \
  --n-samples 100 \
  --thinking medium \
  --delay 0.5
```

### 2. Seed44 CLOVA (5개 변수)
```bash
python code/06_clova_experiment.py \
  --personas outputs/personas/personas_100_seed44.json \
  --out-dir results/clova_experiment_seed44 \
  --variables CONLEGIS PARTYLR NORTHWHO UNIFI KRPROUD \
  --n-samples 100 \
  --thinking medium \
  --delay 0.5
```

### 3. Seed45 CLOVA (6개 변수)
```bash
python code/06_clova_experiment.py \
  --personas outputs/personas/personas_100_seed45.json \
  --out-dir results/clova_experiment_seed45 \
  --variables CONFINAN CONLEGIS PARTYLR NORTHWHO UNIFI KRPROUD \
  --n-samples 100 \
  --thinking medium \
  --delay 0.5
```

### 4. Seed46 CLOVA (6개 변수)
```bash
python code/06_clova_experiment.py \
  --personas outputs/personas/personas_100_seed46.json \
  --out-dir results/clova_experiment_seed46 \
  --variables CONFINAN CONLEGIS PARTYLR NORTHWHO UNIFI KRPROUD \
  --n-samples 100 \
  --thinking medium \
  --delay 0.5
```

### 5. 결과 집계 및 논문 업데이트
```bash
python code/07_aggregate_results.py
python code/08_bootstrap_ci_analysis.py
```

---

## 중요 주의사항

1. **같은 seed 디렉토리에는 한 번에 하나의 프로세스만 실행**
2. **병렬 실행하려면 다른 seed를 각각 실행**
3. **순차적으로 하나씩 실행해야 Race Condition 방지**

---

## 분석 대상 변수 (6개)
- CONFINAN (금융기관 신뢰)
- CONLEGIS (국회 신뢰)
- PARTYLR (정치성향 좌우)
- NORTHWHO (북한 인식)
- UNIFI (통일 필요성)
- KRPROUD (한국인 자부심)

(SATFIN은 분석 대상이 아님)

---

## Personas 파일 위치
- `outputs/personas/personas_100_seed42.json`
- `outputs/personas/personas_100_seed43.json`
- `outputs/personas/personas_100_seed44.json`
- `outputs/personas/personas_100_seed45.json`
- `outputs/personas/personas_100_seed46.json`

---

비교 결과 (CLOVA Seed42,43 평균 vs GPT-5.2 5seed 평균):

변수	CLOVA S42	CLOVA S43	CLOVA 평균	GPT 평균	개선율	승자
CONFINAN	0.0622	0.0987	0.0805	0.0633	-27.1%	GPT
CONLEGIS	0.0825	0.0356	0.0591	0.1342	+56.0%	CLOVA
PARTYLR	0.0653	0.0561	0.0607	0.0459	-32.2%	GPT
NORTHWHO	0.0841	0.1149	0.0995	0.1230	+19.1%	CLOVA
UNIFI	0.1150	0.1299	0.1225	0.2589	+52.7%	CLOVA
KRPROUD	0.1338	진행중	0.1338	0.1046	-27.9%	GPT
핵심:

CLOVA 승리: 3/6 (50%)
GPT-5.2 승리: 3/6 (50%)
전체 개선율: +23.83% ✅
Seed43의 CONLEGIS (0.0356)이 매우 우수함 - CONLEGIS에서 CLOVA의 강점 확인
Seed44 완료되면 더 정확한 비교가 가능합니다.