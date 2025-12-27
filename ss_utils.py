"""
공용 유틸리티 (경로/설정/통계)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def ensure_parent_dir(path: str) -> None:
    """파일 경로의 parent 디렉토리를 생성"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_first_int(text: str) -> int | None:
    """
    모델 출력에서 첫 번째 정수를 추출.
    예: "답: 3" -> 3, "10" -> 10, "3\n(보통)" -> 3
    """
    match = re.search(r"-?\d+", text.strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_probabilities(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Cannot normalize: sum <= 0")
    return arr / total


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Jensen-Shannon divergence (자연로그 기반, 0~ln(2)).
    참고: SciPy jensenshannon()은 distance(=sqrt(JS))를 반환.
    """
    p_arr = normalize_probabilities(p)
    q_arr = normalize_probabilities(q)
    m = 0.5 * (p_arr + q_arr)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (kl(p_arr, m) + kl(q_arr, m))


def js_distance(p: Sequence[float], q: Sequence[float]) -> float:
    return math.sqrt(js_divergence(p, q))


@dataclass(frozen=True)
class Benchmark:
    year: int
    analyzable_variables: List[str]
    distributions_pct: Dict[str, Dict[int, float]]

    def categories(self, variable: str) -> List[int]:
        return sorted(self.distributions_pct[variable].keys())

    def distribution_prob(self, variable: str) -> np.ndarray:
        categories = self.categories(variable)
        pct = [self.distributions_pct[variable][c] for c in categories]
        return normalize_probabilities(pct)


def load_benchmark(path: str = "data/kgss_benchmarks_2023.json") -> Benchmark:
    raw = read_json(path)
    distributions_pct: Dict[str, Dict[int, float]] = {}
    for var, dist in raw.get("distributions", {}).items():
        if not dist:
            continue
        distributions_pct[var] = {int(k): float(v) for k, v in dist.items()}

    return Benchmark(
        year=int(raw.get("year", 0)),
        analyzable_variables=list(raw.get("analyzable_variables", [])),
        distributions_pct=distributions_pct,
    )


def format_scale_labels(scale_labels: Mapping[str, str] | None) -> str:
    if not scale_labels:
        return ""
    parts: List[str] = []
    for k, v in scale_labels.items():
        parts.append(f"{k}: {v}")
    return ", ".join(parts)


def observed_distribution(values: Iterable[int], categories: Sequence[int]) -> Dict[int, int]:
    counts = {int(c): 0 for c in categories}
    for v in values:
        if int(v) in counts:
            counts[int(v)] += 1
    return counts


@dataclass(frozen=True)
class NcpCredentials:
    access_key_id: str
    secret_key: str


def load_ncp_credentials() -> NcpCredentials | None:
    """
    NCP API Gateway 인증키를 환경변수에서 로드.

    - NCP_ACCESS_KEY_ID -> X-NCP-APIGW-API-KEY-ID
    - NCP_SECRET_KEY -> X-NCP-APIGW-API-KEY

    둘 중 하나라도 없으면 None을 반환(선택 설정).
    """
    access_key_id = (os.getenv("NCP_ACCESS_KEY_ID") or "").strip()
    secret_key = (os.getenv("NCP_SECRET_KEY") or "").strip()
    if not access_key_id or not secret_key:
        return None
    return NcpCredentials(access_key_id=access_key_id, secret_key=secret_key)


def ncp_api_gateway_headers(creds: NcpCredentials | None = None) -> Dict[str, str]:
    """
    https://api.ncloud-docs.com/docs/common-ncpapi 기준 API Gateway 인증 헤더 생성.
    """
    resolved = creds or load_ncp_credentials()
    if not resolved:
        raise RuntimeError(
            "NCP credentials missing. Set NCP_ACCESS_KEY_ID and NCP_SECRET_KEY in .env (see .env.example)."
        )
    return {
        "X-NCP-APIGW-API-KEY-ID": resolved.access_key_id,
        "X-NCP-APIGW-API-KEY": resolved.secret_key,
    }


def load_clova_studio_api_key() -> str | None:
    """
    CLOVA Studio 콘솔에서 발급한 API Key.

    참고: 실제 헤더 이름은 엔드포인트/문서에 따라 다를 수 있으나,
    일반적으로 X-NCP-CLOVASTUDIO-API-KEY를 사용한다.
    """
    value = (os.getenv("CLOVA_STUDIO_API_KEY") or "").strip()
    return value or None


def clova_studio_headers(
    *,
    include_content_type: bool = True,
) -> Dict[str, str]:
    """
    CLOVA Studio 호출에 사용할 인증 헤더를 구성.

    공통 요청 헤더(문서 기준):
    - Authorization: Bearer <CLOVA_STUDIO_API_KEY>
    - Content-Type: application/json
    """
    api_key = load_clova_studio_api_key()
    if not api_key:
        raise RuntimeError(
            "CLOVA Studio API key missing. Set CLOVA_STUDIO_API_KEY in .env (see .env.example)."
        )

    headers: Dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers
