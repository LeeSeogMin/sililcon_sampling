"""
100개 페르소나 층화 추출 스크립트

KGSS 2023년 실제 분포를 반영하여 대표성 있는 100개 페르소나 생성
"""

import argparse
import json
import os
from typing import List, Dict
import random

# 연령-성별 기준 샘플링 (KGSS 2023 분포 반영)
AGE_GENDER_DISTRIBUTION = {
    "20대": {"남성": 9, "여성": 8},
    "30대": {"남성": 8, "여성": 8},
    "40대": {"남성": 9, "여성": 9},
    "50대": {"남성": 10, "여성": 10},
    "60대 이상": {"남성": 14, "여성": 15}
}

# 교육수준 분포 (연령대별)
EDUCATION_DISTRIBUTION = {
    "20대": {"고졸 이하": 0.20, "대졸": 0.70, "대학원 이상": 0.10},
    "30대": {"고졸 이하": 0.25, "대졸": 0.65, "대학원 이상": 0.10},
    "40대": {"고졸 이하": 0.30, "대졸": 0.60, "대학원 이상": 0.10},
    "50대": {"고졸 이하": 0.40, "대졸": 0.50, "대학원 이상": 0.10},
    "60대 이상": {"고졸 이하": 0.60, "대졸": 0.35, "대학원 이상": 0.05}
}

# 지역 분포
REGION_DISTRIBUTION = {
    "수도권": 0.50,
    "영남권": 0.25,
    "호남권": 0.15,
    "충청/강원": 0.10
}

# 직업 (연령-교육 조합 기반)
OCCUPATION_MAPPING = {
    "20대": {
        "고졸 이하": ["서비스직", "판매직", "단순노무"],
        "대졸": ["사무직", "전문직", "학생"],
        "대학원 이상": ["연구직", "학생", "전문직"]
    },
    "30대": {
        "고졸 이하": ["기능직", "자영업", "서비스직"],
        "대졸": ["사무직", "관리직", "전문직"],
        "대학원 이상": ["전문직", "연구직", "관리직"]
    },
    "40대": {
        "고졸 이하": ["자영업", "기능직", "서비스직"],
        "대졸": ["관리직", "사무직", "전문직"],
        "대학원 이상": ["전문직", "관리직", "연구직"]
    },
    "50대": {
        "고졸 이하": ["자영업", "농업/어업", "기능직"],
        "대졸": ["관리직", "사무직", "자영업"],
        "대학원 이상": ["전문직", "관리직", "연구직"]
    },
    "60대 이상": {
        "고졸 이하": ["은퇴", "농업/어업", "자영업"],
        "대졸": ["은퇴", "자영업", "관리직"],
        "대학원 이상": ["은퇴", "전문직", "관리직"]
    }
}


def weighted_choice(choices: Dict[str, float]) -> str:
    """가중치 기반 랜덤 선택"""
    items = list(choices.keys())
    weights = list(choices.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_personas(num_personas: int = 100) -> List[Dict]:
    """층화 추출을 통한 100개 페르소나 생성"""
    if num_personas != 100:
        raise ValueError("현재 분포 테이블 기준으로 num_personas는 100만 지원합니다.")

    personas = []
    persona_id = 1

    # 1단계: 연령-성별 기준 샘플링
    for age_group, gender_dist in AGE_GENDER_DISTRIBUTION.items():
        for gender, count in gender_dist.items():
            for _ in range(count):
                # 2단계: 교육수준 할당 (연령대별 분포)
                education = weighted_choice(EDUCATION_DISTRIBUTION[age_group])

                # 3단계: 지역 할당 (전체 분포)
                region = weighted_choice(REGION_DISTRIBUTION)

                # 4단계: 직업 할당 (연령-교육 조합 기반)
                occupation_options = OCCUPATION_MAPPING[age_group][education]
                occupation = random.choice(occupation_options)

                # 페르소나 생성
                persona = {
                    "persona_id": f"P{persona_id:03d}",
                    "age_group": age_group,
                    "gender": gender,
                    "education": education,
                    "region": region,
                    "occupation": occupation
                }
                personas.append(persona)
                persona_id += 1

    return personas


def save_personas(personas: List[Dict], output_path: str = "output/personas.json"):
    """페르소나를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(personas)}개 페르소나 저장 완료: {output_path}")


def print_persona_stats(personas: List[Dict]):
    """페르소나 통계 출력"""
    from collections import Counter

    print("\n=== 페르소나 통계 ===\n")

    # 연령-성별 분포
    print("📊 연령-성별 분포:")
    age_gender_counts = Counter([(p['age_group'], p['gender']) for p in personas])
    for (age, gender), count in sorted(age_gender_counts.items()):
        print(f"  {age} {gender}: {count}명")

    # 교육수준 분포
    print("\n📚 교육수준 분포:")
    education_counts = Counter([p['education'] for p in personas])
    for education, count in education_counts.items():
        print(f"  {education}: {count}명 ({count/len(personas)*100:.1f}%)")

    # 지역 분포
    print("\n🗺️ 지역 분포:")
    region_counts = Counter([p['region'] for p in personas])
    for region, count in region_counts.items():
        print(f"  {region}: {count}명 ({count/len(personas)*100:.1f}%)")

    # 직업 분포
    print("\n💼 직업 분포:")
    occupation_counts = Counter([p['occupation'] for p in personas])
    for occupation, count in sorted(occupation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {occupation}: {count}명")

    print(f"\n총 페르소나 수: {len(personas)}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KGSS 2023 분포 기반 100개 페르소나 생성")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (재현성)")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/personas/personas_100.json",
        help="출력 JSON 경로",
    )
    args = parser.parse_args()

    # 랜덤 시드 설정 (재현성)
    random.seed(args.seed)

    # 100개 페르소나 생성
    personas = generate_personas(num_personas=100)

    # 통계 출력
    print_persona_stats(personas)

    # JSON 파일로 저장
    save_personas(personas, args.out)

    # 샘플 출력
    print("\n=== 샘플 페르소나 (처음 5개) ===\n")
    for p in personas[:5]:
        print(f"{p['persona_id']}: {p['age_group']} {p['gender']}, {p['education']}, {p['region']}, {p['occupation']}")
