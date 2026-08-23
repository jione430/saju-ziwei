"""
사주 × 자미두수 계산 엔진 백엔드

- 사주(四柱八字): lunar_python
- 자미두수(紫微斗數): py-iztro
- 진태양시 보정: 127.5°E 기준
- AI 리포트 프롬프트 생성
- Anthropic API를 통한 실제 리포트 생성

주요 기능

1. /api/chart
   → 사주 + 자미두수 명식/명반 계산

2. /api/prompt
   → AI에게 전달할 프롬프트 생성
   → 클라이언트에서 복사해서 Claude 등에 직접 사용 가능

3. /api/report
   → 서버의 Anthropic API를 이용해 실제 리포트 생성

상세 리포트 기능

summary
→ 기본 요약 리포트

detailed
→ 사용자가 선택한 상세 분석 항목을 중심으로 심층 리포트 생성

예:
detail_options = [
    "기질",
    "연애",
    "재물",
    "직업",
    "건강",
    "인간관계",
    "10년운세"
]

detail_question = "이직한다면 어떤 직무가 잘 맞는지 알고 싶어요."
"""


# =========================================================
# THREAD / IMPORT
# =========================================================

import threading

# py-iztro(pythonmonkey/SpiderMonkey)의
# "too much recursion" 문제 방지를 위한 스택 크기 설정
threading.stack_size(64 * 1024 * 1024)


import os

from datetime import datetime, timedelta
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lunar_python import Solar
from py_iztro import Astro

import anthropic

from cities import (
    CITY_LONGITUDE,
    TRUE_SOLAR_REFERENCE_LONGITUDE,
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="사주×자미두수 계산 엔진 API",
    version="0.2.0",
)


# =========================================================
# ANTHROPIC
# =========================================================

ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY"
)

anthropic_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY
)


# =========================================================
# CORS
# =========================================================

# 프로토타입 단계에서는 전체 허용
# 실제 배포 시 프론트엔드 도메인으로 제한 권장
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 사주 오행 매핑
# =========================================================

GAN_TO_WUXING = {
    "甲": "木",
    "乙": "木",
    "丙": "火",
    "丁": "火",
    "戊": "土",
    "己": "土",
    "庚": "金",
    "辛": "金",
    "壬": "水",
    "癸": "水",
}


ZHI_TO_WUXING = {
    "子": "水",
    "丑": "土",
    "寅": "木",
    "卯": "木",
    "辰": "土",
    "巳": "火",
    "午": "火",
    "未": "土",
    "申": "金",
    "酉": "金",
    "戌": "土",
    "亥": "水",
}


# =========================================================
# REQUEST MODELS
# =========================================================

class ChartRequest(BaseModel):

    name: str

    birth_date: str = Field(
        ...,
        description="YYYY-MM-DD, 그레고리력(양력) 입력 기준"
    )

    birth_time: Optional[str] = Field(
        None,
        description="HH:MM, time_unknown=True면 생략 가능"
    )

    time_unknown: bool = False

    birth_city: str

    gender: Literal[
        "여성",
        "남성"
    ]

    calendar_type: Literal[
        "양력",
        "음력"
    ] = "양력"

    is_leap_month: bool = False


class ReportRequest(BaseModel):

    # /api/chart 전체 응답
    chart: dict

    # 프론트에서 저장한 사용자 입력
    #
    # {
    #   name,
    #   gender,
    #   date,
    #   time,
    #   city,
    #   preQuestion
    # }
    user_input: dict

    # summary / detailed
    detail_level: Literal[
        "summary",
        "detailed"
    ] = "summary"

    # 상세 분석에서 사용자가 선택한 항목
    #
    # 예:
    # [
    #   "기질",
    #   "연애",
    #   "직업",
    #   "10년운세"
    # ]
    detail_options: list[str] = []

    # 상세 분석에서 추가로 입력한 질문
    detail_question: str = ""


# =========================================================
# CITY / TRUE SOLAR TIME
# =========================================================

def get_longitude(city: str) -> float:
    """
    도시명으로 경도 조회.

    CITY_LONGITUDE에 등록된 도시가 있으면
    해당 경도를 사용한다.

    찾지 못하면
    TRUE_SOLAR_REFERENCE_LONGITUDE를 사용하여
    보정값 0으로 처리한다.
    """

    for key, lon in CITY_LONGITUDE.items():

        if key in city or city in key:
            return lon

    return TRUE_SOLAR_REFERENCE_LONGITUDE


def true_solar_time_correction(
    dt: datetime,
    longitude: float
) -> datetime:
    """
    127.5E 기준 진태양시 보정.

    경도가 기준보다 동쪽이면 시간을 앞당기고,
    서쪽이면 늦춘다.
    """

    offset_minutes = (
        TRUE_SOLAR_REFERENCE_LONGITUDE
        - longitude
    ) * 4

    return dt - timedelta(
        minutes=offset_minutes
    )


def hour_to_ziwei_index(
    hour: int,
    minute: int
) -> int:
    """
    py-iztro의 시진 인덱스 규칙.

    0 = 조자시
    00~01시 = 0
    01~03시 = 1
    ...
    23~24시 = 12
    """

    h = hour + minute / 60

    boundaries = [
        (0, 1, 0),
        (1, 3, 1),
        (3, 5, 2),
        (5, 7, 3),
        (7, 9, 4),
        (9, 11, 5),
        (11, 13, 6),
        (13, 15, 7),
        (15, 17, 8),
        (17, 19, 9),
        (19, 21, 10),
        (21, 23, 11),
        (23, 24, 12),
    ]

    for start, end, idx in boundaries:

        if start <= h < end:
            return idx

    return 0


# =========================================================
# CHART API
# =========================================================

@app.post("/api/chart")
async def calculate_chart(
    req: ChartRequest
):

    # -----------------------------------------------------
    # 음력
    # -----------------------------------------------------

    if req.calendar_type == "음력":

        raise HTTPException(
            status_code=400,
            detail=(
                "음력 입력은 아직 지원되지 않습니다. "
                "양력으로 입력해 주세요."
            ),
        )


    # -----------------------------------------------------
    # 생년월일
    # -----------------------------------------------------

    try:

        y, mo, d = map(
            int,
            req.birth_date.split("-")
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "birth_date 형식이 올바르지 않습니다 "
                "(YYYY-MM-DD)"
            ),
        )


    # -----------------------------------------------------
    # 출생시간
    # -----------------------------------------------------

    if (
        req.time_unknown
        or not req.birth_time
    ):

        # 명식 계산 최소 요건을 위한 임시 시간
        h, mi = 12, 0

    else:

        try:

            h, mi = map(
                int,
                req.birth_time.split(":")
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "birth_time 형식이 올바르지 않습니다 "
                    "(HH:MM)"
                ),
            )


    # -----------------------------------------------------
    # 도시 / 진태양시
    # -----------------------------------------------------

    longitude = get_longitude(
        req.birth_city
    )

    raw_dt = datetime(
        y,
        mo,
        d,
        h,
        mi
    )

    corrected_dt = true_solar_time_correction(
        raw_dt,
        longitude
    )


    # =====================================================
    # 사주 계산
    # =====================================================

    solar = Solar.fromYmdHms(
        corrected_dt.year,
        corrected_dt.month,
        corrected_dt.day,
        corrected_dt.hour,
        corrected_dt.minute,
        0
    )

    lunar = solar.getLunar()

    bazi = lunar.getEightChar()


    pillars = {

        "year": {
            "gan": bazi.getYearGan(),
            "zhi": bazi.getYearZhi(),
        },

        "month": {
            "gan": bazi.getMonthGan(),
            "zhi": bazi.getMonthZhi(),
        },

        "day": {
            "gan": bazi.getDayGan(),
            "zhi": bazi.getDayZhi(),
        },
    }


    if (
        not req.time_unknown
        and req.birth_time
    ):

        pillars["time"] = {
            "gan": bazi.getTimeGan(),
            "zhi": bazi.getTimeZhi(),
        }

    else:

        pillars["time"] = None


    # =====================================================
    # 오행 분포
    # =====================================================

    wx_count = {
        "木": 0,
        "火": 0,
        "土": 0,
        "金": 0,
        "水": 0,
    }


    for key, pillar in pillars.items():

        if pillar is None:
            continue

        wx_count[
            GAN_TO_WUXING[
                pillar["gan"]
            ]
        ] += 1

        wx_count[
            ZHI_TO_WUXING[
                pillar["zhi"]
            ]
        ] += 1


    # =====================================================
    # 자미두수
    # =====================================================

    gender_cn = (
        "女"
        if req.gender == "여성"
        else "男"
    )


    if (
        req.time_unknown
        or not req.birth_time
    ):

        # 시간 미상
        ziwei_result = None

    else:

        hour_idx = hour_to_ziwei_index(
            corrected_dt.hour,
            corrected_dt.minute
        )

        astro = Astro()


        r = astro.by_solar(
            (
                f"{corrected_dt.year}-"
                f"{corrected_dt.month:02d}-"
                f"{corrected_dt.day:02d}"
            ),
            hour_idx,
            gender_cn,
            True,
            "zh-CN",
        )


        palaces = []


        for p in r.palaces:

            palaces.append({

                "name": p.name,

                "branch":
                    p.earthly_branch,

                "stem":
                    p.heavenly_stem,

                "major_stars": [

                    {
                        "name": s.name,

                        "brightness":
                            s.brightness,

                        "mutagen":
                            s.mutagen,
                    }

                    for s in p.major_stars
                ],

                "minor_stars": [
                    s.name
                    for s in p.minor_stars
                ],

                "is_body":
                    p.is_body_palace,

                "decadal":
                    p.decadal.range,
            })


        ziwei_result = {

            "soul_palace":
                r.earthly_branch_of_soul_palace,

            "body_palace":
                r.earthly_branch_of_body_palace,

            "five_elements_class":
                r.five_elements_class,

            "soul_star":
                r.soul,

            "body_star":
                r.body,

            "palaces":
                palaces,
        }


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "input": {

            "date":
                req.birth_date,

            "time":
                "모름"
                if req.time_unknown
                else req.birth_time,

            "gender":
                req.gender,

            "place":
                req.birth_city,
        },


        "correction": {

            "longitude_used":
                longitude,

            "reference_longitude":
                TRUE_SOLAR_REFERENCE_LONGITUDE,

            "offset_minutes":
                round(
                    (
                        TRUE_SOLAR_REFERENCE_LONGITUDE
                        - longitude
                    ) * 4,
                    1
                ),

            "corrected_time":
                (
                    corrected_dt.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if not req.time_unknown
                    else None
                ),
        },


        "lunar_date":
            (
                f"{lunar.getYear()}년 "
                f"{lunar.getMonth()}월 "
                f"{lunar.getDay()}일"
            ),


        "bazi":
            pillars,


        "wuxing_count":
            wx_count,


        "day_master":
            bazi.getDayGan(),


        "ziwei":
            ziwei_result,
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# AI REPORT
# =========================================================

MAPPING_TABLE = """
| 주제 | 사주 | 자미두수 |
|---|---|---|
| 타고난 기질 | 일간+오행분포 | 명궁 주성 |
| 재물운 | 재성 | 재백궁 |
| 직업/사회운 | 관성 | 관록궁 |
| 학업/정신자산 | 인성 | 부모궁·복덕궁 |
| 대인관계 | 비겁 | 형제궁·노복궁 |
| 표현력/자녀 | 식상 | 자녀궁 |
| 결혼/애정 | 일지 | 부처궁 |
| 건강경향 | 오행 편중 | 질액궁 |
| 인생흐름 | 대운 | 대한 |
"""


# =========================================================
# DETAIL OPTION DEFINITIONS
# =========================================================

DETAIL_OPTION_GUIDES = {

    "기질": """
[상세 항목: 타고난 기질]

반드시 다음을 중심으로 분석한다.

- 일간의 기본 성향
- 오행 분포의 균형과 편중
- 강점과 약점
- 사고방식
- 행동 패턴
- 스트레스를 받을 때 나타나는 경향
- 인간관계에서의 기본 태도
- 자미두수 명궁 및 명궁 주성과의 교차해석

단순히 성격 유형을 나열하지 말고,
어떤 간지/오행/명궁/주성이 근거인지 함께 설명한다.
""",


    "연애": """
[상세 항목: 연애·결혼]

다음 내용을 구체적으로 분석한다.

- 연애에서의 기본 성향
- 상대에게 중요하게 생각하는 요소
- 애정 표현 방식
- 관계에서 갈등이 생기는 패턴
- 잘 맞는 관계의 특징
- 결혼관
- 배우자궁 및 부처궁
- 자미두수의 부처궁 주성
- 사주와 자미두수가 보여주는 공통점과 차이
- 향후 연애 흐름을 볼 수 있는 경우 해당 시기

확정적인 결혼 여부나 이혼 여부를 단정하지 않는다.
""",


    "재물": """
[상세 항목: 재물운]

다음 내용을 구체적으로 분석한다.

- 돈을 다루는 기본 성향
- 재성의 구조
- 소비와 저축 성향
- 안정적인 수입과 변동성 있는 수입 중 어떤 방식이 더 어울리는지
- 재물과 직업의 연결
- 재백궁과 주요 별
- 재물운에서 주의할 점
- 장기적인 재정 습관에 대한 현실적인 조언

투자 종목이나 구체적인 금융상품을 추천하지 않는다.
""",


    "직업": """
[상세 항목: 직업·커리어]

다음 내용을 가장 깊게 분석한다.

- 직업적 강점
- 일하는 방식
- 조직생활에서의 특징
- 독립적인 업무와 협업 중 어느 환경이 더 맞는지
- 직업 선택에서 중요한 조건
- 관성 및 식상/재성/인성 등의 관계
- 관록궁과 주요 별
- 사회적 성취 방식
- 어떤 종류의 업무에서 능력을 발휘하기 쉬운지
- 장기 커리어 방향
- 직업을 선택할 때 피해야 할 환경
- 사주와 자미두수에서 공통으로 나타나는 직업적 특징

가능하다면 추상적인 직업명만 나열하지 말고
업무 환경과 업무 방식까지 설명한다.
""",


    "건강": """
[상세 항목: 건강 경향]

오행의 균형 및 질액궁을 참고하여
생활 습관 관점에서 분석한다.

- 전반적인 컨디션 관리 경향
- 생활 리듬
- 스트레스와 피로 관리
- 오행 편중과 관련해 전통 명리학에서 보는 경향
- 질액궁의 특징
- 생활에서 실천할 수 있는 관리 방법

질병이나 의학적 진단을 절대 단정하지 않는다.
의료행위나 치료법을 권하지 않는다.
""",


    "인간관계": """
[상세 항목: 인간관계]

다음 내용을 분석한다.

- 대인관계의 기본 성향
- 친해지는 방식
- 사람을 대하는 방식
- 갈등 발생 패턴
- 친구 관계
- 조직 안에서의 관계
- 도움을 주고받는 방식
- 형제궁 및 노복궁
- 사주와 자미두수의 공통점과 차이

관계에 대한 단정적인 예언보다는
반복적으로 나타날 수 있는 경향을 설명한다.
""",


    "학업": """
[상세 항목: 학업·성장]

다음 내용을 분석한다.

- 공부하는 방식
- 정보를 습득하는 방식
- 집중력과 지속력에 대한 경향
- 인성 및 식상의 관계
- 부모궁·복덕궁
- 새로운 기술을 배울 때의 강점
- 장기적인 자기계발 방식
- 어떤 학습 환경에서 효율이 높을 가능성이 있는지
""",


    "10년운세": """
[상세 항목: 향후 10년]

향후 10년의 흐름을 중심으로 분석한다.

가능한 경우 반드시
사주 대운과 자미두수 대한을 함께 고려한다.

- 전체적인 10년 흐름
- 중요한 전환점
- 2~3년 단위의 흐름
- 직업
- 재물
- 연애/인간관계
- 성장과 변화
- 주의할 시기
- 활용하기 좋은 시기

나이를 계산할 수 있는 정보가 부족하다면
무리해서 정확한 나이를 만들어내지 않는다.
제공된 데이터가 허용하는 범위 안에서 해석한다.
""",


    "대운": """
[상세 항목: 대운·대한]

사주의 대운과 자미두수 대한을 비교하는 데 집중한다.

- 현재 운의 위치
- 주요 대운
- 대운이 바뀌는 시점
- 자미두수 대한과의 관계
- 두 체계가 같은 방향을 가리키는 시기
- 서로 다른 신호를 보이는 시기
- 중요한 선택을 하기 좋은 흐름과 신중해야 할 흐름

계산되지 않은 대운/대한 정보를 임의로 만들어내지 않는다.
제공된 구조화 데이터에 근거해서 해석한다.
""",


    "올해": """
[상세 항목: 올해 운세]

현재 시점의 한 해 흐름을 중심으로 분석한다.

- 올해 전체 분위기
- 직업/학업
- 재물
- 연애
- 인간관계
- 변화와 기회
- 주의할 점
- 현실적으로 활용할 수 있는 행동 전략

현재 연도를 데이터와 사용자 입력에서 확인할 수 없는 경우
임의로 특정 연도를 만들어내지 않는다.
""",


    "질문": """
[상세 항목: 사용자 질문 집중 분석]

사용자가 입력한 질문을 가장 중요한 분석 대상으로 삼는다.

질문의 답을 먼저 제시하고,
그 다음에 사주와 자미두수의 어떤 데이터가
그 답을 뒷받침하는지 설명한다.

단순한 점괘식 답변이 아니라
근거 → 해석 → 현실적인 조언 순서로 작성한다.
""",
}


# =========================================================
# OPTION ALIAS
# =========================================================

DETAIL_OPTION_ALIASES = {

    "기질": "기질",
    "타고난 기질": "기질",
    "성격": "기질",

    "연애": "연애",
    "연애운": "연애",
    "결혼": "연애",
    "연애·결혼": "연애",

    "재물": "재물",
    "재물운": "재물",
    "돈": "재물",

    "직업": "직업",
    "직업운": "직업",
    "커리어": "직업",

    "건강": "건강",
    "건강운": "건강",

    "인간관계": "인간관계",
    "대인관계": "인간관계",

    "학업": "학업",
    "학업·성장": "학업",
    "공부": "학업",

    "10년운세": "10년운세",
    "향후10년": "10년운세",
    "향후 10년": "10년운세",

    "대운": "대운",
    "대운·대한": "대운",

    "올해": "올해",
    "올해운세": "올해",
    "올해 운세": "올해",

    "질문": "질문",
    "사용자질문": "질문",
}


def normalize_detail_options(
    options: list[str]
) -> list[str]:
    """
    프론트에서 넘어온 상세 옵션을
    백엔드 내부 표준 이름으로 변환한다.

    중복은 제거한다.
    """

    result = []

    for option in options or []:

        if not isinstance(
            option,
            str
        ):
            continue

        normalized = (
            DETAIL_OPTION_ALIASES.get(
                option.strip()
            )
        )

        if normalized and normalized not in result:

            result.append(
                normalized
            )

    return result


# =========================================================
# DETAIL PROMPT BUILDER
# =========================================================

def build_detail_instruction(
    detail_options: list[str],
    detail_question: str,
    pre_question: str
) -> str:
    """
    사용자가 선택한 상세 항목을
    실제 AI 프롬프트에 삽입한다.
    """

    normalized_options = (
        normalize_detail_options(
            detail_options
        )
    )


    sections = []


    for option in normalized_options:

        guide = DETAIL_OPTION_GUIDES.get(
            option
        )

        if guide:

            sections.append(
                guide.strip()
            )


    # 사용자 질문이 있는데
    # 질문 옵션을 직접 선택하지 않은 경우에도
    # 질문을 분석 대상으로 포함한다.

    if (
        pre_question
        and "질문" not in normalized_options
    ):

        sections.append(
            """
[상세 항목: 사용자가 미리 남긴 질문]

아래 질문을 상세 분석의 중요한 축으로 사용한다.

질문:
"""
            + pre_question
            + """

질문에 대한 답변을 먼저 명확하게 제시하고,
그 뒤에 사주와 자미두수의 근거를 설명한다.
"""
        )


    if detail_question:

        sections.append(
            """
[상세 항목: 추가 요청사항]

사용자가 상세 분석을 위해 추가로 요청한 내용:

"""
            + detail_question
            + """

위 요청사항을 별도의 중요한 분석 기준으로 반영한다.
질문의 내용과 직접 관련된 사주/자미두수 근거를 찾아
구체적으로 설명한다.
"""
        )


    if not sections:

        return """
[상세 분석 항목]

사용자가 특정 상세 항목을 선택하지 않았다.

기본 리포트의 모든 핵심 영역을 조금 더 깊게 분석하되,
각 영역에서 새로운 근거와 정보를 추가한다.

특히 타고난 기질, 연애, 재물, 직업,
건강, 인간관계, 향후 흐름을 균형 있게 다룬다.
""".strip()


    return "\n\n".join(
        sections
    )


# =========================================================
# REPORT PROMPT
# =========================================================

def build_report_prompt(
    chart: dict,
    user_input: dict,
    detail_level: str,
    detail_options: list[str] | None = None,
    detail_question: str = ""
) -> str:

    if detail_options is None:
        detail_options = []


    has_ziwei = (
        chart.get("ziwei")
        is not None
    )


    pre_question = (
        user_input.get(
            "preQuestion"
        )
        or ""
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    if detail_level == "summary":

        length_rule = """
[분량 규칙]

전체 약 1,200~2,000자 내외의
빠르게 읽을 수 있는 요약 리포트로 작성한다.

각 섹션은 1~2개의 짧은 문단으로 구성한다.

핵심 정보만 전달하고,
같은 내용을 반복하지 않는다.
""".strip()


    # =====================================================
    # DETAILED
    # =====================================================

    else:

        length_rule = """
[분량 규칙]

전체적으로 충분히 상세한 리포트를 작성한다.

단순히 문장을 길게 늘리지 말고,
각 분석 항목에서 실제 구조화 데이터의 근거를
추가로 제시하여 깊이를 만든다.

사용자가 선택한 상세 항목은
선택하지 않은 항목보다 훨씬 깊게 다룬다.

선택한 각 항목은 최소 3~5개의 문단 또는
그에 준하는 충분한 설명을 제공한다.

단, 데이터가 없는 부분은 억지로 만들어내지 않는다.

전체 분량은 약 5~20페이지 상당을 목표로 하되,
정보가 없는 부분을 반복해서 분량을 늘리지 않는다.
""".strip()


    # =====================================================
    # DETAIL INSTRUCTION
    # =====================================================

    detail_instruction = ""


    if detail_level == "detailed":

        detail_instruction = build_detail_instruction(
            detail_options,
            detail_question,
            pre_question
        )


    # =====================================================
    # USER QUESTION
    # =====================================================

    user_question_block = ""


    if pre_question:

        user_question_block = f"""
[사용자가 미리 남긴 질문]

"{pre_question}"
"""


    # =====================================================
    # ADDITIONAL DETAIL QUESTION
    # =====================================================

    additional_question_block = ""


    if detail_question:

        additional_question_block = f"""
[상세 버전에서 사용자가 추가한 요청]

"{detail_question}"
"""


    # =====================================================
    # DETAIL OPTION LIST
    # =====================================================

    normalized_options = (
        normalize_detail_options(
            detail_options
        )
    )


    selected_options_text = (
        ", ".join(
            normalized_options
        )
        if normalized_options
        else "선택된 특정 항목 없음"
    )


    # =====================================================
    # COMMON RULES
    # =====================================================

    cross_rule = (

        """
사주와 자미두수를 항상 같은 주제 안에서
교차 비교한다.

예를 들어 직업운을 설명할 때는
사주의 관성/식상/재성/인성 등의 구조와
자미두수의 관록궁 및 주요 별을 함께 살핀다.
"""

        if has_ziwei

        else

        """
자미두수 데이터가 없으므로
사주 데이터만으로 해석한다.

시간을 알면 자미두수 교차분석이 가능하다는 점을
리포트 서두에서 간단히 안내한다.
"""
    )


    # =====================================================
    # SECTION RULE
    # =====================================================

    if detail_level == "summary":

        section_rule = """
[섹션 구성]

다음 순서를 따른다.

## 한눈에 보는 요약
## 타고난 기질
## 인생의 진짜 황금기
## 연애운
## 재물운
## 직업운
## 건강운
## 개운법
## 향후 10년 디테일 운세
## 향후 10년 대운 분석
## 사주와 자미두수: 일치와 차이
"""

        if pre_question:

            section_rule += """
## 질문에 대한 답변
"""

        section_rule += """
## 종합 조언
"""


    else:

        section_rule = """
[섹션 구성]

기본적으로 다음 순서를 따른다.

## 한눈에 보는 요약
## 타고난 기질
## 인생의 진짜 황금기
## 연애운
## 재물운
## 직업운
## 건강운
## 개운법
## 향후 10년 디테일 운세
## 향후 10년 대운 분석
## 사주와 자미두수: 일치와 차이
"""

        if pre_question:

            section_rule += """
## 질문에 대한 답변
"""

        section_rule += """
## 종합 조언
"""


        section_rule += """
선택된 상세 항목은 해당 기본 섹션 안에서
특히 깊게 확장한다.

예를 들어 "직업"을 선택했다면
직업운 섹션을 가장 깊게 작성한다.

"연애"를 선택했다면
연애운 섹션을 가장 깊게 작성한다.

"10년운세"를 선택했다면
향후 10년 디테일 운세와
향후 10년 대운 분석을 특히 깊게 작성한다.

선택되지 않은 영역은
기본적인 해석만 제공한다.
"""


    # =====================================================
    # FINAL PROMPT
    # =====================================================

    return f"""
당신은 명리학(사주)과 자미두수에 모두 정통한 해석가입니다.

아래 입력된 구조화 데이터는
이미 검증된 계산 엔진이 산출한 결과입니다.

당신은 계산을 다시 하지 않고
제공된 데이터를 해석하는 역할만 수행합니다.

==================================================
[사주 데이터]
==================================================

{chart.get("bazi")}

오행분포:
{chart.get("wuxing_count")}

일간:
{chart.get("day_master")}


==================================================
[자미두수 데이터]
==================================================

{
    chart.get("ziwei")
    if has_ziwei
    else
    "태어난 시간을 몰라 자미두수는 계산되지 않음 - 사주만으로 해석할 것"
}


==================================================
[사용자 기본정보]
==================================================

이름:
{user_input.get("name")}

성별:
{user_input.get("gender")}

생년월일:
{user_input.get("date")}

시간:
{user_input.get("time")}

출생지:
{user_input.get("city")}


{user_question_block}


{additional_question_block}


==================================================
[리포트 모드]
==================================================

현재 모드:
{detail_level}

상세 분석 선택 항목:
{selected_options_text}


==================================================
[사주 × 자미두수 매핑 규칙]
==================================================

{MAPPING_TABLE if has_ziwei else "(자미두수 데이터 없음 - 사주 단독 해석)"}


==================================================
[상세 분석 지시]
==================================================

{
    detail_instruction
    if detail_level == "detailed"
    else
    "현재는 요약 버전이므로 특정 항목을 과도하게 확장하지 않는다."
}


==================================================
[작성 규칙]
==================================================

1. 전문용어는 처음 등장할 때
괄호 안에 쉬운 설명을 덧붙인다.
또한, 한자는 모두 한글로 바꾸어 가독성을 높인다.

2.
{cross_rule}

3.
두 체계가 같은 신호를 주면
"두 체계 모두 ~라고 말합니다"와 같이
공통점을 명확하게 강조한다.

두 체계가 서로 다른 신호를 주면
억지로 하나로 통일하지 않고
각각의 해석을 구분하여 설명한다.

4.
건강, 사고, 이혼 등 민감한 주제는
"경향이 있다", "주의가 필요할 수 있다"
수준으로만 표현한다.

확정적인 예언을 하지 않는다.

5.
의학적 진단을 하지 않는다.

법률 자문이나 투자 자문으로
해석될 수 있는 표현도 사용하지 않는다.

6.
"인생의 진짜 황금기"와
"향후 10년 대운 분석"에서는
가능한 경우 사주 대운과
자미두수 대한을 함께 비교한다.

단,
구조화 데이터에 존재하지 않는
대운/대한 정보를 임의로 만들어내지 않는다.

7.
"향후 10년 디테일 운세"는
가능한 경우 앞으로의 흐름을
2~3년 단위로 나누어 설명한다.

8.
"개운법"은 미신적이거나
값비싼 소비를 유도하지 않는다.

색상, 방향, 생활 습관,
시간 관리, 마음가짐처럼
부담 없이 실천할 수 있는 조언을
3~5개 정도 제시한다.

9.
사용자가 선택한 상세 항목은
다른 항목보다 훨씬 깊게 분석한다.

10.
상세 항목을 분석할 때는
가능한 한 다음 구조를 따른다.

"근거 → 해석 → 실제 생활에서 나타날 수 있는 모습 → 조언"

11.
단순한 성격 묘사나
인터넷에서 흔히 볼 수 있는
일반적인 운세 문구를 반복하지 않는다.

반드시 제공된 사주/자미두수 데이터와
연결하여 설명한다.

12.
데이터에 없는 사실을 만들어내지 않는다.

특히 정확한 대운 시작 나이,
특정 사건 발생,
특정 연도에 반드시 일어나는 사건 등은
데이터로 확인되지 않는다면 단정하지 않는다.

13.
톤은 친근하지만 가볍지 않게 작성한다.

확신을 주되 단정하지 않는다.

일반인이 읽었을 때
"그래서 나는 실제로 어떻게 행동하면 되는지"
알 수 있도록 작성한다.

14.
마크다운 취소선(~~텍스트~~)을 사용하지 않는다.

15.
마크다운 구분선(---)을 사용하지 않는다.

16.
섹션은 반드시 ## 제목으로 구분한다.


{section_rule}


{length_rule}


17.
같은 문장이나 표현을 반복하여
분량을 억지로 늘리지 않는다.

각 섹션은 가능한 한
서로 다른 데이터 근거와 새로운 정보를 제공한다.


==================================================
[최종 지시]
==================================================

위 데이터를 바탕으로
사용자가 실제로 읽었을 때 도움이 되는
사주 × 자미두수 통합 리포트를 작성하세요.

특히 상세 모드에서는
사용자가 선택한 항목과 추가 질문을
가장 중요한 분석 대상으로 삼으세요.
""".strip()


# =========================================================
# PROMPT API
# =========================================================

@app.post("/api/prompt")
def generate_prompt(
    req: ReportRequest
):
    """
    AI 프롬프트 텍스트 생성.

    API 키를 사용하지 않는다.

    사용자가 생성된 프롬프트를 복사하여
    Claude 등의 AI 챗봇에 직접 붙여넣을 수 있다.
    """

    try:

        prompt = build_report_prompt(
            chart=req.chart,
            user_input=req.user_input,
            detail_level=req.detail_level,
            detail_options=req.detail_options,
            detail_question=req.detail_question,
        )

        return {
            "prompt": prompt
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "프롬프트 생성 중 오류가 발생했습니다: "
                + str(e)
            ),
        )


# =========================================================
# REPORT API
# =========================================================

@app.post("/api/report")
def generate_report(
    req: ReportRequest
):

    if not ANTHROPIC_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "서버에 ANTHROPIC_API_KEY "
                "환경변수가 설정되지 않았습니다."
            ),
        )


    try:

        prompt = build_report_prompt(
            chart=req.chart,
            user_input=req.user_input,
            detail_level=req.detail_level,
            detail_options=req.detail_options,
            detail_question=req.detail_question,
        )


        # -------------------------------------------------
        # 토큰 수
        # -------------------------------------------------

        if req.detail_level == "detailed":

            max_tokens = 16000

        else:

            max_tokens = 3000


        # -------------------------------------------------
        # Anthropic API
        # -------------------------------------------------

        message = anthropic_client.messages.create(

            model="claude-sonnet-5",

            max_tokens=max_tokens,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )


        # -------------------------------------------------
        # 응답 텍스트
        # -------------------------------------------------

        text = "".join(

            block.text

            for block in message.content

            if block.type == "text"

        )


        return {
            "report": text
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"리포트 생성 실패: {str(e)}"
            ),
        )
