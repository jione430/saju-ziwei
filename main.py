"""
사주 × 자미두수 계산 엔진 백엔드
- 사주(四柱八字): lunar_python (6tail 만세력 포팅, 절기 기준)
- 자미두수(紫微斗數): py-iztro (삼합파, iztro 알고리즘 포팅)
- 진태양시 보정: 127.5°E(한반도 지리적 중심) 기준
- AI 리포트: Anthropic API를 서버에서 호출 (API 키는 환경변수로 보관, 클라이언트에 노출 안 함)
"""
import threading
threading.stack_size(64 * 1024 * 1024)  # py-iztro(pythonmonkey/SpiderMonkey)가 워커 스레드에서
                                          # "too much recursion" 오류를 내는 문제 해결용 (기본 스택 크기 부족)

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timedelta

from lunar_python import Solar
from py_iztro import Astro
import anthropic

from cities import CITY_LONGITUDE, TRUE_SOLAR_REFERENCE_LONGITUDE

app = FastAPI(title="사주×자미두수 계산 엔진 API", version="0.1.0")

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 프로토타입 단계 CORS: 실제 배포 시 프론트엔드 도메인으로 제한할 것
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GAN_TO_WUXING = {'甲':'木','乙':'木','丙':'火','丁':'火','戊':'土','己':'土','庚':'金','辛':'金','壬':'水','癸':'水'}
ZHI_TO_WUXING = {'子':'水','丑':'土','寅':'木','卯':'木','辰':'土','巳':'火','午':'火','未':'土','申':'金','酉':'金','戌':'土','亥':'水'}


class ChartRequest(BaseModel):
    name: str
    birth_date: str = Field(..., description="YYYY-MM-DD, 그레고리력(양력) 입력 기준")
    birth_time: Optional[str] = Field(None, description="HH:MM, time_unknown=True면 생략 가능")
    time_unknown: bool = False
    birth_city: str
    gender: Literal["여성", "남성"]
    calendar_type: Literal["양력", "음력"] = "양력"
    is_leap_month: bool = False  # calendar_type=음력일 때만 사용


def get_longitude(city: str) -> float:
    """도시명으로 경도 조회. 목록에 없으면 기준 경도(보정 0)로 폴백."""
    for key, lon in CITY_LONGITUDE.items():
        if key in city or city in key:
            return lon
    return TRUE_SOLAR_REFERENCE_LONGITUDE


def true_solar_time_correction(dt: datetime, longitude: float) -> datetime:
    """127.5E 기준 진태양시 보정. 경도가 기준보다 동쪽이면 시간을 앞당기고, 서쪽이면 늦춘다."""
    offset_minutes = (TRUE_SOLAR_REFERENCE_LONGITUDE - longitude) * 4
    return dt - timedelta(minutes=offset_minutes)


def hour_to_ziwei_index(hour: int, minute: int) -> int:
    """py-iztro의 시진 인덱스 규칙에 맞춤 (0=조자시 00~01시 ... 12=야자시 23~24시)."""
    h = hour + minute / 60
    boundaries = [
        (0, 1, 0), (1, 3, 1), (3, 5, 2), (5, 7, 3), (7, 9, 4), (9, 11, 5),
        (11, 13, 6), (13, 15, 7), (15, 17, 8), (17, 19, 9), (19, 21, 10),
        (21, 23, 11), (23, 24, 12),
    ]
    for start, end, idx in boundaries:
        if start <= h < end:
            return idx
    return 0


@app.post("/api/chart")
async def calculate_chart(req: ChartRequest):
    if req.calendar_type == "음력":
        # 음력 입력은 lunar_python의 Lunar.fromYmd 경로로 별도 처리 필요.
        # 1차 버전은 양력 입력만 지원, 음력 입력은 다음 단계에서 추가.
        raise HTTPException(status_code=400, detail="음력 입력은 아직 지원되지 않습니다. 양력으로 입력해 주세요.")

    try:
        y, mo, d = map(int, req.birth_date.split("-"))
    except ValueError:
        raise HTTPException(status_code=400, detail="birth_date 형식이 올바르지 않습니다 (YYYY-MM-DD)")

    if req.time_unknown or not req.birth_time:
        h, mi = 12, 0  # 명식 계산 최소 요건 충족용 임시값. 시주/시궁은 응답에서 제외됨.
    else:
        try:
            h, mi = map(int, req.birth_time.split(":"))
        except ValueError:
            raise HTTPException(status_code=400, detail="birth_time 형식이 올바르지 않습니다 (HH:MM)")

    longitude = get_longitude(req.birth_city)
    raw_dt = datetime(y, mo, d, h, mi)
    corrected_dt = true_solar_time_correction(raw_dt, longitude)

    # ---- 사주 계산 ----
    solar = Solar.fromYmdHms(corrected_dt.year, corrected_dt.month, corrected_dt.day,
                              corrected_dt.hour, corrected_dt.minute, 0)
    lunar = solar.getLunar()
    bazi = lunar.getEightChar()

    pillars = {
        "year": {"gan": bazi.getYearGan(), "zhi": bazi.getYearZhi()},
        "month": {"gan": bazi.getMonthGan(), "zhi": bazi.getMonthZhi()},
        "day": {"gan": bazi.getDayGan(), "zhi": bazi.getDayZhi()},
    }
    if not req.time_unknown and req.birth_time:
        pillars["time"] = {"gan": bazi.getTimeGan(), "zhi": bazi.getTimeZhi()}
    else:
        pillars["time"] = None

    wx_count = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
    for key, p in pillars.items():
        if p is None:
            continue
        wx_count[GAN_TO_WUXING[p["gan"]]] += 1
        wx_count[ZHI_TO_WUXING[p["zhi"]]] += 1

    # ---- 자미두수 계산 ----
    gender_cn = "女" if req.gender == "여성" else "男"
    if req.time_unknown or not req.birth_time:
        ziwei_result = None  # 시간 미상 시 자미두수는 시궁 확정 불가 -> 프론트에서 "시간 입력 필요" 안내
    else:
        hour_idx = hour_to_ziwei_index(corrected_dt.hour, corrected_dt.minute)
        astro = Astro()
        r = astro.by_solar(f"{corrected_dt.year}-{corrected_dt.month}-{corrected_dt.day}",
                            hour_idx, gender_cn, True, "zh-CN")
        palaces = []
        for p in r.palaces:
            palaces.append({
                "name": p.name,
                "branch": p.earthly_branch,
                "stem": p.heavenly_stem,
                "major_stars": [{"name": s.name, "brightness": s.brightness, "mutagen": s.mutagen}
                                 for s in p.major_stars],
                "minor_stars": [s.name for s in p.minor_stars],
                "is_body": p.is_body_palace,
                "decadal": p.decadal.range,
            })
        ziwei_result = {
            "soul_palace": r.earthly_branch_of_soul_palace,
            "body_palace": r.earthly_branch_of_body_palace,
            "five_elements_class": r.five_elements_class,
            "soul_star": r.soul,
            "body_star": r.body,
            "palaces": palaces,
        }

    return {
        "input": {
            "date": req.birth_date,
            "time": "모름" if req.time_unknown else req.birth_time,
            "gender": req.gender,
            "place": req.birth_city,
        },
        "correction": {
            "longitude_used": longitude,
            "reference_longitude": TRUE_SOLAR_REFERENCE_LONGITUDE,
            "offset_minutes": round((TRUE_SOLAR_REFERENCE_LONGITUDE - longitude) * 4, 1),
            "corrected_time": corrected_dt.strftime("%Y-%m-%d %H:%M") if not req.time_unknown else None,
        },
        "lunar_date": f"{lunar.getYear()}년 {lunar.getMonth()}월 {lunar.getDay()}일",
        "bazi": pillars,
        "wuxing_count": wx_count,
        "day_master": bazi.getDayGan(),
        "ziwei": ziwei_result,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------- AI 리포트 생성 ----------------

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


class ReportRequest(BaseModel):
    chart: dict  # /api/chart 응답 전체
    user_input: dict  # {name, gender, date, time, city, preQuestion}
    detail_level: Literal["summary", "detailed"] = "summary"


def build_report_prompt(chart: dict, user_input: dict, detail_level: str) -> str:
    has_ziwei = chart.get("ziwei") is not None
    pre_question = user_input.get("preQuestion") or ""

    length_rule = (
        "12. 분량: 전체 A4 기준 5~20페이지 상당(공백 포함 약 6,000~18,000자)의 상세 리포트로 작성한다. "
        "각 섹션은 최소 3~5문단 이상, 구체적인 근거(어떤 궁/주성/간지/십성 때문에 이렇게 해석했는지)를 매번 명시하며 채운다."
        if detail_level == "detailed" else
        "12. 분량: 화면에서 빠르게 읽을 수 있는 요약본으로, 전체 1,200~2,000자 내외로 작성한다. "
        "각 섹션은 1~2문단, 핵심만 짧고 명확하게 전달한다."
    )

    return f"""당신은 명리학(사주)과 자미두수에 모두 정통한 해석가입니다.
아래 입력된 구조화 데이터는 이미 검증된 계산 엔진이 산출한 결과이며, 당신은 이 데이터를 계산하지 않고 해석만 수행합니다.

[사주 데이터]
{chart.get('bazi')} / 오행분포: {chart.get('wuxing_count')} / 일간: {chart.get('day_master')}

[자미두수 데이터]
{chart.get('ziwei') if has_ziwei else '태어난 시간을 몰라 자미두수는 계산되지 않음 - 사주만으로 해석할 것'}

[사용자 기본정보]
이름={user_input.get('name')}, 성별={user_input.get('gender')}, 생년월일={user_input.get('date')}, 시간={user_input.get('time')}, 출생지={user_input.get('city')}
{f'''
[사용자가 미리 남긴 질문]
"{pre_question}"
''' if pre_question else ''}

[매핑 규칙]
{MAPPING_TABLE if has_ziwei else '(자미두수 데이터 없음 - 매핑/교차해석 생략, 사주 단독 해석으로 진행)'}

[작성 규칙]
1. 전문용어는 처음 등장 시 괄호로 쉬운 설명을 덧붙인다.
2. {'위 매핑 규칙에 따라 사주와 자미두수를 항상 같은 주제 안에서 교차 비교한다.' if has_ziwei else '자미두수 데이터가 없으므로 사주 데이터만으로 해석하고, 시간을 알면 더 정확한 자미두수 교차분석이 가능하다는 점을 리포트 서두에 안내한다.'}
3. 두 체계가 같은 신호를 주면 "두 체계 모두 ~라고 말합니다"로 강조하고, 다른 신호를 주면 양쪽을 있는 그대로 제시한다. 억지로 하나로 통일하지 않는다.
4. 건강/사고/이혼 등 민감한 주제는 "경향이 있다/주의가 필요한 시기다" 수준으로만 서술하고 확정적 예언을 하지 않는다.
5. 의학적 진단, 법률·투자 자문으로 해석될 수 있는 문장은 쓰지 않는다.
6. "인생의 진짜 황금기"와 "향후 10년 대운 분석"은 반드시 사주 대운(大運)과 자미두수 대한(大限)을 나이대별로 겹쳐서 구체적인 나이 구간을 짚어가며 서술한다. 나이 구간을 쓸 때 물결표(~)를 두 번 겹쳐 쓰지 않는다 (마크다운 취소선으로 오해될 수 있음).
7. "향후 10년 디테일 운세"는 앞으로의 10년을 2~3년 단위로 끊어서 각 구간에 어떤 흐름이 오는지 구체적으로 서술한다.
8. "개운법"은 미신적이거나 값비싼 소비를 유도하는 내용 없이, 색상·방향·습관·마음가짐처럼 부담 없이 실천할 수 있는 조언 위주로 3~5개 제시한다.
9. 마크다운 ## 제목을 사용해 다음 섹션 순서를 그대로 따른다:
   한눈에 보는 요약 / 타고난 기질 / 인생의 진짜 황금기 / 연애운 / 재물운 / 직업운 / 건강운 / 개운법 / 향후 10년 디테일 운세 / 향후 10년 대운 분석 / 사주와 자미두수: 일치와 차이 {'/ 질문에 대한 답변' if pre_question else ''} / 종합 조언
10. {'"질문에 대한 답변" 섹션에서는 사용자가 남긴 질문에 대해 위에서 분석한 내용을 근거로 구체적으로 답변한다.' if pre_question else '사용자가 남긴 질문이 없으므로 질문 답변 섹션은 생략한다.'}
11. 톤: 친근하지만 가볍지 않게, 확신을 주되 단정하지 않게. 일반인도 쉽게 이해할 수 있도록.
{length_rule}
13. 분량을 채우기 위해 같은 문장이나 표현을 반복하지 않는다. 각 섹션은 서로 다른 데이터 근거와 새로운 정보를 제공해야 한다.
14. 마크다운 취소선(~~텍스트~~)과 구분선(---)은 절대 사용하지 않는다. 섹션 구분은 오직 ## 제목으로만 한다.

지금부터 리포트를 작성하세요."""


@app.post("/api/prompt")
def generate_prompt(req: ReportRequest):
    """API 키/과금 없이, 프롬프트 텍스트만 만들어서 돌려준다.
    사용자가 이 텍스트를 복사해서 claude.ai 등에 직접 붙여넣어 무료로 해석을 받는 용도."""
    prompt = build_report_prompt(req.chart, req.user_input, req.detail_level)
    return {"prompt": prompt}


@app.post("/api/report")
def generate_report(req: ReportRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="서버에 ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")

    prompt = build_report_prompt(req.chart, req.user_input, req.detail_level)
    max_tokens = 16000 if req.detail_level == "detailed" else 3000

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return {"report": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 실패: {str(e)}")
