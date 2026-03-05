# /mnt/data/app.py
# ------------------------------------------------------------
# 🎮 Thinking Gym — Game Planner Edition (Full Script)
# - Step 1~4 화면 포맷(메타/3파트 요약/키워드/기사기반 4문항 + 선택 렌즈 질문)
# - Question Object: intent 포함
# - RSS '연관 기사' 추천: tags overlap + 반대 성향(카테고리 opposite)
# - 저장: questions(객체) + answers([{question_id, answer}]) + planner_summary_json
# - 평가(Planner): 질문별 점수 + 모범답안 + 레이다 5축 + 데일리 인사이트
# ------------------------------------------------------------

import json
import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta

import requests
import trafilatura
import streamlit as st
import feedparser
from github import Github
from dateutil import tz
from openai import OpenAI

KST = tz.gettz("Asia/Seoul")


# ---------------------------
# Utils
# ---------------------------
def _now_kst_iso() -> str:
    return datetime.now(tz=KST).isoformat(timespec="seconds")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_json_from_text(text: str):
    """Extract first JSON object from text (best-effort)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _ensure_token_usage():
    if "token_usage" not in st.session_state:
        st.session_state["token_usage"] = {
            "article_id": None,
            "summary": None,
            "questions": None,
            "eval": None,
        }


def _usage_to_str(u):
    if not u:
        return "아직 없음"
    if isinstance(u, dict):
        pt = u.get("prompt_tokens")
        ct = u.get("completion_tokens")
        tt = u.get("total_tokens")
    else:
        pt = getattr(u, "prompt_tokens", None)
        ct = getattr(u, "completion_tokens", None)
        tt = getattr(u, "total_tokens", None)
    return f"prompt={pt}, completion={ct}, total={tt}"


# ---------------------------
# Tag extraction (improved)
# - used for RSS related recommendations
# ---------------------------
_KO_JOSA_SUFFIXES = [
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "에게서",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로", "랑", "이나", "나",
    "들", "께", "밖에", "마저", "조차", "라도",
]

_STOPWORDS = set(
    """
    기사 요약 내용 관련 오늘 이번 해당 통해 대한 그리고 하지만 또한
    있다 없다 된다 했다 한다 한 것 등 수 로 의 에서 으로 에게 보다
    the a an to of in on for with from at by as is are was were be been
    this that these those it its their them they you your we our

    game games gaming gamer gamers
    게임 게임들 게임의 게임은 게임이 게임을 게임에 게임에서 게임으로 게임과 게임와
    good group gamesindustry biz now newly wants invest bought back catalogues catalogue
    """.split()
)


def _strip_korean_josa(token: str) -> str:
    if not token:
        return token
    t = token
    for suf in sorted(_KO_JOSA_SUFFIXES, key=len, reverse=True):
        if len(t) >= 3 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t


def _normalize_tokens_keep_hyphen(text: str):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"[^0-9A-Za-z가-힣\-\s]+", " ", text)  # keep hyphen
    parts = [p.strip() for p in text.split() if p.strip()]
    cleaned = []
    for p in parts:
        t = p.lower().strip("-")
        t = _strip_korean_josa(t)
        t = t.strip("-")
        if t and len(t) >= 2:
            cleaned.append(t)
    return cleaned


def extract_tags_from_text(text: str, max_tags: int = 7) -> list:
    """
    Lightweight tags extractor:
    - strips simple Korean josa
    - removes stopwords
    - keeps hyphen tokens like 'triple-i'
    - de-duplicates and reduces near-duplicates (games/game)
    """
    tokens = _normalize_tokens_keep_hyphen(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
    if not tokens:
        return []

    freq = Counter(tokens)

    # small canonicalization for game/games
    def canon(t: str) -> str:
        if t in ("game", "games"):
            return "games"
        return t

    seen = set()
    out = []
    for tok, _c in freq.most_common(60):
        t = canon(tok)
        if t in seen:
            continue
        if t in _STOPWORDS:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tags:
            break
    return out


def jaccard(a, b) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


# ---------------------------
# Config
# ---------------------------
def load_cfg():
    return {
        "token": st.secrets["GITHUB_TOKEN"],
        "repo_name": st.secrets["REPO_NAME"],
        "branch": st.secrets.get("BRANCH", "main"),
        "path_articles": st.secrets.get("ARTICLES_PATH", "data/articles.json"),
        "path_sessions": st.secrets.get("SESSIONS_PATH", "data/sessions.json"),
        "path_sources": st.secrets.get("SOURCES_PATH", "data/sources.json"),
        "path_eval": st.secrets.get("EVAL_PATH", "data/eval.json"),
    }


# ---------------------------
# GitHub JSON Store
# ---------------------------
class GitHubJSONStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gh = Github(cfg["token"])
        self.repo = self.gh.get_repo(cfg["repo_name"])
        self.branch = cfg["branch"]

    def _get(self, path):
        try:
            f = self.repo.get_contents(path, ref=self.branch)
            content = f.decoded_content.decode("utf-8")
            return json.loads(content), f.sha
        except Exception:
            return None, None

    def _put(self, path, obj, message):
        data = json.dumps(obj, ensure_ascii=False, indent=2)
        existing, sha = self._get(path)
        if sha:
            self.repo.update_file(path, message, data, sha, branch=self.branch)
        else:
            self.repo.create_file(path, message, data, branch=self.branch)

    def read_json(self, path, default):
        obj, _ = self._get(path)
        return obj if obj is not None else default

    def write_json(self, path, obj, message):
        self._put(path, obj, message)


@st.cache_resource
def get_store():
    cfg = load_cfg()
    return GitHubJSONStore(cfg), cfg


# ---------------------------
# Default data
# ---------------------------
def default_sources():
    return {
        "categories": [
            {"name": "Game Industry", "rss": []},
            {"name": "Business", "rss": []},
            {"name": "AI/Tech", "rss": []},
        ],
        "manual_urls": [],
    }


def default_eval():
    # legacy rule-scoring (kept for backward compatibility / simple metric)
    return {
        "criteria": [
            {"name": "원인 분석", "desc": "현상의 근본 원인을 설명했는가", "weight": 20},
            {"name": "시간 축", "desc": "단기/중기/장기 구분이 있는가", "weight": 20},
            {"name": "2차 파급", "desc": "간접 영향까지 고려했는가", "weight": 20},
            {"name": "리스크", "desc": "부정 시나리오/불확실성을 언급했는가", "weight": 20},
            {"name": "게임 연결", "desc": "게임 산업 전략과 연결했는가", "weight": 20},
        ]
    }


def ensure_bootstrap(store, cfg):
    sources = store.read_json(cfg["path_sources"], None)
    if sources is None:
        store.write_json(cfg["path_sources"], default_sources(), "bootstrap sources")

    eval_cfg = store.read_json(cfg["path_eval"], None)
    if eval_cfg is None:
        store.write_json(cfg["path_eval"], default_eval(), "bootstrap eval")

    articles = store.read_json(cfg["path_articles"], None)
    if articles is None:
        store.write_json(cfg["path_articles"], {"articles": []}, "bootstrap articles")

    sessions = store.read_json(cfg["path_sessions"], None)
    if sessions is None:
        store.write_json(cfg["path_sessions"], {"sessions": []}, "bootstrap sessions")


# ---------------------------
# OpenAI Client
# ---------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------------------------
# Article fetch
# ---------------------------
def fetch_article_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ThinkingGym/2.0)"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
    return (text or "").strip()


# ---------------------------
# RSS Collect
# ---------------------------
def collect_from_rss(store, cfg, category_name, rss_url, max_items=10):
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})
    existing = {a["id"]: a for a in articles_db["articles"]}

    feed = feedparser.parse(rss_url)
    new_count = 0

    for entry in feed.entries[:max_items]:
        link = entry.get("link", "").strip()
        title = entry.get("title", "").strip()
        published = entry.get("published", "") or entry.get("updated", "")

        if not link or not title:
            continue

        aid = _sha1(link)
        if aid in existing:
            continue

        tags = extract_tags_from_text(title, max_tags=7)

        existing[aid] = {
            "id": aid,
            "title": title,
            "url": link,
            "category": category_name,
            "published": published,
            "collected_at": _now_kst_iso(),
            "used_in_session": False,
            "added_by": "auto",
            "tags": tags,
        }
        new_count += 1

    articles_db["articles"] = sorted(existing.values(), key=lambda x: x.get("collected_at", ""), reverse=True)
    store.write_json(cfg["path_articles"], articles_db, f"collect rss ({category_name})")
    return new_count


# ---------------------------
# Planner Summary (Step 1~2)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def make_planner_summary(article: dict) -> dict:
    """
    Output JSON:
    - article_meta
    - summary: market_insight/system_tech/ux
    - keywords: 3
    """
    url = article.get("url", "")
    title = article.get("title", "")
    category = article.get("category", None)
    published = article.get("published", None)

    body = fetch_article_text(url)
    if not body:
        return {
            "article_meta": {"title": title, "url": url, "published_at": published, "category": category},
            "summary": {"market_insight": [], "system_tech": [], "ux": []},
            "keywords": [],
            "caveats": ["요약 실패: 본문 추출 불가(차단/구조 문제 가능)"],
        }

    body = body[:14000]

    prompt = f"""
[ROLE]
너는 '시니어 게임 디자이너 및 콘텐츠 전략가'다.
언어는 한국어 중심, 톤은 냉철하고 분석적이지만 성장 코칭 스타일.
GDD, DAU, Retention, LTV, Economy Balance, UX/UI, Core Loop 등의 용어를 적절히 사용한다.

[CONTEXT]
- 기사 제목: {title}
- 기사 URL: {url}
- 발행일(있으면): {published}
- 카테고리(있으면): {category}
- 기사 원문:
{body}

[INSTRUCTIONS]
1) 기사를 게임 기획자 관점으로 3파트 요약:
   - 시장 인사이트(Market Insight)
   - 시스템 및 기술(System & Tech)
   - 유저 경험(UX)
2) 각 파트는 2~4문장(또는 bullet 2~4개) 수준으로 "바로 기획에 쓰일 관점"으로 작성.
3) 핵심 키워드 3개를 뽑고, why_it_matters(근거 1~2문장) 포함.
4) 기사에 없는 내용 단정 금지. 불확실하면 '알 수 없음' 표기.
5) 아래 JSON 스키마대로만 출력(그 외 텍스트 금지).

[OUTPUT JSON SCHEMA]
{{
  "article_meta": {{
    "title": "string",
    "url": "string",
    "published_at": "string|null",
    "category": "string|null"
  }},
  "summary": {{
    "market_insight": ["string","string","string"],
    "system_tech": ["string","string","string"],
    "ux": ["string","string","string"]
  }},
  "keywords": [
    {{"keyword":"string","why_it_matters":"string"}},
    {{"keyword":"string","why_it_matters":"string"}},
    {{"keyword":"string","why_it_matters":"string"}}
  ],
  "caveats": ["string"]
}}
""".strip()

    client = get_openai_client()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    _ensure_token_usage()
    try:
        st.session_state["token_usage"]["summary"] = getattr(res, "usage", None)
    except Exception:
        pass

    parsed = _safe_json_from_text(res.choices[0].message.content)
    if not parsed:
        return {
            "article_meta": {"title": title, "url": url, "published_at": published, "category": category},
            "summary": {"market_insight": [], "system_tech": [], "ux": []},
            "keywords": [],
            "caveats": ["AI 요약 JSON 파싱 실패", res.choices[0].message.content[:400]],
        }
    return parsed


def session_tags_from_summary(planner_summary: dict) -> list:
    """
    Prefer keywords as tags. Fallback to token tags from title + summary.
    """
    tags = []
    try:
        kws = planner_summary.get("keywords") or []
        tags = [k.get("keyword", "").strip() for k in kws if isinstance(k, dict)]
        tags = [t for t in tags if t]
    except Exception:
        tags = []

    if tags:
        # normalize tags lightly
        out = []
        seen = set()
        for t in tags:
            tt = t.strip()
            if not tt:
                continue
            low = tt.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(tt)
        return out[:7]

    # fallback: title + summary text tokens
    meta = planner_summary.get("article_meta") or {}
    title = meta.get("title", "")
    summ = planner_summary.get("summary") or {}
    blob = title + " " + " ".join((summ.get("market_insight") or []) + (summ.get("system_tech") or []) + (summ.get("ux") or []))
    return extract_tags_from_text(blob, max_tags=7)


# ---------------------------
# Planner Questions (Step 3~4)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def generate_planner_questions(planner_summary_json: dict, selected_lenses: list) -> dict:
    """
    Create:
    - 4 article-based questions with points 10/20/30/40
    - optional role-lens questions (each 10) grounded to article
    Question Object includes intent.
    """
    prompt = f"""
[ROLE]
너는 '시니어 게임 디자이너 및 콘텐츠 전략가'다.
기사 요약을 바탕으로, 세션마다 달라지는 "기사 밀착형" 질문을 만든다.

[INPUT]
- planner_summary_json:
{json.dumps(planner_summary_json, ensure_ascii=False)}

- selected_lenses:
{json.dumps(selected_lenses, ensure_ascii=False)}
# 가능한 값: "balance_abuse" | "psych_retention" | "bm_ops"
# 없으면 []

[INSTRUCTIONS]

0) 매우 중요: 반드시 한국어로만 작성한다. (prompt/intent/rubric/answer_format_hint/reference 모두 한국어)
   - 영문 고유명사(회사명/게임명/제품명)는 원문 그대로 유지 가능
   - 그 외 문장/설명은 한국어

1) 기사 기반 질문 4개(필수) 생성:
   - 사실 확인형: type="fact", points=10
   - 논리 추론형: type="logic", points=20
   - 시스템 분석형: type="system", points=30
   - 역설계/적용형: type="reverse_apply", points=40

2) 선택된 렌즈가 있으면, 렌즈마다 1개씩 "기사와 어울리는" 심화 질문을 추가:
   - type="role_lens", points=10, role_key에 렌즈값 저장
   - 질문은 planner_summary_json의 keywords 또는 summary 섹션(시장/시스템/UX)과 반드시 연결되어야 함
   - '악용 시나리오는?' 같은 일반 템플릿 문장만 쓰면 실패(기사 요소를 문장에 포함)

3) 모든 질문은 아래 Question Object 필드를 포함:
   - question_id (string)
   - type ("fact"|"logic"|"system"|"reverse_apply"|"role_lens")
   - points (int)
   - prompt (string)
   - intent (string)  # 이 질문을 왜 던졌는지 1문장
   - answer_format_hint (string)
   - rubric (3~5개 string 배열)
   - tags (레이다용; 아래 5개 중 관련된 것 2~4개)
     - SystemicThinking, UserCentric, BusinessAcumen, DetailLogic, TrendInsight
   - role_key ("balance_abuse"|"psych_retention"|"bm_ops"|null)
   - grounding (object)
     - from_keywords (array)
     - from_summary_section ("market_insight"|"system_tech"|"ux"|null)
     - reference (string)

4) 출력은 반드시 JSON만.

[OUTPUT JSON SCHEMA]
{{
  "questions": [
    {{
      "question_id":"string",
      "type":"fact|logic|system|reverse_apply|role_lens",
      "points": 10,
      "prompt":"string",
      "intent":"string",
      "answer_format_hint":"string",
      "rubric":["string","string","string"],
      "tags":["SystemicThinking","UserCentric"],
      "role_key": null,
      "grounding": {{
        "from_keywords": ["string"],
        "from_summary_section": "market_insight|system_tech|ux|null",
        "reference": "string"
      }}
    }}
  ],
  "total_points": 0
}}
""".strip()

    client = get_openai_client()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    _ensure_token_usage()
    try:
        st.session_state["token_usage"]["questions"] = getattr(res, "usage", None)
    except Exception:
        pass

    text = res.choices[0].message.content
    parsed = _safe_json_from_text(text)
    if not parsed or "questions" not in parsed:
        return {"questions": [], "total_points": 0, "error": "AI 질문 JSON 파싱 실패", "raw": text[:500]}

    try:
        parsed["total_points"] = int(sum(int(q.get("points", 0)) for q in parsed["questions"]))
    except Exception:
        parsed["total_points"] = 0
    return parsed


# ---------------------------
# Planner Evaluation (Step 5~8)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def evaluate_planner(planner_summary_json: dict, questions: list, answers: list, recent_sessions: list | None):
    """
    Returns JSON with:
    - per_question scoring (0..points)
    - model answers
    - radar 5 metrics (1..10)
    - daily insight (trend/weakness/task)
    """
    prompt = f"""
[ROLE]
너는 '시니어 게임 디자이너 및 콘텐츠 전략가'다.
사용자 답변을 냉정하게 채점하고, 실무 기획 역량이 오르도록 코칭한다.

[INPUT]
- planner_summary_json:
{json.dumps(planner_summary_json, ensure_ascii=False)}

- questions:
{json.dumps(questions, ensure_ascii=False)}

- answers:
{json.dumps(answers, ensure_ascii=False)}

- recent_sessions (최근 3~10개, 없으면 null):
{json.dumps(recent_sessions, ensure_ascii=False) if recent_sessions is not None else "null"}

[SCORING RULES]
1) 질문별 점수: 0 ~ 해당 질문 points 범위에서 채점.
2) role_lens(type="role_lens") 질문은 다음 분해 채점(총 10점 내):
   - 기사 연결성(0~3): keywords/summary 근거로 연결했나
   - 직무 관점 품질(0~4): 렌즈 관점(악용/리텐션/BM)의 깊이
   - 구체성(0~3): 지표/루프/싱크/예외처리 등 설계 디테일
3) 답변이 의미없거나 매우 짧으면(예: 80자 미만 or 'asdf', '테스트') 해당 질문 점수는 0에 가깝게.
4) 각 질문마다:
   - feedback: 좋은 점/부족한 점/개선 방향(짧고 날카롭게)
   - model_answer: 모범 답안(핵심 구조를 보여주되 길지 않게)
5) 레이다 5지표(1~10):
   - Systemic Thinking, User-Centric, Business Acumen, Detail & Logic, Trend Insight
6) 데일리 인사이트:
   - 성장 확인: 최근 3일간(가능하면) 무엇이 올랐는지
   - 약점 보완: 가장 약한 축과 이유
   - 추천 과제: 다음 세션에서 다루면 좋은 기사 주제 1개
7) 출력은 반드시 JSON만.

[OUTPUT JSON SCHEMA]
{{
  "question_results": [
    {{
      "question_id": "string",
      "score": 0,
      "max_score": 0,
      "feedback": "string",
      "model_answer": "string"
    }}
  ],
  "total_score": 0,
  "radar": {{
    "SystemicThinking": 1,
    "UserCentric": 1,
    "BusinessAcumen": 1,
    "DetailLogic": 1,
    "TrendInsight": 1
  }},
  "daily_insight": {{
    "growth_check": "string",
    "weakness_fix": "string",
    "recommended_next_topic": "string"
  }}
}}
""".strip()

    client = get_openai_client()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
    )

    _ensure_token_usage()
    try:
        st.session_state["token_usage"]["eval"] = getattr(res, "usage", None)
    except Exception:
        pass

    text = res.choices[0].message.content
    parsed = _safe_json_from_text(text)
    if not parsed:
        return {
            "question_results": [],
            "total_score": 0,
            "radar": {
                "SystemicThinking": 1,
                "UserCentric": 1,
                "BusinessAcumen": 1,
                "DetailLogic": 1,
                "TrendInsight": 1,
            },
            "daily_insight": {
                "growth_check": "평가 JSON 파싱 실패",
                "weakness_fix": "원문을 확인하세요.",
                "recommended_next_topic": "게임 퍼블리싱/포트폴리오 전략",
            },
            "raw": text[:500],
        }
    return parsed


# ---------------------------
# Related Articles Recommendation (RSS)
# ---------------------------
def get_opposite_category(cat: str) -> str:
    """
    Simple opposite mapping:
    - AI/Tech ↔ Business
    - Game Industry ↔ Business (or AI/Tech). choose Business as "business angle"
    """
    if cat == "AI/Tech":
        return "Business"
    if cat == "Business":
        return "AI/Tech"
    if cat == "Game Industry":
        return "Business"
    return "Business"


def recommend_related_articles(store, cfg, current_article: dict, current_tags: list, limit_each: int = 5):
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})
    candidates = [a for a in articles_db.get("articles", []) if a.get("id") != current_article.get("id")]

    # ensure each candidate has tags
    for a in candidates:
        if not isinstance(a.get("tags"), list) or len(a.get("tags", [])) == 0:
            a["tags"] = extract_tags_from_text(a.get("title", ""), max_tags=7)

    # 1) overlap picks
    scored = []
    for a in candidates:
        sim = jaccard(current_tags, a.get("tags", []))
        if sim > 0:
            scored.append((sim, a))
    scored.sort(key=lambda x: (x[0], x[1].get("collected_at", "")), reverse=True)
    overlap_pick = [a for _, a in scored[:limit_each]]

    # 2) opposite category picks (prefer low overlap but still relevant)
    opp_cat = get_opposite_category(current_article.get("category", ""))
    opp = [a for a in candidates if a.get("category") == opp_cat]
    opp_scored = []
    for a in opp:
        sim = jaccard(current_tags, a.get("tags", []))
        opp_scored.append((1.0 - sim, a))
    opp_scored.sort(key=lambda x: (x[0], x[1].get("collected_at", "")), reverse=True)
    opposite_pick = [a for _, a in opp_scored[:limit_each]]

    return overlap_pick, opposite_pick


# ---------------------------
# Save Session
# ---------------------------
def save_session(
    store,
    cfg,
    article: dict,
    planner_summary_json: dict,
    session_tags: list,
    questions: list,
    answers: list,
    eval_pack: dict | None,
):
    sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})

    # build a compact final summary
    # pick first 2-3 answers
    answer_texts = [a.get("answer", "").strip() for a in answers if a.get("answer")]
    final_summary = ""
    if answer_texts:
        final_summary = "\n".join([f"- {t[:240]}" for t in answer_texts[:3]])

    session = {
        "session_id": _sha1(article.get("id", "") + _now_kst_iso()),
        "date": _now_kst_iso(),
        "article": {
            "id": article.get("id", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "category": article.get("category", ""),
            "published": article.get("published", ""),
        },
        "tags": session_tags or [],
        "planner_summary_json": planner_summary_json,
        "questions": questions,
        "answers": answers,

        # evaluation pack (planner)
        "eval_pack": eval_pack or {},
        "final_summary": final_summary,
    }

    sessions_db["sessions"].insert(0, session)
    store.write_json(cfg["path_sessions"], sessions_db, "add planner session")

    # mark article used + keep tags
    for a in articles_db.get("articles", []):
        if a.get("id") == article.get("id"):
            a["used_in_session"] = True
            if not isinstance(a.get("tags"), list) or len(a.get("tags", [])) == 0:
                a["tags"] = extract_tags_from_text(a.get("title", ""), max_tags=7)
    store.write_json(cfg["path_articles"], articles_db, "mark article used")

    return session


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Thinking Gym — Game Planner Edition", layout="wide")

store, cfg = get_store()
ensure_bootstrap(store, cfg)

sources = store.read_json(cfg["path_sources"], default_sources())
_ensure_token_usage()

st.title("🧠 Thinking Gym — Game Planner Edition")
st.caption("뉴스 1개로 게임 기획자 관점 사고를 훈련하고 기록을 쌓습니다. (데이터는 GitHub에 저장)")

tab_collect, tab_add, tab_list, tab_session, tab_growth, tab_settings = st.tabs(
    ["📡 자동 수집", "➕ 수동 추가", "🗂 기사 목록", "✍️ 세션", "📈 성장", "⚙️ 설정"]
)

# -------- 자동 수집 --------
with tab_collect:
    st.subheader("카테고리별 RSS 자동 수집")
    sources = store.read_json(cfg["path_sources"], default_sources())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 카테고리 & RSS")
        for cat in sources["categories"]:
            with st.expander(f"{cat['name']} (RSS {len(cat['rss'])}개)", expanded=False):
                new_rss = st.text_input(f"{cat['name']} RSS 추가", key=f"rss_add_{cat['name']}")
                if st.button(f"➕ {cat['name']}에 RSS 추가", key=f"btn_add_{cat['name']}") and new_rss:
                    cat["rss"].append(new_rss.strip())
                    store.write_json(cfg["path_sources"], sources, "add rss")
                    st.success("추가됨! (저장 완료)")
                    st.rerun()
                if cat["rss"]:
                    st.write(cat["rss"])

    with col2:
        st.markdown("### 지금 수집하기")
        cat_names = [c["name"] for c in sources["categories"]]
        sel = st.selectbox("수집할 카테고리", cat_names)
        max_items = st.slider("RSS당 가져올 글 수", 3, 30, 10)

        if st.button("🚀 수집 실행"):
            cat = next(c for c in sources["categories"] if c["name"] == sel)
            if not cat["rss"]:
                st.warning("이 카테고리에 RSS가 없습니다. 먼저 RSS를 추가하세요.")
            else:
                total_new = 0
                for url in cat["rss"]:
                    try:
                        total_new += collect_from_rss(store, cfg, sel, url, max_items=max_items)
                    except Exception as e:
                        st.error(f"RSS 수집 실패: {url}\n{e}")
                st.success(f"수집 완료! 새 글 {total_new}개")
                st.rerun()

# -------- 수동 추가 --------
with tab_add:
    st.subheader("내가 찾은 기사/블로그 글을 추가")
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})

    url = st.text_input("추가할 글 URL")
    title = st.text_input("제목(자동 추출이 어려우면 직접 입력)")
    cat_names = [c["name"] for c in sources["categories"]]
    category = st.selectbox("카테고리", cat_names)

    if st.button("➕ 기사 추가"):
        if not url or not title:
            st.warning("URL과 제목을 입력하세요.")
        else:
            aid = _sha1(url.strip())
            existing_ids = {a["id"] for a in articles_db["articles"]}
            if aid in existing_ids:
                st.info("이미 저장된 글입니다.")
            else:
                tags = extract_tags_from_text(title.strip(), max_tags=7)
                articles_db["articles"].insert(
                    0,
                    {
                        "id": aid,
                        "title": title.strip(),
                        "url": url.strip(),
                        "category": category,
                        "published": "",
                        "collected_at": _now_kst_iso(),
                        "used_in_session": False,
                        "added_by": "manual",
                        "tags": tags,
                    },
                )
                store.write_json(cfg["path_articles"], articles_db, "add manual article")
                st.success("추가 완료!")
                st.rerun()

# -------- 기사 목록 --------
with tab_list:
    st.subheader("기사 목록")
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        cat_names = ["(전체)"] + [c["name"] for c in sources["categories"]]
        filter_cat = st.selectbox("카테고리 필터", cat_names)
    with col2:
        only_unused = st.checkbox("세션 안한 글만", value=True)
    with col3:
        q = st.text_input("검색(제목)", placeholder="예: Unity, NVIDIA ...")

    filtered = []
    for a in articles_db["articles"]:
        if filter_cat != "(전체)" and a.get("category") != filter_cat:
            continue
        if only_unused and a.get("used_in_session"):
            continue
        if q and q.lower() not in a.get("title", "").lower():
            continue
        filtered.append(a)

    st.write(f"표시 중: {len(filtered)}개")

    for a in filtered[:60]:
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{a.get('title','')}**  \n{a.get('url','')}")
        cols[1].write(a.get("category", ""))
        cols[2].write("✅ 사용" if a.get("used_in_session") else "🟡 미사용")

        if st.button("이 기사로 세션 시작", key=f"start_{a['id']}"):
            st.session_state["selected_article_id"] = a["id"]
            st.session_state["planner_summary_url"] = None
            st.session_state["planner_summary"] = None
            st.session_state["planner_questions_pack"] = None
            st.session_state["last_saved_session_id"] = None
            st.session_state["token_usage"] = {"article_id": a["id"], "summary": None, "questions": None, "eval": None}
            st.rerun()

# -------- 세션 --------
with tab_session:
    st.subheader("세션 (Game Planner Edition)")

    selected_id = st.session_state.get("selected_article_id")

    def _find_article_by_id(article_id: str):
        db = store.read_json(cfg["path_articles"], {"articles": []})
        for x in db["articles"]:
            if x.get("id") == article_id:
                return x
        return None

    if not selected_id:
        st.info("먼저 ‘기사 목록’에서 ‘이 기사로 세션 시작’을 눌러주세요.")
    else:
        article = _find_article_by_id(selected_id)
        if not article:
            st.error("선택한 기사를 articles.json에서 찾지 못했습니다.")
        else:
            st.markdown(f"### 📰 {article.get('title','')}")
            st.write(article.get("url", ""))
            st.divider()

            if "OPENAI_API_KEY" not in st.secrets:
                st.error("OPENAI_API_KEY가 Streamlit Secrets에 없습니다.")
            else:
                # Step 1~2 Summary
                if st.session_state.get("planner_summary") is None or st.session_state.get("planner_summary_url") != article.get("url"):
                    st.session_state["planner_summary_url"] = article.get("url")
                    with st.spinner("Step 2 요약 생성 중..."):
                        st.session_state["planner_summary"] = make_planner_summary(article)

                planner_summary = st.session_state.get("planner_summary") or {}
                meta = planner_summary.get("article_meta") or {}
                summ = planner_summary.get("summary") or {}
                keywords = planner_summary.get("keywords") or []
                caveats = planner_summary.get("caveats") or []

                st.markdown("### 📝 Step 1. 기사 메타 데이터")
                st.write(f"**Title:** {meta.get('title','')}")
                st.write(f"**URL:** {meta.get('url','')}")
                st.write(f"**Category:** {meta.get('category', '')}")

                st.markdown("### 🔍 Step 2. AI 요약 (3파트)")
                st.markdown("**1) 시장 인사이트 (Market Insight)**")
                for x in summ.get("market_insight", []) or []:
                    st.write(f"- {x}")

                st.markdown("**2) 시스템 및 기술 (System & Tech)**")
                for x in summ.get("system_tech", []) or []:
                    st.write(f"- {x}")

                st.markdown("**3) 유저 경험 (UX)**")
                for x in summ.get("ux", []) or []:
                    st.write(f"- {x}")

                st.markdown("### 🔑 핵심 키워드 3개")
                for k in keywords:
                    if isinstance(k, dict):
                        st.write(f"- **{k.get('keyword','')}**: {k.get('why_it_matters','')}")

                if caveats:
                    st.caption("caveats: " + " / ".join(caveats[:2]))

                # tags used for related recommendation (prefer keywords)
                session_tags = session_tags_from_summary(planner_summary)
                if session_tags:
                    st.caption("tags: " + ", ".join(session_tags))

                st.divider()

                # Token usage display
                st.markdown("### 🧾 이번 세션 토큰 사용량")
                tu = st.session_state.get("token_usage", {})
                if tu.get("article_id") != selected_id:
                    u_sum = u_q = u_eval = None
                else:
                    u_sum = tu.get("summary")
                    u_q = tu.get("questions")
                    u_eval = tu.get("eval")

                c1, c2, c3 = st.columns(3)
                c1.markdown("**요약(summary)**")
                c1.caption(_usage_to_str(u_sum))
                c2.markdown("**질문 생성(questions)**")
                c2.caption(_usage_to_str(u_q))
                c3.markdown("**평가(eval)**")
                c3.caption(_usage_to_str(u_eval))

                st.divider()

                # --- START: Step3 자동 생성 + Step4 렌즈 추가 생성 UI (교체 블록) ---
                
                # (1) Step 3: 기사 기반 4문항 자동 생성
                st.markdown("### ❓ Step 3. 질문 생성 (기사 기반 4문항)")
                
                # base 질문팩이 없거나, 기사 URL이 바뀌었으면 자동 생성
                base_key_url = st.session_state.get("base_questions_url")
                if st.session_state.get("base_questions_pack") is None or base_key_url != article.get("url"):
                    st.session_state["base_questions_url"] = article.get("url")
                    with st.spinner("기사 기반 4문항 자동 생성 중..."):
                        st.session_state["base_questions_pack"] = generate_planner_questions(planner_summary, selected_lenses=[])
                
                # base 질문들
                base_pack = st.session_state.get("base_questions_pack") or {}
                base_questions = base_pack.get("questions", []) or []
                
                # 렌즈 질문(추가분) 저장소
                if "lens_questions" not in st.session_state or st.session_state.get("lens_questions_url") != article.get("url"):
                    st.session_state["lens_questions_url"] = article.get("url")
                    st.session_state["lens_questions"] = []  # 기사 변경 시 렌즈 질문 초기화
                
                lens_questions = st.session_state.get("lens_questions") or []
                
                # (2) Step 4: 선택형 렌즈(체크박스) + "렌즈 질문만" 추가 생성
                st.markdown("### 🛠️ Step 4. 선택형 질문 (Role Lens)")
                st.caption("체크 후 아래 버튼을 누르면, 기사 내용과 밀착된 '렌즈 질문'만 추가 생성됩니다. (기존 4문항은 유지)")
                
                colA, colB, colC = st.columns(3)
                with colA:
                    lens_balance = st.checkbox("밸런스/어뷰징 렌즈", value=False)
                    st.caption("(인수/퍼블리싱 시스템 도입 시 악용/어뷰징 관점)")
                with colB:
                    lens_psych = st.checkbox("심리/리텐션 렌즈", value=False)
                    st.caption("(유저 심리/동기/피드백 루프 관점)")
                with colC:
                    lens_bm = st.checkbox("BM/운영 렌즈", value=False)
                    st.caption("(인플레 방지 + 수익 기여 + 라이브 운영 관점)")
                
                selected_lenses = []
                if lens_balance: selected_lenses.append("balance_abuse")
                if lens_psych: selected_lenses.append("psych_retention")
                if lens_bm: selected_lenses.append("bm_ops")
                
                # 렌즈 질문만 추가 생성 버튼
                if st.button("➕ 렌즈 질문 추가 생성", key="btn_add_lens_questions"):
                    if not selected_lenses:
                        st.warning("렌즈를 최소 1개 선택하세요.")
                    else:
                        with st.spinner("선택한 렌즈 질문만 생성 중..."):
                            lens_pack = generate_role_lens_questions(planner_summary, selected_lenses)
                            new_lens_qs = lens_pack.get("questions", []) or []
                
                            # 중복 방지(같은 question_id)
                            existing_ids = {q.get("question_id") for q in lens_questions if isinstance(q, dict)}
                            for q in new_lens_qs:
                                if q.get("question_id") not in existing_ids:
                                    lens_questions.append(q)
                
                            st.session_state["lens_questions"] = lens_questions
                            st.success(f"렌즈 질문 {len(new_lens_qs)}개 추가 생성 완료!")
                
                # (3) 최종 질문 리스트 = base 4문항 + 렌즈 질문(추가)
                qs = base_questions + lens_questions
                total_pts = sum(int(q.get("points", 0)) for q in qs if isinstance(q, dict))
                
                st.caption(f"총 배점: {total_pts}점 (렌즈 질문은 선택 시 +10점씩 추가)")
                
                show_intent = st.checkbox("intent(출제 의도) 보기", value=True)
                
                # Answers (⚠️ key가 question_id 기반이라 기존 답변 유지됨)
                answers = []
                for qobj in qs:
                    qid = qobj.get("question_id")
                    pts = qobj.get("points", 0)
                    qtype = qobj.get("type", "")
                    st.markdown(f"#### [{pts}점] ({qtype}) {qobj.get('prompt','')}")
                    if show_intent:
                        st.caption(f"intent: {qobj.get('intent','')}")
                    hint = qobj.get("answer_format_hint") or "자유 형식"
                    st.caption(f"답변 힌트: {hint}")
                
                    ans = st.text_area("내 답변", height=150, key=f"ans_{qid}")
                    answers.append({"question_id": qid, "answer": ans})
                
                st.divider()
                
                # 저장/평가 버튼(기존 로직 유지)
                left, right = st.columns([1, 1])
                
                with left:
                    if st.button("💾 저장만", key="btn_save_only"):
                        session = save_session(
                            store=store,
                            cfg=cfg,
                            article=article,
                            planner_summary_json=planner_summary,
                            session_tags=session_tags,
                            questions=qs,
                            answers=answers,
                            eval_pack=None,
                        )
                        st.session_state["last_saved_session_id"] = session.get("session_id")
                        st.success("저장 완료! (평가 없음)")
                
                with right:
                    if st.button("🧠 평가 + 저장", key="btn_eval_and_save"):
                        sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
                        recent = sessions_db.get("sessions", [])[:10]
                        recent_min = []
                        for s in recent:
                            ep = (s.get("eval_pack") or {})
                            radar = ep.get("radar") if isinstance(ep, dict) else None
                            if radar:
                                recent_min.append({"date": s.get("date"), "radar": radar})
                        if not recent_min:
                            recent_min = None
                
                        with st.spinner("AI가 채점/피드백/모범답안을 생성 중..."):
                            eval_pack = evaluate_planner(planner_summary, qs, answers, recent_min)
                
                        session = save_session(
                            store=store,
                            cfg=cfg,
                            article=article,
                            planner_summary_json=planner_summary,
                            session_tags=session_tags,
                            questions=qs,
                            answers=answers,
                            eval_pack=eval_pack,
                        )
                        st.session_state["last_saved_session_id"] = session.get("session_id")
                        st.success("평가+저장 완료!")
                        st.json(eval_pack)

# --- END: Step3 자동 생성 + Step4 렌즈 추가 생성 UI (교체 블록) ---

                            # show eval summary
                            st.markdown("### ✅ 평가 요약")
                            st.json(eval_pack)

                            # related recommendations
                            overlap_pick, opposite_pick = recommend_related_articles(
                                store=store, cfg=cfg, current_article=article, current_tags=session_tags, limit_each=5
                            )
                            st.divider()
                            st.markdown("## 🔗 RSS '연관 기사' 추천")
                            st.caption("오늘 읽은 기사 tags[]와 겹치거나, 반대되는 성향의 기사를 추천합니다.")
                            cL, cR = st.columns(2)
                            with cL:
                                st.markdown("### ✅ tags가 겹치는 기사")
                                if not overlap_pick:
                                    st.write("(추천 없음)")
                                else:
                                    for a in overlap_pick:
                                        sim = jaccard(session_tags, a.get("tags", []))
                                        st.markdown(
                                            f"- **{a.get('title','')}**  \n  {a.get('url','')}  \n  _overlap={sim:.2f}, category={a.get('category','')}_"
                                        )
                            with cR:
                                st.markdown("### 🌓 반대 성향 기사")
                                if not opposite_pick:
                                    st.write("(추천 없음)")
                                else:
                                    for a in opposite_pick:
                                        sim = jaccard(session_tags, a.get("tags", []))
                                        st.markdown(
                                            f"- **{a.get('title','')}**  \n  {a.get('url','')}  \n  _overlap={sim:.2f}, category={a.get('category','')}_"
                                        )

# -------- 성장 --------
with tab_growth:
    st.subheader("성장 대시보드 (Planner)")
    sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
    sessions = sessions_db.get("sessions", [])

    if not sessions:
        st.info("아직 기록이 없습니다. 세션을 저장해보세요.")
    else:
        # show last 10 radar trends if available
        radar_sessions = []
        for s in sessions[:30]:
            ep = s.get("eval_pack") or {}
            radar = ep.get("radar") if isinstance(ep, dict) else None
            if radar:
                radar_sessions.append({"date": s.get("date", "")[:10], **radar})

        c1, c2 = st.columns(2)
        with c1:
            st.metric("총 세션 수", str(len(sessions)))
        with c2:
            # simple streak
            dates = sorted({(s.get("date", "")[:10]) for s in sessions if s.get("date")}, reverse=True)
            streak = 1
            for i in range(1, len(dates)):
                d0 = datetime.fromisoformat(dates[i - 1])
                d1 = datetime.fromisoformat(dates[i])
                if (d0 - d1).days == 1:
                    streak += 1
                else:
                    break
            st.metric("연속 학습(일)", str(streak))

        st.divider()

        if radar_sessions:
            try:
                import pandas as pd

                df = pd.DataFrame(radar_sessions).head(10)
                st.markdown("### 최근 10개 레이다 점수")
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.info("pandas가 없으면 레이다 표는 일부 제한될 수 있습니다.")
        else:
            st.info("아직 평가(Eval) 데이터가 없습니다. '평가+저장'을 한 번 실행해보세요.")

        st.divider()

        st.markdown("### 최근 세션")
        for s in sessions[:10]:
            art = s.get("article", {})
            title = art.get("title", "")
            url = art.get("url", "")
            date = s.get("date", "")[:10]
            tags = s.get("tags", [])
            ep = s.get("eval_pack") or {}

            with st.expander(f"{date} | {title}"):
                if url:
                    st.markdown(f"**원문 링크:** {url}")
                if tags:
                    st.caption("tags: " + ", ".join(tags))

                # --- START: Step 2 요약(읽기 쉬운 버전) ---
                planner = s.get("planner_summary_json", {}) or {}
                summ = (planner.get("summary") or {})
                kws = (planner.get("keywords") or [])
                
                st.markdown("**Step 2 요약(읽기 쉬운 버전)**")
                
                st.markdown("**시장 인사이트**")
                for x in (summ.get("market_insight") or []):
                    st.write(f"- {x}")
                
                st.markdown("**시스템/기술**")
                for x in (summ.get("system_tech") or []):
                    st.write(f"- {x}")
                
                st.markdown("**UX**")
                for x in (summ.get("ux") or []):
                    st.write(f"- {x}")
                
                st.markdown("**키워드 3개**")
                for k in kws:
                    if isinstance(k, dict):
                        kw = k.get("keyword", "")
                        why = k.get("why_it_matters", "")
                        if kw:
                            st.write(f"- **{kw}**: {why}")
                # --- END: Step 2 요약(읽기 쉬운 버전) ---

                st.markdown("**질문 & 내 답변**")
                qmap = {q.get("question_id"): q for q in (s.get("questions") or []) if isinstance(q, dict)}
                ans = s.get("answers") or []
                if isinstance(ans, list) and ans and isinstance(ans[0], dict):
                    for a in ans:
                        qid = a.get("question_id")
                        q = qmap.get(qid, {})
                        st.markdown(f"- **[{qid}] {q.get('prompt', qid)}**")
                        if q.get("intent"):
                            st.caption(f"intent: {q.get('intent')}")
                        st.write(a.get("answer", ""))
                        st.write("---")
                        # --- START: 평가 결과(핵심 요약) ---
                        if ep:
                            st.markdown("**평가 요약**")
                        
                            total = ep.get("total_score")
                            radar = ep.get("radar") or {}
                            daily = ep.get("daily_insight") or {}
                            qres = ep.get("question_results") or []
                        
                            if total is not None:
                                st.write(f"- 총점: **{total}**")
                        
                            if radar:
                                st.write("- 레이다(5축):")
                                st.write(
                                    f"  - SystemicThinking: {radar.get('SystemicThinking')}, "
                                    f"UserCentric: {radar.get('UserCentric')}, "
                                    f"BusinessAcumen: {radar.get('BusinessAcumen')}, "
                                    f"DetailLogic: {radar.get('DetailLogic')}, "
                                    f"TrendInsight: {radar.get('TrendInsight')}"
                                )
                        
                            if qres:
                                st.write("- 질문별 점수(상위 3개):")
                                for r in qres[:3]:
                                    st.write(f"  - {r.get('question_id')}: {r.get('score')}/{r.get('max_score')}")
                        
                            if daily:
                                st.write("- 데일리 인사이트:")
                                if daily.get("growth_check"):
                                    st.write(f"  - 성장 확인: {daily.get('growth_check')}")
                                if daily.get("weakness_fix"):
                                    st.write(f"  - 약점 보완: {daily.get('weakness_fix')}")
                                if daily.get("recommended_next_topic"):
                                    st.write(f"  - 추천 과제: {daily.get('recommended_next_topic')}")
                        # --- END: 평가 결과(핵심 요약) ---

                if s.get("final_summary"):
                    st.markdown("**내 답변 요약**")
                    st.code(s.get("final_summary"))

# -------- 설정 --------
with tab_settings:
    st.subheader("설정")
    st.markdown("### 카테고리/소스")
    st.json(sources)
    st.markdown("### 저장 경로")
    st.json(
        {
            "repo": cfg["repo_name"],
            "branch": cfg["branch"],
            "articles_path": cfg["path_articles"],
            "sessions_path": cfg["path_sessions"],
            "sources_path": cfg["path_sources"],
        }
    )
