# /mnt/data/app.py
import requests
import trafilatura
import json
import hashlib
import re
from collections import Counter
from openai import OpenAI
from datetime import datetime, timedelta
from dateutil import tz

import streamlit as st
import feedparser
from github import Github

KST = tz.gettz("Asia/Seoul")


# ---------------------------
# Utils
# ---------------------------
def _now_kst_iso():
    return datetime.now(tz=KST).isoformat(timespec="seconds")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_json_from_text(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _ensure_token_usage():
    if "token_usage" not in st.session_state:
        st.session_state["token_usage"] = {"article_id": None, "summary": None, "eval": None}


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


def _normalize_tokens(text: str):
    """
    아주 가벼운 태그 추출용 토큰화.
    - 한글/영문/숫자/특수 섞인 기사 요약/제목에서도 대충 돌아가게.
    """
    if not text:
        return []
    text = text.lower()
    # 한글/영문/숫자만 남기고 나머지는 공백 처리
    text = re.sub(r"[^0-9a-z가-힣\s]+", " ", text)
    parts = [p.strip() for p in text.split() if p.strip()]
    return parts


_STOPWORDS = set(
    """
    기사 요약 내용 관련 오늘 이번 해당 통해 대한 그리고 하지만 또한
    있다 없다 된다 했다 한다 한 것 등 수 로 의 에서 으로 에게 보다
    the a an to of in on for with from at by as is are was were be been
    this that these those it its their them they you your we our
    """.split()
)


def extract_tags_from_text(text: str, max_tags: int = 7):
    """
    - 목적: "연관 기사 추천"용 tags[] 생성
    - 정확한 NLP 태깅이 아니라, '겹치는 주제'를 찾을 수 있을 정도의 간단 태그.
    """
    tokens = _normalize_tokens(text)
    tokens = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
    if not tokens:
        return []
    freq = Counter(tokens)
    # 너무 흔한 숫자만 토큰은 제외
    cand = []
    for tok, c in freq.most_common(50):
        if tok.isdigit():
            continue
        cand.append(tok)
        if len(cand) >= max_tags:
            break
    return cand


def jaccard(a, b):
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

        # 제목 기반 1차 태그(가벼운 기본값) — 연관 기사 추천에 활용
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
            "tags": tags,  # ✅ 추가
        }
        new_count += 1

    articles_db["articles"] = sorted(existing.values(), key=lambda x: x["collected_at"], reverse=True)
    store.write_json(cfg["path_articles"], articles_db, f"collect rss ({category_name})")
    return new_count


# ---------------------------
# Rule scoring (simple)
# ---------------------------
def simple_scoring(answer_text: str, eval_cfg):
    text = (answer_text or "").strip()
    length_score = min(len(text) / 400, 1.0)

    keyword_hits = 0
    for kw in ["단기", "중기", "장기", "리스크", "위험", "기회", "bm", "ua", "ltv", "서버", "운영", "라이브", "경쟁"]:
        if kw in text.lower():
            keyword_hits += 1
    kw_score = min(keyword_hits / 8, 1.0)

    total = int((0.6 * length_score + 0.4 * kw_score) * 100)

    per = {}
    for c in eval_cfg["criteria"]:
        per[c["name"]] = int(total * (c["weight"] / 100))
    return total, per


def make_final_summary(article, answers_by_id):
    """
    answers_by_id: dict(question_id -> answer_text)
    """
    # 기존 템플릿 감성 유지 (짧게)
    lines = []
    lines.append(f"- 기사: {article.get('title','')}")
    # 대표 3개만 뽑아서 보여주기 (ID 기준)
    a = (answers_by_id.get("q1") or "").strip()
    b = (answers_by_id.get("q2") or "").strip()
    c = (answers_by_id.get("q3") or "").strip()

    if a:
        lines.append(f"- 핵심 판단: {a[:220]}")
    if b:
        lines.append(f"- 근거/가정: {b[:220]}")
    if c:
        lines.append(f"- 다음 행동/검증: {c[:220]}")
    return "\n".join(lines)


# ---------------------------
# OpenAI Client
# ---------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------------------------
# Article fetch + summary
# ---------------------------
def fetch_article_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ThinkingGym/1.0)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
    return (text or "").strip()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def make_korean_summary(url: str) -> str:
    body = fetch_article_text(url)
    if not body:
        return "요약 실패: 본문을 추출하지 못했습니다. (사이트 차단/구조 문제 가능)"

    body = body[:12000]

    prompt = f"""
너는 뉴스 요약 전문가다.
아래 기사 내용을 한국어로 요약해줘.

규칙:
- 핵심 요약 5개 불릿(•)
- 숫자/고유명사(회사/제품/인물) 유지
- 과장/추측 금지. 기사에 없는 내용은 '알 수 없음'
- 마지막 줄에 '한 줄 결론:' 추가

기사 본문:
{body}
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

    return res.choices[0].message.content.strip()


# ---------------------------
# AI Evaluation (기존 유지: 입력은 text, 출력은 JSON)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def evaluate_thinking(article_title: str, summary_ko: str, user_answer: str):
    client = get_openai_client()

    prompt = f"""
다음 뉴스 기사에 대해 사용자의 분석을 평가하고, AI 기준 모범 답안을 작성해라.

기사 제목:
{article_title}

기사 요약(한글):
{summary_ko}

사용자 분석(여러 답변 포함):
{user_answer}

요구:
1) AI 분석(기사 기반으로, 게임 산업 관점까지 포함)
2) 사용자 분석 평가(좋은 점 / 부족한 점 / 개선 팁)
3) AI 모범 답안(짧게)

채점(0~5):
- cause: 원인 분석
- timeline: 시간 축(단기/중기/장기)
- impact: 2차 파급
- risk: 리스크/불확실성
- game: 게임 산업 연결

반드시 JSON으로만 출력하라.

채점 규칙(매우 중요):
- 사용자 분석이 80자 미만이거나 "테스트", "asdf" 같은 의미 없는 내용이면 ai_score의 모든 값은 0으로 해라.
- 아래 예시는 "형식"만 참고해라. 숫자(4,3,4,2,3)는 절대 복사하지 마라.

형식:
{{
  "ai_analysis": "...",
  "ai_feedback": "좋은 점/부족한 점/개선 팁",
  "ai_score": {{"cause":4,"timeline":3,"impact":4,"risk":2,"game":3}},
  "ai_model_answer": "..."
}}
""".strip()

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    _ensure_token_usage()
    try:
        st.session_state["token_usage"]["eval"] = getattr(res, "usage", None)
    except Exception:
        pass

    text = res.choices[0].message.content
    parsed = _safe_json_from_text(text)

    if parsed and isinstance(parsed, dict):
        parsed.setdefault("ai_analysis", text)
        parsed.setdefault("ai_feedback", "")
        parsed.setdefault("ai_model_answer", "")
        score = parsed.get("ai_score", {})
        if not isinstance(score, dict):
            score = {}
        for k in ["cause", "timeline", "impact", "risk", "game"]:
            score.setdefault(k, 0)
        parsed["ai_score"] = score
        return parsed

    return {
        "ai_analysis": text,
        "ai_feedback": "",
        "ai_score": {"cause": 0, "timeline": 0, "impact": 0, "risk": 0, "game": 0},
        "ai_model_answer": "",
    }


# ---------------------------
# RSS "연관 기사" 추천
# - 오늘 읽은 기사 tags[]와 겹치거나
# - 반대되는 성향(카테고리 대비 opposite category) 기사 추천
# ---------------------------
def get_opposite_category(cat: str) -> str:
    """
    아주 단순한 '반대 성향' 정의:
    - AI/Tech ↔ Business
    - Game Industry ↔ AI/Tech
    - Business ↔ AI/Tech
    """
    if cat == "AI/Tech":
        return "Business"
    if cat == "Business":
        return "AI/Tech"
    if cat == "Game Industry":
        return "AI/Tech"
    return "Business"


def recommend_related_articles(store, cfg, current_article: dict, current_tags: list, limit_each: int = 5):
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})
    candidates = [a for a in articles_db.get("articles", []) if a.get("id") != current_article.get("id")]

    # 후보 태그가 없으면 제목 기반으로 임시 생성
    for a in candidates:
        if "tags" not in a or not isinstance(a["tags"], list) or len(a["tags"]) == 0:
            a["tags"] = extract_tags_from_text(a.get("title", ""), max_tags=7)

    # 1) tags overlap 추천
    scored = []
    for a in candidates:
        sim = jaccard(current_tags, a.get("tags", []))
        if sim > 0:
            scored.append((sim, a))
    scored.sort(key=lambda x: (x[0], x[1].get("collected_at", "")), reverse=True)
    overlap_pick = [a for _, a in scored[:limit_each]]

    # 2) opposite 성향 추천: category를 반대로 잡고, 그 중에서 tags가 '덜 겹치는' 것도 섞음
    opp_cat = get_opposite_category(current_article.get("category", ""))
    opp = [a for a in candidates if a.get("category") == opp_cat and not a.get("used_in_session")]
    opp_scored = []
    for a in opp:
        sim = jaccard(current_tags, a.get("tags", []))
        # "반대 성향"이므로 너무 겹치지 않는 것도 가치가 있음: (1 - sim) 기준
        opp_scored.append((1.0 - sim, a))
    opp_scored.sort(key=lambda x: (x[0], x[1].get("collected_at", "")), reverse=True)
    opposite_pick = [a for _, a in opp_scored[:limit_each]]

    return overlap_pick, opposite_pick


# ---------------------------
# Save session
# - questions: Question Object 배열 (intent 포함)
# - answers: [{question_id, answer}]
# - session.tags: 이번 기사 태그
# ---------------------------
def save_session(store, cfg, article, questions, answers, eval_cfg, article_summary_ko: str, ai_pack: dict, session_tags: list):
    sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})

    # answers(list[dict]) -> combined text
    combined = "\n".join([(a.get("answer") or "") for a in answers if a.get("answer")]).strip()
    total, per = simple_scoring(combined, eval_cfg)

    answers_by_id = {a.get("question_id"): (a.get("answer") or "") for a in answers if a.get("question_id")}

    session = {
        "session_id": _sha1(article["id"] + _now_kst_iso()),
        "date": _now_kst_iso(),
        "article": {
            "id": article.get("id", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "category": article.get("category", ""),
            "published": article.get("published", ""),
        },
        "tags": session_tags or [],  # ✅ 추가: 오늘 읽은 기사 태그
        "article_summary_ko": article_summary_ko or "",
        "questions": questions,  # ✅ Question Object 배열(intent 포함)
        "answers": answers,      # ✅ [{question_id, answer}]

        "ai_analysis": ai_pack.get("ai_analysis", ""),
        "ai_score": ai_pack.get("ai_score", {}),
        "ai_feedback": ai_pack.get("ai_feedback", ""),
        "ai_model_answer": ai_pack.get("ai_model_answer", ""),

        "score_total": total,
        "score_by_criteria": per,
        "final_summary": make_final_summary(article, answers_by_id),
    }

    sessions_db["sessions"].insert(0, session)
    store.write_json(cfg["path_sessions"], sessions_db, "add session")

    # 기사 used 표시 + tags 저장(없었을 경우)
    for a in articles_db["articles"]:
        if a.get("id") == article.get("id"):
            a["used_in_session"] = True
            if "tags" not in a or not isinstance(a["tags"], list) or len(a["tags"]) == 0:
                a["tags"] = session_tags or []
    store.write_json(cfg["path_articles"], articles_db, "mark article used (+tags)")

    return session


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Thinking Gym", layout="wide")

store, cfg = get_store()
ensure_bootstrap(store, cfg)

sources = store.read_json(cfg["path_sources"], default_sources())
eval_cfg = store.read_json(cfg["path_eval"], default_eval())

st.title("🧠 Thinking Gym")
st.caption("매일 10~15분, 뉴스 1개로 전략 사고를 훈련하고 기록을 쌓습니다. (데이터는 GitHub에 저장)")

_ensure_token_usage()

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
                articles_db["articles"].insert(0, {
                    "id": aid,
                    "title": title.strip(),
                    "url": url.strip(),
                    "category": category,
                    "published": "",
                    "collected_at": _now_kst_iso(),
                    "used_in_session": False,
                    "added_by": "manual",
                    "tags": tags,  # ✅ 추가
                })
                store.write_json(cfg["path_articles"], articles_db, "add manual article (+tags)")
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

    for a in filtered[:50]:
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{a.get('title','')}**  \n{a.get('url','')}")
        cols[1].write(a.get("category", ""))
        cols[2].write("✅ 사용" if a.get("used_in_session") else "🟡 미사용")

        if st.button("이 기사로 세션 시작", key=f"start_{a['id']}"):
            st.session_state["selected_article_id"] = a["id"]
            st.session_state["summary_url"] = None
            st.session_state["summary_ko"] = None
            st.session_state["summary_tags"] = None  # ✅ 추가: 이번 기사 tags 캐시
            st.session_state["token_usage"] = {"article_id": a["id"], "summary": None, "eval": None}
            st.rerun()

# -------- 세션 --------
with tab_session:
    st.subheader("세션 (10~15분)")

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

            st.markdown("### 📄 기사 요약(한글)")

            if "OPENAI_API_KEY" not in st.secrets:
                st.error("OPENAI_API_KEY가 Streamlit Secrets에 없습니다.")
                summary_ko = ""
            else:
                current_url = article.get("url", "")
                prev_url = st.session_state.get("summary_url")

                if st.session_state.get("summary_ko") is None or prev_url != current_url:
                    st.session_state["summary_url"] = current_url
                    with st.spinner("AI가 기사 요약 중..."):
                        try:
                            st.session_state["summary_ko"] = make_korean_summary(current_url)
                        except Exception as e:
                            st.session_state["summary_ko"] = None
                            st.error(f"요약 생성 실패: {e}")

                    # ✅ 요약 갱신 시 tags도 새로 만들기
                    st.session_state["summary_tags"] = extract_tags_from_text(st.session_state.get("summary_ko") or "", max_tags=7)

                summary_ko = st.session_state.get("summary_ko") or ""
                if summary_ko:
                    st.write(summary_ko)

            # ✅ tags 표시
            session_tags = st.session_state.get("summary_tags") or []
            if session_tags:
                st.caption("tags: " + ", ".join(session_tags))

            st.divider()

            st.markdown("### 🧾 이번 세션 토큰 사용량")

            tu = st.session_state.get("token_usage", {})
            if tu.get("article_id") != selected_id:
                u_sum = None
                u_eval = None
            else:
                u_sum = tu.get("summary")
                u_eval = tu.get("eval")

            c1, c2 = st.columns(2)
            c1.markdown("**요약(summary)**")
            c1.caption(_usage_to_str(u_sum))

            c2.markdown("**평가(eval)**")
            c2.caption(_usage_to_str(u_eval))

            st.divider()

            # ✅ Question Object (intent 포함)
            st.markdown("### 질문 (고정 3개)")

            questions = [
                {
                    "question_id": "q1",
                    "prompt": "이 변화의 근본 원인은 무엇인가?",
                    "intent": "이 질문은 표면 현상(뉴스 헤드라인) 아래의 '원인/구조'를 파악하는 훈련입니다.",
                },
                {
                    "question_id": "q2",
                    "prompt": "단기/중기/장기 영향은 어떻게 다를까?",
                    "intent": "이 질문은 같은 사건도 시간 축에 따라 KPI/리스크/기회가 달라진다는 점을 점검하기 위한 것입니다.",
                },
                {
                    "question_id": "q3",
                    "prompt": "게임 산업(특히 운영/BM/UA/라이브) 관점에서 시사점은?",
                    "intent": "이 질문은 기사 내용을 '게임 사업/라이브 운영 의사결정'으로 번역하는 능력을 확인하기 위한 것입니다.",
                },
            ]

            # intent 표시 옵션
            show_intent = st.checkbox("출제 의도(intent) 보기", value=True)

            answers = []
            for qobj in questions:
                if show_intent:
                    st.caption(f"intent: {qobj['intent']}")
                ans = st.text_area(qobj["prompt"], height=120, key=f"ans_{qobj['question_id']}")
                answers.append({"question_id": qobj["question_id"], "answer": ans})

            st.divider()

            # ✅ 저장 버튼
            if st.button("✅ 논의 종료 & 기록 저장"):
                user_answer_all = "\n".join([(a.get("answer") or "") for a in answers]).strip()

                ai_pack = {"ai_analysis": "", "ai_score": {}, "ai_feedback": "", "ai_model_answer": ""}

                if summary_ko and user_answer_all and "OPENAI_API_KEY" in st.secrets:
                    with st.spinner("AI가 당신의 답변을 평가/비교 중..."):
                        try:
                            ai_pack = evaluate_thinking(
                                article_title=article.get("title", ""),
                                summary_ko=summary_ko,
                                user_answer=user_answer_all
                            )
                        except Exception as e:
                            st.error(f"AI 평가 실패: {e}")

                session = save_session(
                    store=store,
                    cfg=cfg,
                    article=article,
                    questions=questions,
                    answers=answers,
                    eval_cfg=eval_cfg,
                    article_summary_ko=summary_ko,
                    ai_pack=ai_pack,
                    session_tags=session_tags,
                )

                st.success("저장 완료! 성장 탭에서 확인하세요.")
                st.json({
                    "score_total": session.get("score_total"),
                    "rule_score_by_criteria": session.get("score_by_criteria"),
                    "ai_score": session.get("ai_score"),
                    "tags": session.get("tags"),
                })

                # ✅ RSS '연관 기사' 추천 섹션 (요청 반영)
                st.divider()
                st.markdown("## 🔗 연관 기사 추천 (RSS)")
                st.caption("오늘 읽은 기사 tags[]가 겹치거나, 반대되는 성향의 기사를 추천합니다.")

                overlap_pick, opposite_pick = recommend_related_articles(
                    store=store,
                    cfg=cfg,
                    current_article=article,
                    current_tags=session_tags,
                    limit_each=5,
                )

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("### ✅ tags 겹치는 기사")
                    if not overlap_pick:
                        st.write("(추천 없음) — tags가 부족하거나 후보가 없을 수 있어요.")
                    else:
                        for a in overlap_pick:
                            sim = jaccard(session_tags, a.get("tags", []))
                            st.markdown(f"- **{a.get('title','')}**  \n  {a.get('url','')}  \n  _overlap={sim:.2f}, category={a.get('category','')}_")

                with colB:
                    st.markdown("### 🌓 반대 성향 기사")
                    if not opposite_pick:
                        st.write("(추천 없음) — 반대 카테고리 후보가 부족할 수 있어요.")
                    else:
                        for a in opposite_pick:
                            sim = jaccard(session_tags, a.get("tags", []))
                            st.markdown(f"- **{a.get('title','')}**  \n  {a.get('url','')}  \n  _overlap={sim:.2f}, category={a.get('category','')}_")

                st.rerun()

# -------- 성장 --------
with tab_growth:
    st.subheader("성장 대시보드")

    sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
    sessions = sessions_db.get("sessions", [])

    if not sessions:
        st.info("아직 기록이 없습니다. 세션을 한 번 저장해보세요.")
    else:
        now = datetime.now(tz=KST)
        last7 = []
        for s in sessions:
            try:
                dt = datetime.fromisoformat(s["date"])
                if now - dt <= timedelta(days=7):
                    last7.append(s.get("score_total", 0))
            except Exception:
                pass
        avg7 = sum(last7) / len(last7) if last7 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("최근 7일 평균 점수(규칙기반)", f"{avg7:.1f}")
        c2.metric("총 세션 수", str(len(sessions)))

        dates = sorted({(s.get("date", "")[:10]) for s in sessions if s.get("date")}, reverse=True)
        streak = 1
        for i in range(1, len(dates)):
            d0 = datetime.fromisoformat(dates[i - 1])
            d1 = datetime.fromisoformat(dates[i])
            if (d0 - d1).days == 1:
                streak += 1
            else:
                break
        c3.metric("연속 학습(일)", str(streak))

        try:
            import pandas as pd
            scores = [s.get("score_total", 0) for s in sessions[:20]]
            df = pd.DataFrame({
                "session": list(range(len(scores))),
                "score": scores
            })
            st.markdown("### 최근 20개 세션 사고력 성장 그래프(규칙기반)")
            st.line_chart(df.set_index("session"))
        except Exception as e:
            st.info(f"(그래프 비활성) pandas 설치 필요 또는 오류: {e}")

        st.markdown("### 최근 세션")
        for s in sessions[:10]:
            art = s.get("article", {})
            title = art.get("title", "")
            url = art.get("url", "")
            date = s.get("date", "")[:10]
            total = s.get("score_total", 0)

            with st.expander(f"{date} | {title} | {total}점"):
                if url:
                    st.markdown(f"**원문 링크:** {url}")

                if s.get("tags"):
                    st.caption("tags: " + ", ".join(s.get("tags")))

                if s.get("article_summary_ko"):
                    st.write("**기사 요약(한글)**")
                    st.code(s["article_summary_ko"])

                st.write("**내 답변**")
                ans = s.get("answers", [])
                # ✅ 새로운 포맷: list[dict]
                if isinstance(ans, list) and ans and isinstance(ans[0], dict):
                    out = []
                    qmap = {q.get("question_id"): q for q in (s.get("questions") or []) if isinstance(q, dict)}
                    for a in ans:
                        qid = a.get("question_id")
                        qprompt = (qmap.get(qid) or {}).get("prompt", qid)
                        out.append(f"[{qid}] {qprompt}\n{a.get('answer','')}")
                    st.code("\n\n".join(out))
                # 구버전 포맷: list[str]
                elif isinstance(ans, list) and ans:
                    st.code("\n\n".join([x for x in ans if x]))
                else:
                    st.write("(없음)")

                st.write("**AI 분석**")
                st.write(s.get("ai_analysis", ""))

                st.write("**AI 모범 답안(짧게)**")
                st.write(s.get("ai_model_answer", ""))

                st.write("**사용자 분석 평가(피드백)**")
                st.write(s.get("ai_feedback", ""))

                st.write("**AI 평가 점수**")
                st.json(s.get("ai_score", {}))

                st.write("**규칙 기반 점수(기존)**")
                st.json(s.get("score_by_criteria", {}))

                st.write("**최종 요약(내 답변 템플릿)**")
                st.code(s.get("final_summary", ""))

# -------- 설정 --------
with tab_settings:
    st.subheader("설정")
    st.markdown("### 평가 기준(5개) — 나중에 수정 가능")
    st.json(eval_cfg)

    st.markdown("### 카테고리/소스")
    st.json(sources)
