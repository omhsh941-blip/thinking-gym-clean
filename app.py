import requests
import trafilatura
import json
import hashlib
import re
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
def evaluate_thinking(article_title, summary, user_answer):

    client = get_openai_client()

    prompt = f"""
다음 뉴스 기사에 대한 사용자의 분석을 평가하라.

기사 제목
{article_title}

기사 요약
{summary}

사용자 분석
{user_answer}

1. 기사 핵심 분석
2. 사용자 분석 평가
3. 모범 전략 분석

그리고 아래 기준으로 점수도 매겨라 (0~5점)

- 원인 분석
- 시간 축
- 2차 파급
- 리스크
- 게임 산업 연결

반드시 JSON으로 출력

{{
"ai_analysis":"...",
"score":{{
"cause":4,
"timeline":3,
"impact":4,
"risk":2,
"game":3
}}
}}
"""

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

text = res.choices[0].message.content

import re
match = re.search(r'\{.*\}', text, re.S)

if match:
    import json
    return json.loads(match.group())

return {
    "ai_analysis": text,
    "score": {
        "cause": 0,
        "timeline": 0,
        "impact": 0,
        "risk": 0,
        "game": 0
    }
    }

import json
return json.loads(res.choices[0].message.content)

# ---------------------------
# Config
# ---------------------------
def load_cfg():
    return {
        "token": st.secrets["GITHUB_TOKEN"],
        "repo_name": st.secrets["REPO_NAME"],  # 예: "omhsh941-blip/thinking-gym-clean"
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
        "manual_urls": []
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

        existing[aid] = {
            "id": aid,
            "title": title,
            "url": link,
            "category": category_name,
            "published": published,
            "collected_at": _now_kst_iso(),
            "used_in_session": False,
            "added_by": "auto",
        }
        new_count += 1

    articles_db["articles"] = sorted(existing.values(), key=lambda x: x["collected_at"], reverse=True)
    store.write_json(cfg["path_articles"], articles_db, f"collect rss ({category_name})")
    return new_count


# ---------------------------
# Simple scoring (rule-based)
# ---------------------------
def simple_scoring(answer_text: str, eval_cfg):
    text = (answer_text or "").strip()
    length_score = min(len(text) / 400, 1.0)

    keyword_hits = 0
    for kw in ["단기", "중기", "장기", "리스크", "위험", "기회", "BM", "UA", "LTV", "서버", "운영", "라이브", "경쟁"]:
        if kw in text:
            keyword_hits += 1
    kw_score = min(keyword_hits / 8, 1.0)

    total = int((0.6 * length_score + 0.4 * kw_score) * 100)

    per = {}
    for c in eval_cfg["criteria"]:
        per[c["name"]] = int(total * (c["weight"] / 100))
    return total, per


def make_final_summary(article, answers):
    a = (answers[0] if len(answers) > 0 else "").strip()
    b = (answers[1] if len(answers) > 1 else "").strip()
    c = (answers[2] if len(answers) > 2 else "").strip()

    lines = []
    lines.append(f"- 기사: {article['title']}")
    if a:
        lines.append(f"- 핵심 판단: {a[:220]}")
    if b:
        lines.append(f"- 근거/가정: {b[:220]}")
    if c:
        lines.append(f"- 다음 행동/검증: {c[:220]}")
    return "\n".join(lines)


def save_session(store, cfg, article, questions, answers, eval_cfg, article_summary_ko: str):
    sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})
    articles_db = store.read_json(cfg["path_articles"], {"articles": []})

    combined = "\n".join([a for a in answers if a])
    total, per = simple_scoring(combined, eval_cfg)

    # AI 분석 생성
ai_eval = evaluate_thinking(
    article["title"],
    article_summary_ko,
    "\n".join(answers)
)

session = {
        "session_id": _sha1(article["id"] + _now_kst_iso()),
        "date": _now_kst_iso(),
        "article": article,
        "article_summary_ko": article_summary_ko or "",
        "questions": questions,
        "answers": answers,

        "ai_analysis": ai_eval["ai_analysis"],
        "ai_score": ai_eval["score"],

        "score_total": total,
        "score_by_criteria": per,
        "final_summary": make_final_summary(article, answers),
}

sessions_db["sessions"].insert(0, session)
store.write_json(cfg["path_sessions"], sessions_db, "add session")

# mark article used
for a in articles_db["articles"]:
    if a["id"] == article["id"]:
        a["used_in_session"] = True
store.write_json(cfg["path_articles"], articles_db, "mark article used")

return session


# ---------------------------
# OpenAI Summary (Korean)
# ---------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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
    return res.choices[0].message.content.strip()


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Thinking Gym", layout="wide")

store, cfg = get_store()
ensure_bootstrap(store, cfg)

sources = store.read_json(cfg["path_sources"], default_sources())
eval_cfg = store.read_json(cfg["path_eval"], default_eval())
articles_db = store.read_json(cfg["path_articles"], {"articles": []})
sessions_db = store.read_json(cfg["path_sessions"], {"sessions": []})

st.title("🧠 Thinking Gym")
st.caption("매일 10~15분, 뉴스 1개로 전략 사고를 훈련하고 기록을 쌓습니다. (데이터는 GitHub에 저장)")

tab_collect, tab_add, tab_list, tab_session, tab_growth, tab_settings = st.tabs(
    ["📡 자동 수집", "➕ 수동 추가", "🗂 기사 목록", "✍️ 세션", "📈 성장", "⚙️ 설정"]
)

# -------- 자동 수집 --------
with tab_collect:
    st.subheader("카테고리별 RSS 자동 수집")

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
                articles_db["articles"].insert(0, {
                    "id": aid,
                    "title": title.strip(),
                    "url": url.strip(),
                    "category": category,
                    "published": "",
                    "collected_at": _now_kst_iso(),
                    "used_in_session": False,
                    "added_by": "manual",
                })
                store.write_json(cfg["path_articles"], articles_db, "add manual article")
                st.success("추가 완료!")
                st.rerun()

# -------- 기사 목록 --------
with tab_list:
    st.subheader("기사 목록")

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
        if filter_cat != "(전체)" and a["category"] != filter_cat:
            continue
        if only_unused and a.get("used_in_session"):
            continue
        if q and q.lower() not in a["title"].lower():
            continue
        filtered.append(a)

    st.write(f"표시 중: {len(filtered)}개")

    for a in filtered[:50]:
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{a['title']}**  \n{a['url']}")
        cols[1].write(a["category"])
        cols[2].write("✅ 사용" if a.get("used_in_session") else "🟡 미사용")

        if st.button("이 기사로 세션 시작", key=f"start_{a['id']}"):
            st.session_state["selected_article_id"] = a["id"]
            # 요약 상태 초기화
            st.session_state["summary_url"] = None
            st.session_state["summary_ko"] = None
            st.success("세션 탭으로 이동하세요.")
            st.rerun()

# -------- 세션 --------
with tab_session:
    st.subheader("세션 (10~15분)")

    selected_id = st.session_state.get("selected_article_id")

    def _find_article_by_id(article_id: str):
        db = store.read_json(cfg["path_articles"], {"articles": []})
        for x in db["articles"]:
            if x["id"] == article_id:
                return x
        return None

    if not selected_id:
        st.info("먼저 ‘기사 목록’에서 ‘이 기사로 세션 시작’을 눌러주세요.")
    else:
        article = _find_article_by_id(selected_id)

        if not article:
            st.error("선택한 기사를 articles.json에서 찾지 못했습니다.")
        else:
            st.markdown(f"### 📰 {article['title']}")
            st.write(article["url"])
            st.divider()

            # --- 요약 자동 생성 ---
            st.markdown("### 📄 기사 요약(한글)")

            if "OPENAI_API_KEY" not in st.secrets:
                st.error("OPENAI_API_KEY가 Streamlit Secrets에 없습니다.")
                summary_ko = ""
            else:
                current_url = article["url"]
                prev_url = st.session_state.get("summary_url")

                if st.session_state.get("summary_ko") is None or prev_url != current_url:
                    st.session_state["summary_url"] = current_url
                    with st.spinner("AI가 기사 요약 중..."):
                        try:
                            st.session_state["summary_ko"] = make_korean_summary(current_url)
                        except Exception as e:
                            st.session_state["summary_ko"] = None
                            st.error(f"요약 생성 실패: {e}")

                summary_ko = st.session_state.get("summary_ko") or ""
                if summary_ko:
                    st.write(summary_ko)

            st.divider()

            # --- 질문 ---
            st.markdown("### 질문 (고정 3개)")
            questions = [
                "1) 이 변화의 근본 원인은 무엇인가?",
                "2) 단기/중기/장기 영향은 어떻게 다를까?",
                "3) 게임 산업(특히 운영/BM/UA/라이브) 관점에서 시사점은?"
            ]

            a1 = st.text_area(questions[0], height=120)
            a2 = st.text_area(questions[1], height=120)
            a3 = st.text_area(questions[2], height=120)

            if st.button("✅ 논의 종료 & 기록 저장"):
                session = save_session(
                    store, cfg, article, questions, [a1, a2, a3], eval_cfg,
                    article_summary_ko=summary_ko
                )
                st.success("저장 완료! 성장 탭에서 확인하세요.")
                st.json({
                    "total": session["score_total"],
                    "by_criteria": session["score_by_criteria"],
                    "final_summary": session["final_summary"],
                    "article_summary_ko": session["article_summary_ko"]
                })
                st.rerun()

# -------- 성장 --------
with tab_growth:
    st.subheader("성장 대시보드")

    sessions = sessions_db["sessions"]
    if not sessions:
        st.info("아직 기록이 없습니다. 세션을 한 번 저장해보세요.")
    else:
        now = datetime.now(tz=KST)
        last7 = []
        for s in sessions:
            dt = datetime.fromisoformat(s["date"])
            if now - dt <= timedelta(days=7):
                last7.append(s["score_total"])
        avg7 = sum(last7) / len(last7) if last7 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("최근 7일 평균 점수", f"{avg7:.1f}")
        c2.metric("총 세션 수", str(len(sessions)))

        dates = sorted({s["date"][:10] for s in sessions}, reverse=True)
        streak = 1
        for i in range(1, len(dates)):
            d0 = datetime.fromisoformat(dates[i - 1])
            d1 = datetime.fromisoformat(dates[i])
            if (d0 - d1).days == 1:
                streak += 1
            else:
                break
        c3.metric("연속 학습(일)", str(streak))
        import pandas as pd
        
        scores = [s["score_total"] for s in sessions[:20]]
        
        df = pd.DataFrame({
            "session": range(len(scores)),
            "score": scores
        })
        
        st.markdown("### 최근 20개 세션 사고력 성장 그래프")
        st.line_chart(df.set_index("session"))
        st.markdown("### 최근 세션")
        for s in sessions[:10]:
            with st.expander(f"{s['date'][:10]} | {s['article']['title']} | {s['score_total']}점"):
                if s.get("article_summary_ko"):
                    st.write("**기사 요약(한글)**")
                    st.code(s["article_summary_ko"])
                st.write("**최종 요약**")
                st.code(s["final_summary"])
                st.write("**항목별 점수**")
                st.json(s["score_by_criteria"])
                st.write("**AI 분석**")
                st.write(s.get("ai_analysis",""))
                st.write("**AI 평가 점수**")
                st.json(s.get("ai_score",{}))

# -------- 설정 --------
with tab_settings:
    st.subheader("설정")
    st.markdown("### 평가 기준(5개) — 나중에 수정 가능")
    st.json(eval_cfg)

    st.markdown("### 카테고리/소스")
    st.json(sources)
