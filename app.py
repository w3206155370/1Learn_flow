from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
STORE_PATH = DATA_DIR / "learnflow_store.json"
MODEL_ID = os.getenv("MODEL_ID", "deepseek-flash")

DEFAULT_STORE: dict[str, Any] = {
    "profile": {
        "goal": "掌握 Python 与 AI Agent 开发",
        "level": "入门到初级",
        "minutes_per_day": 60,
        "interests": "Python、AI Agent、工具调用、项目实战",
    },
    "materials": [],
    "tasks": [],
    "reviews": [],
    "audit_log": [],
}


def load_store() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return json.loads(json.dumps(DEFAULT_STORE, ensure_ascii=False))
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in DEFAULT_STORE.items():
            data.setdefault(key, json.loads(json.dumps(value, ensure_ascii=False)))
        return data
    except (OSError, json.JSONDecodeError):
        st.warning("本地数据文件无法读取，已使用空白数据启动。原文件未删除。")
        return json.loads(json.dumps(DEFAULT_STORE, ensure_ascii=False))


def save_store(store: dict[str, Any]) -> None:
    tmp_path = STORE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    tmp_path.replace(STORE_PATH)


def add_log(store: dict[str, Any], action: str, detail: str, status: str = "success") -> None:
    store["audit_log"].insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail[:500],
        "status": status,
    })
    store["audit_log"] = store["audit_log"][:200]


def get_client() -> OpenAI | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def extract_json(text: str) -> dict[str, Any]:
    """Parse model JSON, tolerating a fenced code block or surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型没有返回可解析的 JSON 对象")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("模型返回的数据必须是 JSON 对象")
    return obj


def analyze_material(client: OpenAI, model: str, profile: dict[str, Any], title: str, content: str) -> dict[str, Any]:
    system = """你是 LearnFlow 的个性化学习顾问。必须根据用户目标、水平、时间和材料内容做判断，不要编造材料里没有的事实。只返回合法 JSON，不要 Markdown。字段：relevance_score(0-100整数), priority(高/中/低), decision(现在学习/稍后学习/暂不推荐), reason(推荐理由), summary(不超过150字摘要), prerequisites(字符串数组), study_actions(2-4个具体动作数组), key_points(3-5个知识点数组), quiz(3个对象数组，每个含question、answer、explanation)。如果材料不相关，仍需解释原因，study_actions可以为空数组。"""
    user = {
        "学习画像": profile,
        "材料标题": title,
        "材料正文": content[:18000],
        "要求": "判断材料对当前目标的价值。优先考虑目标匹配度、学习阶段、先修知识、实际可执行性。",
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    message = response.choices[0].message.content or "{}"
    result = extract_json(message)
    result["relevance_score"] = max(0, min(100, int(result.get("relevance_score", 50))))
    if result.get("priority") not in {"高", "中", "低"}:
        result["priority"] = "中"
    if result.get("decision") not in {"现在学习", "稍后学习", "暂不推荐"}:
        result["decision"] = "稍后学习"
    for key in ["prerequisites", "study_actions", "key_points", "quiz"]:
        if not isinstance(result.get(key), list):
            result[key] = []
    return result


def create_demo_analysis(profile: dict[str, Any], title: str, content: str) -> dict[str, Any]:
    """Offline demo fallback. Clearly identified as heuristic, not an LLM result."""
    goal_words = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", profile.get("goal", "")))
    content_words = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", title + " " + content[:2000]))
    overlap = len(goal_words & content_words)
    score = min(95, 45 + overlap * 8)
    if score >= 70:
        decision, priority = "现在学习", "高"
    elif score >= 50:
        decision, priority = "稍后学习", "中"
    else:
        decision, priority = "暂不推荐", "低"
    snippet = re.sub(r"\s+", " ", content).strip()
    return {
        "relevance_score": score,
        "priority": priority,
        "decision": decision,
        "reason": "演示模式使用关键词重合度做初步判断；正式个性化评估需要配置 DeepSeek API。请将此结果视为演示，不是语义评估。",
        "summary": (snippet[:140] + "…") if len(snippet) > 140 else snippet,
        "prerequisites": ["确认材料中的核心术语", "核对材料是否匹配当前学习阶段"],
        "study_actions": ["用自己的话复述材料核心观点", "选择一个概念写一个小例子", "记录仍不理解的问题"],
        "key_points": [title or "材料主题", "目标匹配度需要人工复核", "将信息转化为可执行行动"],
        "quiz": [
            {"question": "这份材料与你当前学习目标的关系是什么？", "answer": "需要根据学习目标和材料内容判断。", "explanation": "把材料内容与当前目标逐项对照。"},
            {"question": "阅读后，你能产出什么可验证的成果？", "answer": "例如一个例子、一段代码或一份简短解释。", "explanation": "用输出检验理解，而非仅凭读完判断掌握。"},
            {"question": "有哪些内容还需要进一步验证？", "answer": "记录不确定的概念、事实或实践步骤。", "explanation": "把不确定之处转换成后续检索或复习任务。"},
        ],
        "demo_mode": True,
    }


def normalize_tasks(items: list[Any], material_id: str, title: str) -> list[dict[str, Any]]:
    out = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append({
                "id": f"task-{datetime.now().timestamp()}-{len(out)}",
                "content": text,
                "status": "pending",
                "source_material_id": material_id,
                "source_title": title,
                "created_at": date.today().isoformat(),
                "completed_at": None,
            })
    return out


def schedule_review(review: dict[str, Any], remembered: bool) -> dict[str, Any]:
    """Simple transparent spacing rule: 3 days if remembered, tomorrow if not."""
    updated = dict(review)
    updated["last_reviewed"] = date.today().isoformat()
    updated["review_count"] = int(updated.get("review_count", 0)) + 1
    if remembered:
        updated["next_review"] = (date.today() + timedelta(days=3)).isoformat()
        updated["mastery"] = min(5, int(updated.get("mastery", 0)) + 1)
        updated["last_result"] = "记住了"
    else:
        updated["next_review"] = (date.today() + timedelta(days=1)).isoformat()
        updated["mastery"] = max(0, int(updated.get("mastery", 0)) - 1)
        updated["last_result"] = "还不熟"
    return updated


def main() -> None:
    st.set_page_config(page_title="LearnFlow · 个性化 AI 学习助手", page_icon="🧭", layout="wide")
    st.title("🧭 LearnFlow")
    st.caption("个性化 AI 学习规划与知识管理智能体 · 从筛选信息到复习检验的学习闭环")

    store = load_store()
    client = get_client()
    if client is None:
        st.info("当前为离线演示模式。配置 DEEPSEEK_API_KEY 后，材料筛选与复习题将由 DeepSeek 生成；离线结果会明确标注为演示估算。")
    else:
        st.success(f"已连接 DeepSeek · 模型：{MODEL_ID}")

    with st.sidebar:
        st.header("我的学习画像")
        with st.form("profile_form"):
            goal = st.text_area("当前学习目标", value=store["profile"].get("goal", ""), height=90)
            level = st.selectbox("当前水平", ["零基础", "入门到初级", "中级", "进阶"], index=["零基础", "入门到初级", "中级", "进阶"].index(store["profile"].get("level", "入门到初级")) if store["profile"].get("level") in ["零基础", "入门到初级", "中级", "进阶"] else 1)
            minutes = st.slider("每天可用学习时间（分钟）", 15, 240, int(store["profile"].get("minutes_per_day", 60)), step=15)
            interests = st.text_area("偏好 / 已有知识", value=store["profile"].get("interests", ""), height=80)
            save_profile = st.form_submit_button("保存学习画像", use_container_width=True)
        if save_profile:
            store["profile"] = {"goal": goal.strip(), "level": level, "minutes_per_day": minutes, "interests": interests.strip()}
            add_log(store, "更新学习画像", f"目标：{goal.strip()}")
            save_store(store)
            st.success("学习画像已保存")

        st.divider()
        st.caption("数据存储在本地 data/learnflow_store.json。演示数据不会上传；调用 API 时，只有当前提交的材料和学习画像会发送给模型。")

    tab_filter, tab_plan, tab_review, tab_audit = st.tabs(["🔎 信息筛选", "🗂 学习计划", "🧠 复习中心", "🧾 审计记录"])

    with tab_filter:
        st.subheader("信息筛选")
        st.write("粘贴文章、课程笔记或技术文档。LearnFlow 会根据你的学习目标判断是否值得现在投入时间。")
        with st.form("material_form", clear_on_submit=False):
            material_title = st.text_input("材料标题", placeholder="例如：Agent Loop 入门教程")
            material_url = st.text_input("来源链接（选填）", placeholder="https://...")
            material_content = st.text_area("材料正文", height=220, placeholder="粘贴你希望评估的内容……")
            submitted = st.form_submit_button("分析这份材料", type="primary", use_container_width=True)
        if submitted:
            if material_title.strip() == "" or material_content.strip() == "":
                st.error("请填写材料标题和正文。")
            else:
                with st.spinner("正在结合你的学习目标分析材料……"):
                    try:
                        if client is not None:
                            analysis = analyze_material(client, MODEL_ID, store["profile"], material_title, material_content)
                            mode = "DeepSeek 分析"
                        else:
                            analysis = create_demo_analysis(store["profile"], material_title, material_content)
                            mode = "离线演示估算"
                        material_id = f"mat-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                        record = {"id": material_id, "title": material_title.strip(), "url": material_url.strip(), "content": material_content.strip()[:18000], "analysis": analysis, "created_at": datetime.now().isoformat(timespec="seconds"), "mode": mode}
                        store["materials"].insert(0, record)
                        add_log(store, "筛选材料", f"{material_title}；分数 {analysis.get('relevance_score', 0)}；判断 {analysis.get('decision', '')}；模式 {mode}")
                        save_store(store)
                        st.session_state["last_material_id"] = material_id
                        st.success("分析已保存。可以检查结果后，将建议转化为学习任务或复习卡。")
                    except Exception as exc:
                        st.error(f"分析失败：{exc}")

        selected_id = st.session_state.get("last_material_id")
        material_to_show = next((m for m in store["materials"] if m["id"] == selected_id), None)
        if material_to_show is not None:
            a = material_to_show["analysis"]
            st.divider()
            st.subheader("最近一次分析")
            if a.get("demo_mode"):
                st.warning("此结果来自离线演示规则，不是模型语义判断。")
            c1, c2, c3 = st.columns(3)
            c1.metric("目标相关度", f"{a.get('relevance_score', 0)}/100")
            c2.metric("优先级", a.get("priority", "中"))
            c3.metric("建议", a.get("decision", "稍后学习"))
            st.markdown("**推荐理由**")
            st.write(a.get("reason", ""))
            st.markdown("**内容摘要**")
            st.write(a.get("summary", ""))
            st.markdown("**关键知识点**")
            for point in a.get("key_points", []):
                st.markdown(f"- {point}")
            st.markdown("**建议学习动作**")
            actions = a.get("study_actions", [])
            if len(actions) == 0:
                st.write("模型暂未给出学习动作，可以根据目标手动创建任务。")
            else:
                for action in actions:
                    st.markdown(f"- {action}")
            col_add, col_review = st.columns(2)
            if col_add.button("将学习动作加入计划", use_container_width=True):
                new_tasks = normalize_tasks(actions, material_to_show["id"], material_to_show["title"])
                existing = {t["content"] for t in store["tasks"]}
                to_add = [t for t in new_tasks if t["content"] not in existing]
                store["tasks"].extend(to_add)
                add_log(store, "创建学习任务", f"来自材料：{material_to_show['title']}；新增 {len(to_add)} 项")
                save_store(store)
                st.success(f"已新增 {len(to_add)} 项任务。")
            if col_review.button("将知识点生成复习卡", use_container_width=True):
                existing_questions = {r["question"] for r in store["reviews"]}
                added = 0
                for q in a.get("quiz", []):
                    if isinstance(q, dict) and q.get("question") and q["question"] not in existing_questions:
                        store["reviews"].append({"id": f"rev-{datetime.now().timestamp()}-{added}", "material_id": material_to_show["id"], "material_title": material_to_show["title"], "question": str(q.get("question", "")), "answer": str(q.get("answer", "")), "explanation": str(q.get("explanation", "")), "created_at": date.today().isoformat(), "next_review": date.today().isoformat(), "review_count": 0, "mastery": 0, "last_result": "未复习"})
                        added += 1
                add_log(store, "生成复习卡", f"材料：{material_to_show['title']}；新增 {added} 张")
                save_store(store)
                st.success(f"已新增 {added} 张复习卡，可在“复习中心”练习。")

        if store["materials"]:
            st.divider()
            st.subheader("历史材料")
            for m in store["materials"][:10]:
                score = m.get("analysis", {}).get("relevance_score", "—")
                with st.expander(f"{m['title']} · 相关度 {score}/100 · {m['created_at'][:10]}"):
                    st.write(m.get("analysis", {}).get("reason", ""))
                    if m.get("url"):
                        st.markdown(f"来源：{m['url']}")

    with tab_plan:
        st.subheader("学习计划与进度")
        total = len(store["tasks"])
        complete = sum(1 for t in store["tasks"] if t.get("status") == "completed")
        in_progress = sum(1 for t in store["tasks"] if t.get("status") == "in_progress")
        c1, c2, c3 = st.columns(3)
        c1.metric("任务总数", total)
        c2.metric("已完成", complete)
        c3.metric("进行中", in_progress)
        if total > 0:
            st.progress(complete / total, text=f"完成进度：{complete}/{total}")
        with st.form("manual_task_form", clear_on_submit=True):
            manual_task = st.text_input("手动添加学习任务", placeholder="例如：用自己的话解释 tool_calls")
            manual_submit = st.form_submit_button("添加任务")
        if manual_submit and manual_task.strip():
            store["tasks"].append({"id": f"manual-{datetime.now().timestamp()}", "content": manual_task.strip(), "status": "pending", "source_material_id": None, "source_title": "手动添加", "created_at": date.today().isoformat(), "completed_at": None})
            add_log(store, "手动添加任务", manual_task.strip())
            save_store(store)
            st.rerun()
        if not store["tasks"]:
            st.info("还没有学习任务。先在“信息筛选”中分析材料并创建任务，或在这里手动添加。")
        else:
            for task in store["tasks"]:
                with st.container(border=True):
                    left, right = st.columns([3, 1])
                    left.markdown(f"**{task['content']}**")
                    left.caption(f"来源：{task.get('source_title', '手动添加')} · 创建于 {task.get('created_at', '')}")
                    statuses = ["pending", "in_progress", "completed"]
                    labels = {"pending": "待开始", "in_progress": "进行中", "completed": "已完成"}
                    current = task.get("status", "pending")
                    new_status = right.selectbox("状态", statuses, index=statuses.index(current) if current in statuses else 0, format_func=lambda x: labels[x], key=f"status-{task['id']}")
                    if new_status != current:
                        task["status"] = new_status
                        task["completed_at"] = datetime.now().isoformat(timespec="seconds") if new_status == "completed" else None
                        add_log(store, "更新任务状态", f"{task['content']} -> {labels[new_status]}")
                        save_store(store)
                        st.rerun()

    with tab_review:
        st.subheader("复习中心")
        due_reviews = [r for r in store["reviews"] if r.get("next_review", "") <= date.today().isoformat()]
        c1, c2 = st.columns(2)
        c1.metric("复习卡总数", len(store["reviews"]))
        c2.metric("今日到期", len(due_reviews))
        if not store["reviews"]:
            st.info("还没有复习卡。请先在“信息筛选”分析材料，再点击“将知识点生成复习卡”。")
        else:
            for review in due_reviews:
                with st.container(border=True):
                    st.caption(f"来源：{review.get('material_title', '')} · 复习次数：{review.get('review_count', 0)} · 熟悉度：{review.get('mastery', 0)}/5")
                    st.markdown(f"### {review['question']}")
                    if st.button("显示参考答案", key=f"reveal-btn-{review['id']}"):
                        st.session_state[f"show-answer-{review['id']}"] = True

                    if st.session_state.get(f"show-answer-{review['id']}", False):
                        st.markdown("**参考答案**")
                        st.write(review.get("answer", ""))
                        if review.get("explanation"):
                            st.markdown("**解释**")
                            st.write(review["explanation"])
                        yes, no = st.columns(2)
                        if yes.button("我记住了（3 天后复习）", key=f"yes-{review['id']}"):
                            updated = schedule_review(review, True)
                            store["reviews"] = [updated if r["id"] == review["id"] else r for r in store["reviews"]]
                            add_log(store, "复习打卡", f"掌握：{review['question'][:100]}")
                            save_store(store)
                            st.rerun()
                        if no.button("还不熟（明天复习）", key=f"no-{review['id']}"):
                            updated = schedule_review(review, False)
                            store["reviews"] = [updated if r["id"] == review["id"] else r for r in store["reviews"]]
                            add_log(store, "复习打卡", f"待加强：{review['question'][:100]}")
                            save_store(store)
                            st.rerun()
            future_reviews = [r for r in store["reviews"] if r.get("next_review", "") > date.today().isoformat()]
            if future_reviews:
                with st.expander(f"未来复习卡（{len(future_reviews)} 张）"):
                    for review in future_reviews:
                        st.write(f"{review['next_review']} · {review['question']}")

    with tab_audit:
        st.subheader("工具与学习操作记录")
        st.caption("记录学习画像、材料筛选、任务变更与复习打卡，帮助你检查系统实际做过什么。")
        if not store["audit_log"]:
            st.info("还没有操作记录。")
        else:
            st.dataframe(store["audit_log"], use_container_width=True, hide_index=True)
        payload = json.dumps(store, ensure_ascii=False, indent=2)
        st.download_button(
            "下载 learnflow_store.json",
            payload,
            file_name="learnflow_store.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.caption("LearnFlow MVP · 本地 JSON 存储 · 结果需要用户判断与复核；离线模式结果会明确标注，不应视为真正的 AI 语义分析。")


if __name__ == "__main__":
    main()
