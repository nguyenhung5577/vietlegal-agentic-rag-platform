import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import generate_stream, orchestrate, retrieve


SAMPLE_QUESTIONS = [
    "Công ty trách nhiệm hữu hạn một thành viên là gì?",
    "Luật Doanh nghiệp điều chỉnh những loại hình doanh nghiệp nào?",
    "Doanh nghiệp có được kinh doanh ngành nghề chưa ghi trong Giấy chứng nhận đăng ký doanh nghiệp không?",
    "Công ty TNHH hai thành viên trở lên khác công ty cổ phần ở điểm nào về thành viên/cổ đông và chuyển nhượng vốn?",
    "Một công ty TNHH muốn chuyển đổi thành công ty cổ phần để gọi vốn. Các vấn đề pháp lý chính cần kiểm tra là gì?",
    "Tôi muốn biết cách tránh bị xử phạt khi không đăng ký thay đổi nội dung doanh nghiệp.",
    "Tranh chấp giữa cổ đông thì cơ quan đăng ký kinh doanh có giải quyết không?",
]


def _set_sample_question(question: str) -> None:
    st.session_state["query_input"] = question


def _split_context_blocks(context: str) -> list[str]:
    blocks = re.split(r"\n=+\n", context)
    return [block.strip() for block in blocks if block.strip()]


def _run_retrieval(query: str):
    cache = st.session_state.setdefault("retrieval_cache", {})
    if query in cache:
        return cache[query]

    timings = {}

    t0 = time.perf_counter()
    plan = orchestrate(query)
    timings["orchestrator"] = time.perf_counter() - t0

    if plan.get("out_of_scope"):
        return plan, "", timings

    t1 = time.perf_counter()
    context = retrieve(plan, query)
    timings["retriever"] = time.perf_counter() - t1
    result = (plan, context, timings)
    cache[query] = result
    return result


st.set_page_config(page_title="VietLegal RAG", layout="wide")

st.title("VietLegal RAG")
st.caption("Tra cứu pháp luật Việt Nam với agentic query planning, hybrid retrieval, rerank và câu trả lời có căn cứ.")

with st.sidebar:
    st.header("VietLegalRAG")
    st.write(
        "Demo RAG pháp lý tiếng Việt cho tra cứu điều luật, tư vấn tình huống đầu tư/doanh nghiệp "
        "và kiểm tra căn cứ pháp lý trong tập văn bản đã nạp."
    )
    st.divider()
    st.subheader("Có thể tra cứu")
    st.markdown(
        """
- Luật Doanh nghiệp 2020 và luật sửa đổi 2025
- Luật Đầu tư 2025
- Nghị định 31/2021/NĐ-CP
- Nghị định 47/2021/NĐ-CP
- Nghị định 153/2020/NĐ-CP
- Nghị định 168/2025/NĐ-CP
- Các phụ lục, danh mục ngành nghề và điều khoản liên quan trong dữ liệu
        """.strip()
    )
    st.divider()
    st.caption("Hệ thống là demo kỹ thuật RAG, không thay thế ý kiến tư vấn pháp lý chính thức.")


default_query = (
    "Một nhà đầu tư Hàn Quốc muốn góp 60% vốn vào một công ty Việt Nam hoạt động "
    "trong ngành giáo dục. Hệ thống cần kiểm tra những nhóm điều kiện pháp lý nào trước khi kết luận?"
)
if "query_input" not in st.session_state:
    st.session_state["query_input"] = default_query

query = st.text_area("Câu hỏi", key="query_input", height=105)

with st.expander("Câu hỏi mẫu", expanded=True):
    cols = st.columns(2)
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        cols[idx % 2].button(
            sample,
            key=f"sample_question_{idx}",
            use_container_width=True,
            on_click=_set_sample_question,
            args=(sample,),
        )

if st.button("Tra cứu", type="primary", disabled=not query.strip()):
    query = query.strip()

    with st.spinner("Đang phân tích câu hỏi và truy xuất văn bản..."):
        try:
            plan, context, timings = _run_retrieval(query)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Orchestrator", f"{timings.get('orchestrator', 0):.2f}s")
    metric_cols[1].metric("Retriever", f"{timings.get('retriever', 0):.2f}s")
    generator_metric = metric_cols[2].empty()
    generator_metric.metric("Generator", "streaming")
    total_metric = metric_cols[3].empty()
    total_metric.metric("Total", f"{sum(timings.values()):.2f}s")

    st.subheader("Answer")
    if plan.get("out_of_scope"):
        st.markdown(plan.get("out_of_scope_reply", "Câu hỏi nằm ngoài phạm vi tra cứu pháp luật Việt Nam."))
    elif not context.strip():
        st.warning("Không tìm thấy văn bản pháp luật liên quan trong cơ sở dữ liệu.")
    else:
        t2 = time.perf_counter()
        try:
            answer = st.write_stream(generate_stream(query, context, plan["query_type"]))
        except Exception as exc:
            st.error(f"Generator failed: {exc}")
            st.stop()
        timings["generator"] = time.perf_counter() - t2
        generator_metric.metric("Generator", f"{timings['generator']:.2f}s")
        total_metric.metric("Total", f"{sum(timings.values()):.2f}s")
        st.session_state["last_answer"] = answer

    with st.expander("Retrieval plan", expanded=False):
        st.json(plan)

    with st.expander("Retrieved context", expanded=False):
        blocks = _split_context_blocks(context)
        if not blocks:
            st.warning("No retrieved context.")
        for idx, block in enumerate(blocks, start=1):
            title = block.splitlines()[0][:140] if block.splitlines() else f"Result {idx}"
            with st.expander(f"{idx}. {title}", expanded=idx <= 2):
                st.text(block)
