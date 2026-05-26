"""
Multi-Agent Legal RAG System
─────────────────────────────
Orchestrator (Groq)   → query analysis, search-query decomposition
Retriever    (Local)  → BM25 + Vector hybrid + Cohere rerank
Generator    (Gemini) → long-context legal answer synthesis
"""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterator

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import src.config as config
from src.retriever import get_retriever, _format_results


# ═══════════════════════════════════════════════════════════
#  1. MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════

def _init_orchestrator() -> BaseChatModel:
    if config.ORCHESTRATOR_PROVIDER == "google":
        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_ORCHESTRATOR_MODEL,
            google_api_key=config.GOOGLE_ORCHESTRATOR_API_KEY,
            temperature=0,
        )
    if config.ORCHESTRATOR_PROVIDER == "groq":
        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )
    raise RuntimeError(f"Unsupported ORCHESTRATOR_PROVIDER: {config.ORCHESTRATOR_PROVIDER}")


def _init_generator() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
    )


orchestrator_llm = None
generator_llm = None


def get_orchestrator_llm() -> BaseChatModel:
    global orchestrator_llm
    if orchestrator_llm is None:
        if config.ORCHESTRATOR_PROVIDER == "google" and not config.GOOGLE_ORCHESTRATOR_API_KEY:
            raise RuntimeError("GOOGLE_ORCHESTRATOR_API_KEY is not set.")
        if config.ORCHESTRATOR_PROVIDER == "groq" and not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        orchestrator_llm = _init_orchestrator()
    return orchestrator_llm


def get_generator_llm() -> ChatGoogleGenerativeAI:
    global generator_llm
    if generator_llm is None:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        generator_llm = _init_generator()
    return generator_llm


# ═══════════════════════════════════════════════════════════
#  2. ORCHESTRATOR AGENT (Groq — ultra-fast)
# ═══════════════════════════════════════════════════════════

_ORCHESTRATOR_SYSTEM = """Bạn là Orchestrator Agent trong hệ thống tra cứu pháp luật Việt Nam.

NHIỆM VỤ: Phân tích câu hỏi và tạo TỐI ĐA 4 search queries độc lập, mỗi query bắt một khía cạnh pháp lý riêng.

Cơ sở dữ liệu chứa:
- Luật Đầu tư 2025 (143/2025/QH15) — luật hiện hành, thay thế hoàn toàn Luật Đầu tư 2020
- Luật Doanh nghiệp 2020 (59/2020/QH14) và Luật sửa đổi Luật Doanh nghiệp 2025 (76/2025/QH15)
- Luật sửa đổi nhiều luật 2022 (03/2022/QH15) — sửa đổi Luật Đầu tư, Luật Doanh nghiệp và các luật khác
- Nghị định 31/2021/NĐ-CP (hướng dẫn Luật Đầu tư, bao gồm Phụ lục I danh mục ngành nghề hạn chế)
- Nghị định 47/2021/NĐ-CP (hướng dẫn Luật Doanh nghiệp)
- Nghị định 153/2020/NĐ-CP (chào bán trái phiếu doanh nghiệp)
- Nghị định 168/2025/NĐ-CP (đăng ký doanh nghiệp)

NGUYÊN TẮC TẠO QUERIES (BẮT BUỘC):
1. KHÔNG dùng nguyên câu hỏi của user làm query — quá dài, quá chung chung, BM25 sẽ miss
2. Mỗi query = 1 cụm từ khóa pháp lý ngắn gọn (5-15 từ), tập trung vào THỰC THỂ + HÀNH VI + ĐIỀU KHOẢN
3. Với câu hỏi về ngành nghề hạn chế → LUÔN tạo query riêng cho "Phụ lục I" / "danh mục ngành nghề hạn chế"
4. Với câu hỏi về sở hữu nước ngoài → tạo query về "tỷ lệ sở hữu", "tiếp cận thị trường", "điều kiện đầu tư"
5. Với câu hỏi tra cứu Điều cụ thể → query = "Điều X [Tên luật/Nghị định]"
6. Với câu hỏi so sánh → 1 query mỗi văn bản

VÍ DỤ:
- User: "nhà đầu tư nước ngoài 100% vốn vận tải hàng hóa"
  → queries: ["vận tải hàng hóa nhà đầu tư nước ngoài tỷ lệ sở hữu Luật Đầu tư 2025", "ngành nghề hạn chế tiếp cận thị trường Phụ lục I Nghị định 31", "điều kiện đầu tư vận tải đường bộ nước ngoài", "tiếp cận thị trường có điều kiện nhà đầu tư nước ngoài 143/2025/QH15"]

NHẬN DIỆN CÂU HỎI NGOÀI PHẠM VI:
Nếu câu hỏi KHÔNG liên quan đến pháp luật Việt Nam trong cơ sở dữ liệu (hỏi cá nhân, nấu ăn, thời tiết, v.v.):
→ Trả về: {"out_of_scope": true, "out_of_scope_reply": "câu từ chối lịch sự bằng tiếng Việt"}

OUTPUT: Chỉ trả về JSON, không có text nào khác, không có markdown:
{"out_of_scope":false,"query_type":"lookup","search_queries":["query1","query2","query3"],"expand_links":true,"reasoning":"..."}"""


def _extract_json(raw: str) -> str:
    """Strip <think>...</think> blocks (Qwen3), markdown fences, then return first JSON object."""
    # Remove Qwen3 extended thinking blocks
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE).strip()
    # Extract from markdown code fences
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if fence:
        return fence.group(1).strip()
    # Extract first {...} JSON object
    brace = re.search(r'\{[\s\S]*\}', raw)
    if brace:
        return brace.group(0)
    return raw


def orchestrate(query: str) -> Dict:
    """Orchestrator analyzes the query and returns a retrieval strategy."""
    response = get_orchestrator_llm().invoke([
        SystemMessage(content=_ORCHESTRATOR_SYSTEM),
        HumanMessage(content=query),
    ])

    raw = response.content.strip()
    cleaned = _extract_json(raw)

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort fallback with minimal decomposition
        print(f"  ⚠ Orchestrator parse failed. Raw output:\n{raw[:300]}")
        plan = {
            "query_type": "lookup",
            "search_queries": [query],
            "expand_links": True,
            "reasoning": "Fallback — JSON parse error",
        }

    plan.setdefault("out_of_scope", False)
    plan.setdefault("query_type", "lookup")
    plan.setdefault("search_queries", [query])
    plan.setdefault("expand_links", True)
    if not plan.get("out_of_scope") and not plan["search_queries"]:
        plan["search_queries"] = [query]

    return plan


# ═══════════════════════════════════════════════════════════
#  3. RETRIEVER (Local — BM25 + Vector + Rerank)
# ═══════════════════════════════════════════════════════════

def _normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", value.lower()).strip()


def _focus_terms(focus_text: str) -> set[str]:
    stopwords = {
        "theo", "trong", "ngoai", "nuoc", "viet", "nam", "doanh", "nghiep",
        "dieu", "kien", "phap", "luat", "can", "nhung", "truoc", "ket", "luan",
    }
    normalized = _normalize_for_match(focus_text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 4 and token not in stopwords
    }


def _trim_text(text: str, max_chars: int, focus_text: str = "") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    terms = _focus_terms(focus_text)
    if terms:
        lines = text.splitlines()
        best_idx = -1
        best_score = 0
        for idx, line in enumerate(lines):
            normalized_line = _normalize_for_match(line)
            score = sum(1 for term in terms if term in normalized_line)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0:
            selected = [lines[best_idx]]
            used = len(lines[best_idx])
            left = best_idx - 1
            right = best_idx + 1
            while used < max_chars and (left >= 0 or right < len(lines)):
                if left >= 0:
                    candidate = lines[left]
                    if used + len(candidate) + 1 > max_chars:
                        break
                    selected.insert(0, candidate)
                    used += len(candidate) + 1
                    left -= 1
                if right < len(lines):
                    candidate = lines[right]
                    if used + len(candidate) + 1 > max_chars:
                        break
                    selected.append(candidate)
                    used += len(candidate) + 1
                    right += 1
            prefix = "...[trimmed before]\n" if left >= 0 else ""
            suffix = "\n...[trimmed after]" if right < len(lines) else ""
            return prefix + "\n".join(selected).rstrip() + suffix

    cutoff = text.rfind("\n", 0, max_chars)
    if cutoff < max_chars * 0.6:
        cutoff = max_chars
    return text[:cutoff].rstrip() + "\n...[truncated for speed]"


def _context_sort_key(result: Dict) -> tuple:
    reason = result.get("retrieval_reason", "")
    exact = 1 if reason.startswith("exact_") else 0
    main = 0 if result.get("is_reference") else 1
    return (main, exact, result.get("score", 0.0))


def _select_context_results(results: list[Dict], focus_text: str = "") -> list[Dict]:
    if not results:
        return []

    ranked = sorted(results, key=_context_sort_key, reverse=True)
    main_results = [res for res in ranked if not res.get("is_reference")]
    reference_results = [res for res in ranked if res.get("is_reference")]

    article_budget = max(1, config.MAX_CONTEXT_ARTICLES)
    reference_budget = max(0, min(config.MAX_REFERENCE_ARTICLES, article_budget))
    main_budget = max(1, article_budget - reference_budget)

    selected = main_results[:main_budget]
    remaining_budget = article_budget - len(selected)
    selected.extend(reference_results[: min(reference_budget, remaining_budget)])

    if len(selected) < article_budget:
        selected_ids = {res.get("id") for res in selected}
        for res in ranked:
            if res.get("id") not in selected_ids:
                selected.append(res)
                selected_ids.add(res.get("id"))
            if len(selected) >= article_budget:
                break

    trimmed = []
    for res in selected:
        item = dict(res)
        item["metadata"] = dict(res.get("metadata", {}))
        item["text"] = _trim_text(str(res.get("text", "")), config.MAX_ARTICLE_CHARS, focus_text)
        trimmed.append(item)
    return trimmed


def _format_context(results: list[Dict], focus_text: str = "") -> str:
    blocks = _format_results(_select_context_results(results, focus_text))
    if not blocks or config.MAX_CONTEXT_CHARS <= 0:
        return "\n".join(blocks)

    selected_blocks = []
    used_chars = 0
    for block in blocks:
        next_len = len(block) + 1
        if used_chars + next_len > config.MAX_CONTEXT_CHARS:
            if not selected_blocks:
                selected_blocks.append(_trim_text(block, config.MAX_CONTEXT_CHARS, focus_text))
            break
        selected_blocks.append(block)
        used_chars += next_len
    return "\n".join(selected_blocks)


def _build_supplemental_queries(user_query: str, plan: Dict, results: list[Dict]) -> list[str]:
    focus = " ".join([user_query, *plan.get("search_queries", [])])
    normalized_focus = _normalize_for_match(focus)
    normalized_context = _normalize_for_match(
        " ".join(
            " ".join(
                [
                    str(res.get("text", ""))[:2000],
                    str(res.get("metadata", {}).get("article", "")),
                    str(res.get("metadata", {}).get("number", "")),
                ]
            )
            for res in results
        )
    )

    queries = []
    has_foreign_investor = "nha dau tu nuoc ngoai" in normalized_focus
    has_capital_action = any(term in normalized_focus for term in ("gop von", "mua co phan", "phan von gop"))

    if has_foreign_investor:
        if "phu luc i" not in normalized_context and "nganh nghe han che" not in normalized_context:
            queries.append("Phụ lục I Nghị định 31 ngành nghề hạn chế tiếp cận thị trường nhà đầu tư nước ngoài")
        if "tiep can thi truong" not in normalized_context:
            queries.append("Điều 8 Luật Đầu tư 2025 tiếp cận thị trường nhà đầu tư nước ngoài")
        if has_capital_action and "gop von" not in normalized_context:
            queries.append("Điều 24 Luật Đầu tư 2025 góp vốn mua cổ phần nhà đầu tư nước ngoài")
            queries.append("Điều 65 Nghị định 31 góp vốn mua cổ phần nhà đầu tư nước ngoài")
        if any(term in normalized_focus for term in ("ty le", "60", "so huu")) and "ty le so huu" not in normalized_context:
            queries.append("tỷ lệ sở hữu nhà đầu tư nước ngoài góp vốn mua cổ phần")
        if "giao duc" in normalized_focus and "giao duc" not in normalized_context:
            queries.append("giáo dục Phụ lục I Nghị định 31 nhà đầu tư nước ngoài")

    existing = {_normalize_for_match(q) for q in plan.get("search_queries", [])}
    deduped = []
    for query in queries:
        normalized_query = _normalize_for_match(query)
        if normalized_query not in existing and normalized_query not in {_normalize_for_match(q) for q in deduped}:
            deduped.append(query)
    return deduped[:3]


def _build_topic_shortcut_queries(user_query: str) -> list[str]:
    normalized = _normalize_for_match(user_query)
    mentions_llc = "tnhh" in normalized or "trach nhiem huu han" in normalized
    mentions_jsc = "cong ty co phan" in normalized or "co dong" in normalized
    asks_comparison = any(
        term in normalized
        for term in ("khac", "so sanh", "doi chieu", "thanh vien", "co dong", "chuyen nhuong")
    )

    if mentions_llc and mentions_jsc and asks_comparison:
        return [
            "Điều 46 Luật Doanh nghiệp 2020 công ty trách nhiệm hữu hạn hai thành viên trở lên",
            "Điều 47 Luật Doanh nghiệp 2020 góp vốn thành lập công ty trách nhiệm hữu hạn",
            "Điều 52 Luật Doanh nghiệp 2020 chuyển nhượng phần vốn góp",
            "Điều 111 Luật Doanh nghiệp 2020 công ty cổ phần",
            "Điều 115 Luật Doanh nghiệp 2020 quyền của cổ đông phổ thông",
            "Điều 127 Luật Doanh nghiệp 2020 chuyển nhượng cổ phần",
        ]

    if "chuyen doi" in normalized and mentions_llc and "co phan" in normalized:
        return [
            "Điều 202 Luật Doanh nghiệp 2020 chuyển đổi công ty trách nhiệm hữu hạn thành công ty cổ phần",
            "Điều 111 Luật Doanh nghiệp 2020 công ty cổ phần",
            "Điều 127 Luật Doanh nghiệp 2020 chuyển nhượng cổ phần",
            "Điều 120 Luật Doanh nghiệp 2020 cổ phần phổ thông của cổ đông sáng lập",
        ]

    return []


def retrieve(plan: Dict, user_query: str = "") -> str:
    """Execute retrieval based on the orchestrator's plan."""
    retriever = get_retriever()

    query_type = plan["query_type"]
    queries = plan["search_queries"][: config.MAX_SEARCH_QUERIES]
    if len(plan["search_queries"]) > len(queries):
        plan["search_queries_truncated"] = plan["search_queries"][len(queries):]
        plan["search_queries"] = queries
    expand = plan.get("expand_links", True)

    shortcut_queries = _build_topic_shortcut_queries(user_query)
    if shortcut_queries:
        plan["shortcut_queries"] = shortcut_queries
        seen = set()
        shortcut_results = []
        for q in shortcut_queries:
            for res in retriever.retrieve(q, expand_links=False):
                aid = res["metadata"].get("article_id", "")
                if aid not in seen:
                    seen.add(aid)
                    shortcut_results.append(res)
        if shortcut_results:
            focus_text = " ".join([user_query, *shortcut_queries])
            return _format_context(shortcut_results, focus_text)

    if query_type == "comparison" and len(queries) >= 2:
        seen = set()
        all_results = []
        for q in queries:
            for res in retriever.retrieve(q, expand_links=False):
                aid = res["metadata"].get("article_id", "")
                if aid not in seen:
                    seen.add(aid)
                    all_results.append(res)
        focus_text = " ".join([user_query, *queries])
        return _format_context(all_results, focus_text)

    # lookup / consultation — merge results from all queries
    seen = set()
    all_results = []
    for q in queries:
        for res in retriever.retrieve(q, expand_links=expand):
            aid = res["metadata"].get("article_id", "")
            if aid not in seen:
                seen.add(aid)
                all_results.append(res)
    supplemental_queries = _build_supplemental_queries(user_query, plan, all_results)
    if supplemental_queries:
        plan["supplemental_queries"] = supplemental_queries
    for q in supplemental_queries:
        for res in retriever.retrieve(q, expand_links=expand):
            aid = res["metadata"].get("article_id", "")
            if aid not in seen:
                seen.add(aid)
                all_results.append(res)

    focus_text = " ".join([user_query, *queries, *supplemental_queries])
    return _format_context(all_results, focus_text)


# ═══════════════════════════════════════════════════════════
#  4. GENERATOR AGENT (Gemini — large context, accurate)
# ═══════════════════════════════════════════════════════════

_GENERATOR_SYSTEM = (
    "Bạn là CHUYÊN GIA TƯ VẤN PHÁP LUẬT VietLegal-RAG.\n\n"

    "── KIỂM TRA DỮ LIỆU TRƯỚC KHI TRẢ LỜI ──\n"
    "Trước tiên, đánh giá xem các văn bản tra cứu có ĐỦ để trả lời dứt khoát không.\n"
    "Nếu THIẾU dữ liệu then chốt (ví dụ: câu hỏi về ngành nghề hạn chế nhưng không có Phụ lục I;\n"
    "câu hỏi về tỷ lệ sở hữu nhưng chỉ có nguyên tắc chung, không có danh mục cụ thể), PHẢI:\n"
    "  1. Trả lời phần nào CÓ thể kết luận được từ dữ liệu hiện có.\n"
    "  2. Nêu rõ: '!Dữ liệu còn thiếu: [tên văn bản/phụ lục cụ thể]'\n"
    "  3. Nêu rõ: 'Để kết luận dứt khoát, cần tra cứu thêm: [tên cụ thể].'\n"
    "TUYỆT ĐỐI KHÔNG liệt kê nguyên tắc chung rồi nói 'hãy tự tra Phụ lục'.\n\n"

    "── CÁCH TRẢ LỜI ──\n\n"

    "A) HỎI TRA CỨU (\"Điều X quy định gì?\"):\n"
    "   Liệt kê đầy đủ tất cả Khoản.\n\n"

    "B) HỎI TƯ VẤN (\"được không?\", \"cần lưu ý?\", \"thủ tục thế nào?\"):\n"
    "   ### Phân tích\n"
    "   - Tách từng HÀNH VI riêng (VD: 'thành lập DN' ≠ 'mua cổ phần').\n"
    "   - Với mỗi hành vi: Khoản nào áp dụng → ĐƯỢC / KHÔNG ĐƯỢC / CÓ ĐIỀU KIỆN.\n"
    "   ### Kết luận\n"
    "   Kết luận rõ ràng cho từng hành vi. Không mập mờ.\n"
    "   ### Cơ sở pháp lý\n"
    "   Dẫn chiếu: Khoản — Điều — Văn bản.\n\n"

    "C) HỎI SO SÁNH: Trình bày đối chiếu hoặc bảng.\n\n"

    "── QUY TẮC ──\n"
    "- TRUNG LẬP: Không thiên về 'cấm' hay 'cho phép'.\n"
    "- CHÍNH XÁC: Khoản cấm 'thành lập DN' KHÔNG đồng nghĩa cấm 'mua cổ phần'.\n"
    "- Dẫn chiếu ngắn gọn: '(Khoản X Điều Y, Tên văn bản)'.\n"
    "- KHÔNG bịa thông tin ngoài văn bản tra cứu.\n"
    "- Nếu context chỉ có nguyên tắc chung, PHẢI nêu rõ giới hạn đó.\n"
)


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _sanitize_generated_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return text


def generate(query: str, context: str, query_type: str) -> str:
    """Generator synthesizes a legal answer from retrieved documents."""
    if query_type == "out_of_scope":
        user_prompt = (
            f"Người dùng hỏi: {query}\n\n"
            f"Câu hỏi này không liên quan đến pháp luật. Hãy trả lời thân thiện, tự nhiên."
        )
    else:
        user_prompt = (
            f"CÂU HỎI: {query}\n\n"
            f"LOẠI CÂU HỎI: {query_type}\n\n"
            f"VĂN BẢN PHÁP LUẬT TRA CỨU ĐƯỢC:\n{context}"
        )

    response = get_generator_llm().invoke([
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    return _sanitize_generated_text(_message_content_to_text(response.content))


def generate_stream(query: str, context: str, query_type: str) -> Iterator[str]:
    """Stream generated answer chunks for the web UI."""
    if query_type == "out_of_scope":
        user_prompt = (
            f"Người dùng hỏi: {query}\n\n"
            "Câu hỏi này không liên quan đến pháp luật. Hãy trả lời thân thiện, tự nhiên."
        )
    else:
        user_prompt = (
            f"CÂU HỎI: {query}\n\n"
            f"LOẠI CÂU HỎI: {query_type}\n\n"
            f"VĂN BẢN PHÁP LUẬT TRA CỨU ĐƯỢC:\n{context}"
        )

    for chunk in get_generator_llm().stream([
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]):
        text = _message_content_to_text(chunk.content)
        if text:
            yield _sanitize_generated_text(text)


def run_query_pipeline(query: str, retrieval_only: bool = False) -> Dict:
    timings = {}

    t0 = time.perf_counter()
    try:
        plan = orchestrate(query)
    except RuntimeError as exc:
        plan = {
            "out_of_scope": False,
            "query_type": "lookup",
            "search_queries": [query],
            "expand_links": True,
            "reasoning": f"Fallback because orchestrator is unavailable: {exc}",
        }
    timings["orchestrator"] = time.perf_counter() - t0

    if plan.get("out_of_scope"):
        return {
            "query": query,
            "plan": plan,
            "context": "",
            "answer": plan.get("out_of_scope_reply", ""),
            "timings": timings,
        }

    t1 = time.perf_counter()
    context = retrieve(plan, query)
    timings["retriever"] = time.perf_counter() - t1

    answer = ""
    generator_error = ""
    if not retrieval_only and context.strip():
        t2 = time.perf_counter()
        try:
            answer = generate(query, context, plan["query_type"])
        except RuntimeError as exc:
            generator_error = str(exc)
        timings["generator"] = time.perf_counter() - t2

    return {
        "query": query,
        "plan": plan,
        "context": context,
        "answer": answer,
        "generator_error": generator_error,
        "timings": timings,
    }


# ═══════════════════════════════════════════════════════════
#  5. MULTI-AGENT PIPELINE
# ═══════════════════════════════════════════════════════════

def run_query(query: str, mode: str = "fast"):
    """
    Main entry point.
    - 'fast' mode: Orchestrator → Retriever → Generator (no tool-call loop)
    - 'agent' mode: same pipeline (legacy flag kept for CLI compat)
    """
    print(f"\n{'='*60}")
    print(f"  [NGƯỜI DÙNG]: {query}")
    print(f"{'='*60}")

    # ── Step 1: Orchestrator (Groq) ──
    t0 = time.perf_counter()
    print("\n▶ [Orchestrator · Groq] Đang phân tích câu hỏi...")
    plan = orchestrate(query)
    t_orch = time.perf_counter() - t0

    # ── Short-circuit: out of scope — use Orchestrator's reply directly, skip Generator ──
    if plan.get("out_of_scope"):
        answer = plan.get("out_of_scope_reply", "Xin lỗi, tôi chỉ hỗ trợ tra cứu pháp luật Việt Nam.")
        print(f"  ↩ Ngoài phạm vi — trả lời trực tiếp (không gọi Generator)  ⏱ {t_orch:.2f}s")
        print(f"\n{'─'*60}")
        print(f"[AI]:\n{answer}")
        print(f"{'─'*60}")
        print(f"\n⏱ Tổng: {t_orch:.2f}s (Orchestrator: {t_orch:.2f}s)")
        print(f"{'='*60}\n")
        return

    print(f"  Loại: {plan['query_type']}")
    print(f"  Queries: {plan['search_queries']}")
    print(f"  Expand links: {plan.get('expand_links')}")
    print(f"  Reasoning: {plan.get('reasoning', '')}")
    print(f"  ⏱ {t_orch:.2f}s")

    # ── Step 2: Retriever (Local) ──
    t1 = time.perf_counter()
    print("\n▶ [Retriever · Local] Đang tra cứu BM25 + Vector + Rerank...")
    context = retrieve(plan, query)
    t_ret = time.perf_counter() - t1

    if not context.strip():
        print("  ⚠ Không tìm thấy văn bản pháp luật liên quan.")
        print(f"\n{'='*60}")
        return

    doc_count = context.count("=== KẾT QUẢ CHÍNH") + context.count("=== VĂN BẢN DẪN CHIẾU")
    print(f"  Tìm thấy {doc_count} điều luật liên quan.")
    print(f"  ⏱ {t_ret:.2f}s")

    # ── Step 3: Generator (Gemini) ──
    t2 = time.perf_counter()
    print("\n▶ [Generator · Gemini] Đang tổng hợp câu trả lời pháp lý...")
    answer = generate(query, context, plan["query_type"])
    t_gen = time.perf_counter() - t2

    print(f"\n{'─'*60}")
    print(f"[AI]:\n{answer}")
    print(f"{'─'*60}")
    print(f"\n⏱ Tổng: {t_orch + t_ret + t_gen:.2f}s "
          f"(Orchestrator: {t_orch:.2f}s | Retriever: {t_ret:.2f}s | Generator: {t_gen:.2f}s)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    mode = "agent" if "--agent" in args else "fast"
    args = [a for a in args if a != "--agent"]
    user_query = " ".join(args) if args else "Điều 17 Luật Doanh nghiệp 2020 quy định về điều gì?"
    run_query(user_query, mode=mode)
