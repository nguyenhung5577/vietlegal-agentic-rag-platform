"""
Multi-Agent Legal RAG System
─────────────────────────────
Orchestrator (Groq)   → query analysis, search-query decomposition
Retriever    (Local)  → BM25 + Vector hybrid + Cohere rerank
Generator    (Gemini) → long-context legal answer synthesis
"""

import json
import re
import time
from typing import Dict

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import src.config as config
from src.retriever import get_retriever, _format_results


# ═══════════════════════════════════════════════════════════
#  1. MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════

def _init_orchestrator() -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,
    )


def _init_generator() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
    )


orchestrator_llm = _init_orchestrator()
generator_llm = _init_generator()


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
    response = orchestrator_llm.invoke([
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

def retrieve(plan: Dict) -> str:
    """Execute retrieval based on the orchestrator's plan."""
    retriever = get_retriever()

    query_type = plan["query_type"]
    queries = plan["search_queries"]
    expand = plan.get("expand_links", True)

    if query_type == "comparison" and len(queries) >= 2:
        seen = set()
        all_results = []
        for q in queries:
            for res in retriever.retrieve(q, expand_links=False):
                aid = res["metadata"].get("article_id", "")
                if aid not in seen:
                    seen.add(aid)
                    all_results.append(res)
        return "\n".join(_format_results(all_results))

    # lookup / consultation — merge results from all queries
    seen = set()
    all_results = []
    for q in queries:
        for res in retriever.retrieve(q, expand_links=expand):
            aid = res["metadata"].get("article_id", "")
            if aid not in seen:
                seen.add(aid)
                all_results.append(res)
    return "\n".join(_format_results(all_results))


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

    response = generator_llm.invoke([
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    return response.content


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
    context = retrieve(plan)
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
