import os
import re
import yaml
from typing import Dict, List
from dataclasses import dataclass, asdict


MAX_NODE_CHARS = 1800


@dataclass
class LegalNode:
    text: str
    metadata: Dict
    article_id: str


def _safe_id(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    return re.sub(r"[^0-9A-Za-zÀ-ỹ_/-]+", "_", value).replace("/", "_")


class HierarchicalLegalIngestor:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.chapter_pattern = re.compile(r"^##\s+(Chương\s+.+)$", re.MULTILINE | re.IGNORECASE)
        self.article_pattern = re.compile(
            r"^(?:###\s+|\*\*)(Điều\s+\d+\..+?)(?:\*\*)?$",
            re.MULTILINE | re.IGNORECASE,
        )
        self.clause_pattern = re.compile(r"\n(\d+\.|[a-zđ]\))\s", re.IGNORECASE)
        self.phuluc_pattern = re.compile(r"^(PHỤ LỤC\s+[IVXLCDM]+[^\n]*)", re.MULTILINE | re.IGNORECASE)
        self.table_row_pattern = re.compile(r"^Theo mục\s+'(.+?)'.*?có các thông tin chi tiết sau:\s*(.+)$", re.DOTALL)
        self.section_pattern = re.compile(r"^([A-ZĐ]\.\s+.+|\d+\.\s+.+|[IVXLCDM]+\.\s+.+)$")

    def _extract_doc_title(self, content: str, frontmatter: dict) -> str:
        doc_num = frontmatter.get("number", "")
        issued_date = frontmatter.get("issued_date", "")
        year = issued_date[:4] if issued_date else ""

        match = re.search(
            r"^#\s+(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH)\s*$\n+\s*(.+?)$",
            content,
            re.MULTILINE,
        )
        if match:
            doc_type = match.group(1).strip()
            doc_name = match.group(2).strip()
            if doc_type == "LUẬT":
                title = f"Luật {doc_name.capitalize()}"
                return f"{title} {year}" if year else title
            return f"{doc_type.capitalize()} {doc_num}" if doc_num else doc_type.capitalize()

        return f"Văn bản {doc_num}"

    def parse_file(self, file_path: str) -> List[LegalNode]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        frontmatter = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                main_content = parts[2]
            else:
                main_content = content
        else:
            main_content = content

        doc_num = frontmatter.get("number", "Unknown")
        doc_title = self._extract_doc_title(main_content, frontmatter)
        frontmatter["title"] = doc_title

        legal_body, appendix_body = self._split_legal_body_and_appendices(main_content)
        nodes = self._parse_articles(legal_body, frontmatter, doc_num, doc_title)
        nodes.extend(self._parse_phuluc(appendix_body, frontmatter, doc_num, doc_title))
        return nodes

    def _split_legal_body_and_appendices(self, content: str) -> tuple[str, str]:
        first_appendix = self.phuluc_pattern.search(content)
        if not first_appendix:
            return content, ""
        return content[: first_appendix.start()].rstrip(), content[first_appendix.start() :].lstrip()

    def _parse_articles(self, content: str, frontmatter: dict, doc_num: str, doc_title: str) -> List[LegalNode]:
        nodes = []
        for chap_title, chap_content in self._split_with_headers(content, self.chapter_pattern):
            for art_title, art_content in self._split_with_headers(chap_content, self.article_pattern):
                art_match = re.search(r"Điều\s+(\d+)", art_title, re.IGNORECASE)
                if not art_match:
                    continue

                art_num = art_match.group(1)
                base_id = _safe_id(f"{doc_num}_A{art_num}")
                chunks = self._split_article_content(art_content)

                for i, chunk_text in enumerate(chunks, start=1):
                    if not chunk_text.strip():
                        continue
                    contextual_text = f"[{doc_title} ({doc_num})] [{chap_title}] [{art_title}]\n{chunk_text.strip()}"
                    metadata = {
                        **frontmatter,
                        "chapter": chap_title,
                        "article": art_title,
                        "article_number": art_num,
                        "clause_idx": i,
                        "article_id": base_id,
                        "chunk_type": "article",
                        "cross_references": ";".join(self._extract_references(chunk_text)),
                    }
                    nodes.append(LegalNode(text=contextual_text, metadata=metadata, article_id=f"{base_id}_C{i}"))
        return nodes

    def _split_article_content(self, text: str) -> List[str]:
        if len(text) <= MAX_NODE_CHARS:
            return [text]

        clauses = self._split_by_clauses(text)
        chunks = []
        for clause in clauses:
            chunks.extend(self._split_long_text(clause, MAX_NODE_CHARS))
        return chunks

    def _parse_phuluc(self, content: str, frontmatter: dict, doc_num: str, doc_title: str) -> List[LegalNode]:
        nodes = []
        if not content.strip():
            return nodes

        pl_matches = list(self.phuluc_pattern.finditer(content))
        for i, pl_match in enumerate(pl_matches):
            pl_title = pl_match.group(1).strip()
            pl_start = pl_match.end()
            pl_end = pl_matches[i + 1].start() if i + 1 < len(pl_matches) else len(content)
            pl_body = content[pl_start:pl_end].strip()

            roman = re.search(r"PHỤ LỤC\s+([IVXLCDM]+)", pl_title, re.IGNORECASE)
            pl_token = f"PLUC_{roman.group(1)}" if roman else f"PLUC_{i + 1}"
            units = self._split_appendix_units(pl_body)

            for j, unit in enumerate(units, start=1):
                unit_title = self._appendix_unit_title(unit, j)
                base_id = _safe_id(f"{doc_num}_{pl_token}_U{j:04d}")
                text = f"[{doc_title} ({doc_num})] [{pl_title}] [{unit_title}]\n{unit.strip()}"
                metadata = {
                    **frontmatter,
                    "chapter": pl_title,
                    "article": f"{pl_title} - {unit_title}",
                    "appendix": pl_title,
                    "article_id": base_id,
                    "chunk_type": "appendix",
                    "cross_references": "",
                }
                nodes.append(LegalNode(text=text, metadata=metadata, article_id=f"{base_id}_C0"))

        return nodes

    def _split_appendix_units(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: List[str] = []
        buffer: List[str] = []

        def flush_buffer():
            if buffer:
                units.extend(self._split_long_text("\n\n".join(buffer), MAX_NODE_CHARS))
                buffer.clear()

        for paragraph in paragraphs:
            if paragraph.startswith("Theo mục"):
                flush_buffer()
                units.extend(self._split_long_text(paragraph, MAX_NODE_CHARS))
                continue

            first_line = paragraph.splitlines()[0].strip()
            starts_section = bool(self.section_pattern.match(first_line))
            if starts_section and buffer:
                flush_buffer()

            if sum(len(p) for p in buffer) + len(paragraph) > MAX_NODE_CHARS:
                flush_buffer()
            buffer.append(paragraph)

        flush_buffer()
        return units

    def _appendix_unit_title(self, text: str, idx: int) -> str:
        table_match = self.table_row_pattern.match(text)
        if table_match:
            return table_match.group(1).strip()[:120]

        for line in text.splitlines():
            clean = line.strip(" *")
            if clean:
                return clean[:120]
        return f"Mục {idx}"

    def _split_with_headers(self, text: str, pattern: re.Pattern) -> List[tuple]:
        matches = list(pattern.finditer(text))
        if not matches:
            return [("Nội dung không phân chương", text)]

        results = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            header = match.group(1).strip()
            body = text[match.end() : end].strip()
            results.append((header, body))
        return results

    def _split_by_clauses(self, text: str) -> List[str]:
        parts = self.clause_pattern.split(text)
        if len(parts) < 2:
            return [text]

        clauses = []
        intro = parts[0].strip()
        if intro:
            clauses.append(intro)

        for i in range(1, len(parts), 2):
            marker = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            clause = f"{marker} {body}".strip()
            if clause:
                clauses.append(clause)
        return clauses

    def _split_long_text(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]

        parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        chunks = []
        current = ""
        for part in parts:
            if len(part) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
                continue

            candidate = f"{current}\n{part}".strip() if current else part
            if len(candidate) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks

    def _extract_references(self, text: str) -> List[str]:
        refs = []
        doc_map = {
            "Luật Doanh nghiệp": "59/2020/QH14",
            "Nghị định 31": "31/2021/NĐ-CP",
            "Nghị định 168": "168/2025/NĐ-CP",
            "Nghị định 153": "153/2020/NĐ-CP",
            "Luật Đầu tư": "143/2025/QH15",
        }

        pattern = re.compile(
            r"Điều\s+(\d+)(?:\s+(?:của\s+)?(?:Nghị định|Luật|Văn bản|Thông tư)\s+([\w\s\d/.\-Đđ]+))?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            art_num = match.group(1)
            doc_ref_raw = match.group(2)
            doc_id = "CURRENT"
            if doc_ref_raw:
                doc_ref = doc_ref_raw.strip()
                for name, num in doc_map.items():
                    if name.lower() in doc_ref.lower():
                        doc_id = num
                        break
                else:
                    num_match = re.search(r"(\d+/\d+/[A-ZĐ\-]+)", doc_ref)
                    doc_id = num_match.group(1) if num_match else doc_ref
            refs.append(_safe_id(f"{doc_id}_A{art_num}"))

        for doc_name, doc_id in doc_map.items():
            if doc_name.lower() in text.lower():
                refs.append(doc_id)

        return list(set(refs))


def main():
    import json

    ingestor = HierarchicalLegalIngestor("data/markdown")
    all_nodes = []
    for filename in os.listdir("data/markdown"):
        if filename.endswith(".md"):
            print(f"Processing {filename}...")
            all_nodes.extend(ingestor.parse_file(os.path.join("data/markdown", filename)))

    print(f"Total chunks created: {len(all_nodes)}")
    output_sample = [asdict(n) for n in all_nodes[:5]]
    with open("data/ingested_sample.json", "w", encoding="utf-8") as f:
        json.dump(output_sample, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
