import os
import re
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class LegalNode:
    text: str
    metadata: Dict
    article_id: str

class HierarchicalLegalIngestor:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        # Regex for Chapters, Articles, and Clauses
        self.chapter_pattern = re.compile(r'^## (Chương .+)$', re.MULTILINE)
        self.article_pattern = re.compile(r'^(?:### |\*\*)(Điều \d+\..+?)(?:\*\*|)$', re.MULTILINE)
        self.clause_pattern = re.compile(r'\n(\d+\.|[a-z]\))\s')
        # Regex for Appendices (plain-text headers, no ## prefix)
        self.phuluc_pattern = re.compile(r'^(PHỤ LỤC\s+[IVX]+[^\n]*)', re.MULTILINE)
        # Mục A./B./C. lines — capture the full line
        self.muc_pattern = re.compile(r'^([A-Z]\.\s+[^\n]+)', re.MULTILINE)

    def _extract_doc_title(self, content: str, frontmatter: dict) -> str:
        """Extract human-readable title like 'Luật Doanh nghiệp 2020' from heading."""
        doc_num = frontmatter.get('number', '')
        issued_date = frontmatter.get('issued_date', '')
        year = issued_date[:4] if issued_date else ''

        match = re.search(r'^#\s+(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH)\s*$\n+\s*(.+?)$', content, re.MULTILINE)
        if match:
            doc_type = match.group(1).strip()
            doc_name = match.group(2).strip()

            if doc_type == "LUẬT":
                title = f"Luật {doc_name.capitalize()}"
                return f"{title} {year}" if year else title
            else:
                type_name = doc_type.capitalize()
                return f"{type_name} {doc_num}" if doc_num else type_name

        return f"Văn bản {doc_num}"

    def parse_file(self, file_path: str) -> List[LegalNode]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract frontmatter
        frontmatter = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                main_content = parts[2]
        else:
            main_content = content

        nodes = []
        doc_num = frontmatter.get('number', 'Unknown')
        doc_title = self._extract_doc_title(main_content, frontmatter)
        frontmatter['title'] = doc_title
        
        # Split by Chapter
        chapters = self._split_with_headers(main_content, self.chapter_pattern)
        
        for chap_title, chap_content in chapters:
            # Split Chapter by Article
            articles = self._split_with_headers(chap_content, self.article_pattern)
            
            for art_title, art_content in articles:
                # Clean Article Title for ID
                art_match = re.search(r'Điều (\d+)', art_title)
                art_num = art_match.group(1) if art_match else "unknown"
                article_id = f"{doc_num}_A{art_num}".replace('/', '_')
                
                # Further split into Clauses if necessary (simple heuristic: length > 1000)
                if len(art_content) > 1000:
                    clauses = self._split_by_clauses(art_content)
                    for i, clause_text in enumerate(clauses):
                        contextual_text = f"[{doc_title} ({doc_num})] [{chap_title}] [{art_title}] \n{clause_text.strip()}"
                        unique_id = f"{article_id}_C{i+1}"
                        
                        metadata = {
                            **frontmatter,
                            "chapter": chap_title,
                            "article": art_title,
                            "clause_idx": i + 1,
                            "article_id": article_id,  # Keep the base article_id for logical grouping
                            "cross_references": ";".join(self._extract_references(clause_text))
                        }
                        nodes.append(LegalNode(text=contextual_text, metadata=metadata, article_id=unique_id))
                else:
                    contextual_text = f"[{doc_title} ({doc_num})] [{chap_title}] [{art_title}] \n{art_content.strip()}"
                    unique_id = f"{article_id}_C0"
                    metadata = {
                        **frontmatter,
                        "chapter": chap_title,
                        "article": art_title,
                        "article_id": article_id,
                        "cross_references": ";".join(self._extract_references(art_content))
                    }
                    nodes.append(LegalNode(text=contextual_text, metadata=metadata, article_id=unique_id))
        
        # Parse Appendices (Phụ lục) — plain-text sections ignored by chapter/article parser
        nodes.extend(self._parse_phuluc(main_content, frontmatter, doc_num, doc_title))

        return nodes

    def _parse_phuluc(self, content: str, frontmatter: dict, doc_num: str, doc_title: str) -> List[LegalNode]:
        """Parse Phụ lục sections (PHỤ LỤC I, II, ...) that use plain-text headers."""
        nodes = []
        pl_matches = list(self.phuluc_pattern.finditer(content))
        if not pl_matches:
            return nodes

        for i, pl_match in enumerate(pl_matches):
            pl_title = pl_match.group(1).strip()
            pl_start = pl_match.end()
            pl_end = pl_matches[i + 1].start() if i + 1 < len(pl_matches) else len(content)
            pl_body = content[pl_start:pl_end].strip()

            # Build a safe ID token: "PHỤ LỤC I" → "PLUC_I", "PHỤ LỤC II" → "PLUC_II"
            roman = re.search(r'[IVX]+', pl_title)
            pl_id_token = f"PLUC_{roman.group(0)}" if roman else "PLUC"

            # Split by Mục (A., B., C. ...)
            muc_matches = list(self.muc_pattern.finditer(pl_body))

            if not muc_matches:
                # No Mục sub-division — index the whole appendix as one node
                article_id = f"{doc_num}_{pl_id_token}".replace('/', '_')
                text = f"[{doc_title} ({doc_num})] [{pl_title}]\n{pl_body}"
                metadata = {
                    **frontmatter,
                    "chapter": pl_title,
                    "article": pl_title,
                    "article_id": article_id,
                    "cross_references": "",
                }
                nodes.append(LegalNode(text=text, metadata=metadata, article_id=f"{article_id}_C0"))
            else:
                for j, muc_match in enumerate(muc_matches):
                    muc_title = muc_match.group(1).strip()
                    muc_start = muc_match.end()
                    muc_end = muc_matches[j + 1].start() if j + 1 < len(muc_matches) else len(pl_body)
                    muc_body = pl_body[muc_start:muc_end].strip()

                    article_id = f"{doc_num}_{pl_id_token}_MUC_{j+1}".replace('/', '_')
                    article_label = f"{pl_title} - {muc_title}"

                    text = f"[{doc_title} ({doc_num})] [{pl_title}] [{muc_title}]\n{muc_body}"
                    metadata = {
                        **frontmatter,
                        "chapter": pl_title,
                        "article": article_label,
                        "article_id": article_id,
                        "cross_references": "",
                    }
                    nodes.append(LegalNode(text=text, metadata=metadata, article_id=f"{article_id}_C0"))

        return nodes

    def _split_with_headers(self, text: str, pattern: re.Pattern) -> List[tuple]:
        """Splits text and keeps the headers."""
        matches = list(pattern.finditer(text))
        if not matches:
            return [("Nội dung không phân chương/điều", text)]
        
        results = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            header = matches[i].group(1)
            content = text[matches[i].end():end]
            results.append((header, content))
        return results

    def _split_by_clauses(self, text: str) -> List[str]:
        """Splits article content into clauses."""
        parts = self.clause_pattern.split(text)
        # parts will contain [intro, "1.", clause1, "2.", clause2, ...]
        if len(parts) < 2:
            return [text]
        
        clauses = [parts[0]] # Include the intro of the article
        for i in range(1, len(parts), 2):
            marker = parts[i]
            body = parts[i+1] if i+1 < len(parts) else ""
            clauses.append(f"{marker} {body}")
        return [c for c in clauses if c.strip()]

    def _extract_references(self, text: str) -> List[str]:
        """Extracts and canonicalizes references like 'Điều 5 Nghị định 168'."""
        refs = []
        # Mapping common names to doc numbers for better linking
        doc_map = {
            "Luật Doanh nghiệp": "59/2020/QH14",
            "Nghị định 31": "31/2021/NĐ-CP",
            "Nghị định 168": "168/2025/NĐ-CP",
            "Nghị định 153": "153/2020/NĐ-CP",
            "Luật Đầu tư": "61/2020/QH14"
        }
        
        # 1. Pattern for specific articles: Điều [Số] [Loại văn bản] [Số hiệu/Tên]
        pattern = re.compile(r'Điều (\d+)(?:\s+(?:của\s+)?(?:Nghị định|Luật|Văn bản|Thông tư)\s+([\w\s\d/.\-]+))?')
        
        for match in pattern.finditer(text):
            art_num = match.group(1)
            doc_ref_raw = match.group(2)
            
            doc_id = "CURRENT"
            if doc_ref_raw:
                doc_ref = doc_ref_raw.strip()
                # Try to map common names
                found = False
                for name, num in doc_map.items():
                    if name.lower() in doc_ref.lower():
                        doc_id = num
                        found = True
                        break
                if not found:
                    # Try to extract a number-like pattern if no name match
                    num_match = re.search(r'(\d+/\d+/[A-Z\-]+)', doc_ref)
                    if num_match:
                        doc_id = num_match.group(1)
                    else:
                        doc_id = doc_ref # Fallback to raw text
            
            refs.append(f"{doc_id}_A{art_num}".replace('/', '_').replace('.', '_'))

        # 2. Pattern for general document references (e.g., "theo quy định của Luật Đầu tư")
        # Only extract if it's one of our known documents to avoid noise
        for doc_name, doc_id in doc_map.items():
            if doc_name.lower() in text.lower():
                # Add a general reference to the document (no specific article)
                refs.append(doc_id)
                
        return list(set(refs))

def main():
    import json
    ingestor = HierarchicalLegalIngestor("data/markdown")
    all_nodes = []
    
    # Process files
    for filename in os.listdir("data/markdown"):
        if filename.endswith(".md"):
            print(f"Processing {filename}...")
            nodes = ingestor.parse_file(os.path.join("data/markdown", filename))
            all_nodes.extend(nodes)
            
    print(f"Total chunks created: {len(all_nodes)}")
    
    # In a real scenario, we would now pipe these nodes to LlamaIndex persistence.
    # For now, let's output a summary or first few nodes to verify.
    output_sample = [asdict(n) for n in all_nodes[:5]]
    with open("data/ingested_sample.json", "w", encoding="utf-8") as f:
        json.dump(output_sample, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
