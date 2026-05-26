import subprocess
import re
import os
import glob
import json
import argparse
from pathlib import Path
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

def extract_metadata(html_content):
    meta = {}
    if not BeautifulSoup: return meta
    
    soup = BeautifulSoup(html_content, 'html.parser')
    for style_or_script in soup(["style", "script"]):
        style_or_script.decompose()
        
    text = soup.get_text(separator=' ')
    head_text = text[:5000]
    
    # 1. Issuing Authority - Pick the FIRST one that appears in the document
    auth_patterns = [
        r'QUỐC\s+HỘI',
        r'CHÍNH\s+PHỦ',
        r'ỦY\s+BAN\s+THƯỜNG\s+VỤ\s+QUỐC\s+HỘI',
        r'BỘ\s+[A-ZÀ-ỹ\s]{4,}',
        r'TỔNG\s+CỤC\s+[A-ZÀ-ỹ\s]{4,}',
        r'VĂN\s+PHÒNG\s+([A-ZÀ-ỹ\s]+)',
        r'UBND\s+[A-ZÀ-ỹ\s]+|ỦY\s+BAN\s+NHÂN\s+DÂN\s+[A-ZÀ-ỹ\s]+'
    ]
    
    matches = []
    for pattern in auth_patterns:
        for m in re.finditer(pattern, head_text, re.IGNORECASE):
            # Extract and clean (take only the first line of the match if multi-line)
            val = m.group(0).split('\n')[0].strip()
            if 5 < len(val) < 150:
                matches.append((m.start(), val.upper()))
    
    if matches:
        # Sort by position and pick the first one
        matches.sort()
        meta["authority"] = matches[0][1]

    # 2. Document Number (e.g., Số: 31/2021/NĐ-CP, Luật số 59/2020/QH14)
    # Support Vietnamese characters like Đ in NĐ-CP
    num_match = re.search(r'(?:Số|Luật số|Nghị định số|Thông tư số|Quyết định số|Văn bản hợp nhất số|VBHN|VBL)\s*[:]?\s*([0-9\d]{1,4}/[0-9A-Za-zÀ-ỹ\-/]+)', head_text, re.IGNORECASE)
    if num_match:
        meta["number"] = num_match.group(1).strip()
    else:
        num_match_alt = re.search(r'Số\s*[:]?\s*([0-9A-Za-zÀ-ỹ\-/]{5,})', head_text, re.IGNORECASE)
        if num_match_alt:
            meta["number"] = num_match_alt.group(1).strip()

    # 3. Issued Date
    date_match = re.search(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', head_text, re.IGNORECASE)
    if date_match:
        d, m, y = date_match.groups()
        meta["issued_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    return meta

def linearize_html_tables(soup, metadata):
    if not soup: return
    tables = soup.find_all('table')
    number = metadata.get('number', metadata.get('source', 'văn bản'))
    doc_ref = f"Nghị định {number}" if 'NĐ' in str(number) else f"Văn bản {number}"

    for table in tables:
        rows = table.find_all('tr')
        table_text = table.get_text().upper()
        
        # Identify and REMOVE signer tables (User requested removal)
        if any(k in table_text for k in ["THỦ TƯỚNG", "CHỦ TỊCH", "BỘ TRƯỞNG", "THỨ TRƯỞNG"]):
            if len(rows) <= 5: 
                table.decompose() # Remove from document
                continue

        # Header detection
        header_row = None
        data_rows = []
        for row in rows:
            cells = row.find_all(['th', 'td'])
            cell_vals = [c.get_text(strip=True) for c in cells]
            if not header_row and any(cell_vals):
                header_row = cell_vals
                continue
            data_rows.append(cell_vals)

        if not header_row: continue

        # Context (Section Title)
        context_prefix = ""
        prev = table.find_previous(['h1', 'h2', 'h3', 'p'])
        if prev:
            context_prefix = prev.get_text(strip=True)
            if len(context_prefix) > 120:
                context_prefix = context_prefix[:120] + "..."

        replacement_div = soup.new_tag("div")
        for data in data_rows:
            if not any(data): continue
            
            items = []
            primary_subject = ""
            for h, v in zip(header_row, data):
                if not h or not v or v in [' ', '-', '']: continue
                
                if h.upper() in ["TỈNH", "THÀNH PHỐ", "ĐỊA BÀN", "TÊN CHẤT", "NGÀNH, NGHỀ"] and not primary_subject:
                    primary_subject = f"tại {v}"
                elif not primary_subject:
                    primary_subject = f"mục '{v}'"
                
                items.append(f"{h}: {v}")
            
            if items:
                intro = f"Theo phụ lục của {doc_ref}"
                if context_prefix: intro = f"Theo mục '{context_prefix}' của {doc_ref}"
                
                sentence = f"{intro}, {primary_subject}, có các thông tin chi tiết sau: " + ", ".join(items) + "."
                p_tag = soup.new_tag("p")
                p_tag.string = sentence
                replacement_div.append(p_tag)

        table.replace_with(replacement_div)

def clean_markdown(content):
    # Strip ALL HTML remnants
    content = re.sub(r'(?s)<table[^>]*>.*?</table>', '', content)
    content = re.sub(r'<(div|span|p|tbody|tr|td|thead|colgroup|col)[^>]*>', '', content)
    content = re.sub(r'</(div|span|p|tbody|tr|td|thead|colgroup|col)>', '', content)
    
    # Unit Normalization (Legal & Technical units like m2, m3, km2)
    # Handles superscript remnants from Pandoc (e.g. ^2^, ^3^)
    content = re.sub(r'([mk])m\^([23])\^', r'\1m\2', content)
    content = re.sub(r'([mk])m\s+([23])', r'\1m\2', content) # m 2 -> m2
    content = re.sub(r'([mk])m([²³])', r'\1m\2', content) # Direct unicode m² -> m²
    # Standardize common units to m2, m3 for model consistency
    content = re.sub(r'm[²]', 'm2', content) 
    content = re.sub(r'm[³]', 'm3', content)
    content = re.sub(r'km[²]', 'km2', content)
    
    # Structured Headings
    content = re.sub(r'^(LUẬT|NGHỊ ĐỊNH)(.*?)$', r'# \1\2', content, flags=re.M)
    content = re.sub(r'^(Chương\s+[IVXLCDM]+\b[^\n]*)$', r'## \1', content, flags=re.M | re.I)
    content = re.sub(r'^(Điều\s+\d+\..*)$', r'### \1', content, flags=re.M | re.I)
    
    # Fix escaping
    content = re.sub(r'\\([.)\-\*])', r'\1', content)
    
    # Remove Boilerplate
    content = re.sub(r'(?i)^.*CỘNG H?ÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM.*$\n?', '', content, flags=re.M)
    content = re.sub(r'(?i)^.*Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc.*$\n?', '', content, flags=re.M)
    
    # Spacing
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()

ROOT = Path(__file__).resolve().parents[1]


def convert_odt_to_md(input_file, output_dir=None):
    input_path = Path(input_file)
    output_root = Path(output_dir) if output_dir else input_path.parent.parent / "markdown"
    output_file = output_root / f"{input_path.stem}.md"
    os.makedirs(output_file.parent, exist_ok=True)

    # 1. Convert ODT to HTML
    try:
        html_p = subprocess.run(['pandoc', str(input_path), '-t', 'html', '--wrap=none'], capture_output=True, text=True, check=True)
        html_src = html_p.stdout
    except: return

    # 2. Extract Metadata & Process Tables
    meta = extract_metadata(html_src)
    meta['source'] = input_path.name
    
    if BeautifulSoup:
        soup = BeautifulSoup(html_src, 'html.parser')
        linearize_html_tables(soup, meta)
        processed_html = str(soup)
    else:
        processed_html = html_src
    
    # 3. Convert Processed HTML to Markdown
    try:
        tmp = f"{output_file}.tmp.html"
        with open(tmp, 'w', encoding='utf-8') as f: f.write(processed_html)
        subprocess.run(['pandoc', tmp, '-t', 'gfm', '--wrap=none', '-o', str(output_file)], check=True)
        os.remove(tmp)
    except: return

    # 4. Cleanup & Final Assembly
    with open(output_file, 'r', encoding='utf-8') as f:
        md = f.read()

    md = clean_markdown(md)
    # Generate Frontmatter
    fm = "---\ntype: \"legal_document\"\n"
    for k, v in meta.items():
        if v: fm += f"{k}: \"{v}\"\n"
    fm += "---\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fm + md)
    print(f"Successfully processed: {output_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw ODT legal documents to Markdown.")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "markdown"))
    args = parser.parse_args()

    files = glob.glob(str(Path(args.input_dir) / "*.odt"))
    for f in files:
        convert_odt_to_md(f, output_dir=args.output_dir)
