# tools/rag/markdown_splitter.py
"""
Markdown 结构感知切块（纯函数，无 IO，便于单测）。

策略：
  1. 按 ATX 标题(#..######)切成"节(section)"，记录标题路径；标题前的内容归入"前言"节。
  2. 节内按空行切段落，贪心累加进一个 chunk，逼近 max_chars 就收尾输出；
     每个 chunk 前缀 "[节标题]" 以保留章节上下文，供检索命中后回读。
  3. 单段本身超过预算时，用固定窗口 + overlap 硬切兼底，绝不丢内容。

输出：[{"content": str, "metadata": {source, section, chunk_index, total_chunks}}]
"""
import re
from typing import Any, Dict, List

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _hard_split(text: str, budget: int, overlap: int) -> List[str]:
    """超长段落固定窗口滑窗兼底，overlap 保留上下文衔接。"""
    budget = max(50, budget)
    step = max(1, budget - max(0, overlap))
    pieces: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        pieces.append(text[i:i + budget])
        i += step
    return pieces


def _sections(markdown: str) -> List[tuple]:
    """把 markdown 拆成 [(section_title, section_body), ...]。"""
    sections: List[tuple] = []
    title = ""          # 首个标题之前的内容归入空标题"前言"节
    buf: List[str] = []
    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            body = "\n".join(buf).strip("\n")
            if body:
                sections.append((title, body))
            title = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    tail = "\n".join(buf).strip("\n")
    if tail:
        sections.append((title, tail))
    return sections


def split(
    markdown: str,
    source: str,
    max_chars: int = 600,
    overlap: int = 120,
) -> List[Dict[str, Any]]:
    """把 Markdown 切成带元数据的文本块列表。"""
    if not markdown or not markdown.strip():
        return []

    chunks: List[Dict[str, Any]] = []

    def emit(title: str, text: str) -> None:
        prefix = f"[{title}]\n" if title else ""
        body = text.strip()
        if not body:
            return
        content = prefix + body
        chunks.append({
            "content": content,
            "metadata": {"source": source, "section": title},
        })

    for title, body in _sections(markdown):
        budget = max(80, max_chars - len(title) - 3)  # 预留 "[title]\n" 前缀
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]

        current = ""
        for para in paragraphs:
            para = para.strip("\n")
            candidate = (current + "\n\n" + para) if current else para
            if len(candidate) <= budget:
                current = candidate
                continue
            # 当前累积块先收尾
            if current:
                emit(title, current)
                current = ""
            # 段落本身超长 → 硬切；否则作为新块起点
            if len(para) > budget:
                for piece in _hard_split(para, budget, overlap):
                    emit(title, piece)
            else:
                current = para
        if current:
            emit(title, current)

    # 统一补 chunk_index / total_chunks
    total = len(chunks)
    for idx, c in enumerate(chunks):
        c["metadata"]["chunk_index"] = idx
        c["metadata"]["total_chunks"] = total
    return chunks
