# tools/rag/rag_ingest.py
"""
入库编排层：把"一个路径(文件或目录)"跑成"解析→切块→向量化→写入向量库"的完整流水线。

适配自研 MinerUClient(tools/rag/mineru.py) 的**批量接口**：
    upload_and_parse(paths) -> [batch_id, ...]          # 同步、按50分片
    get_parse_result(batch_id) -> {data:{extract_result:[...]}}  # 需其返回结果(见文末建议)
本模块负责：批量上传、轮询到终态、下载结果包取 full.md、再逐文件切块入库。

设计约束：
  - 解析=注入的 mineru 客户端；切块=markdown_splitter；向量=EmbeddingModel；存储=ChromaVectorStore。
  - 同步的 requests 调用一律用 asyncio.to_thread 桥接，避免阻塞事件循环。
  - 文件与结果的对应关系用 data_id(与客户端 "abcd_{全局序号}" 约定一致)，而非文件名，避免同名跨目录串块。
  - 单文件失败不中断整批。不暴露 CLI 入口。
"""
import asyncio
import io
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from tools.rag.markdown_splitter import split

DEFAULT_EXTS = ".pdf,.png,.jpg,.jpeg,.jp2,.webp,.gif,.bmp,.docx,.pptx,.xlsx"
_STATE_DONE = "done"
_STATE_FAILED = "failed"


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _supported_exts(exts: Optional[List[str]]) -> set:
    if exts:
        return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    raw = os.getenv("INGEST_EXTS", DEFAULT_EXTS)
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _data_id(index: int) -> str:
    # 必须与 MinerUClient._upload_batch 里的 data_id 规则保持一致
    return f"abcd_{index}"


@dataclass
class FileResult:
    name: str
    ok: bool
    chunks: int = 0
    error: Optional[str] = None


@dataclass
class IngestSummary:
    files_ok: int = 0
    files_failed: int = 0
    chunks_added: int = 0
    results: List[FileResult] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"入库完成：成功 {self.files_ok} 个文件，失败 {self.files_failed} 个，共写入 {self.chunks_added} 个文本块。"
        ]
        for r in self.results:
            if r.ok:
                lines.append(f"  ✓ {r.name}：{r.chunks} 块")
            else:
                lines.append(f"  ✗ {r.name}：{r.error}")
        return "\n".join(lines)


def expand_targets(path: str, exts: Optional[List[str]] = None) -> List[Path]:
    """把入参展开成待处理文件列表；目录递归并按扩展名过滤，文件直接返回。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    allowed = _supported_exts(exts)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in allowed)


# ---------------- MinerU 结果获取 ----------------

def _iter_extract_items(payload: Any) -> List[Dict[str, Any]]:
    """把 get_parse_result 的返回归一化成 extract_result 列表（兼容外层带/不带 data）。"""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return []
    return data.get("extract_result") or []


async def _poll_results(mineru, batch_ids: List[str], total: int,
                        timeout: int, interval: int) -> Dict[str, Dict[str, Any]]:
    """轮询所有 batch，直到每个文件到达 done/failed 或超时。返回 {data_id: item}。"""
    outcomes: Dict[str, Dict[str, Any]] = {}
    deadline = time.monotonic() + timeout
    while len(outcomes) < total:
        for bid in batch_ids:
            try:
                payload = await asyncio.to_thread(mineru.get_parse_result, bid)
            except Exception:  # noqa: BLE001 单次查询异常按瞬态处理，下轮重试
                continue
            for item in _iter_extract_items(payload):
                did = item.get("data_id")
                if did and did not in outcomes and item.get("state") in (_STATE_DONE, _STATE_FAILED):
                    outcomes[did] = item
        if len(outcomes) >= total:
            break
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(interval)
    return outcomes


def _extract_md_from_zip(zip_bytes: bytes) -> str:
    """从结果包(zip)中取出 full.md 的文本。"""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    target = "full.md" if "full.md" in names else next(
        (n for n in names if Path(n).name == "full.md"), None)
    if target is None:
        raise ValueError(f"结果包内未找到 full.md，包含：{names[:20]}")
    return zf.read(target).decode("utf-8", errors="replace")


async def _fetch_markdown(zip_url: str) -> str:
    """下载 full_zip_url(公开 CDN，无需鉴权) 并提取 Markdown。"""
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(zip_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"下载结果包失败 HTTP {resp.status}")
            raw = await resp.read()
    return _extract_md_from_zip(raw)


# ---------------- 编排主流程 ----------------

async def ingest_path(
    path: str,
    store,
    embedding,
    mineru,
    reindex: bool = True,
    max_chars: Optional[int] = None,
    overlap: Optional[int] = None,
    exts: Optional[List[str]] = None,
    poll_timeout: Optional[int] = None,
    poll_interval: Optional[int] = None,
) -> IngestSummary:
    """核心编排：批量解析 → 逐文件切块入库；单文件失败不中断整批。"""
    max_chars = max_chars if max_chars is not None else _env_int("INGEST_MAX_CHARS", 600)
    overlap = overlap if overlap is not None else _env_int("INGEST_OVERLAP", 120)
    poll_timeout = poll_timeout if poll_timeout is not None else _env_int("MINERU_POLL_TIMEOUT", 300)
    poll_interval = poll_interval if poll_interval is not None else _env_int("MINERU_POLL_INTERVAL", 3)

    paths = expand_targets(path, exts)
    summary = IngestSummary()
    if not paths:
        return summary

    # 1) 批量上传（客户端内部按50分片，返回每片 batch_id）
    str_paths = [str(p) for p in paths]
    try:
        batch_ids = await asyncio.to_thread(mineru.upload_and_parse, str_paths)
    except Exception as e:  # noqa: BLE001 上传/申请链接阶段整体失败：所有文件记为失败
        for p in paths:
            summary.files_failed += 1
            summary.results.append(FileResult(name=p.name, ok=False,
                                              error=f"{type(e).__name__}: {e}"))
        return summary

    # 2) 轮询直到全部到终态（或超时）
    outcomes = await _poll_results(mineru, list(batch_ids), len(paths), poll_timeout, poll_interval)

    # 3) 逐文件取正文 → 切块 → 向量化 → 入库
    for idx, p in enumerate(paths):
        did = _data_id(idx)
        source_key = str(p.resolve())
        result = FileResult(name=p.name, ok=False)
        try:
            item = outcomes.get(did)
            if item is None:
                raise TimeoutError(f"未在 {poll_timeout}s 内完成（可能 get_parse_result 未返回结果）")
            if item.get("state") == _STATE_FAILED:
                raise RuntimeError(f"解析失败：{item.get('err_msg') or '未知原因'}")
            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise RuntimeError("任务完成但没有结果包地址 full_zip_url")

            markdown = await _fetch_markdown(zip_url)
            chunks = split(markdown, source=source_key, max_chars=max_chars, overlap=overlap)
            for c in chunks:
                c["metadata"]["file_name"] = p.name

            if not chunks:
                result.ok = True
                summary.files_ok += 1
                summary.results.append(result)
                continue

            texts = [c["content"] for c in chunks]
            vectors = await embedding.encode_batched(texts)

            if reindex:
                await store.delete_by_source(source_key)
            await store.upsert_documents(chunks, vectors)

            result.ok = True
            result.chunks = len(chunks)
            summary.files_ok += 1
            summary.chunks_added += len(chunks)
        except Exception as e:  # noqa: BLE001 单文件隔离
            result.error = f"{type(e).__name__}: {e}"
            summary.files_failed += 1
        summary.results.append(result)

    return summary
