import hashlib
import os

import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio

from dotenv import load_dotenv

load_dotenv()

class ChromaVectorStore:
    """
    Chroma 向量存储层
    """
    def __init__(
        self,
        dim: Optional[int] = None,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None
    ):
        self.dim = os.getenv("EMBEDDING_DIMENSION")
        self.collection_name = collection_name or os.getenv("RAG_COLLECTION_NAME", "knowledge_base")
        persist_dir = persist_dir or os.getenv("RAG_PERSIST_DIR", "./chroma_kb")

        # 初始化 Chroma 客户端
        if persist_dir:
            # 磁盘持久化模式，重启数据保留
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            # 内存模式，开发调试用
            self.client = chromadb.EphemeralClient()

        # 创建/获取集合，使用余弦相似度
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _make_id(doc: Dict[str, Any], fallback_index: int) -> str:
        """稳定 id：同一(来源, 块序号)恒定 → 可幂等 upsert、按来源删除。"""
        meta = doc.get("metadata", {}) or {}
        source = str(meta.get("source", "unknown"))
        chunk_index = meta.get("chunk_index", fallback_index)
        key = f"{source}::{chunk_index}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata 只接受 str/int/float/bool/None，剔除其它类型。"""
        clean = {}
        for k, v in (meta or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
        return clean

    async def add_documents(self, documents: List[Dict[str, Any]], vectors: np.ndarray):
        """
        批量入库（新增）。
        :param documents: [{"content": "文本", "metadata": {"source": "xxx", "chunk_index": 0}}]
        :param vectors: 对应文档的向量矩阵
        """
        def _add_sync():
            ids = [self._make_id(doc, idx) for idx, doc in enumerate(documents)]
            contents = [doc["content"] for doc in documents]
            metas = [self._sanitize_meta(doc.get("metadata", {})) for doc in documents]
            embeddings = vectors.tolist()

            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metas,
                embeddings=embeddings
            )

        # 同步操作放线程池，不阻塞事件循环
        await asyncio.to_thread(_add_sync)

    async def upsert_documents(self, documents: List[Dict[str, Any]], vectors: np.ndarray):
        """
        幂等入库：相同(来源,块序号)的 id 已存在则覆盖，否则新增。
        """
        def _upsert_sync():
            ids = [self._make_id(doc, idx) for idx, doc in enumerate(documents)]
            contents = [doc["content"] for doc in documents]
            metas = [self._sanitize_meta(doc.get("metadata", {})) for doc in documents]
            embeddings = vectors.tolist()

            self.collection.upsert(
                ids=ids,
                documents=contents,
                metadatas=metas,
                embeddings=embeddings
            )

        await asyncio.to_thread(_upsert_sync)

    async def delete_by_source(self, source: str):
        """按来源(文件)删除其全部历史块，供重灌(reindex)先清旧。"""
        def _del_sync():
            self.collection.delete(where={"source": source})

        await asyncio.to_thread(_del_sync)

    async def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        相似度检索
        :return: [{"content": "...", "metadata": {...}, "score": 0~1相似度}]
        """
        def _search_sync():
            result = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k
            )

            output = []
            docs = result["documents"][0]
            metas = result["metadatas"][0]
            distances = result["distances"][0]

            for idx in range(len(docs)):
                # Chroma 余弦距离 0~2 → 转换为相似度 0~1（越大越相似）
                similarity = 1 - distances[idx] / 2
                output.append({
                    "content": docs[idx],
                    "metadata": metas[idx] or {},
                    "score": round(similarity, 4)
                })
            return output

        return await asyncio.to_thread(_search_sync)

    @property
    def size(self) -> int:
        """文档总数"""
        return self.collection.count()
