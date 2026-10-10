import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List

load_dotenv()

class EmbeddingModel:

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.dimension = os.getenv("EMBEDDING_DIMENSION")

    async def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量文本转向量
        :param texts:
        :return:
        """
        resp = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        # 提取所有的 embedding转化为 numpy
        # 把嵌套 list 转成二维 ndarray，形状为 (N, dim)，N 是条数，dim 是向量维度；并指定 float32 节省内存、加快运算。
        # vectors 是一个 (N, dim) 的浮点矩阵，每一行是一个句子的嵌入向量。
        vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)

        # 归一化，保证后续内积 = 余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        return vectors

    async def encode_single(self, text: str) -> np.ndarray:
        """单条文本转向量"""
        vectors = await self.encode([text])
        return vectors[0]

    async def encode_batched(self, texts: List[str], batch: int = None) -> np.ndarray:
        """
        分批向量化：规避 OpenAI 兼容网关单次 input 条数上限。
        返回按输入顺序拼接的 (N, dim)；空输入返回 (0, 0)。
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if batch is None:
            try:
                batch = int(os.getenv("INGEST_EMBED_BATCH", 10))
            except (TypeError, ValueError):
                batch = 10
        batch = max(1, batch)
        outs = []
        for i in range(0, len(texts), batch):
            outs.append(await self.encode(texts[i:i + batch]))
        return np.concatenate(outs, axis=0)