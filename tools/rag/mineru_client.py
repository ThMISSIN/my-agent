import os
from typing import Optional, List
import requests
from dotenv import load_dotenv

load_dotenv()


def _resolve_bool(explicit: Optional[bool], env_key: str, default: bool) -> bool:
    """显式传参优先；否则读 .env 字符串(==true 才算真)；都没有则用官方默认。"""
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


class MinerUClient:
    # MinerU file-urls/batch 单次申请上传链接不超过 50 个，超出需分片
    BATCH_LIMIT = 50

    def __init__(
            self,
            base_url: Optional[str] = None,
            token: Optional[str] = None,
            model_version: Optional[str] = None,
            enable_ocr: Optional[bool] = None,
            enable_formula: Optional[bool] = None,
            enable_table: Optional[bool] = None,
            language: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv('MINERU_BASE_URL') # or "https://mineru.net/api/v4"
        self.api_url = f"{self.base_url}/file-urls/batch"
        self.token = token or os.getenv("MINERU_TOKEN")
        self.model_version = model_version or os.getenv("MINERU_MODEL_VERSION")
        self.enable_ocr = _resolve_bool(enable_ocr, "MINERU_ENABLE_OCR", False)
        self.enable_formula = _resolve_bool(enable_formula, "MINERU_ENABLE_FORMULA", True)
        self.enable_table = _resolve_bool(enable_table, "MINERU_ENABLE_TABLE", True)
        self.language = language or os.getenv("MINERU_LANGUAGE")

    def upload_and_parse(self, file_path_list: List[str]) -> List[str]:
        """批量上传本地文件到 MinerU；按 50 个/批分片，返回每片的 batch_id 列表供后续查询。"""
        if not file_path_list:
            raise ValueError("file_path_list 为空，没有可上传的文件")
        batch_ids: List[str] = []
        for start in range(0, len(file_path_list), self.BATCH_LIMIT):
            chunk = file_path_list[start:start + self.BATCH_LIMIT]
            batch_ids.append(self._upload_batch(chunk, start))
        return batch_ids

    def _upload_batch(self, file_chunk: List[str], start_index: int) -> str:
        """对单片(<=50个文件)申请上传链接并逐个 PUT，成功返回该片的 batch_id。"""
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        files_payload = []
        for i, file_path in enumerate(file_chunk):
            files_payload.append({
                "name": os.path.basename(file_path),
                # 全局序号保证跨片 data_id 不重复
                "data_id": f"abcd_{start_index + i}",
                "is_ocr": self.enable_ocr,
            })
        data = {
            "files": files_payload,
            "model_version": self.model_version,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
        }
        response = requests.post(self.api_url, headers=header, json=data)
        if response.status_code != 200:
            raise RuntimeError(
                f"申请上传链接失败：HTTP {response.status_code} {response.text[:200]}")
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(
                f"申请上传链接失败：code={result.get('code')} msg={result.get('msg')}")
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        for idx, upload_url in enumerate(urls):
            file_path = file_chunk[idx]
            with open(file_path, 'rb') as f:
                res_upload = requests.put(upload_url, data=f)
                if res_upload.status_code != 200:
                    raise RuntimeError(
                        f"文件上传失败：{file_path} HTTP {res_upload.status_code}")
        return batch_id

    def get_parse_result(self, batch_id: str) -> dict:
        """单次查询该 batch 的解析结果，返回 data（含 extract_result 列表）；轮询/超时由编排层负责。"""
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        res = requests.get(url, headers=header)
        if res.status_code != 200:
            raise RuntimeError(
                f"查询解析结果失败：HTTP {res.status_code} {res.text[:200]}")
        body = res.json()
        if body.get("code") != 0:
            raise RuntimeError(
                f"查询解析结果失败：code={body.get('code')} msg={body.get('msg')}")
        return body.get("data", {})

if __name__ == '__main__':
    mineru = MinerUClient()
    # mineru.upload_and_parse(["C:\\Users\\zkt11\\Desktop\\乱\\郑凯桐.pdf", "C:\\Users\\zkt11\\Desktop\\乱\\郑凯桐-2.pdf"])

    batch_id = "89615095-9a7f-4ca8-8bb8-8d5e49242598"
    result = mineru.get_parse_result(batch_id)
    print(result)
