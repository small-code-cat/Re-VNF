# noise_service.py
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from demo.filter.noiser_vllm import QueryNoiser  # ← 改成你这段类所在的真实 import

app = FastAPI(title="QueryNoiser Service")

# --------- 配置区：启动时加载一次模型（很关键）---------
MODEL_PATH = "/path/to/your/model"   # ← 改成你的模型路径
IS_COT = True                        # ← 默认是否cot
noiser: Optional[QueryNoiser] = None


class NoiseReq(BaseModel):
    query: str
    images: List[str]               # 这里直接传 image path 列表（最简）
    max_retries: int = 1


class NoiseResp(BaseModel):
    predicted_order: List[int]


@app.on_event("startup")
def _startup():
    global noiser
    noiser = QueryNoiser(MODEL_PATH, is_cot=IS_COT)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/find_noise", response_model=NoiseResp)
def find_noise(req: NoiseReq):
    if not noiser:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        order = noiser.find_noise(req.query, req.images, max_retries=req.max_retries)
        return NoiseResp(predicted_order=order)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# pip install fastapi uvicorn pydantic
# uvicorn noise_service:app --host 0.0.0.0 --port 8003