import sys
from pathlib import Path


SRC_CORE = Path(__file__).resolve().parent / "src" / "core"
if str(SRC_CORE) not in sys.path:
    sys.path.insert(0, str(SRC_CORE))

from set.runtime import ensure_project_runtime

ensure_project_runtime()

from fastapi import FastAPI

from api.routes_agent import router as agent_router
from api.routes_llm import router as llm_router

app = FastAPI(title="LLM Client Service")

app.include_router(agent_router)
app.include_router(llm_router)


@app.get("/health")
def health():
    return {"status": "ok"}
