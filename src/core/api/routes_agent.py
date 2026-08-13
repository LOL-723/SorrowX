from fastapi import APIRouter, Form, HTTPException

from llm.Agent.AgentRuntime import AgentRequest, get_agent_runtime
from schemas.llm import AgentResponse


router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/ask", response_model=AgentResponse)
def Agent_Ask(
    message: str = Form(...),
    session_id: str = Form(...),
):
    try:
        result = get_agent_runtime().run(
            AgentRequest(goal=message, session_id=session_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent request failed: {str(e)}"
        )

    if result.status != "finished":
        raise HTTPException(status_code=500, detail=result.error or "agent run failed")

    return AgentResponse(
        run_id=result.run_id,
        status=result.status,
        message=result.answer,
    )
