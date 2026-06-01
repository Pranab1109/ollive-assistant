import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.graph.assistant_graph import assistant_graph
from backend.app.services.observability import trace_repo
from backend.app.services.evaluator import EvaluatorService
from backend.app.tools.hospital_tools import _read_appointments, _write_appointments, _get_doctors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evergreen Medical AI Assistant Platform",
    description="Production-grade AI Receptionist Evaluation and Observability System",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    def mask(val: str) -> str:
        if not val:
            return "empty"
        if len(val) <= 8:
            return "***"
        return f"{val[:4]}...{val[-4:]}"
    logger.info("========================================")
    logger.info("APPLICATION STARTUP CONFIGURATION:")
    logger.info(f"GEMINI_API_KEY: {mask(settings.GEMINI_API_KEY)}")
    logger.info(f"HF_SPACE_MODEL_URL: {settings.HF_SPACE_MODEL_URL or 'empty'}")
    logger.info(f"HF_TOKEN: {mask(settings.HF_TOKEN)}")
    logger.info(f"OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"OSS_MODEL_NAME: {settings.OSS_MODEL_NAME}")
    logger.info("========================================")

# CORS configurations for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for requests
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    session_id: str
    model_type: str  # "oss" or "frontier"

class ChatResponse(BaseModel):
    response: str
    query_id: str
    session_id: str
    model_type: str
    should_retry: Optional[bool] = False
    retry_delay: Optional[float] = 0.0

# Cache to track evaluation runner state
_eval_running_status = {"status": "idle", "message": "No evaluations currently running."}

# --- API ROUTES ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Executes the hospital assistant state graph for a query.
    """
    # Create unique tracking ID
    query_id = str(uuid.uuid4())
    
    # Get last query text
    user_query = ""
    for msg in reversed(request.messages):
        if msg["role"] == "user":
            user_query = msg["content"]
            break
            
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in history.")

    # Initialize Trace directly using the router's query_id
    model_name = "gemini-2.5-flash" if request.model_type == "frontier" else settings.OSS_MODEL_NAME
    trace_repo.create_trace(
        session_id=request.session_id,
        model=model_name,
        query=user_query,
        query_id=query_id
    )

    # Populate initial Graph State
    state = {
        "messages": request.messages,
        "session_id": request.session_id,
        "query_id": query_id,
        "model_type": request.model_type,
        "current_response": "",
        "tool_calls": [],
        "tool_results": [],
        "refusal_message": None,
        "step_count": 0,
        "should_retry": False,
        "retry_delay": 0.0
    }

    try:
        # Run state graph asynchronously
        final_state = await assistant_graph.ainvoke(state)
        final_response = final_state.get("current_response", "Sorry, I could not generate a response.")
        
        # Save trace outcome
        trace_repo.finalize_trace(query_id, final_response, success=True)
        
        return ChatResponse(
            response=final_response,
            query_id=query_id,
            session_id=request.session_id,
            model_type=request.model_type,
            should_retry=final_state.get("should_retry", False),
            retry_delay=final_state.get("retry_delay", 0.0)
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint state execution: {e}")
        trace_repo.finalize_trace(query_id, "", success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/traces/{query_id}")
async def get_trace_endpoint(query_id: str):
    """
    Returns the node execution timeline, guardrail outputs, and latency metrics for a query.
    """
    trace = trace_repo.get_trace(query_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace with query_id '{query_id}' not found.")
    return trace

@app.get("/api/session/traces/{session_id}")
async def get_session_traces_endpoint(session_id: str):
    """
    Returns all query execution logs for an active chat session.
    """
    return trace_repo.get_session_traces(session_id)

@app.get("/api/appointments")
async def get_appointments_endpoint():
    """
    Lists all doctor appointments booked in the system.
    Filters to only show confirmed appointments.
    """
    appointments = _read_appointments()
    return [app for app in appointments if app.get("status", "confirmed") == "confirmed"]

@app.post("/api/appointments/{booking_id}/cancel")
async def cancel_appointment_endpoint(booking_id: str):
    """
    Cancels an existing appointment by its Booking ID.
    """
    from datetime import datetime
    appointments = _read_appointments()
    
    found = False
    for app in appointments:
        if app.get("id", "").upper() == booking_id.upper() and app.get("status") == "confirmed":
            app["status"] = "cancelled"
            app["cancelled_at"] = datetime.now().isoformat()
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail=f"No active appointment found with ID '{booking_id}'.")
    
    _write_appointments(appointments)
    return {"message": f"Appointment {booking_id} has been cancelled.", "id": booking_id}

@app.get("/api/doctors")
async def get_doctors_endpoint():
    """
    Returns the full doctor roster with schedules from the database.
    """
    return _get_doctors()

# --- EVALUATION ENGINE API ---

async def background_eval_task():
    global _eval_running_status
    try:
        logger.info("Starting background evaluation run...")
        await EvaluatorService.run_evaluation()
        _eval_running_status = {"status": "idle", "message": "Evaluation completed successfully."}
        logger.info("Background evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error in background evaluation: {e}")
        _eval_running_status = {"status": "failed", "message": f"Evaluation failed: {str(e)}"}

@app.post("/api/evals/run")
async def run_evals_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers the automated A/B evaluation suite in the background.
    """
    global _eval_running_status
    if _eval_running_status["status"] == "running":
        return {"status": "running", "message": "An evaluation is already running."}
        
    _eval_running_status = {"status": "running", "message": "Automated evaluations running in background."}
    background_tasks.add_task(background_eval_task)
    return _eval_running_status

@app.get("/api/evals/status")
async def get_evals_status_endpoint():
    """
    Fetches the running status of the background evaluation task.
    """
    return _eval_running_status

@app.get("/api/evals")
async def get_evals_results_endpoint():
    """
    Returns the latest compiled benchmark scores and summaries.
    """
    results = EvaluatorService.get_latest_results()
    if not results:
        return {"message": "No evaluations have been run yet. Call /api/evals/run to start."}
    return results

@app.get("/api/evals/pdf")
async def download_pdf_report_endpoint():
    """
    Generates and returns the print-friendly 1-page PDF evaluation report.
    """
    try:
        pdf_path = EvaluatorService.generate_pdf_report()
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF report could not be generated.")
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename="hospital_agent_evaluation_report.pdf"
        )
    except Exception as e:
        logger.error(f"Error serving PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report PDF: {str(e)}")

# --- SERVING STATIC FRONTEND BUILD ---

frontend_dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))

if os.path.exists(frontend_dist_path):
    logger.info(f"Serving static frontend build from {frontend_dist_path}")
    
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")
    
    static_assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
    os.makedirs(static_assets_path, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_assets_path), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"message": "API endpoint not found"})
        
        index_file = os.path.join(frontend_dist_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse(content={"message": "Frontend build files found but index.html missing"})
else:
    logger.warning(
        f"Frontend build folder not found at {frontend_dist_path}. "
        "Frontend will not be served from FastAPI. Exposing API routes only."
    )
    
    static_assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
    os.makedirs(static_assets_path, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_assets_path), name="static")

    @app.get("/")
    async def root_fallback():
        return {
            "message": "AI Receptionist API server is running. Frontend static directory not found. Please compile frontend first."
        }
