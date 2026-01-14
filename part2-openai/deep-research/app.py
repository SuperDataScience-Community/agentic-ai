from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
from research_workflow import run_research_pipeline

app = FastAPI()


class ResearchRequest(BaseModel):
    topic: str


async def research_stream_generator(topic: str):
    """Generator that yields progress updates during research"""
    
    # Send initial message
    yield f"data: {json.dumps({'status': 'progress', 'message': 'Breaking down topic into subtopics...'})}\n\n"
    await asyncio.sleep(1)
    
    yield f"data: {json.dumps({'status': 'progress', 'message': 'Researching multiple sources...'})}\n\n"
    await asyncio.sleep(1)
    
    yield f"data: {json.dumps({'status': 'progress', 'message': 'Optimizing research findings...'})}\n\n"
    await asyncio.sleep(1)
    
    yield f"data: {json.dumps({'status': 'progress', 'message': 'Generating final report...'})}\n\n"
    
    # Run the actual research pipeline
    final_report = await run_research_pipeline(topic)
    
    # Send final report
    yield f"data: {json.dumps({'status': 'complete', 'message': 'Research complete!', 'report': final_report})}\n\n"


@app.post("/research")
async def research(request: ResearchRequest):
    """Streaming endpoint for research queries"""
    return StreamingResponse(
        research_stream_generator(request.topic),
        media_type="text/event-stream"
    )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
