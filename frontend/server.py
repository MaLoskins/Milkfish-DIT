# Optimized frontend server
import os
import sys
import json
import asyncio
import uuid
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import aiofiles

# Import project modules
try:
    from main import VideoGenerator, VideoParameters
    from config import load_config
    from prompts import get_prompt_types
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Global state
app_config = None
active_tasks: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global app_config
    
    # Startup
    print("Starting Milkfish DIT Server...")
    
    # Initialize configuration
    class AppConfig:
        def __init__(self):
            self.config = load_config()
            self.video_generator = VideoGenerator(self.config)
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.parent_dir = parent_dir
    
    app_config = AppConfig()
    
    # Ensure output directory exists
    output_dir = parent_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Clean up temp files
    for f in Path(".").glob("*.log"):
        try: f.unlink()
        except: pass
    
    print(f"Server ready. Parent directory: {parent_dir}")
    
    yield
    
    # Shutdown
    print("Shutting down server...")
    
    # Mark running tasks as interrupted
    for task in active_tasks.values():
        if task["status"] == "running":
            task["status"] = "interrupted"
            task["error"] = "Server shutdown"
    
    # Shutdown executor
    app_config.executor.shutdown(wait=True)
    
    print("Server shutdown complete.")

# Create FastAPI app
app = FastAPI(
    title="Milkfish DIT",
    description="AI-powered video generation service",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    prompt_type: str = Field(default="did_you_know")
    model: str = Field(default="Flux", pattern="^(Flux|SD)$")
    voice: str = Field(default="raspy")
    fps: int = Field(default=24, ge=1, le=60)
    aspect_ratio: List[int] = Field(default=[9, 16])
    transition_duration: float = Field(default=0.5, ge=0, le=5)
    pan_effect: bool = True
    zoom_effect: bool = True
    subtitles: bool = True
    subtitle_style: str = Field(default="modern")
    subtitle_animation: str = Field(default="phrase")
    highlight_keywords: bool = True
    
    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError('Invalid aspect ratio')
        return v

# Static file endpoints
static_files = {
    "/": ("index.html", {"Cache-Control": "public, max-age=3600"}),
    "/style.css": ("style.css", {"Cache-Control": "public, max-age=3600"}),
    "/script.js": ("script.js", {"Cache-Control": "no-cache, no-store, must-revalidate"})
}

for route, (filename, headers) in static_files.items():
    @app.get(route, include_in_schema=False)
    async def serve_static(request: Request, fn=filename, h=headers):
        return FileResponse(fn, headers=h)

# API endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/config")
async def get_configuration():
    """Get configuration options"""
    voices = list(app_config.config.voice_mappings.keys()) if app_config else [
        "raspy", "upbeat", "expressive", "deep", "authoritative",
        "trustworthy", "friendly", "intense"
    ]
    
    return {
        "prompt_types": get_prompt_types(),
        "models": ["Flux", "SD"],
        "voices": voices,
        "subtitle_styles": ["modern", "minimal", "bold", "classic", "dynamic"],
        "subtitle_animations": ["phrase", "word", "word-by-word", "typewriter"],
        "subtitle_positions": ["bottom", "top", "middle"],
        "transition_styles": ["fade", "slide", "zoom"]
    }

@app.post("/api/generate")
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation"""
    if not app_config or not app_config.video_generator:
        raise HTTPException(503, "Service unavailable")
    
    # Validate voice
    if request.voice not in app_config.config.voice_mappings:
        raise HTTPException(400, f"Invalid voice: {request.voice}")
    
    task_id = str(uuid.uuid4())
    
    # Initialize task
    active_tasks[task_id] = {
        "id": task_id,
        "status": "starting",
        "progress": 0,
        "stage": "Initializing",
        "topic": request.topic,
        "created_at": datetime.now().isoformat(),
        "output_dir": None,
        "error": None
    }
    
    # Start generation in background
    background_tasks.add_task(run_generation, task_id, request)
    
    return {"task_id": task_id, "message": "Video generation started"}

async def run_generation(task_id: str, request: GenerateRequest):
    """Run video generation"""
    try:
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["progress"] = 5
        
        # Create video parameters
        video_params = VideoParameters(
            fps=request.fps,
            aspect_ratio=tuple(request.aspect_ratio),
            transition_duration=request.transition_duration,
            pan_effect=request.pan_effect,
            zoom_effect=request.zoom_effect,
            subtitles=request.subtitles,
            subtitle_style=request.subtitle_style,
            subtitle_animation=request.subtitle_animation,
            highlight_keywords=request.highlight_keywords
        )
        
        # Get voice ID
        voice_id = app_config.config.voice_mappings.get(request.voice)
        
        # Calculate run_id
        output_path = parent_dir / "output"
        run_id = len([d for d in output_path.iterdir() if d.is_dir()]) + 1
        
        # Run generation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app_config.executor,
            generate_video_sync,
            task_id,
            request,
            voice_id,
            run_id,
            video_params
        )
        
        if result.success:
            active_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "stage": "Completed",
                "output_dir": result.output_dir,
                "video_path": os.path.join(result.output_dir, "video", "final_video.mp4")
            })
        else:
            active_tasks[task_id].update({
                "status": "failed",
                "error": result.error
            })
            
    except Exception as e:
        active_tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })
    finally:
        # Clean up after 1 hour
        await asyncio.sleep(3600)
        if task_id in active_tasks:
            del active_tasks[task_id]

def generate_video_sync(task_id, request, voice_id, run_id, video_params):
    """Synchronous video generation with progress monitoring"""
    # Monitor progress
    def monitor_progress():
        while task_id in active_tasks and active_tasks[task_id]["status"] == "running":
            if active_tasks[task_id].get("output_dir"):
                output_dir = Path(active_tasks[task_id]["output_dir"])
                if output_dir.exists():
                    # Update progress based on file creation
                    if (output_dir / "texts" / "paragraph.txt").exists():
                        active_tasks[task_id]["stage"] = "Generating images"
                        active_tasks[task_id]["progress"] = 30
                    if list((output_dir / "images").glob("*.png")):
                        active_tasks[task_id]["stage"] = "Generating audio"
                        active_tasks[task_id]["progress"] = 60
                    if (output_dir / "audio" / "paragraph.mp3").exists():
                        active_tasks[task_id]["stage"] = "Creating video"
                        active_tasks[task_id]["progress"] = 80
            time.sleep(2)
    
    import threading
    import time
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    # Fix image generation path
    original_method = app_config.video_generator._generate_images
    
    def fixed_generate_images(self, model, run_dir, images_dir):
        import subprocess
        import logging
        logger = logging.getLogger("VideoGenerator")
        
        try:
            image_gen_path = app_config.parent_dir / "image_generation.py"
            cmd = [sys.executable, str(image_gen_path), "--model", model, "--run-dir", run_dir]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Image generation failed:\n{result.stderr}")
                return False
            
            import glob
            if not glob.glob(os.path.join(images_dir, "*.png")):
                logger.error("No images were generated")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return False
    
    # Apply fix
    app_config.video_generator._generate_images = fixed_generate_images.__get__(
        app_config.video_generator, type(app_config.video_generator)
    )
    
    # Store output directory in task
    active_tasks[task_id]["output_dir"] = None
    
    # Generate video
    result = app_config.video_generator.generate_single_video(
        topic=request.topic,
        prompt_type=request.prompt_type,
        model=request.model,
        voice=request.voice,
        voice_id=voice_id,
        output_base_dir=str(parent_dir / "output"),
        run_id=run_id,
        video_params=video_params
    )
    
    # Update task with output directory
    if result.output_dir:
        active_tasks[task_id]["output_dir"] = result.output_dir
    
    return result

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get task status"""
    if task_id not in active_tasks:
        raise HTTPException(404, "Task not found")
    
    task = active_tasks[task_id]
    
    return {
        "id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
        "stage": task["stage"],
        "created_at": task["created_at"],
        "error": task.get("error")
    }

@app.get("/api/videos")
async def list_videos():
    """List all videos"""
    videos = []
    output_dir = parent_dir / "output"
    
    if not output_dir.exists():
        return {"videos": []}
    
    for dir_path in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not dir_path.is_dir():
            continue
        
        video_path = dir_path / "video" / "final_video.mp4"
        metadata_path = dir_path / "metadata.json"
        
        if video_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Get video duration
                duration = None
                try:
                    import subprocess
                    result = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        duration = float(result.stdout.strip())
                except:
                    pass
                
                videos.append({
                    "id": dir_path.name,
                    "topic": metadata.get("topic", "Unknown"),
                    "prompt_type": metadata.get("prompt_type", "Unknown"),
                    "model": metadata.get("model", "Unknown"),
                    "voice": metadata.get("voice", "Unknown"),
                    "created_at": metadata.get("timestamp", "Unknown"),
                    "size": video_path.stat().st_size,
                    "duration": duration,
                    "video_params": metadata.get("video_params", {}),
                    "status": "completed"
                })
            except Exception as e:
                print(f"Error reading {dir_path}: {e}")
    
    return {"videos": videos}

@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """Stream video file"""
    video_path = parent_dir / "output" / video_id / "video" / "final_video.mp4"
    
    if not video_path.exists():
        # Check if still generating
        for task in active_tasks.values():
            if task.get("output_dir", "").endswith(video_id) and task["status"] == "running":
                raise HTTPException(202, "Video is still being generated")
        raise HTTPException(404, "Video not found")
    
    # Stream the video
    async def iterfile():
        async with aiofiles.open(video_path, 'rb') as f:
            while chunk := await f.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Length": str(video_path.stat().st_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{video_id}.mp4"'
        }
    )

@app.get("/api/video/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Get video thumbnail"""
    # Try to find first image
    images_dir = parent_dir / "output" / video_id / "images"
    if images_dir.exists():
        images = sorted(images_dir.glob("*.png"))
        if images:
            # Create thumbnail
            cache_dir = parent_dir / "output" / ".thumbnails"
            cache_dir.mkdir(exist_ok=True)
            cached = cache_dir / f"{video_id}.jpg"
            
            if cached.exists():
                return FileResponse(cached, media_type="image/jpeg",
                                  headers={"Cache-Control": "public, max-age=86400"})
            
            try:
                from PIL import Image
                with Image.open(images[0]) as img:
                    img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                    img.save(cached, "JPEG", quality=85, optimize=True)
                
                return FileResponse(cached, media_type="image/jpeg",
                                  headers={"Cache-Control": "public, max-age=86400"})
            except:
                pass
    
    return JSONResponse(content=None, status_code=204)

@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video"""
    video_dir = parent_dir / "output" / video_id
    
    if not video_dir.exists():
        raise HTTPException(404, "Video not found")
    
    # Check if currently generating
    for task_id, task in active_tasks.items():
        if task.get("output_dir", "").endswith(video_id):
            if task["status"] == "running":
                raise HTTPException(409, "Cannot delete video while generating")
            del active_tasks[task_id]
            break
    
    try:
        shutil.rmtree(video_dir)
        
        # Delete thumbnail
        cached = parent_dir / "output" / ".thumbnails" / f"{video_id}.jpg"
        if cached.exists():
            cached.unlink()
        
        return {"message": "Video deleted successfully", "id": video_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete video: {str(e)}")

@app.get("/api/tasks")
async def list_tasks():
    """List active tasks"""
    return {
        "tasks": [
            {
                "id": task["id"],
                "status": task["status"],
                "progress": task["progress"],
                "stage": task["stage"],
                "topic": task["topic"],
                "created_at": task["created_at"]
            }
            for task in active_tasks.values()
        ]
    }

@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task"""
    if task_id not in active_tasks:
        raise HTTPException(404, "Task not found")
    
    task = active_tasks[task_id]
    
    if task["status"] != "running":
        raise HTTPException(400, f"Task is not running (status: {task['status']})")
    
    task["status"] = "cancelled"
    task["error"] = "Cancelled by user"
    
    # Clean up output directory
    if task.get("output_dir"):
        try:
            shutil.rmtree(task["output_dir"])
        except:
            pass
    
    return {"message": "Task cancelled", "id": task_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)