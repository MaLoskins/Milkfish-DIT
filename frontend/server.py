# Optimized frontend server with enhanced progress tracking
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
    
    # Initialize task with detailed progress tracking
    active_tasks[task_id] = {
        "id": task_id,
        "status": "starting",
        "progress": 0,
        "stage": "Initializing",
        "topic": request.topic,
        "created_at": datetime.now().isoformat(),
        "output_dir": None,
        "error": None,
        "details": {
            "text_generation": {
                "paragraph": False,
                "descriptions_count": 0,
                "descriptions_completed": 0
            },
            "image_generation": {
                "total": 0,
                "completed": 0,
                "current": None
            },
            "tts": {
                "started": False,
                "completed": False
            },
            "video": {
                "started": False,
                "completed": False
            }
        }
    }
    
    # Start generation in background
    background_tasks.add_task(run_generation, task_id, request)
    
    return {"task_id": task_id, "message": "Video generation started"}

async def run_generation(task_id: str, request: GenerateRequest):
    """Run video generation with enhanced progress tracking"""
    try:
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["progress"] = 2
        active_tasks[task_id]["stage"] = "Starting generation"
        
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
        
        # Run generation with enhanced monitoring
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app_config.executor,
            generate_video_sync_enhanced,
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
                "stage": "Video generation complete!",
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

def generate_video_sync_enhanced(task_id, request, voice_id, run_id, video_params):
    """Enhanced synchronous video generation with detailed progress tracking"""
    import threading
    import time
    import logging
    from queue import Queue
    
    # Create a custom logger handler to capture progress
    progress_queue = Queue()
    
    class ProgressHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            # Parse progress messages
            if "Generated paragraph" in msg:
                progress_queue.put(("text", "paragraph_complete", None))
            elif "Extracted" in msg and "image descriptions" in msg:
                try:
                    count = int(msg.split("Extracted ")[1].split(" image")[0])
                    progress_queue.put(("text", "descriptions_complete", count))
                except:
                    pass
            elif "Image description" in msg:
                try:
                    parts = msg.split("Image description ")[1].split("/")
                    current = int(parts[0])
                    total = int(parts[1])
                    progress_queue.put(("text", "description_progress", (current, total)))
                except:
                    pass
            elif "Generating image" in msg:
                try:
                    parts = msg.split("Generating image ")[1].split("/")
                    current = int(parts[0])
                    total = int(parts[1])
                    progress_queue.put(("image", "progress", (current, total)))
                except:
                    pass
            elif "Processing image" in msg:
                try:
                    parts = msg.split("Processing image ")[1].split("/")
                    current = int(parts[0])
                    total = int(parts[1])
                    progress_queue.put(("image", "progress", (current, total)))
                except:
                    pass
            elif "Uploading to ElevenLabs" in msg:
                progress_queue.put(("tts", "uploading", None))
            elif "Processing audio" in msg:
                progress_queue.put(("tts", "processing", None))
            elif "Downloading audio file" in msg:
                progress_queue.put(("tts", "downloading", None))
            elif "Audio generation complete" in msg:
                progress_queue.put(("tts", "complete", None))
            elif "Video:" in msg:
                progress_queue.put(("video", "progress", msg.split("Video: ")[1]))
            elif "Video creation complete" in msg:
                progress_queue.put(("video", "complete", None))
    
    # Add our handler to the logger
    logger = logging.getLogger("VideoGenerator")
    progress_handler = ProgressHandler()
    progress_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(progress_handler)
    
    # Monitor progress from multiple sources
    def monitor_progress():
        while task_id in active_tasks and active_tasks[task_id]["status"] == "running":
            # Check progress queue
            while not progress_queue.empty():
                try:
                    category, event, data = progress_queue.get_nowait()
                    update_task_progress(task_id, category, event, data)
                except:
                    pass
            
            # Also monitor progress file if available
            if active_tasks[task_id].get("output_dir"):
                output_dir = Path(active_tasks[task_id]["output_dir"])
                progress_file = output_dir / "progress.json"
                
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            progress_data = json.load(f)
                        
                        # Update main progress if file has newer data
                        file_progress = progress_data.get("progress", 0)
                        if file_progress > active_tasks[task_id]["progress"]:
                            active_tasks[task_id]["progress"] = file_progress
                            active_tasks[task_id]["stage"] = progress_data.get("stage", "Processing")
                            
                            # Merge details
                            if "details" in progress_data:
                                for key, value in progress_data["details"].items():
                                    if key in active_tasks[task_id]["details"]:
                                        active_tasks[task_id]["details"][key].update(value)
                    except:
                        pass
            
            time.sleep(0.2)  # Check 5 times per second for smooth updates
    
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    # Override methods to capture progress
    original_generate_text = app_config.video_generator._generate_text
    original_generate_images = app_config.video_generator._generate_images
    original_generate_audio = app_config.video_generator._generate_audio
    original_create_video = app_config.video_generator._create_video
    
    def enhanced_generate_text(self, topic, prompt_type, texts_dir):
        update_task_progress(task_id, "text", "started", None)
        result = original_generate_text(topic, prompt_type, texts_dir)
        if result:
            update_task_progress(task_id, "text", "completed", None)
        return result
    
    def enhanced_generate_images(self, model, run_dir, images_dir):
        update_task_progress(task_id, "image", "started", None)
        
        # Fix the path issue and capture image generation progress
        import subprocess
        import logging
        logger = logging.getLogger("VideoGenerator")
        
        try:
            image_gen_path = app_config.parent_dir / "image_generation.py"
            cmd = [sys.executable, str(image_gen_path), "--model", model, "--run-dir", run_dir]
            
            # Run with real-time output capture
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            # Monitor output for progress updates
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    if "PROGRESS:" in line:
                        try:
                            parts = line.split("PROGRESS:")[1].strip().split("|")
                            if len(parts) >= 2:
                                progress = int(parts[0])
                                stage = parts[1]
                                
                                # Extract image count from stage
                                if "Generating image" in stage:
                                    img_parts = stage.split("Generating image ")[1].split("/")
                                    current = int(img_parts[0])
                                    total = int(img_parts[1])
                                    update_task_progress(task_id, "image", "progress", (current, total))
                        except:
                            pass
                    else:
                        logger.debug(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                logger.error(f"Image generation failed:\n{stderr}")
                return False
            
            import glob
            images = glob.glob(os.path.join(images_dir, "*.png"))
            if not images:
                logger.error("No images were generated")
                return False
            
            update_task_progress(task_id, "image", "completed", len(images))
            return True
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return False
    
    def enhanced_generate_audio(self, texts_dir, audio_dir, voice_id):
        update_task_progress(task_id, "tts", "started", None)
        result = original_generate_audio(texts_dir, audio_dir, voice_id)
        if result:
            update_task_progress(task_id, "tts", "completed", None)
        return result
    
    def enhanced_create_video(self, dirs, video_params):
        update_task_progress(task_id, "video", "started", None)
        result = original_create_video(dirs, video_params)
        if result:
            update_task_progress(task_id, "video", "completed", None)
        return result
    
    # Apply enhanced methods
    app_config.video_generator._generate_text = enhanced_generate_text.__get__(
        app_config.video_generator, type(app_config.video_generator)
    )
    app_config.video_generator._generate_images = enhanced_generate_images.__get__(
        app_config.video_generator, type(app_config.video_generator)
    )
    app_config.video_generator._generate_audio = enhanced_generate_audio.__get__(
        app_config.video_generator, type(app_config.video_generator)
    )
    app_config.video_generator._create_video = enhanced_create_video.__get__(
        app_config.video_generator, type(app_config.video_generator)
    )
    
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
    
    # Clean up handler
    logger.removeHandler(progress_handler)
    
    return result

def update_task_progress(task_id, category, event, data):
    """Update task progress based on category and event"""
    if task_id not in active_tasks:
        return
    
    task = active_tasks[task_id]
    details = task["details"]
    
    # Calculate overall progress based on phase weights
    # Text: 0-20%, Images: 20-70%, TTS: 70-80%, Video: 80-100%
    
    if category == "text":
        if event == "started":
            task["stage"] = "Generating text content"
            task["progress"] = 5
        elif event == "paragraph_complete":
            details["text_generation"]["paragraph"] = True
            task["stage"] = "Paragraph generated, extracting image descriptions"
            task["progress"] = 10
        elif event == "description_progress":
            current, total = data
            details["text_generation"]["descriptions_count"] = total
            details["text_generation"]["descriptions_completed"] = current
            task["stage"] = f"Extracting image descriptions ({current}/{total})"
            task["progress"] = 10 + int((current / total) * 8)  # 10-18%
        elif event == "descriptions_complete":
            details["text_generation"]["descriptions_count"] = data
            details["text_generation"]["descriptions_completed"] = data
            task["stage"] = f"Text generation complete ({data} descriptions)"
            task["progress"] = 20
        elif event == "completed":
            task["progress"] = 20
    
    elif category == "image":
        if event == "started":
            task["stage"] = "Starting image generation"
            task["progress"] = 22
        elif event == "progress":
            current, total = data
            details["image_generation"]["total"] = total
            details["image_generation"]["completed"] = current
            details["image_generation"]["current"] = current
            task["stage"] = f"Generating image {current}/{total}"
            # Images take up 20-70% of progress
            task["progress"] = 20 + int((current / total) * 50)
        elif event == "completed":
            total = data
            details["image_generation"]["total"] = total
            details["image_generation"]["completed"] = total
            task["stage"] = f"Image generation complete ({total} images)"
            task["progress"] = 70
    
    elif category == "tts":
        if event == "started":
            details["tts"]["started"] = True
            task["stage"] = "Starting text-to-speech"
            task["progress"] = 72
        elif event == "uploading":
            task["stage"] = "Uploading to ElevenLabs"
            task["progress"] = 74
        elif event == "processing":
            task["stage"] = "Processing audio"
            task["progress"] = 76
        elif event == "downloading":
            task["stage"] = "Downloading audio file"
            task["progress"] = 78
        elif event == "completed" or event == "complete":
            details["tts"]["completed"] = True
            task["stage"] = "Audio generation complete"
            task["progress"] = 80
    
    elif category == "video":
        if event == "started":
            details["video"]["started"] = True
            task["stage"] = "Starting video creation"
            task["progress"] = 82
        elif event == "progress":
            task["stage"] = f"Video: {data}"
            # Estimate progress between 82-98%
            if "Processing image" in str(data):
                try:
                    parts = str(data).split("Processing image ")[1].split("/")
                    current = int(parts[0])
                    total = int(parts[1])
                    task["progress"] = 82 + int((current / total) * 10)  # 82-92%
                except:
                    pass
            elif "Applying transitions" in str(data):
                task["progress"] = 93
            elif "Adding subtitles" in str(data):
                task["progress"] = 95
            elif "Encoding video" in str(data):
                task["progress"] = 97
        elif event == "completed" or event == "complete":
            details["video"]["completed"] = True
            task["stage"] = "Video creation complete!"
            task["progress"] = 100

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get detailed task status"""
    if task_id not in active_tasks:
        raise HTTPException(404, "Task not found")
    
    task = active_tasks[task_id]
    
    return {
        "id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
        "stage": task["stage"],
        "created_at": task["created_at"],
        "error": task.get("error"),
        "details": task.get("details", {})
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
                "created_at": task["created_at"],
                "details": task.get("details", {})
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