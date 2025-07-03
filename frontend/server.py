# Enhanced frontend server with improved efficiency and error handling

import os
import sys
import json
import asyncio
import uuid
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Add parent directory to path to import project modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import aiofiles

# Import existing modules with proper error handling
try:
    from main import VideoGenerator, VideoParameters, ProcessManager
    from config import load_config
    from prompts import get_prompt_types
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are in the parent directory")
    sys.exit(1)


# Import for lifespan
from contextlib import asynccontextmanager

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Milkfish DIT Server...")
    
    # Ensure output directory exists
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Clean up old temporary files
    temp_files = list(Path(".").glob("*.log"))
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except:
            pass
    
    print(f"Server ready. Configuration loaded: {app_config.initialized}")
    print(f"Parent directory: {parent_dir}")
    
    yield
    
    # Shutdown
    print("Shutting down server...")
    
    # Clean up any running tasks
    for task_id, task in active_tasks.items():
        if task["status"] == "running":
            task["status"] = "interrupted"
            task["error"] = "Server shutdown"
    
    # Shutdown executor
    if hasattr(app_config, 'executor'):
        app_config.executor.shutdown(wait=True)
    
    print("Server shutdown complete.")

# FastAPI app with enhanced configuration
app = FastAPI(
    title="Milkfish DIT",
    description="Professional AI-powered video generation service",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for better compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration and initialization
class AppConfig:
    """Application configuration singleton"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        try:
            self.config = load_config()
            self.video_generator = VideoGenerator(self.config)
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.initialized = True
            self.parent_dir = parent_dir  # Store parent directory
        except Exception as e:
            print(f"Warning: Failed to initialize config: {e}")
            self.config = None
            self.video_generator = None
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.initialized = False
            self.parent_dir = parent_dir
    
    def get_generator(self):
        if not self.initialized or not self.video_generator:
            raise HTTPException(
                status_code=503,
                detail="Video generator not available. Check server configuration."
            )
        return self.video_generator


# Initialize app config
app_config = AppConfig()

# Store active tasks with enhanced tracking
active_tasks: Dict[str, Dict] = {}

# Request models with validation
from pydantic import field_validator

class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    prompt_type: str = Field(default="did_you_know")
    model: str = Field(default="Flux", pattern="^(Flux|SD)$")
    voice: str = Field(default="raspy")
    # Video parameters
    fps: int = Field(default=24, ge=1, le=60)
    aspect_ratio: List[int] = Field(default=[9, 16], min_items=2, max_items=2)
    transition_duration: float = Field(default=0.5, ge=0, le=5)
    pan_effect: bool = True
    zoom_effect: bool = True
    subtitles: bool = True
    subtitle_style: str = Field(default="modern")
    subtitle_animation: str = Field(default="phrase")
    subtitle_position: str = Field(default="bottom")
    highlight_keywords: bool = True
    transition_style: str = Field(default="fade")
    
    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v):
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError('Aspect ratio values must be positive')
        return v
    
    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v):
        if not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()


# Serve static files with caching
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(
        "index.html",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@app.get("/style.css", response_class=FileResponse)
async def read_style():
    return FileResponse(
        "style.css",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@app.get("/script.js", response_class=FileResponse)
async def read_script():
    return FileResponse(
        "script.js",
        headers={"Cache-Control": "public, max-age=3600"}
    )

# Health check endpoint
@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    """Check if the server is healthy and all services are available."""
    return {
        "status": "healthy",
        "services": {
            "config": app_config.config is not None,
            "video_generator": app_config.video_generator is not None,
            "timestamp": datetime.now().isoformat()
        }
    }

# API endpoints
@app.get("/api/config")
@app.head("/api/config")
async def get_configuration():
    """Get available configuration options."""
    try:
        prompt_types = get_prompt_types()
        voices = list(app_config.config.voice_mappings.keys()) if app_config.config else [
            "raspy", "upbeat", "expressive", "deep", "authoritative",
            "trustworthy", "friendly", "intense"
        ]
        
        return {
            "prompt_types": prompt_types,
            "models": ["Flux", "SD"],
            "voices": voices,
            "subtitle_styles": ["modern", "minimal", "bold", "classic", "dynamic"],
            "subtitle_animations": ["phrase", "word", "word-by-word", "typewriter"],
            "subtitle_positions": ["bottom", "top", "middle"],
            "transition_styles": ["fade", "slide", "zoom"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

@app.post("/api/generate")
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation process."""
    video_generator = app_config.get_generator()
    
    # Validate voice exists
    if app_config.config and request.voice not in app_config.config.voice_mappings:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice: {request.voice}"
        )
    
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    active_tasks[task_id] = {
        "id": task_id,
        "status": "starting",
        "progress": 0,
        "stage": "Initializing",
        "topic": request.topic,
        "created_at": datetime.now().isoformat(),
        "output_dir": None,
        "error": None,
        "request": request.model_dump()  # Fixed: use model_dump() instead of dict()
    }
    
    # Start generation in background
    background_tasks.add_task(
        run_video_generation,
        task_id,
        request
    )
    
    return {"task_id": task_id, "message": "Video generation started"}

async def run_video_generation(task_id: str, request: GenerateRequest):
    """Run video generation process with enhanced error handling."""
    try:
        # Update status
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["stage"] = "Preparing generation"
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
            subtitle_position=request.subtitle_position,
            transition_style=request.transition_style,
            highlight_keywords=request.highlight_keywords
        )
        
        # Get voice ID
        voice_id = None
        if app_config.config:
            voice_id = app_config.config.voice_mappings.get(request.voice)
        else:
            # Fallback voice mappings
            fallback_voices = {
                "raspy": "XB0fDUnXU5powFXDhCwa",
                "upbeat": "FGY2WhTYpPnrIDTdsKH5",
                "expressive": "9BWtsMINqrJLrRacOk9x",
                "deep": "nPczCjzI2devNBz1zQrb",
                "authoritative": "onwK4e9ZLuTAKqWW03F9",
                "trustworthy": "pqHfZKP75CvOlQylNhV4",
                "friendly": "XrExE9yKIg1WjnnlVkGX",
                "intense": "N2lVS1w4EtoT3dr4eOWO"
            }
            voice_id = fallback_voices.get(request.voice)
            
        if not voice_id:
            raise ValueError(f"Unknown voice: {request.voice}")
        
        # Prepare output directory
        output_path = Path("../output")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate run_id
        existing_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        run_id = len(existing_dirs) + 1
        
        # Create a progress callback
        def update_progress(stage: str, progress: int):
            if task_id in active_tasks:
                active_tasks[task_id]["stage"] = stage
                active_tasks[task_id]["progress"] = progress
        
        # Update progress based on typical workflow
        update_progress("Generating text content", 10)
        
        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app_config.executor,
            generate_video_sync,
            app_config.get_generator(),
            request,
            voice_id,
            run_id,
            video_params,
            task_id
        )
        
        if result.success:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["progress"] = 100
            active_tasks[task_id]["stage"] = "Completed"
            active_tasks[task_id]["output_dir"] = result.output_dir
            active_tasks[task_id]["video_path"] = os.path.join(result.output_dir, "video", "final_video.mp4")
        else:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = result.error
            
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        active_tasks[task_id]["traceback"] = traceback.format_exc()
        
    finally:
        # Clean up task after 1 hour
        await asyncio.sleep(3600)
        if task_id in active_tasks:
            del active_tasks[task_id]

def generate_video_sync(generator, request, voice_id, run_id, video_params, task_id):
    """Synchronous video generation with progress updates."""
    # Create a custom generator that knows about the parent directory
    class VideoGeneratorWithPath(type(generator)):
        def _generate_images(self, model, run_dir, images_dir):
            """Override to fix the path issue"""
            import subprocess
            import logging
            
            logger = logging.getLogger("VideoGenerator")
            logger.info("Phase 2: Image Generation")
            
            try:
                # Use the correct path to image_generation.py
                image_gen_path = os.path.join(app_config.parent_dir, "image_generation.py")
                
                cmd = [
                    sys.executable,
                    image_gen_path,  # Fixed: use full path
                    "--model", model,
                    "--run-dir", run_dir
                ]
                
                # Add debug flag if in debug mode
                if logger.level <= logging.DEBUG:
                    cmd.append("--debug")
                
                logger.debug(f"Running command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"Image generation failed:\n{result.stderr}")
                    return False
                
                # Verify images were created
                import glob
                image_files = glob.glob(os.path.join(images_dir, "*.png"))
                if not image_files:
                    logger.error("No images were generated")
                    return False
                
                logger.info(f"✓ Generated {len(image_files)} images")
                return True
                
            except subprocess.TimeoutExpired:
                logger.error("Image generation timed out after 10 minutes")
                return False
            except Exception as e:
                logger.error(f"Image generation error: {e}")
                return False
    
    # Create a wrapped generator with the fixed method
    wrapped_generator = generator
    wrapped_generator._generate_images = VideoGeneratorWithPath._generate_images.__get__(wrapped_generator)
    
    # Update progress during generation
    def progress_callback(stage, progress):
        if task_id in active_tasks:
            active_tasks[task_id]["stage"] = stage
            active_tasks[task_id]["progress"] = progress
    
    # Monitor the generation process by checking file creation
    import threading
    
    def monitor():
        while task_id in active_tasks and active_tasks[task_id]["status"] == "running":
            try:
                # Check progress based on output directory
                if active_tasks[task_id].get("output_dir"):
                    output_dir = Path(active_tasks[task_id]["output_dir"])
                    if output_dir.exists():
                        if (output_dir / "texts" / "paragraph.txt").exists():
                            progress_callback("Generating images", 30)
                        if list((output_dir / "images").glob("*.png")):
                            progress_callback("Generating audio", 60)
                        if (output_dir / "audio" / "paragraph.mp3").exists():
                            progress_callback("Creating video", 80)
            except:
                pass
            threading.Event().wait(2)
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    result = wrapped_generator.generate_single_video(
        topic=request.topic,
        prompt_type=request.prompt_type,
        model=request.model,
        voice=request.voice,
        voice_id=voice_id,
        output_base_dir="../output",
        run_id=run_id,
        video_params=video_params
    )
    
    # Store output directory in task
    if task_id in active_tasks:
        active_tasks[task_id]["output_dir"] = result.output_dir
    
    return result

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get generation status with enhanced information."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    # Enhanced progress estimation based on output files
    if task["status"] == "running" and task.get("output_dir"):
        output_path = Path(task["output_dir"])
        if output_path.exists():
            progress_hints = {
                "texts/paragraph.txt": (30, "Generating images"),
                "images/*.png": (60, "Generating audio"),
                "audio/paragraph.mp3": (80, "Creating video"),
                "video/final_video.mp4": (95, "Finalizing")
            }
            
            for file_pattern, (progress, stage) in progress_hints.items():
                if "*" in file_pattern:
                    dir_path, pattern = file_pattern.split("/")
                    if list((output_path / dir_path).glob(pattern)):
                        task["progress"] = progress
                        task["stage"] = stage
                else:
                    if (output_path / file_pattern).exists():
                        task["progress"] = progress
                        task["stage"] = stage
    
    return {
        "id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
        "stage": task["stage"],
        "created_at": task["created_at"],
        "error": task.get("error")
    }

@app.get("/api/videos")
async def list_videos(request: Request):
    """List all generated videos with pagination support."""
    try:
        page = int(request.query_params.get('page', 1))
        per_page = int(request.query_params.get('per_page', 50))
    except:
        page = 1
        per_page = 50
    
    videos = []
    output_dir = Path("../output")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"videos": [], "total": 0, "page": page, "per_page": per_page}
    
    # Find all video directories
    video_dirs = []
    for dir_path in output_dir.iterdir():
        if not dir_path.is_dir():
            continue
            
        video_path = dir_path / "video" / "final_video.mp4"
        metadata_path = dir_path / "metadata.json"
        
        if video_path.exists() and metadata_path.exists():
            try:
                # Get file modification time for sorting
                mtime = metadata_path.stat().st_mtime
                video_dirs.append((mtime, dir_path))
            except:
                continue
    
    # Sort by modification time (newest first)
    video_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # Apply pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_dirs = video_dirs[start_idx:end_idx]
    
    # Build video list
    for _, dir_path in page_dirs:
        try:
            metadata_path = dir_path / "metadata.json"
            video_path = dir_path / "video" / "final_video.mp4"
            
            async with aiofiles.open(metadata_path, mode='r') as f:
                metadata = json.loads(await f.read())
            
            # Get file info
            stat = video_path.stat()
            
            # Get video duration if possible
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
                "size": stat.st_size,
                "duration": duration,
                "video_params": metadata.get("video_params", {}),
                "status": "completed"
            })
            
        except Exception as e:
            print(f"Error reading metadata from {dir_path}: {e}")
    
    return {
        "videos": videos,
        "total": len(video_dirs),
        "page": page,
        "per_page": per_page,
        "pages": (len(video_dirs) + per_page - 1) // per_page
    }

@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """Stream video file efficiently."""
    video_path = Path("../output") / video_id / "video" / "final_video.mp4"
    
    if not video_path.exists():
        # Check if it's a currently generating video
        for task in active_tasks.values():
            if task.get("output_dir", "").endswith(video_id) and task["status"] == "running":
                raise HTTPException(
                    status_code=202,
                    detail="Video is still being generated"
                )
        
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get file size for range requests
    file_size = video_path.stat().st_size
    
    # Stream the video file
    async def iterfile():
        async with aiofiles.open(video_path, 'rb') as f:
            while chunk := await f.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{video_id}.mp4"'
        }
    )

@app.get("/api/video/{video_id}/thumbnail")
async def get_thumbnail(video_id: str):
    """Get video thumbnail with caching."""
    # Check cache first
    cache_dir = Path("../output/.thumbnails")
    cache_dir.mkdir(exist_ok=True)
    cached_thumbnail = cache_dir / f"{video_id}.jpg"
    
    if cached_thumbnail.exists():
        return FileResponse(
            cached_thumbnail,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )
    
    # Try to find first image as thumbnail
    images_dir = Path("../output") / video_id / "images"
    if images_dir.exists():
        images = sorted(images_dir.glob("*.png"))
        if images:
            # Generate thumbnail
            try:
                from PIL import Image
                
                # Open and resize image
                with Image.open(images[0]) as img:
                    # Create thumbnail (max 320x180)
                    img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                    
                    # Save to cache
                    img.save(cached_thumbnail, "JPEG", quality=85, optimize=True)
                
                return FileResponse(
                    cached_thumbnail,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"}
                )
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
    
    # Return 204 No Content if no thumbnail available
    return JSONResponse(content=None, status_code=204)

@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and its directory."""
    video_dir = Path("../output") / video_id
    
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if video is currently being generated
    for task_id, task in active_tasks.items():
        if task.get("output_dir", "").endswith(video_id):
            if task["status"] == "running":
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete video while it's being generated"
                )
            # Remove from active tasks
            del active_tasks[task_id]
            break
    
    try:
        # Delete directory
        shutil.rmtree(video_dir)
        
        # Delete cached thumbnail if exists
        cached_thumbnail = Path("../output/.thumbnails") / f"{video_id}.jpg"
        if cached_thumbnail.exists():
            cached_thumbnail.unlink()
        
        return {"message": "Video deleted successfully", "id": video_id}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete video: {str(e)}"
        )

@app.get("/api/tasks")
async def list_tasks():
    """List all active generation tasks."""
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
        ],
        "total": len(active_tasks)
    }

@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel an active generation task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    if task["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not running (status: {task['status']})"
        )
    
    # Mark as cancelled
    task["status"] = "cancelled"
    task["error"] = "Cancelled by user"
    
    # Clean up output directory if exists
    if task.get("output_dir"):
        try:
            shutil.rmtree(task["output_dir"])
        except:
            pass
    
    return {"message": "Task cancelled", "id": task_id}
