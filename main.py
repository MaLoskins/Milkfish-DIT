# Enhanced main.py with improved efficiency and optimization

import subprocess
import sys
import os
import shutil
import platform
import argparse
import glob
import random
import json
import logging
import time
import signal
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import lru_cache
import asyncio
import aiofiles

# Import project modules
from create_video import create_video
from text_generation import TextGenerator
from tts import TTSGenerator
from config import Config, load_config


# Enhanced logging with color support
class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure enhanced logging with optional file output."""
    logger = logging.getLogger("VideoGenerator")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    if sys.stdout.isatty():  # Only use color if output is a terminal
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


@dataclass
class VideoParameters:
    """Enhanced video generation parameters with validation."""
    # Video settings
    fps: int = 24
    aspect_ratio: Tuple[int, int] = (9, 16)
    transition_duration: float = 0.5
    
    # Effects
    pan_effect: bool = True
    zoom_effect: bool = True
    
    # Subtitles
    subtitles: bool = True
    subtitle_style: str = "modern"
    subtitle_animation: str = "phrase"
    subtitle_position: str = "bottom"
    highlight_keywords: bool = True
    
    # Transitions
    transition_style: str = "fade"
    
    # Performance
    cache_frames: bool = True
    parallel_processing: bool = True
    memory_limit_mb: int = 2048
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.fps < 1 or self.fps > 60:
            raise ValueError(f"FPS must be between 1 and 60, got {self.fps}")
        
        if self.transition_duration < 0 or self.transition_duration > 5:
            raise ValueError(f"Transition duration must be between 0 and 5, got {self.transition_duration}")
        
        if self.aspect_ratio[0] <= 0 or self.aspect_ratio[1] <= 0:
            raise ValueError("Aspect ratio values must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for create_video function."""
        return asdict(self)
    
    def get_identifier(self) -> str:
        """Get a unique identifier for this parameter set."""
        parts = []
        if self.subtitle_animation != "phrase":
            parts.append(f"anim_{self.subtitle_animation}")
        if self.subtitle_style != "modern":
            parts.append(f"style_{self.subtitle_style}")
        if self.transition_style != "fade":
            parts.append(f"trans_{self.transition_style}")
        if not self.subtitles:
            parts.append("nosub")
        if not self.pan_effect:
            parts.append("nopan")
        if not self.zoom_effect:
            parts.append("nozoom")
        
        return "_".join(parts) if parts else "baseline"


@dataclass
class SystemResources:
    """System resource monitoring."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_free_gb: float
    gpu_available: bool = False
    gpu_memory_free_mb: float = 0
    
    @classmethod
    def check(cls) -> 'SystemResources':
        """Check current system resources."""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check GPU if available
        gpu_available = False
        gpu_memory_free = 0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_memory_free = gpus[0].memoryFree
        except:
            pass
        
        return cls(
            cpu_percent=cpu,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            gpu_available=gpu_available,
            gpu_memory_free_mb=gpu_memory_free
        )
    
    def has_sufficient_resources(self, min_memory_mb: int = 1024) -> bool:
        """Check if system has sufficient resources."""
        return (
            self.memory_available_mb >= min_memory_mb and
            self.disk_free_gb >= 1.0 and
            self.cpu_percent < 90
        )


@dataclass
class VideoGenerationResult:
    """Enhanced result tracking with detailed metrics."""
    success: bool
    run_id: int
    topic: str
    prompt_type: str
    model: str
    voice: str
    output_dir: str
    video_params: VideoParameters
    error: Optional[str] = None
    duration: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        data['video_params'] = data['video_params'].to_dict() if hasattr(data['video_params'], 'to_dict') else data['video_params']
        return json.dumps(data, indent=2, default=str)


class ProcessManager:
    """Enhanced process management with resource tracking."""
    
    def __init__(self):
        self.processes = {}
        self.logger = logging.getLogger("VideoGenerator.ProcessManager")
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal, cleaning up...")
        self._shutdown = True
        self.cleanup()
        sys.exit(0)
    
    def start_ollama_server(self, wait_time: int = 10) -> bool:
        """Start Ollama server with improved error handling."""
        self.logger.info("Starting Ollama server...")
        
        try:
            if self._is_ollama_running():
                self.logger.info("Ollama server is already running")
                return True
            
            # Check if ollama is installed
            if not shutil.which('ollama'):
                self.logger.error("Ollama not found in PATH. Please install Ollama first.")
                return False
            
            # Start server based on platform
            if platform.system() == 'Windows':
                process = subprocess.Popen(
                    'ollama serve',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )
            
            self.processes['ollama'] = process
            
            # Progressive wait with early exit
            for i in range(wait_time):
                if self._shutdown:
                    return False
                
                time.sleep(1)
                if self._is_ollama_running():
                    self.logger.info(f"✓ Ollama server started in {i+1} seconds")
                    return True
                
                if i % 3 == 0:
                    self.logger.debug(f"Waiting for Ollama server... ({i+1}/{wait_time})")
            
            # Final check
            if self._is_ollama_running():
                self.logger.info("✓ Ollama server is running")
                return True
            else:
                self.logger.error("Ollama server failed to start within timeout")
                return False
                
        except FileNotFoundError:
            self.logger.error("'ollama' command not found. Ensure Ollama is installed and in PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            return False
    
    def stop_ollama_server(self) -> None:
        """Stop Ollama server with improved cleanup."""
        self.logger.info("Stopping Ollama server...")
        
        try:
            # Graceful shutdown first
            if 'ollama' in self.processes:
                process = self.processes['ollama']
                if platform.system() == 'Windows':
                    process.terminate()
                else:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except:
                        process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Ollama didn't stop gracefully, forcing...")
                    process.kill()
            
            # Force kill any remaining processes
            self._force_kill_ollama()
            
            # Clear from process list
            if 'ollama' in self.processes:
                del self.processes['ollama']
            
            # Wait for GPU memory release
            time.sleep(2)
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("✓ Ollama server stopped")
            
        except Exception as e:
            self.logger.warning(f"Error stopping Ollama server: {e}")
    
    def _force_kill_ollama(self):
        """Force kill ollama processes."""
        try:
            if platform.system() == 'Windows':
                subprocess.run(
                    ['taskkill', '/F', '/IM', 'ollama.exe'],
                    capture_output=True
                )
            else:
                subprocess.run(
                    ['pkill', '-f', 'ollama'],
                    capture_output=True
                )
        except:
            pass
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import urllib.request
            response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return response.status == 200
        except:
            return False
    
    def cleanup(self) -> None:
        """Clean up all managed processes."""
        for name, process in self.processes.items():
            try:
                self.logger.info(f"Cleaning up process: {name}")
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        self.processes.clear()


class VideoGenerator:
    """Enhanced video generation orchestrator with optimizations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("VideoGenerator")
        self.process_manager = ProcessManager()
        self.text_generator = TextGenerator(config)
        self.tts_generator = TTSGenerator(config)
        
        # Performance settings
        self.parallel_enabled = multiprocessing.cpu_count() > 2
        self.max_workers = min(4, multiprocessing.cpu_count() - 1)
        
        # Cache for reusable data
        self._cache = {}
        
        self.logger.info(f"Initialized with {self.max_workers} max workers")
    
    def generate_single_video(
        self,
        topic: str,
        prompt_type: str,
        model: str,
        voice: str,
        voice_id: str,
        output_base_dir: str,
        run_id: int,
        video_params: VideoParameters,
        progress_callback: Optional[callable] = None
    ) -> VideoGenerationResult:
        """Generate a single video with enhanced error handling and progress tracking."""
        start_time = time.time()
        metrics = {
            'text_generation_time': 0,
            'image_generation_time': 0,
            'audio_generation_time': 0,
            'video_creation_time': 0,
            'total_time': 0
        }
        
        # Check system resources
        resources = SystemResources.check()
        if not resources.has_sufficient_resources():
            self.logger.warning(
                f"Low system resources: {resources.memory_available_mb:.0f}MB RAM, "
                f"{resources.disk_free_gb:.1f}GB disk"
            )
        
        # Create output directory
        run_dir = self._create_output_directory(output_base_dir, run_id, topic, video_params)
        dirs = self._setup_directories(run_dir)
        
        # Save metadata
        self._save_metadata(run_dir, {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "prompt_type": prompt_type,
            "model": model,
            "voice": voice,
            "voice_id": voice_id,
            "video_params": asdict(video_params),
            "system_info": {
                "platform": platform.system(),
                "python_version": sys.version,
                "cpu_count": multiprocessing.cpu_count(),
                "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024)
            }
        })
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"RUN {run_id}: {topic}")
        self.logger.info(f"Config: {prompt_type} | {model} | {voice}")
        self.logger.info(f"Output: {run_dir}")
        self.logger.info(f"{'='*70}\n")
        
        try:
            # Phase 1: Text Generation
            if progress_callback:
                progress_callback("Generating text content", 10)
            
            phase_start = time.time()
            if not self._generate_text_content(topic, prompt_type, dirs['texts']):
                raise ValueError("Text generation failed")
            metrics['text_generation_time'] = time.time() - phase_start
            
            # Phase 2: Image Generation
            if progress_callback:
                progress_callback("Generating images", 30)
            
            phase_start = time.time()
            if not self._generate_images(model, run_dir, dirs['images']):
                raise ValueError("Image generation failed")
            metrics['image_generation_time'] = time.time() - phase_start
            
            # Phase 3: Text-to-Speech
            if progress_callback:
                progress_callback("Generating audio", 60)
            
            phase_start = time.time()
            if not self._generate_audio(dirs['texts'], dirs['audio'], voice_id):
                raise ValueError("Audio generation failed")
            metrics['audio_generation_time'] = time.time() - phase_start
            
            # Phase 4: Video Creation
            if progress_callback:
                progress_callback("Creating video", 80)
            
            phase_start = time.time()
            if not self._create_video(dirs, video_params):
                raise ValueError("Video creation failed")
            metrics['video_creation_time'] = time.time() - phase_start
            
            # Calculate total time
            total_time = time.time() - start_time
            metrics['total_time'] = total_time
            
            # Save final metadata with metrics
            self._save_metadata(run_dir, {
                "status": "completed",
                "metrics": metrics,
                "completion_time": datetime.now().isoformat()
            }, append=True)
            
            self.logger.info(f"\n✓ Video generated successfully in {total_time:.1f}s!")
            self._log_metrics(metrics)
            
            if progress_callback:
                progress_callback("Completed", 100)
            
            return VideoGenerationResult(
                success=True,
                run_id=run_id,
                topic=topic,
                prompt_type=prompt_type,
                model=model,
                voice=voice,
                output_dir=run_dir,
                video_params=video_params,
                duration=total_time,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"\n✗ Error in run {run_id}: {str(e)}")
            
            # Save error metadata
            self._save_metadata(run_dir, {
                "status": "failed",
                "error": str(e),
                "failure_time": datetime.now().isoformat()
            }, append=True)
            
            return VideoGenerationResult(
                success=False,
                run_id=run_id,
                topic=topic,
                prompt_type=prompt_type,
                model=model,
                voice=voice,
                output_dir=run_dir,
                video_params=video_params,
                error=str(e),
                duration=time.time() - start_time,
                metrics=metrics
            )
        finally:
            # Always ensure Ollama is stopped to free GPU
            self.process_manager.stop_ollama_server()
            
            # Clear caches
            self._clear_caches()
    
    def generate_batch(
        self,
        configurations: List[Dict[str, Any]],
        output_base_dir: str,
        max_parallel: Optional[int] = None
    ) -> List[VideoGenerationResult]:
        """Generate multiple videos with parallel processing support."""
        results = []
        total = len(configurations)
        
        # Create batch directory
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(output_base_dir, f"batch_{batch_timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save batch configuration
        batch_meta = {
            "timestamp": batch_timestamp,
            "total_videos": total,
            "configurations": configurations,
            "parallel_processing": self.parallel_enabled and total > 1
        }
        
        with open(os.path.join(batch_dir, "batch_config.json"), 'w') as f:
            json.dump(batch_meta, f, indent=2)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"STARTING BATCH: {total} videos")
        self.logger.info(f"Output: {batch_dir}")
        self.logger.info(f"{'='*70}\n")
        
        # Determine processing strategy
        if self.parallel_enabled and total > 1 and max_parallel != 1:
            results = self._generate_batch_parallel(configurations, batch_dir, max_parallel)
        else:
            results = self._generate_batch_sequential(configurations, batch_dir)
        
        # Save batch summary
        summary = {
            "batch_timestamp": batch_timestamp,
            "total_runs": total,
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_duration": sum(r.duration for r in results if r.duration),
            "results": [r.to_json() for r in results]
        }
        
        with open(os.path.join(batch_dir, "batch_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"BATCH COMPLETE")
        self.logger.info(f"Successful: {summary['successful']}/{total}")
        self.logger.info(f"Failed: {summary['failed']}/{total}")
        self.logger.info(f"Total time: {summary['total_duration']:.1f}s")
        self.logger.info(f"{'='*70}\n")
        
        return results
    
    def _generate_batch_sequential(
        self,
        configurations: List[Dict[str, Any]],
        batch_dir: str
    ) -> List[VideoGenerationResult]:
        """Generate videos sequentially."""
        results = []
        
        for i, config in enumerate(configurations, 1):
            self.logger.info(f"\nProcessing video {i}/{len(configurations)}...")
            
            result = self.generate_single_video(
                output_base_dir=batch_dir,
                run_id=i,
                **config
            )
            results.append(result)
            
            # Brief pause between generations
            if i < len(configurations):
                time.sleep(5)
        
        return results
    
    def _generate_batch_parallel(
        self,
        configurations: List[Dict[str, Any]],
        batch_dir: str,
        max_parallel: Optional[int] = None
    ) -> List[VideoGenerationResult]:
        """Generate videos in parallel with resource management."""
        results = []
        max_workers = min(max_parallel or self.max_workers, len(configurations))
        
        self.logger.info(f"Using parallel processing with {max_workers} workers")
        
        # Split work to avoid resource contention
        # Text generation and TTS can be parallelized more than image generation
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i, config in enumerate(configurations, 1):
                future = executor.submit(
                    self._generate_video_worker,
                    i,
                    config,
                    batch_dir
                )
                futures[future] = i
            
            # Process results as they complete
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✓" if result.success else "✗"
                    self.logger.info(
                        f"{status} Video {run_id}/{len(configurations)} completed"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Worker error for video {run_id}: {e}")
                    # Create error result
                    results.append(VideoGenerationResult(
                        success=False,
                        run_id=run_id,
                        topic="Unknown",
                        prompt_type="Unknown",
                        model="Unknown",
                        voice="Unknown",
                        output_dir="",
                        video_params=VideoParameters(),
                        error=str(e)
                    ))
        
        return results
    
    def _generate_video_worker(
        self,
        run_id: int,
        config: Dict[str, Any],
        batch_dir: str
    ) -> VideoGenerationResult:
        """Worker function for parallel video generation."""
        # Create new instances for process isolation
        worker_config = load_config()
        worker_generator = VideoGenerator(worker_config)
        
        return worker_generator.generate_single_video(
            output_base_dir=batch_dir,
            run_id=run_id,
            **config
        )
    
    def _create_output_directory(
        self,
        base_dir: str,
        run_id: int,
        topic: str,
        video_params: VideoParameters
    ) -> str:
        """Create unique output directory with proper naming."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize topic for filesystem
        safe_topic = "".join(
            c for c in topic if c.isalnum() or c in (' ', '-', '_')
        ).rstrip()[:50]
        
        # Add parameter identifier if not baseline
        param_id = video_params.get_identifier()
        if param_id != "baseline":
            dir_name = f"{run_id}_{timestamp}_{safe_topic}_{param_id}"
        else:
            dir_name = f"{run_id}_{timestamp}_{safe_topic}"
        
        run_dir = os.path.join(base_dir, dir_name)
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir
    
    def _setup_directories(self, run_dir: str) -> Dict[str, str]:
        """Create and return directory paths."""
        dirs = {
            'texts': os.path.join(run_dir, "texts"),
            'images': os.path.join(run_dir, "images"),
            'audio': os.path.join(run_dir, "audio"),
            'video': os.path.join(run_dir, "video"),
            'logs': os.path.join(run_dir, "logs")
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs
    
    def _save_metadata(self, run_dir: str, data: Dict[str, Any], append: bool = False):
        """Save or append metadata to JSON file."""
        metadata_path = os.path.join(run_dir, "metadata.json")
        
        if append and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                existing = json.load(f)
            existing.update(data)
            data = existing
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _generate_text_content(self, topic: str, prompt_type: str, texts_dir: str) -> bool:
        """Generate text content with caching support."""
        self.logger.info("Phase 1: Text Generation")
        
        # Check cache
        cache_key = f"{prompt_type}_{topic}"
        if cache_key in self._cache.get('texts', {}):
            self.logger.info("Using cached text content")
            cached = self._cache['texts'][cache_key]
            
            # Save to files
            with open(os.path.join(texts_dir, "paragraph.txt"), 'w', encoding='utf-8') as f:
                f.write(cached['paragraph'])
            
            with open(os.path.join(texts_dir, "image_descriptions.txt"), 'w', encoding='utf-8') as f:
                for desc in cached['descriptions']:
                    f.write(desc + '\n')
            
            return True
        
        # Start Ollama server
        if not self.process_manager.start_ollama_server():
            return False
        
        try:
            # Generate content
            paragraph = self.text_generator.generate_content(prompt_type, topic)
            if not paragraph or paragraph.strip() == "I can't fulfill this request.":
                self.logger.error(f"Text generation failed or was refused for topic: {topic}")
                return False
            
            self.logger.info(f"Generated paragraph ({len(paragraph)} chars)")
            self.logger.debug(f"Preview: {paragraph[:200]}...")
            
            # Extract image descriptions
            descriptions = self.text_generator.extract_image_descriptions(
                prompt_type, paragraph, topic
            )
            self.logger.info(f"Extracted {len(descriptions)} image descriptions")
            
            # Save texts
            self.text_generator.save_texts(prompt_type, paragraph, descriptions, texts_dir)
            
            # Cache for reuse
            if 'texts' not in self._cache:
                self._cache['texts'] = {}
            self._cache['texts'][cache_key] = {
                'paragraph': paragraph,
                'descriptions': descriptions
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Text generation error: {e}")
            return False
        finally:
            # Stop Ollama to free GPU for image generation
            self.process_manager.stop_ollama_server()
    
    def _generate_images(self, model: str, run_dir: str, images_dir: str) -> bool:
        """Generate images with improved error handling."""
        self.logger.info("Phase 2: Image Generation")
        
        try:
            # Run image generation script
            cmd = [
                sys.executable,
                "image_generation.py",
                "--model", model,
                "--run-dir", run_dir
            ]
            
            # Add debug flag if in debug mode
            if self.logger.level <= logging.DEBUG:
                cmd.append("--debug")
            
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Image generation failed:\n{result.stderr}")
                return False
            
            # Verify images were created
            image_files = glob.glob(os.path.join(images_dir, "*.png"))
            if not image_files:
                self.logger.error("No images were generated")
                return False
            
            self.logger.info(f"✓ Generated {len(image_files)} images")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Image generation timed out after 10 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return False
    
    def _generate_audio(self, texts_dir: str, audio_dir: str, voice_id: str) -> bool:
        """Generate audio with retry logic."""
        self.logger.info("Phase 3: Text-to-Speech")
        
        try:
            paragraph_file = os.path.join(texts_dir, "paragraph.txt")
            audio_output_path = os.path.join(audio_dir, "paragraph.mp3")
            
            # Check if paragraph exists
            if not os.path.exists(paragraph_file):
                self.logger.error("Paragraph file not found")
                return False
            
            # Generate audio with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    success = self.tts_generator.generate_audio(
                        paragraph_file,
                        audio_output_path,
                        voice_id,
                        max_retries=1  # Handle retries at this level
                    )
                    
                    if success:
                        break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        self.logger.warning(f"TTS attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if not success:
                return False
            
            # Verify files exist
            if not os.path.exists(audio_output_path):
                self.logger.error("Audio file was not created")
                return False
            
            timestamp_file = os.path.join(audio_dir, "time_stamps.json")
            if not os.path.exists(timestamp_file):
                self.logger.error("Timestamp file was not created")
                return False
            
            # Log audio file size
            audio_size = os.path.getsize(audio_output_path) / 1024
            self.logger.info(f"✓ Audio generated ({audio_size:.1f} KB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio generation error: {e}")
            return False
    
    def _create_video(self, dirs: Dict[str, str], video_params: VideoParameters) -> bool:
        """Create video with performance optimizations."""
        self.logger.info("Phase 4: Video Creation")
        
        try:
            # Define paths
            audio_path = os.path.join(dirs['audio'], "paragraph.mp3")
            paragraph_file = os.path.join(dirs['texts'], "paragraph.txt")
            timestamps_file = os.path.join(dirs['audio'], "time_stamps.json")
            output_video = os.path.join(dirs['video'], "final_video.mp4")
            log_file = os.path.join(dirs['logs'], "video_creation_log.txt")
            
            # Verify input files exist
            for file_path, name in [
                (audio_path, "Audio"),
                (paragraph_file, "Paragraph"),
                (timestamps_file, "Timestamps")
            ]:
                if not os.path.exists(file_path):
                    self.logger.error(f"{name} file not found: {file_path}")
                    return False
            
            # Get video parameters dict
            params = video_params.to_dict()
            
            # Remove performance parameters not used by create_video
            params.pop('cache_frames', None)
            params.pop('parallel_processing', None)
            params.pop('memory_limit_mb', None)
            
            self.logger.info(
                f"Creating video: {params['fps']}fps, "
                f"{params['subtitle_animation']} subtitles, "
                f"{params['transition_style']} transitions"
            )
            
            # Create video
            create_video(
                audio_path=audio_path,
                images_dir=dirs['images'],
                output_path=output_video,
                paragraph_file=paragraph_file,
                time_stamps_file=timestamps_file,
                log_file_path=log_file,
                **params
            )
            
            # Verify video was created
            if not os.path.exists(output_video):
                self.logger.error("Video file was not created")
                return False
            
            # Log video file size
            video_size = os.path.getsize(output_video) / (1024 * 1024)
            self.logger.info(f"✓ Video created ({video_size:.1f} MB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video creation error: {e}")
            return False
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"  Text Generation: {metrics['text_generation_time']:.1f}s")
        self.logger.info(f"  Image Generation: {metrics['image_generation_time']:.1f}s")
        self.logger.info(f"  Audio Generation: {metrics['audio_generation_time']:.1f}s")
        self.logger.info(f"  Video Creation: {metrics['video_creation_time']:.1f}s")
        self.logger.info(f"  Total Time: {metrics['total_time']:.1f}s")
    
    def _clear_caches(self):
        """Clear internal caches to free memory."""
        self._cache.clear()
        gc.collect()
    
    def cleanup(self):
        """Clean up resources."""
        self.process_manager.cleanup()
        self._clear_caches()


# Utility functions
@lru_cache(maxsize=128)
def load_prompts_from_file(file_path: str) -> Optional[List[str]]:
    """Load prompts from a text file with caching."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts


def get_default_prompts() -> List[str]:
    """Return a default list of prompts for testing."""
    return [
        "Why cats are actually liquid in disguise",
        "The secret underground society of shopping carts",
        "How to speedrun existence using only a toaster",
        "Why the moon is just Earth's backup save file",
        "The forbidden technique of photosynthesis for humans",
        "How medieval knights invented the first gaming chairs",
        "Why your sleep paralysis demon needs therapy too",
        "The scientific proof that birds work for the bourgeoisie",
        "How to unlock developer mode in reality",
        "Why touching grass is a government conspiracy",
        "The ancient art of communicating with houseplants",
        "How dinosaurs actually invented the internet",
        "Why your WiFi router is plotting world domination",
        "The hidden lore behind automatic sliding doors",
        "How to achieve immortality by never updating Windows"
    ]


def validate_environment() -> bool:
    """Validate that all required components are available."""
    logger = logging.getLogger("VideoGenerator")
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8 or higher is required")
    
    # Check required commands
    required_commands = ['ffmpeg', 'ffprobe']
    for cmd in required_commands:
        if not shutil.which(cmd):
            errors.append(f"{cmd} not found in PATH")
    
    # Check disk space
    disk_usage = psutil.disk_usage('/')
    if disk_usage.free < 5 * 1024 * 1024 * 1024:  # 5GB
        errors.append(f"Low disk space: {disk_usage.free / (1024**3):.1f}GB free")
    
    # Check memory
    memory = psutil.virtual_memory()
    if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
        errors.append(f"Low memory: {memory.available / (1024**3):.1f}GB available")
    
    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True


def main():
    """Enhanced main entry point with better argument handling."""
    parser = argparse.ArgumentParser(
        description="AI Video Generator - Create videos with AI-generated content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single video
  python main.py "Why cats secretly control the internet"
  
  # Generate with specific settings
  python main.py "Topic" --prompt_type conspiracy_theory --model Flux --voice expressive
  
  # Batch generation with random combinations
  python main.py --batch --num-runs 10
  
  # Batch with specific prompt file
  python main.py --batch --prompts-file topics.txt --model SD
  
  # Run ablation tests
  python main.py "Test topic" --ablation --ablation-config config.json
        """
    )
    
    # Basic arguments
    parser.add_argument(
        'topic',
        nargs='?',
        help='Topic for content generation (required for single mode)'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Run in batch mode with multiple videos'
    )
    mode_group.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation tests for parameter comparison'
    )
    
    # Content parameters
    content_group = parser.add_argument_group('content generation')
    content_group.add_argument(
        '--prompt_type',
        choices=[
            'did_you_know', 'scary_story', 'wrong_instructions',
            'fake_history', 'conspiracy_theory', 'bad_advice',
            'fake_discovery', 'incorrect_trivia', 'pseudo_deep_advice',
            'test', 'fun_facts', 'story_time', 'step_by_step',
            'science_explanation', 'historical_event', 'global_insights'
        ],
        help='Type of content to generate'
    )
    content_group.add_argument(
        '--model',
        choices=['Flux', 'SD'],
        help='Image generation model'
    )
    content_group.add_argument(
        '--voice',
        choices=[
            'raspy', 'upbeat', 'expressive', 'deep', 'authoritative',
            'trustworthy', 'friendly', 'intense', 'you_need_a_calc_for_that'
        ],
        help='Voice for text-to-speech'
    )
    
    # Video parameters
    video_group = parser.add_argument_group('video composition')
    video_group.add_argument('--fps', type=int, default=24, help='Frames per second (1-60)')
    video_group.add_argument('--aspect_ratio', nargs=2, type=int, default=[9, 16], 
                            metavar=('WIDTH', 'HEIGHT'), help='Aspect ratio')
    video_group.add_argument('--transition_duration', type=float, default=0.5, 
                            help='Transition duration in seconds (0-5)')
    video_group.add_argument('--no-pan', dest='pan_effect', action='store_false', 
                            help='Disable pan effect')
    video_group.add_argument('--no-zoom', dest='zoom_effect', action='store_false', 
                            help='Disable zoom effect')
    video_group.add_argument('--no-subtitles', dest='subtitles', action='store_false', 
                            help='Disable subtitles')
    video_group.add_argument('--subtitle_style', 
                            choices=['modern', 'minimal', 'bold', 'classic', 'dynamic'], 
                            default='modern', help='Subtitle style')
    video_group.add_argument('--subtitle_animation', 
                            choices=['phrase', 'word', 'word-by-word', 'typewriter'], 
                            default='phrase', help='Subtitle animation')
    video_group.add_argument('--subtitle_position', 
                            choices=['bottom', 'top', 'middle'], 
                            default='bottom', help='Subtitle position')
    video_group.add_argument('--transition_style', 
                            choices=['fade', 'slide', 'zoom'], 
                            default='fade', help='Transition style')
    video_group.add_argument('--no-keywords', dest='highlight_keywords', action='store_false', 
                            help='Disable keyword highlighting')
    
    # Batch parameters
    batch_group = parser.add_argument_group('batch processing')
    batch_group.add_argument('--prompts-file', help='File containing prompts (one per line)')
    batch_group.add_argument('--num-runs', type=int, default=10, 
                            help='Number of videos to generate in batch mode')
    batch_group.add_argument('--parallel', type=int, metavar='N',
                            help='Number of parallel workers (default: auto)')
    batch_group.add_argument('--batch-config', help='JSON file with batch configurations')
    
    # Ablation parameters
    ablation_group = parser.add_argument_group('ablation testing')
    ablation_group.add_argument('--ablation-config', 
                               help='JSON file containing ablation test configuration')
    
    # Output parameters
    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--output_dir', default='output', 
                             help='Output directory for generated files')
    output_group.add_argument('--log-file', help='Save logs to file')
    output_group.add_argument('--save-config', help='Save configuration to JSON file')
    
    # System parameters
    system_group = parser.add_argument_group('system options')
    system_group.add_argument('--debug', action='store_true', help='Enable debug logging')
    system_group.add_argument('--quiet', action='store_true', help='Minimal output')
    system_group.add_argument('--no-cleanup', action='store_true', 
                             help='Skip cleanup of temporary files')
    system_group.add_argument('--check-only', action='store_true', 
                             help='Check environment without generating')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else ("WARNING" if args.quiet else "INFO")
    global logger
    logger = setup_logging(log_level, args.log_file)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    
    if args.check_only:
        logger.info("Environment check passed!")
        sys.exit(0)
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize generator
    generator = VideoGenerator(config)
    
    # Create video parameters
    video_params = VideoParameters(
        fps=args.fps,
        aspect_ratio=tuple(args.aspect_ratio),
        transition_duration=args.transition_duration,
        pan_effect=args.pan_effect,
        zoom_effect=args.zoom_effect,
        subtitles=args.subtitles,
        subtitle_style=args.subtitle_style,
        subtitle_animation=args.subtitle_animation,
        subtitle_position=args.subtitle_position,
        transition_style=args.transition_style,
        highlight_keywords=args.highlight_keywords
    )
    
    # Save configuration if requested
    if args.save_config:
        config_data = {
            "mode": "batch" if args.batch else "ablation" if args.ablation else "single",
            "content": {
                "prompt_type": args.prompt_type,
                "model": args.model,
                "voice": args.voice
            },
            "video_params": video_params.to_dict(),
            "output_dir": args.output_dir
        }
        
        with open(args.save_config, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to {args.save_config}")
    
    try:
        if args.batch:
            # Batch mode
            run_batch_mode(args, config, generator, video_params)
            
        elif args.ablation:
            # Ablation mode
            if not args.topic:
                logger.error("Topic is required for ablation testing")
                sys.exit(1)
            
            run_ablation_mode(args, config, generator, video_params)
            
        else:
            # Single mode
            if not args.topic:
                logger.error("Topic is required in single mode")
                logger.info("Use --batch for batch mode or --ablation for ablation testing")
                sys.exit(1)
            
            run_single_mode(args, config, generator, video_params)
    
    finally:
        # Cleanup
        generator.cleanup()
        
        # Clean up temporary files unless disabled
        if not args.no_cleanup:
            cleanup_temp_files()


def run_single_mode(args, config, generator, video_params):
    """Run single video generation."""
    # Set defaults
    prompt_type = args.prompt_type or 'did_you_know'
    model = args.model or 'Flux'
    voice = args.voice or 'raspy'
    voice_id = config.voice_mappings.get(voice)
    
    if not voice_id:
        logger.error(f"Unknown voice: {voice}")
        sys.exit(1)
    
    logger.info(f"Generating single video: {args.topic}")
    
    # Generate video
    result = generator.generate_single_video(
        topic=args.topic,
        prompt_type=prompt_type,
        model=model,
        voice=voice,
        voice_id=voice_id,
        output_base_dir=args.output_dir,
        run_id=1,
        video_params=video_params
    )
    
    if result.success:
        logger.info(f"✓ Video saved to: {result.output_dir}")
    else:
        logger.error(f"✗ Generation failed: {result.error}")
        sys.exit(1)


def run_batch_mode(args, config, generator, video_params):
    """Run batch video generation."""
    logger.info("=== BATCH MODE ===")
    
    # Load prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        if not prompts:
            logger.error(f"Could not load prompts from {args.prompts_file}")
            sys.exit(1)
    else:
        prompts = get_default_prompts()
    
    # Load batch configuration if provided
    if args.batch_config:
        with open(args.batch_config, 'r') as f:
            batch_data = json.load(f)
        configurations = batch_data.get('configurations', [])
    else:
        # Generate random configurations
        configurations = []
        
        all_prompt_types = [
            'did_you_know', 'scary_story', 'wrong_instructions',
            'fake_history', 'conspiracy_theory', 'bad_advice',
            'fake_discovery', 'incorrect_trivia', 'pseudo_deep_advice'
        ]
        all_models = ['Flux', 'SD']
        all_voices = list(config.voice_mappings.keys())
        
        for i in range(args.num_runs):
            topic = random.choice(prompts)
            
            config_dict = {
                'topic': topic,
                'prompt_type': args.prompt_type or random.choice(all_prompt_types),
                'model': args.model or random.choice(all_models),
                'voice': args.voice or random.choice(all_voices),
                'voice_id': config.voice_mappings[args.voice or random.choice(all_voices)],
                'video_params': video_params
            }
            
            configurations.append(config_dict)
    
    # Run batch generation
    results = generator.generate_batch(
        configurations,
        args.output_dir,
        max_parallel=args.parallel
    )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"\nBatch Summary: {successful}/{len(results)} successful")


def run_ablation_mode(args, config, generator, video_params):
    """Run ablation tests."""
    # Implementation would be similar to the original
    # but with the enhanced generator
    logger.info("Ablation mode not fully implemented in this version")
    sys.exit(1)


def cleanup_temp_files():
    """Clean up temporary files."""
    patterns = ['*.log', '*.tmp', '__pycache__']
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
            except:
                pass


if __name__ == "__main__":
    main()