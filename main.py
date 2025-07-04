# Optimized main.py with progress tracking
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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import project modules
from create_video import create_video
from text_generation import TextGenerator
from tts import TTSGenerator
from config import Config, load_config

# Logging setup
class ColoredFormatter(logging.Formatter):
    """Colored console output"""
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m'
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.stdout.isatty():
            record.levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level="INFO", log_file=None):
    """Configure logging"""
    logger = logging.getLogger("VideoGenerator")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
    return logger

logger = setup_logging()

@dataclass
class VideoParameters:
    """Video generation parameters"""
    fps: int = 24
    aspect_ratio: Tuple[int, int] = (9, 16)
    transition_duration: float = 0.5
    pan_effect: bool = True
    zoom_effect: bool = True
    subtitles: bool = True
    subtitle_style: str = "modern"
    subtitle_animation: str = "phrase"
    subtitle_position: str = "bottom"
    highlight_keywords: bool = True
    transition_style: str = "fade"
    
    def __post_init__(self):
        if not 1 <= self.fps <= 60:
            raise ValueError(f"FPS must be 1-60, got {self.fps}")
        if self.transition_duration < 0 or self.transition_duration > 5:
            raise ValueError(f"Transition duration must be 0-5, got {self.transition_duration}")
        if self.aspect_ratio[0] <= 0 or self.aspect_ratio[1] <= 0:
            raise ValueError("Aspect ratio values must be positive")
    
    def to_dict(self):
        return asdict(self)

@dataclass
class VideoGenerationResult:
    """Result tracking"""
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

class ProgressTracker:
    """Tracks and reports progress to frontend"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.current_progress = 0
        self.current_stage = "Initializing"
        self.stage_details = {}
        
    def update(self, progress: int, stage: str, details: Dict[str, Any] = None):
        """Update progress status"""
        self.current_progress = max(0, min(100, progress))
        self.current_stage = stage
        if details:
            self.stage_details.update(details)
        
        # Write to progress file
        progress_data = {
            "progress": self.current_progress,
            "stage": self.current_stage,
            "details": self.stage_details,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write progress file: {e}")

class ProcessManager:
    """Process management"""
    def __init__(self):
        self.processes = {}
        self.logger = logging.getLogger("VideoGenerator.ProcessManager")
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info("Shutdown signal received")
        self._shutdown = True
        self.cleanup()
        sys.exit(0)
    
    def start_ollama_server(self, wait_time=10):
        """Start Ollama server"""
        self.logger.info("Starting Ollama server...")
        
        if self._is_ollama_running():
            self.logger.info("Ollama already running")
            return True
        
        if not shutil.which('ollama'):
            self.logger.error("Ollama not found in PATH")
            return False
        
        try:
            if platform.system() == 'Windows':
                process = subprocess.Popen('ollama serve', shell=True, 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                process = subprocess.Popen(['ollama', 'serve'],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         preexec_fn=os.setsid)
            
            self.processes['ollama'] = process
            
            for i in range(wait_time):
                if self._shutdown: return False
                time.sleep(1)
                if self._is_ollama_running():
                    self.logger.info(f"✓ Ollama started in {i+1}s")
                    return True
            
            return self._is_ollama_running()
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama: {e}")
            return False
    
    def stop_ollama_server(self):
        """Stop Ollama server"""
        self.logger.info("Stopping Ollama...")
        
        if 'ollama' in self.processes:
            try:
                process = self.processes['ollama']
                if platform.system() == 'Windows':
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except:
                process.kill()
            
            del self.processes['ollama']
        
        # Force kill remaining
        try:
            if platform.system() == 'Windows':
                subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'], capture_output=True)
            else:
                subprocess.run(['pkill', '-f', 'ollama'], capture_output=True)
        except:
            pass
        
        time.sleep(2)
        gc.collect()
        self.logger.info("✓ Ollama stopped")
    
    def _is_ollama_running(self):
        try:
            import urllib.request
            response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return response.status == 200
        except:
            return False
    
    def cleanup(self):
        for name, process in self.processes.items():
            try:
                self.logger.info(f"Cleaning up {name}")
                process.terminate()
                process.wait(timeout=5)
            except:
                try: process.kill()
                except: pass
        self.processes.clear()

class VideoGenerator:
    """Main video generator with progress tracking"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("VideoGenerator")
        self.process_manager = ProcessManager()
        self.text_generator = TextGenerator(config)
        self.tts_generator = TTSGenerator(config)
        self.max_workers = min(4, multiprocessing.cpu_count() - 1)
        self.progress_tracker = None
    
    def generate_single_video(self, topic: str, prompt_type: str, model: str, voice: str,
                            voice_id: str, output_base_dir: str, run_id: int,
                            video_params: VideoParameters) -> VideoGenerationResult:
        """Generate a single video with progress tracking"""
        start_time = time.time()
        metrics = {'text_time': 0, 'image_time': 0, 'audio_time': 0, 'video_time': 0}
        
        # Check resources
        mem = psutil.virtual_memory()
        if mem.available < 2 * 1024 * 1024 * 1024:
            self.logger.warning(f"Low memory: {mem.available / (1024**3):.1f}GB")
        
        # Setup directories
        run_dir = self._create_output_dir(output_base_dir, run_id, topic, video_params)
        dirs = {k: os.path.join(run_dir, k) for k in ['texts', 'images', 'audio', 'video', 'logs']}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(run_dir)
        self.progress_tracker.update(5, "Initializing", {"run_id": run_id, "topic": topic})
        
        # Save metadata
        self._save_metadata(run_dir, {
            "run_id": run_id, "timestamp": datetime.now().isoformat(),
            "topic": topic, "prompt_type": prompt_type,
            "model": model, "voice": voice, "voice_id": voice_id,
            "video_params": asdict(video_params)
        })
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"RUN {run_id}: {topic}")
        self.logger.info(f"Config: {prompt_type} | {model} | {voice}")
        self.logger.info(f"Output: {run_dir}")
        self.logger.info(f"{'='*70}\n")
        
        try:
            # Text Generation
            self.progress_tracker.update(10, "Generating text content")
            t = time.time()
            if not self._generate_text(topic, prompt_type, dirs['texts']):
                raise ValueError("Text generation failed")
            metrics['text_time'] = time.time() - t
            
            # Image Generation
            self.progress_tracker.update(30, "Starting image generation")
            t = time.time()
            if not self._generate_images(model, run_dir, dirs['images']):
                raise ValueError("Image generation failed")
            metrics['image_time'] = time.time() - t
            
            # Audio Generation
            self.progress_tracker.update(70, "Generating audio narration")
            t = time.time()
            if not self._generate_audio(dirs['texts'], dirs['audio'], voice_id):
                raise ValueError("Audio generation failed")
            metrics['audio_time'] = time.time() - t
            
            # Video Creation
            self.progress_tracker.update(80, "Creating final video")
            t = time.time()
            if not self._create_video(dirs, video_params):
                raise ValueError("Video creation failed")
            metrics['video_time'] = time.time() - t
            
            # Complete
            duration = time.time() - start_time
            metrics['total_time'] = duration
            
            self._save_metadata(run_dir, {"status": "completed", "metrics": metrics}, append=True)
            self.progress_tracker.update(100, "Video generation complete!")
            
            self.logger.info(f"\n✓ Video generated in {duration:.1f}s")
            self.logger.info(f"  Text: {metrics['text_time']:.1f}s")
            self.logger.info(f"  Images: {metrics['image_time']:.1f}s")
            self.logger.info(f"  Audio: {metrics['audio_time']:.1f}s")
            self.logger.info(f"  Video: {metrics['video_time']:.1f}s")
            
            return VideoGenerationResult(
                success=True, run_id=run_id, topic=topic,
                prompt_type=prompt_type, model=model, voice=voice,
                output_dir=run_dir, video_params=video_params,
                duration=duration, metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"\n✗ Error in run {run_id}: {str(e)}")
            self._save_metadata(run_dir, {"status": "failed", "error": str(e)}, append=True)
            if self.progress_tracker:
                self.progress_tracker.update(0, f"Failed: {str(e)}")
            
            return VideoGenerationResult(
                success=False, run_id=run_id, topic=topic,
                prompt_type=prompt_type, model=model, voice=voice,
                output_dir=run_dir, video_params=video_params,
                error=str(e), duration=time.time() - start_time
            )
        finally:
            self.process_manager.stop_ollama_server()
            gc.collect()
    
    def _create_output_dir(self, base_dir, run_id, topic, params):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in ' -_')[:50].strip()
        dir_name = f"{run_id}_{timestamp}_{safe_topic}"
        path = os.path.join(base_dir, dir_name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def _save_metadata(self, run_dir, data, append=False):
        path = os.path.join(run_dir, "metadata.json")
        if append and os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            existing.update(data)
            data = existing
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _generate_text(self, topic, prompt_type, texts_dir):
        self.logger.info("Phase 1: Text Generation")
        self.progress_tracker.update(12, "Starting Ollama server")
        
        if not self.process_manager.start_ollama_server():
            return False
        
        try:
            # Generate paragraph
            self.progress_tracker.update(15, "Generating paragraph content")
            paragraph = self.text_generator.generate_content(prompt_type, topic)
            if not paragraph or "I can't fulfill this request" in paragraph:
                self.logger.error(f"Text generation failed for topic: {topic}")
                return False
            
            self.logger.info(f"Generated paragraph ({len(paragraph)} chars)")
            self.progress_tracker.update(18, "Paragraph generated", {"paragraph_length": len(paragraph)})
            
            # Extract image descriptions with progress callback
            self.progress_tracker.update(20, "Extracting image descriptions")
            
            def description_progress(current, total):
                progress = 20 + int((current / total) * 8)  # 20-28% range
                self.progress_tracker.update(progress, f"Image description {current}/{total}")
            
            descriptions = self.text_generator.extract_image_descriptions(
                prompt_type, paragraph, topic, progress_callback=description_progress
            )
            self.logger.info(f"Extracted {len(descriptions)} image descriptions")
            self.progress_tracker.update(28, "Image descriptions complete", {"description_count": len(descriptions)})
            
            self.text_generator.save_texts(prompt_type, paragraph, descriptions, texts_dir)
            return True
            
        except Exception as e:
            self.logger.error(f"Text generation error: {e}")
            return False
        finally:
            self.process_manager.stop_ollama_server()
    
    def _generate_images(self, model, run_dir, images_dir):
        self.logger.info("Phase 2: Image Generation")
        self.progress_tracker.update(30, "Preparing image generation")
        
        try:
            cmd = [sys.executable, "image_generation.py", "--model", model, "--run-dir", run_dir]
            if self.logger.level <= logging.DEBUG:
                cmd.append("--debug")
            
            # Run with real-time output capture
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            # Monitor output for progress updates
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    if "PROGRESS:" in line:
                        # Parse progress update from image generation
                        try:
                            parts = line.split("PROGRESS:")[1].strip().split("|")
                            if len(parts) >= 2:
                                progress = int(parts[0])
                                stage = parts[1]
                                # Map image generation progress (0-100) to overall progress (30-70)
                                mapped_progress = 30 + int(progress * 0.4)
                                self.progress_tracker.update(mapped_progress, stage)
                        except:
                            pass
                    else:
                        self.logger.debug(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                self.logger.error(f"Image generation failed:\n{stderr}")
                return False
            
            images = glob.glob(os.path.join(images_dir, "*.png"))
            if not images:
                self.logger.error("No images were generated")
                return False
            
            self.logger.info(f"✓ Generated {len(images)} images")
            self.progress_tracker.update(70, "Image generation complete", {"image_count": len(images)})
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Image generation timeout")
            return False
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return False
    
    def _generate_audio(self, texts_dir, audio_dir, voice_id):
        self.logger.info("Phase 3: Text-to-Speech")
        self.progress_tracker.update(72, "Starting audio generation")
        
        try:
            paragraph_file = os.path.join(texts_dir, "paragraph.txt")
            audio_file = os.path.join(audio_dir, "paragraph.mp3")
            
            if not os.path.exists(paragraph_file):
                self.logger.error("Paragraph file not found")
                return False
            
            # Add progress callback
            def audio_progress(status):
                if status == "uploading":
                    self.progress_tracker.update(74, "Uploading to ElevenLabs")
                elif status == "processing":
                    self.progress_tracker.update(76, "Processing audio")
                elif status == "downloading":
                    self.progress_tracker.update(78, "Downloading audio file")
            
            success = self.tts_generator.generate_audio(
                paragraph_file, audio_file, voice_id, max_retries=3, progress_callback=audio_progress
            )
            
            if not success or not os.path.exists(audio_file):
                return False
            
            self.logger.info(f"✓ Audio generated ({os.path.getsize(audio_file) / 1024:.1f} KB)")
            self.progress_tracker.update(80, "Audio generation complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio generation error: {e}")
            return False
    
    def _create_video(self, dirs, video_params):
        self.logger.info("Phase 4: Video Creation")
        self.progress_tracker.update(82, "Initializing video creation")
        
        try:
            audio_path = os.path.join(dirs['audio'], "paragraph.mp3")
            paragraph_file = os.path.join(dirs['texts'], "paragraph.txt")
            timestamps_file = os.path.join(dirs['audio'], "time_stamps.json")
            output_video = os.path.join(dirs['video'], "final_video.mp4")
            log_file = os.path.join(dirs['logs'], "video_creation_log.txt")
            
            # Verify files exist
            for f, name in [(audio_path, "Audio"), (paragraph_file, "Paragraph"), (timestamps_file, "Timestamps")]:
                if not os.path.exists(f):
                    self.logger.error(f"{name} file not found: {f}")
                    return False
            
            params = video_params.to_dict()
            
            # Add progress callback
            def video_progress(progress, stage):
                # Map video creation progress (0-100) to overall progress (82-100)
                mapped_progress = 82 + int(progress * 0.18)
                self.progress_tracker.update(mapped_progress, f"Video: {stage}")
            
            params['progress_callback'] = video_progress
            
            create_video(
                audio_path=audio_path,
                images_dir=dirs['images'],
                output_path=output_video,
                paragraph_file=paragraph_file,
                time_stamps_file=timestamps_file,
                log_file_path=log_file,
                **params
            )
            
            if not os.path.exists(output_video):
                self.logger.error("Video file was not created")
                return False
            
            self.logger.info(f"✓ Video created ({os.path.getsize(output_video) / (1024*1024):.1f} MB)")
            self.progress_tracker.update(100, "Video creation complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Video creation error: {e}")
            return False
    
    def cleanup(self):
        self.process_manager.cleanup()
        gc.collect()

def get_default_prompts():
    """Default test prompts"""
    return [
        "Why cats are actually liquid in disguise",
        "The secret underground society of shopping carts",
        "How to speedrun existence using only a toaster",
        "Why the moon is just Earth's backup save file",
        "The forbidden technique of photosynthesis for humans"
    ]

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Milkfish DIT - Create videos with AI-generated content",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('topic', nargs='?', help='Topic for content generation')
    parser.add_argument('--prompt_type', help='Type of content to generate')
    parser.add_argument('--model', choices=['Flux', 'SD'], help='Image generation model')
    parser.add_argument('--voice', help='Voice for text-to-speech')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Video parameters
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--aspect_ratio', nargs=2, type=int, default=[9, 16], help='Aspect ratio')
    parser.add_argument('--no-subtitles', dest='subtitles', action='store_false', help='Disable subtitles')
    parser.add_argument('--subtitle_style', default='modern', help='Subtitle style')
    parser.add_argument('--subtitle_animation', default='phrase', help='Subtitle animation')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging("DEBUG" if args.debug else "INFO")
    
    # Validate environment
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        sys.exit(1)
    
    for cmd in ['ffmpeg', 'ffprobe']:
        if not shutil.which(cmd):
            logger.error(f"{cmd} not found in PATH")
            sys.exit(1)
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize generator
    generator = VideoGenerator(config)
    
    # Create video parameters
    video_params = VideoParameters(
        fps=args.fps,
        aspect_ratio=tuple(args.aspect_ratio),
        subtitles=args.subtitles,
        subtitle_style=args.subtitle_style,
        subtitle_animation=args.subtitle_animation
    )
    
    try:
        if not args.topic:
            logger.error("Topic is required")
            sys.exit(1)
        
        # Set defaults
        prompt_type = args.prompt_type or 'did_you_know'
        model = args.model or 'Flux'
        voice = args.voice or 'raspy'
        voice_id = config.voice_mappings.get(voice)
        
        if not voice_id:
            logger.error(f"Unknown voice: {voice}")
            sys.exit(1)
        
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
    
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()