# config.py

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys (loaded from environment)
    elevenlabs_api_key: Optional[str] = None
    
    # Server URLs
    ollama_url: str = "http://localhost:11434"
    comfyui_url: str = "http://127.0.0.1:8188"
    
    # Model settings
    ollama_model: str = "llama2-uncensored"
    default_image_model: str = "Flux"
    
    # Video settings
    video_fps: int = 24
    video_aspect_ratio: tuple = (9, 16)
    
    # Audio settings
    tts_model: str = "eleven_multilingual_v2"
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75
    tts_style: float = 0.5
    tts_use_speaker_boost: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Voice mappings
    voice_mappings: Dict[str, str] = field(default_factory=lambda: {
        "raspy": "XB0fDUnXU5powFXDhCwa",
        "upbeat": "FGY2WhTYpPnrIDTdsKH5",
        "expressive": "9BWtsMINqrJLrRacOk9x",
        "deep": "nPczCjzI2devNBz1zQrb",
        "authoritative": "onwK4e9ZLuTAKqWW03F9",
        "trustworthy": "pqHfZKP75CvOlQylNhV4",
        "friendly": "XrExE9yKIg1WjnnlVkGX",
        "intense": "N2lVS1w4EtoT3dr4eOWO",
        "you_need_a_calc_for_that": "nF6oRtWvs9pNxDYGsbY5"
    })
    
    def __post_init__(self):
        """Load configuration from environment."""
        # Load API key from environment
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        
        # Override with environment variables if set
        if os.getenv('OLLAMA_MODEL'):
            self.ollama_model = os.getenv('OLLAMA_MODEL')
        if os.getenv('COMFYUI_URL'):
            self.comfyui_url = os.getenv('COMFYUI_URL')
        if os.getenv('VIDEO_FPS'):
            self.video_fps = int(os.getenv('VIDEO_FPS'))
        if os.getenv('VIDEO_ASPECT_RATIO'):
            ratio = os.getenv('VIDEO_ASPECT_RATIO').split(':')
            self.video_aspect_ratio = (int(ratio[0]), int(ratio[1]))
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.elevenlabs_api_key:
            print("ERROR: ELEVENLABS_API_KEY environment variable not set")
            print("Please set it using: export ELEVENLABS_API_KEY=your_key_here")
            return False
        return True
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'ollama_url': self.ollama_url,
            'comfyui_url': self.comfyui_url,
            'ollama_model': self.ollama_model,
            'default_image_model': self.default_image_model,
            'video_fps': self.video_fps,
            'video_aspect_ratio': self.video_aspect_ratio,
            'tts_model': self.tts_model,
            'tts_stability': self.tts_stability,
            'tts_similarity_boost': self.tts_similarity_boost,
            'tts_style': self.tts_style,
            'tts_use_speaker_boost': self.tts_use_speaker_boost,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        config = cls()
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file and environment."""
    if config_file and os.path.exists(config_file):
        config = Config.load_from_file(config_file)
    else:
        config = Config()
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid configuration")
    
    return config


# Load environment variables from .env file if it exists
def load_env_file(env_file: str = ".env") -> None:
    """Load environment variables from .env file."""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Auto-load .env file when module is imported
load_env_file()