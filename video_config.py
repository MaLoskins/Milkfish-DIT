# video_config.py
"""Configuration dataclasses for video generation."""

import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Literal, List, Dict, Optional
from pathlib import Path


@dataclass
class SubtitleOptions:
    """Configuration for video subtitles."""
    enabled: bool = True
    style: Literal["modern", "minimal", "bold", "classic", "dynamic"] = "modern"
    animation: Literal["phrase", "word", "word-by-word", "typewriter"] = "phrase"
    position: Literal["bottom", "top", "middle"] = "bottom"
    highlight_keywords: bool = True
    font_size_ratio: float = 0.05
    stroke_width_ratio: float = 0.002
    bg_opacity: float = 0.5
    fade_duration: float = 0.2
    keywords: List[str] = field(default_factory=list)
    
    # Additional options
    max_chars_per_line: int = 40
    max_lines: int = 2
    typewriter_speed: float = 0.05  # seconds per character
    word_highlight_color: str = "#FFD700"  # Gold for keywords
    
    def __post_init__(self):
        if not self.keywords:
            self.keywords = [
                'key', 'critical', 'important', 'essential', 'crucial',
                'significant', 'major', 'primary', 'fundamental', 'vital',
                'must', 'never', 'always', 'secret', 'shocking', 'amazing'
            ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleOptions':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TransitionOptions:
    """Configuration for video transitions."""
    style: Literal["fade", "slide", "zoom", "wipe", "random"] = "fade"
    duration: float = 0.5
    fade_color: str = "#000000"
    slide_direction: Literal["left", "right", "up", "down"] = "left"
    zoom_factor: float = 1.2
    
    def get_transition_params(self) -> Dict:
        """Get parameters for the selected transition style."""
        if self.style == "fade":
            return {"duration": self.duration, "color": self.fade_color}
        elif self.style == "slide":
            return {"duration": self.duration, "direction": self.slide_direction}
        elif self.style == "zoom":
            return {"duration": self.duration, "factor": self.zoom_factor}
        else:
            return {"duration": self.duration}


@dataclass
class EffectOptions:
    """Configuration for image effects."""
    pan_effect: bool = True
    zoom_effect: bool = True
    pan_speed: float = 1.0  # Multiplier for pan speed
    zoom_speed: float = 1.0  # Multiplier for zoom speed
    ken_burns_intensity: float = 0.2  # How much to zoom/pan (0.0-1.0)
    
    # Advanced effects
    vignette: bool = False
    vignette_intensity: float = 0.3
    color_correction: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0


@dataclass
class VideoConfig:
    """Main configuration for video generation."""
    # Basic settings
    fps: int = 24
    aspect_ratio: Tuple[int, int] = (9, 16)  # Width, Height ratio
    base_dimension: int = 720
    
    # Quality settings
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "medium"  # ultrafast, fast, medium, slow, veryslow
    crf: int = 23  # Constant Rate Factor (0-51, lower = better quality)
    threads: int = 4
    
    # Component configurations
    subtitles: SubtitleOptions = field(default_factory=SubtitleOptions)
    transitions: TransitionOptions = field(default_factory=TransitionOptions)
    effects: EffectOptions = field(default_factory=EffectOptions)
    
    # Performance settings
    cache_processed_images: bool = True
    image_quality: int = 95
    max_memory_mb: int = 2048  # Maximum memory usage for image processing
    
    # Output settings
    output_format: str = "mp4"
    include_metadata: bool = True
    
    def get_video_dimensions(self) -> Tuple[int, int]:
        """Calculate even video dimensions from aspect ratio."""
        w, h = self.aspect_ratio
        if w < h:  # Portrait
            width = self.base_dimension
            height = width * h // w
        else:  # Landscape
            height = self.base_dimension
            width = height * w // h
        
        # Ensure even dimensions (required for many codecs)
        return width + width % 2, height + height % 2
    
    def get_export_params(self) -> Dict:
        """Get parameters for video export."""
        return {
            "codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "fps": self.fps,
            "preset": self.preset,
            "threads": self.threads,
            "ffmpeg_params": ["-crf", str(self.crf)]
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if self.fps < 1 or self.fps > 60:
            errors.append(f"FPS must be between 1 and 60, got {self.fps}")
        
        if self.base_dimension < 360 or self.base_dimension > 4096:
            errors.append(f"Base dimension must be between 360 and 4096, got {self.base_dimension}")
        
        if self.crf < 0 or self.crf > 51:
            errors.append(f"CRF must be between 0 and 51, got {self.crf}")
        
        if self.transitions.duration < 0 or self.transitions.duration > 5:
            errors.append(f"Transition duration must be between 0 and 5 seconds")
        
        return errors
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "fps": self.fps,
            "aspect_ratio": self.aspect_ratio,
            "base_dimension": self.base_dimension,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "preset": self.preset,
            "crf": self.crf,
            "threads": self.threads,
            "subtitles": self.subtitles.to_dict(),
            "transitions": asdict(self.transitions),
            "effects": asdict(self.effects),
            "cache_processed_images": self.cache_processed_images,
            "image_quality": self.image_quality,
            "max_memory_mb": self.max_memory_mb,
            "output_format": self.output_format,
            "include_metadata": self.include_metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VideoConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create component objects
        if 'subtitles' in data:
            data['subtitles'] = SubtitleOptions.from_dict(data['subtitles'])
        if 'transitions' in data:
            data['transitions'] = TransitionOptions(**data['transitions'])
        if 'effects' in data:
            data['effects'] = EffectOptions(**data['effects'])
        
        # Convert aspect ratio to tuple
        if 'aspect_ratio' in data and isinstance(data['aspect_ratio'], list):
            data['aspect_ratio'] = tuple(data['aspect_ratio'])
        
        return cls(**data)
    
    @classmethod
    def get_preset(cls, preset_name: str) -> 'VideoConfig':
        """Get a preset configuration."""
        presets = {
            "fast": cls(
                fps=24,
                preset="fast",
                crf=25,
                subtitles=SubtitleOptions(animation="phrase"),
                transitions=TransitionOptions(duration=0.25),
                effects=EffectOptions(ken_burns_intensity=0.1)
            ),
            "quality": cls(
                fps=30,
                preset="slow",
                crf=19,
                base_dimension=1080,
                subtitles=SubtitleOptions(animation="typewriter"),
                transitions=TransitionOptions(duration=0.5),
                effects=EffectOptions(ken_burns_intensity=0.3, color_correction=True)
            ),
            "minimal": cls(
                fps=24,
                preset="medium",
                subtitles=SubtitleOptions(enabled=False),
                transitions=TransitionOptions(style="fade", duration=0.1),
                effects=EffectOptions(pan_effect=False, zoom_effect=False)
            )
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        return presets[preset_name]


# Convenience function for backward compatibility
def get_video_config(**kwargs) -> VideoConfig:
    """Create a video configuration with optional overrides."""
    config = VideoConfig()
    
    # Apply any provided overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config