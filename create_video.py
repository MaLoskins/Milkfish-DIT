# create_video.py
"""Enhanced video creation pipeline with improved reliability and performance."""

import os
import json
import re
import logging
import gc
import glob
import platform
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

try:
    from moviepy import *
except ImportError:
    import moviepy.editor as mpy
    globals().update({k: getattr(mpy, k) for k in dir(mpy) if not k.startswith('_')})

from video_config import VideoConfig, SubtitleOptions, TransitionOptions, EffectOptions

class VideoCreationError(Exception):
    """Custom exception for video creation errors."""
    pass

def setup_logger(log_file="video_creation_log.txt"):
    """Configure logger with file and console handlers."""
    logger = logging.getLogger("VideoCreator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in [logging.FileHandler(log_file, 'w', 'utf-8'), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

FONT_MAPPINGS = {
    'Windows': {
        'dirs': [os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
                 os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')],
        'fonts': {"Arial": ["arial.ttf", "Arial.ttf", "ArialMT.ttf"],
                  "Arial-Bold": ["arialbd.ttf", "Arial-Bold.ttf", "Arial Bold.ttf"],
                  "Impact": ["impact.ttf", "Impact.ttf"],
                  "Georgia": ["georgia.ttf", "Georgia.ttf"],
                  "Verdana": ["verdana.ttf", "Verdana.ttf"],
                  "Calibri": ["calibri.ttf", "Calibri.ttf"],
                  "Consolas": ["consolas.ttf", "Consolas.ttf"],
                  "Segoe UI": ["segoeui.ttf", "Segoe UI.ttf"]}},
    'Linux': {
        'dirs': ["/usr/share/fonts/truetype/liberation/", "/usr/share/fonts/truetype/dejavu/",
                 "/usr/share/fonts/truetype/", "/usr/local/share/fonts/", os.path.expanduser("~/.fonts/")],
        'fonts': {"Arial": ["LiberationSans-Regular.ttf", "DejaVuSans.ttf", "FreeSans.ttf"],
                  "Arial-Bold": ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "FreeSans-Bold.ttf"],
                  "Georgia": ["LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "FreeSerif.ttf"],
                  "Verdana": ["DejaVuSans.ttf", "Verdana.ttf"],
                  "Impact": ["LiberationSans-Bold.ttf", "Impact.ttf"],
                  "Calibri": ["Carlito-Regular.ttf", "DejaVuSans.ttf"]}},
    'Darwin': {
        'dirs': ["/System/Library/Fonts/", "/Library/Fonts/", os.path.expanduser("~/Library/Fonts/"),
                 "/System/Library/Fonts/Supplemental/"],
        'fonts': {"Arial": ["Arial.ttf", "ArialMT.ttf", "Helvetica.ttc"],
                  "Arial-Bold": ["Arial Bold.ttf", "Arial-BoldMT.ttf", "Helvetica Bold.ttf"],
                  "Georgia": ["Georgia.ttf"], "Verdana": ["Verdana.ttf"],
                  "Impact": ["Impact.ttf"], "Calibri": ["Calibri.ttf", "Arial.ttf"]}}
}

@lru_cache(maxsize=32)
def find_font_path(font_name):
    """Find font file path on the system with caching."""
    mapping = FONT_MAPPINGS.get(platform.system(), {})
    for d in mapping.get('dirs', []):
        if os.path.exists(d):
            for f in mapping.get('fonts', {}).get(font_name, []):
                p = os.path.join(d, f)
                if os.path.exists(p): return p
            ttf = list(Path(d).glob("*.ttf"))
            if ttf: return str(ttf[0])
    return None

@dataclass
class TimestampData:
    """Parsed timestamp data from ElevenLabs."""
    characters: List[str]
    start_times: np.ndarray
    end_times: np.ndarray
    
    @classmethod
    def from_json(cls, path):
        """Load timestamp data from JSON with error handling."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            c, s, e = data.get('characters', []), data.get('character_start_times_seconds', []), data.get('character_end_times_seconds', [])
            if c and s and e and len(c) == len(s) == len(e):
                return cls(c, np.array(s, dtype=np.float32), np.array(e, dtype=np.float32))
        except Exception as e:
            logging.error(f"Failed to load timestamps from {path}: {e}")
        return None
    
    def validate(self):
        """Validate timestamp data integrity."""
        if not len(self.characters): return False
        for i in range(len(self.start_times) - 1):
            if self.start_times[i] > self.end_times[i]: return False
            if self.end_times[i] - self.start_times[i + 1] > 0.1: return False
        return True

class ImageCache:
    """Memory-efficient image cache with automatic cleanup."""
    def __init__(self, max_memory_mb=2048):
        self._cache, self.max_memory_mb, self.current_memory_mb = {}, max_memory_mb, 0
        self.logger = logging.getLogger("VideoCreator.ImageCache")
    
    def load_image(self, path):
        """Load and cache image with memory management."""
        if path in self._cache: return self._cache[path]
        try:
            clip = ImageClip(path)
            memory_mb = (clip.size[0] * clip.size[1] * 4) / (1024 * 1024)
            if self.current_memory_mb + memory_mb > self.max_memory_mb: self.clear()
            self._cache[path] = clip
            self.current_memory_mb += memory_mb
            self.logger.debug(f"Loaded image: {path} ({memory_mb:.1f} MB)")
            return clip
        except Exception as e:
            self.logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def clear(self):
        """Clear cache and free memory."""
        for clip in self._cache.values():
            try: clip.close()
            except: pass
        self._cache.clear()
        self.current_memory_mb = 0
        gc.collect()
        self.logger.debug("Image cache cleared")

class SubtitleGenerator:
    """Handles subtitle generation with various styles and animations."""
    def __init__(self, video_size, options):
        self.video_size, self.options = video_size, options
        self.logger = logging.getLogger("VideoCreator.Subtitles")
        self._style_cache = {}
    
    def parse_text_segments(self, text, timestamps):
        """Parse text into words, phrases and sentences with timestamps."""
        chars, starts, ends = timestamps.characters, timestamps.start_times, timestamps.end_times
        char_timing = [{'char': c, 'start': s, 'end': e} for c, s, e in zip(chars, starts, ends)]
        words = self._parse_words(chars, starts, ends)
        phrases = self._parse_phrases(words)
        sentences = self._parse_sentences(chars, starts, ends)
        return {'words': words, 'phrases': phrases, 'sentences': sentences, 'characters_with_timing': char_timing}
    
    def _parse_words(self, chars, starts, ends):
        """Parse words with improved boundary detection."""
        words, current_word, word_start_idx = [], [], None
        for i, char in enumerate(chars):
            is_boundary = char.isspace() or char in '.,!?;:()[]{}\'"-–—/\\'
            if not is_boundary:
                if word_start_idx is None: word_start_idx = i
                current_word.append(char)
            elif current_word:
                words.append({'text': ''.join(current_word), 'start': starts[word_start_idx],
                             'end': ends[i-1], 'start_idx': word_start_idx, 'end_idx': i-1})
                current_word, word_start_idx = [], None
        if current_word and word_start_idx is not None:
            words.append({'text': ''.join(current_word), 'start': starts[word_start_idx],
                         'end': ends[-1], 'start_idx': word_start_idx, 'end_idx': len(chars)-1})
        return words
    
    def _parse_phrases(self, words):
        """Parse phrases with improved grouping logic."""
        phrases, current_phrase = [], []
        for i, word in enumerate(words):
            current_phrase.append(word)
            phrase_text = ' '.join(w['text'] for w in current_phrase)
            duration = word['end'] - current_phrase[0]['start'] if current_phrase else 0
            should_end = (len(phrase_text) > self.options.max_chars_per_line or len(current_phrase) >= 5 or
                         duration >= 3.0 or any(word['text'].endswith(p) for p in '.!?,;:') or i == len(words) - 1)
            if should_end and current_phrase:
                phrases.append({'text': phrase_text, 'start': current_phrase[0]['start'],
                               'end': current_phrase[-1]['end'], 'start_idx': current_phrase[0]['start_idx'],
                               'end_idx': current_phrase[-1]['end_idx']})
                current_phrase = []
        return phrases
    
    def _parse_sentences(self, chars, starts, ends):
        """Parse sentences with improved detection."""
        sentences, sent_start = [], 0
        for i, char in enumerate(chars):
            is_sentence_end = char in '.!?' and (i == len(chars) - 1 or
                (i + 1 < len(chars) and (chars[i + 1].isspace() or chars[i + 1] in '"\'')))
            if is_sentence_end:
                sent_text = ''.join(chars[sent_start:i+1]).strip()
                if sent_text and len(sent_text.split()) >= 2:
                    sentences.append({'text': sent_text, 'start': starts[sent_start], 'end': ends[i],
                                    'start_idx': sent_start, 'end_idx': i})
                sent_start = i + 1
                while sent_start < len(chars) and chars[sent_start].isspace(): sent_start += 1
        if sent_start < len(chars):
            sent_text = ''.join(chars[sent_start:]).strip()
            if sent_text and len(sent_text.split()) >= 2:
                sentences.append({'text': sent_text, 'start': starts[sent_start], 'end': ends[-1],
                                'start_idx': sent_start, 'end_idx': len(chars) - 1})
        return sentences
    
    def create_subtitle_clip(self, text, start, duration):
        """Create styled subtitle clip with error handling."""
        try:
            styles = {"modern": {"fonts": ["Arial", "Calibri", "Segoe UI"], "color": "white", "stroke": "black"},
                     "minimal": {"fonts": ["Arial", "Calibri"], "color": "white", "stroke": "#333333"},
                     "bold": {"fonts": ["Arial-Bold", "Impact"], "color": "white", "stroke": "black"},
                     "classic": {"fonts": ["Georgia", "Arial"], "color": "#FFFFCC", "stroke": "black"},
                     "dynamic": {"fonts": ["Verdana", "Arial"], "color": "#FFFFFF", "stroke": "#000000"}}
            style = styles.get(self.options.style, styles["modern"])
            font_path = next((find_font_path(f) for f in style["fonts"] if find_font_path(f)), None)
            font_size = int(self.video_size[1] * self.options.font_size_ratio)
            params = {"text": text, "font_size": font_size, "color": style["color"], "stroke_color": style["stroke"],
                     "stroke_width": max(1, int(self.video_size[1] * self.options.stroke_width_ratio)),
                     "text_align": 'center', "method": 'caption', "size": (int(self.video_size[0] * 0.9), None)}
            if font_path: params["font"] = font_path
            clip = TextClip(**params)
            positions = {"bottom": ("center", int(self.video_size[1] * 0.85)),
                        "top": ("center", int(self.video_size[1] * 0.1)), "middle": ("center", "center")}
            clip = clip.with_position(positions.get(self.options.position, positions["bottom"]))
            clip = clip.with_start(start).with_duration(duration)
            fade = min(self.options.fade_duration, duration / 4)
            if fade > 0: clip = clip.with_effects([vfx.CrossFadeIn(fade), vfx.CrossFadeOut(fade)])
            return clip
        except Exception as e:
            self.logger.warning(f"Failed to create subtitle for '{text[:30]}...': {e}")
            return None
    
    def create_typewriter_subtitle(self, phrase, char_timing):
        """Create typewriter effect subtitle."""
        start_idx, end_idx, start_time = phrase['start_idx'], phrase['end_idx'], phrase['start']
        duration = phrase['end'] - phrase['start']
        def make_frame(t):
            current_time = start_time + t
            visible_text = ''.join(char_timing[i]['char'] for i in range(start_idx, end_idx + 1)
                                  if i < len(char_timing) and current_time >= char_timing[i]['start']).strip()
            if visible_text:
                temp_clip = self.create_subtitle_clip(visible_text, 0, 1)
                if temp_clip:
                    frame = temp_clip.get_frame(0)
                    temp_clip.close()
                    return frame
            return np.zeros((self.video_size[1], self.video_size[0], 4), dtype=np.uint8)
        try:
            return VideoClip(make_frame, duration=duration).with_start(start_time)
        except Exception as e:
            self.logger.warning(f"Failed to create typewriter effect: {e}")
            return None
    
    def add_subtitles(self, video, segments):
        """Add subtitles to video with the configured animation style."""
        subtitle_clips = []
        if self.options.animation == "typewriter":
            for phrase in segments['phrases']:
                clip = self.create_typewriter_subtitle(phrase, segments['characters_with_timing'])
                if clip: subtitle_clips.append(clip)
        else:
            items = segments['words'] if self.options.animation in ["word", "word-by-word"] else segments['phrases']
            for item in items:
                text = item['text']
                if self.options.highlight_keywords:
                    for kw in self.options.keywords:
                        if kw.lower() in text.lower() and self.options.animation == "phrase":
                            text = text.replace(kw, kw.upper())
                clip = self.create_subtitle_clip(text, item['start'], max(0.1, item['end'] - item['start'] + 0.05))
                if clip: subtitle_clips.append(clip)
        if subtitle_clips:
            self.logger.info(f"Added {len(subtitle_clips)} subtitle clips")
            return CompositeVideoClip([video] + subtitle_clips)
        self.logger.warning("No subtitle clips were created")
        return video

class ImageProcessor:
    """Handles image processing and effects."""
    def __init__(self, video_size, config):
        self.video_size, self.config = video_size, config
        self.logger = logging.getLogger("VideoCreator.ImageProcessor")
        self.cache = ImageCache(config.max_memory_mb)
    
    def process_image_for_display(self, path, duration):
        """Process image with appropriate effect."""
        clip = self.cache.load_image(path)
        if not clip: return None
        try:
            is_landscape = clip.size[0] > clip.size[1]
            effect_type = ("ken_burns" if is_landscape and self.config.effects.pan_effect else
                          "zoom" if not is_landscape and self.config.effects.zoom_effect else "resize_center")
            processed_clip = self._apply_image_effect(clip, duration, effect_type)
            if self.config.effects.vignette: processed_clip = self._apply_vignette(processed_clip)
            if self.config.effects.color_correction: processed_clip = self._apply_color_correction(processed_clip)
            return processed_clip
        except Exception as e:
            self.logger.error(f"Failed to process image {path}: {e}")
            return None
    
    def _apply_image_effect(self, clip, duration, effect_type):
        """Apply specific effect to image."""
        vw, vh = self.video_size
        iw, ih = clip.size
        if effect_type == "ken_burns":
            scale = vh / ih * (1 + self.config.effects.ken_burns_intensity)
            clip = clip.resized((int(iw * scale), int(ih * scale)))
            x_range = clip.size[0] - vw
            def position_func(t):
                progress = 0.5 * (1 - np.cos(np.pi * t / duration))
                return (-x_range * progress * self.config.effects.pan_speed, (vh - clip.size[1]) // 2)
            clip = clip.with_position(position_func)
        elif effect_type == "zoom":
            base_scale = vh / ih
            def zoom_func(t):
                return base_scale * (1.0 + self.config.effects.ken_burns_intensity * t / duration * self.config.effects.zoom_speed)
            clip = clip.resized(zoom_func).with_position('center')
        else:
            scale = min(vw / iw, vh / ih)
            clip = clip.resized((int(iw * scale), int(ih * scale))).with_position('center')
        return CompositeVideoClip([clip], size=self.video_size).with_duration(duration)
    
    def _apply_vignette(self, clip):
        """Apply vignette effect to clip."""
        def vignette_filter(frame):
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
            vignette = np.clip(1 - dist / np.sqrt((w/2)**2 + (h/2)**2) * self.config.effects.vignette_intensity, 0, 1)
            return (frame * vignette[:, :, np.newaxis]).astype(np.uint8)
        return clip.image_transform(vignette_filter)
    
    def _apply_color_correction(self, clip):
        """Apply color correction to clip."""
        def color_correct(frame):
            img = frame.astype(np.float32) / 255.0
            img = img * self.config.effects.brightness
            img = (img - 0.5) * self.config.effects.contrast + 0.5
            if self.config.effects.saturation != 1.0:
                gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
                img[..., :3] = gray + (img[..., :3] - gray) * self.config.effects.saturation
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        return clip.image_transform(color_correct)
    
    def cleanup(self):
        """Clean up resources."""
        self.cache.clear()

class VideoCreator:
    """Main video creation orchestrator."""
    def __init__(self, config, logger):
        self.config, self.logger = config, logger
        self.image_processor = self.subtitle_generator = None
    
    def create_video(self, audio_path, images_dir, output_path, paragraph_file, time_stamps_file, **kwargs):
        """Create video from components with error handling."""
        for k, v in kwargs.items():
            for obj in [self.config, self.config.subtitles, self.config.transitions, self.config.effects]:
                if hasattr(obj, k): setattr(obj, k, v)
        errors = self.config.validate()
        if errors: raise VideoCreationError(f"Invalid configuration: {', '.join(errors)}")
        video_size = self.config.get_video_dimensions()
        self.image_processor = ImageProcessor(video_size, self.config)
        self.subtitle_generator = SubtitleGenerator(video_size, self.config.subtitles)
        audio_clip = None
        try:
            self.logger.info(f"Loading audio from {audio_path}")
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            self.logger.info(f"Audio duration: {audio_duration:.2f}s")
            with open(paragraph_file, 'r', encoding='utf-8') as f: paragraph_text = f.read().strip()
            timestamps = TimestampData.from_json(time_stamps_file)
            if timestamps and timestamps.validate():
                self.logger.info(f"Loaded {len(timestamps.characters)} character timestamps")
            else:
                self.logger.warning("No timestamps available, subtitles will be disabled")
                self.config.subtitles.enabled = False
                timestamps = None
            image_files = sorted([f for ext in ["png", "jpg", "jpeg", "webp"] 
                                 for f in glob.glob(os.path.join(images_dir, f"*.{ext}"))],
                                key=lambda f: int(m.group(1)) if (m := re.search(r'(\d+)', os.path.basename(f))) else 0)
            if not image_files: raise VideoCreationError("No images found in directory")
            self.logger.info(f"Found {len(image_files)} images")
            video = self._create_image_sequence(image_files, audio_duration)
            video = video.with_audio(audio_clip)
            if self.config.subtitles.enabled and timestamps:
                segments = self.subtitle_generator.parse_text_segments(paragraph_text, timestamps)
                video = self.subtitle_generator.add_subtitles(video, segments)
            self._export_video(video, output_path)
            self.logger.info(f"Video creation completed: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Video creation failed: {e}")
            raise VideoCreationError(f"Failed to create video: {e}")
        finally:
            if audio_clip: audio_clip.close()
            if self.image_processor: self.image_processor.cleanup()
            gc.collect()
    
    def _create_image_sequence(self, image_files, audio_duration):
        """Create video sequence from images with transitions."""
        num_images = len(image_files)
        if num_images == 1:
            clip = self.image_processor.process_image_for_display(image_files[0], audio_duration)
            if not clip: raise VideoCreationError("Failed to process single image")
            return clip
        transition_duration = self.config.transitions.duration
        base_duration = (audio_duration - (num_images - 1) * transition_duration) / num_images
        clips = []
        for i, path in enumerate(image_files):
            duration = audio_duration if i == num_images - 1 else base_duration + transition_duration
            clip = self.image_processor.process_image_for_display(path, duration)
            if clip:
                clips.append(clip)
                self.logger.debug(f"Image {i}: duration={duration:.2f}s")
            else:
                self.logger.warning(f"Failed to process image {i}")
        if not clips: raise VideoCreationError("No images could be processed")
        if len(clips) > 1 and transition_duration > 0:
            self.logger.info(f"Applying {self.config.transitions.style} transitions ({transition_duration}s)")
            background = self.image_processor.process_image_for_display(image_files[-1], audio_duration)
            concatenated = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
            final_video = CompositeVideoClip([background, concatenated])
        else:
            final_video = concatenate_videoclips(clips, method="compose")
        return final_video.with_duration(audio_duration)
    
    def _export_video(self, video, output_path):
        """Export video with configured settings."""
        self.logger.info(f"Exporting video to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_params = self.config.get_export_params()
        export_params['logger'] = None
        video.write_videofile(output_path, **export_params)
        if not os.path.exists(output_path): raise VideoCreationError("Output video was not created")
        self.logger.info(f"Video exported successfully: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")

def create_video(audio_path, images_dir, output_path, paragraph_file, time_stamps_file, fps=24,
                aspect_ratio=(9, 16), transition_duration=0.5, pan_effect=True, zoom_effect=True,
                log_file_path="video_creation_log.txt", subtitles=True, subtitle_style="modern",
                highlight_keywords=True, subtitle_animation="phrase", subtitle_position="bottom",
                transition_style="fade"):
    """Create video from images and audio with subtitles."""
    logger = setup_logger(log_file_path)
    config = VideoConfig(fps=fps, aspect_ratio=aspect_ratio,
                        transitions=TransitionOptions(style=transition_style, duration=transition_duration),
                        effects=EffectOptions(pan_effect=pan_effect, zoom_effect=zoom_effect),
                        subtitles=SubtitleOptions(enabled=subtitles, style=subtitle_style,
                                                animation=subtitle_animation, position=subtitle_position,
                                                highlight_keywords=highlight_keywords))
    creator = VideoCreator(config, logger)
    return creator.create_video(audio_path, images_dir, output_path, paragraph_file, time_stamps_file)