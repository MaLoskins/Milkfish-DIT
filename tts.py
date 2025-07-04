# tts.py

import os
import json
import base64
import time
import logging
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
import requests
import sys
from config import Config


class TTSGenerator:
    """Handles text-to-speech generation using ElevenLabs API with progress reporting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("VideoGenerator.TTS")
        
        # Validate API key
        if not self.config.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key not configured")
    
    def generate_audio(
        self,
        paragraph_file: str,
        output_path: str,
        voice_id: str,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Generate audio from text file with retry logic and progress reporting."""
        try:
            # Read the paragraph
            with open(paragraph_file, "r", encoding="utf-8") as f:
                paragraph = f.read().strip()
            
            if not paragraph:
                self.logger.error("Empty paragraph file")
                return False
            
            self.logger.info(f"Generating audio for {os.path.basename(paragraph_file)} with voice {voice_id}")
            self.logger.debug(f"Text length: {len(paragraph)} characters")
            
            # Report initial progress
            if progress_callback:
                progress_callback("uploading")
            
            # Attempt generation with retries
            for attempt in range(max_retries):
                try:
                    success = self._call_elevenlabs_api(paragraph, output_path, voice_id, progress_callback)
                    if success:
                        return True
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        wait_time = self.config.retry_delay * (attempt + 1) * 2
                        self.logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
            
            self.logger.error("All TTS generation attempts failed")
            return False
            
        except Exception as e:
            self.logger.error(f"TTS generation error: {e}")
            return False
    
    def _call_elevenlabs_api(
        self,
        text: str,
        output_path: str,
        voice_id: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Make API call to ElevenLabs with progress reporting."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "xi-api-key": self.config.elevenlabs_api_key
        }
        
        data = {
            "text": text,
            "model_id": self.config.tts_model,
            "voice_settings": {
                "stability": self.config.tts_stability,
                "similarity_boost": self.config.tts_similarity_boost,
                "style": self.config.tts_style,
                "use_speaker_boost": self.config.tts_use_speaker_boost
            }
        }
        
        self.logger.debug("Making API request to ElevenLabs...")
        
        # Report processing progress
        if progress_callback:
            progress_callback("processing")
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Process response
        response_dict = response.json()
        
        # Report download progress
        if progress_callback:
            progress_callback("downloading")
        
        # Decode audio
        if "audio_base64" not in response_dict:
            self.logger.error("No audio data in response")
            return False
        
        audio_bytes = base64.b64decode(response_dict["audio_base64"])
        
        # Save audio file
        audio_directory = os.path.dirname(output_path)
        os.makedirs(audio_directory, exist_ok=True)
        
        self.logger.info(f"Saving audio to: {output_path}")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        # Save timestamps
        timestamps_path = os.path.join(audio_directory, "time_stamps.json")
        timestamps = response_dict.get("alignment")
        
        if timestamps:
            self.logger.info(f"Saving timestamps to: {timestamps_path}")
            with open(timestamps_path, "w") as f:
                json.dump(timestamps, f, indent=4)
            
            # Save additional metadata
            metadata = {
                "text_length": len(text),
                "voice_id": voice_id,
                "model": self.config.tts_model,
                "audio_duration": timestamps.get("duration_seconds", 0),
                "character_count": len(timestamps.get("characters", [])),
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = os.path.join(audio_directory, "tts_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            self.logger.warning("No timestamps in API response")
            # Create minimal timestamps file to prevent downstream errors
            with open(timestamps_path, "w") as f:
                json.dump({
                    "characters": [],
                    "character_start_times_seconds": [],
                    "character_end_times_seconds": []
                }, f)
        
        # Verify files were created
        if not os.path.exists(output_path):
            self.logger.error("Audio file was not created")
            return False
        
        if not os.path.exists(timestamps_path):
            self.logger.error("Timestamps file was not created")
            return False
        
        file_size = os.path.getsize(output_path)
        self.logger.info(f"✓ TTS conversion successful. Audio size: {file_size / 1024:.1f} KB")
        
        return True
    
    def estimate_duration(self, text: str) -> float:
        """Estimate audio duration based on text length."""
        # Rough estimate: 150 words per minute
        words = len(text.split())
        return (words / 150) * 60  # Convert to seconds


# Legacy function for backward compatibility
def eleven_labs_API(paragraph_file: str, output_path: str, voice: str):
    """Legacy function for TTS generation."""
    from config import load_config
    config = load_config()
    
    if not config.elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")
    
    generator = TTSGenerator(config)
    success = generator.generate_audio(paragraph_file, output_path, voice)
    
    if not success:
        raise RuntimeError("TTS generation failed")


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-Speech Generator")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("output_file", help="Output MP3 file")
    parser.add_argument("--voice", default="XB0fDUnXU5powFXDhCwa", help="Voice ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    from config import load_config
    config = load_config()
    
    # Initialize generator
    generator = TTSGenerator(config)
    
    # Progress callback for testing
    def progress_cb(status):
        print(f"TTS Progress: {status}")
    
    # Generate audio
    success = generator.generate_audio(args.input_file, args.output_file, args.voice, progress_callback=progress_cb)
    
    if success:
        print("✓ Audio generation successful")
    else:
        print("✗ Audio generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()