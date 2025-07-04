# image_generation.py

import os
import sys
import random
import argparse
import time
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import comfy_api
from config import Config, load_config


@dataclass
class ImageGenerationResult:
    """Result of image generation attempt."""
    success: bool
    index: int
    prompt: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    seed: Optional[int] = None


class ImageGenerator:
    """Handles image generation using ComfyUI API with progress reporting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("VideoGenerator.ImageGen")
        self.positive_prompts: List[str] = []
        self.negative_prompts: List[str] = []
        self.model = "Flux"
        self.seeds: List[int] = []
    
    def load_prompts(self, descriptions_file: str) -> None:
        """Load image descriptions from file."""
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                self.positive_prompts = [line.strip() for line in f if line.strip()]
            
            # Default negative prompt
            negative_string = (
                'low quality, jpeg artifacts, blurry, watermark, '
                'black and white, grainy texture'
            )
            self.negative_prompts = [negative_string for _ in self.positive_prompts]
            
            # Generate random seeds
            self.seeds = [random.randint(1, 10000000) for _ in self.positive_prompts]
            
            self.logger.info(f"Loaded {len(self.positive_prompts)} image descriptions")
            
        except FileNotFoundError:
            self.logger.error(f"Could not find {descriptions_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            raise
    
    def set_model(self, model: str) -> None:
        """Set the generation model."""
        if model not in ["Flux", "SD"]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.logger.info(f"Using model: {self.model}")
    
    def generate_images(self, output_dir: str) -> List[ImageGenerationResult]:
        """Generate all images with progress reporting."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        total_images = len(self.positive_prompts)
        self.logger.info(f"Generating {total_images} images using {self.model} model...")
        
        # Report initial progress
        self._report_progress(0, f"Starting {self.model} image generation")
        
        for i in range(total_images):
            # Report progress for each image
            progress_percent = int((i / total_images) * 100)
            self._report_progress(progress_percent, f"Generating image {i+1}/{total_images}")
            
            result = self._generate_single_image(i, output_dir)
            results.append(result)
            
            if result.success:
                self.logger.info(f"✓ Image {i+1}/{total_images} generated successfully")
            else:
                self.logger.error(f"✗ Image {i+1}/{total_images} failed: {result.error}")
            
            # Brief pause between generations to avoid overload
            if i < total_images - 1:
                time.sleep(0.5)
        
        # Final progress report
        self._report_progress(100, f"Completed generating {total_images} images")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Generation complete: {successful}/{len(results)} successful")
        
        return results
    
    def _report_progress(self, percent: int, stage: str):
        """Report progress to stdout for parent process to capture."""
        # Use a special format that the parent process can parse
        print(f"PROGRESS:{percent}|{stage}", flush=True)
    
    def _generate_single_image(self, index: int, output_dir: str) -> ImageGenerationResult:
        """Generate a single image with error handling."""
        output_path = os.path.join(output_dir, f"image_{index}.png")
        
        try:
            # Get dimensions
            width, height = self._get_random_orientation()
            
            # Prepare prompts
            if self.model == "Flux":
                # Flux combines prompts
                full_prompt = self.positive_prompts[index]
                if self.negative_prompts[index]:
                    full_prompt = f"{full_prompt}, NOT {self.negative_prompts[index]}"
                
                self.logger.debug(f"Generating Flux image {index+1}")
                self.logger.debug(f"Prompt: {full_prompt[:100]}...")
                self.logger.debug(f"Dimensions: {width}x{height}")
                
                success = comfy_api.generate_image_flux(
                    prompt_text=full_prompt,
                    output_path=output_path,
                    seed=self.seeds[index],
                    width=width,
                    height=height
                )
                
            else:  # SD
                self.logger.debug(f"Generating SD image {index+1}")
                self.logger.debug(f"Positive: {self.positive_prompts[index][:100]}...")
                self.logger.debug(f"Negative: {self.negative_prompts[index][:50]}...")
                self.logger.debug(f"Dimensions: {width}x{height}")
                
                success = comfy_api.generate_image_sd(
                    prompt_text=self.positive_prompts[index],
                    output_path=output_path,
                    negative_prompt=self.negative_prompts[index],
                    seed=self.seeds[index],
                    width=width,
                    height=height
                )
            
            if success and os.path.exists(output_path):
                return ImageGenerationResult(
                    success=True,
                    index=index,
                    prompt=self.positive_prompts[index],
                    output_path=output_path,
                    dimensions=(width, height),
                    seed=self.seeds[index]
                )
            else:
                return ImageGenerationResult(
                    success=False,
                    index=index,
                    prompt=self.positive_prompts[index],
                    error="Generation failed or file not created"
                )
                
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                index=index,
                prompt=self.positive_prompts[index],
                error=str(e)
            )
    
    def _get_random_orientation(self) -> Tuple[int, int]:
        """Get random image dimensions based on orientation."""
        orientation = random.choice(['horizontal', 'vertical'])
        
        if self.model == "Flux":
            if orientation == 'horizontal':
                return (1024, 920)
            else:
                return (720, 1216)
        else:  # SD
            if orientation == 'horizontal':
                return (768, 512)
            else:
                return (512, 768)
    
    def check_comfyui_connection(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            import urllib.request
            response = urllib.request.urlopen(f"{self.config.comfyui_url}/", timeout=5)
            return response.status == 200
        except:
            return False
    
    def save_generation_metadata(self, output_dir: str, results: List[ImageGenerationResult]) -> None:
        """Save metadata about the generation process."""
        metadata = {
            "model": self.model,
            "total_images": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "images": []
        }
        
        for result in results:
            image_data = {
                "index": result.index,
                "success": result.success,
                "prompt": result.prompt[:200] + "..." if len(result.prompt) > 200 else result.prompt
            }
            
            if result.success:
                image_data.update({
                    "output_path": os.path.basename(result.output_path),
                    "dimensions": result.dimensions,
                    "seed": result.seed
                })
            else:
                image_data["error"] = result.error
            
            metadata["images"].append(image_data)
        
        metadata_path = os.path.join(output_dir, "image_generation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved generation metadata to {metadata_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Image Generator using ComfyUI")
    parser.add_argument(
        '--model',
        default='Flux',
        choices=['Flux', 'SD'],
        help='Model to use for image generation'
    )
    parser.add_argument(
        '--run-dir',
        default='output',
        help='The base directory for the current run'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("VideoGenerator.ImageGen")
    
    # Load configuration
    config = load_config()
    
    # Initialize generator
    generator = ImageGenerator(config)
    
    # Check ComfyUI connection
    logger.info("Checking ComfyUI connection...")
    if not generator.check_comfyui_connection():
        logger.error(f"ComfyUI is not running at {config.comfyui_url}")
        logger.error("Please start ComfyUI before running image generation")
        sys.exit(1)
    logger.info("✓ ComfyUI is running")
    
    # Define paths
    base_run_dir = args.run_dir
    texts_dir = os.path.join(base_run_dir, "texts")
    images_dir = os.path.join(base_run_dir, "images")
    descriptions_path = os.path.join(texts_dir, "image_descriptions.txt")
    
    # Load prompts
    try:
        generator.load_prompts(descriptions_path)
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        sys.exit(1)
    
    # Set model
    generator.set_model(args.model)
    
    # Generate images
    logger.info(f"Saving images to: {images_dir}")
    
    try:
        results = generator.generate_images(images_dir)
        
        # Save metadata
        generator.save_generation_metadata(images_dir, results)
        
        # Check if any images were generated
        successful = sum(1 for r in results if r.success)
        if successful == 0:
            logger.error("No images were successfully generated")
            sys.exit(1)
        
        logger.info("\nImage generation completed!")
        
    except Exception as e:
        logger.error(f"Fatal error during image generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()