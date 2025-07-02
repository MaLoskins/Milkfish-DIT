# text_generation.py

import os
import sys
import re
import time
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from langchain_ollama import OllamaLLM
from prompts import prompts
from config import Config


@dataclass
class TextGenerationResult:
    """Result of text generation."""
    paragraph: str
    descriptions: List[str]
    word_count: int
    sentence_count: int


class TextGenerator:
    """Handles text generation using Ollama LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("VideoGenerator.TextGen")
        self.llm = None
        self._init_llm()
    
    def _init_llm(self) -> None:
        """Initialize the language model."""
        try:
            self.llm = OllamaLLM(
                model=self.config.ollama_model,
                base_url=self.config.ollama_url
            )
            self.logger.info(f"Initialized LLM with model: {self.config.ollama_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def generate_content(self, prompt_type: str, topic: str, max_retries: int = 3) -> str:
        """Generate content based on prompt type and topic with retry logic."""
        prompt_templates = prompts.get(prompt_type)
        if not prompt_templates:
            raise ValueError(f"Invalid prompt type: '{prompt_type}'")
        
        content_prompt = prompt_templates["content"].format(topic=topic)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Generating content (attempt {attempt + 1}/{max_retries})")
                
                # Invoke the language model
                response = self.llm.invoke(content_prompt)
                
                # Clean and validate response
                paragraph = self._clean_paragraph(response)
                
                if self._validate_paragraph(paragraph):
                    self.logger.info(f"Successfully generated paragraph ({len(paragraph)} chars)")
                    return paragraph
                else:
                    self.logger.warning(f"Invalid paragraph generated on attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
        
        raise ValueError("Failed to generate valid content after all retries")
    
    def extract_image_descriptions(
        self,
        prompt_type: str,
        paragraph: str,
        topic: str,
        max_descriptions: int = 20
    ) -> List[str]:
        """Extract image descriptions from the paragraph."""
        try:
            # Get sentences from paragraph
            sentences = self._split_into_sentences(paragraph)
            self.logger.info(f"Processing {len(sentences)} sentences for image descriptions")
            
            all_descriptions = []
            previous_descriptions = set()
            
            for sentence in sentences:
                # Determine image count based on sentence length
                image_count = self._calculate_image_count(sentence)
                self.logger.debug(f"Sentence: '{sentence[:50]}...' - Images: {image_count}")
                
                # Generate descriptions for this sentence
                for _ in range(image_count):
                    if len(all_descriptions) >= max_descriptions:
                        break
                    
                    description = self._generate_single_description(
                        prompt_type, sentence, paragraph, topic
                    )
                    
                    if description and description not in previous_descriptions:
                        previous_descriptions.add(description)
                        all_descriptions.append(description)
                
                if len(all_descriptions) >= max_descriptions:
                    break
            
            self.logger.info(f"Generated {len(all_descriptions)} unique image descriptions")
            return all_descriptions
            
        except Exception as e:
            self.logger.error(f"Error extracting image descriptions: {e}")
            raise
    
    def save_texts(
        self,
        prompt_type: str,
        paragraph: str,
        descriptions: List[str],
        output_dir: str = "output/texts"
    ) -> Dict[str, str]:
        """Save generated texts to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save paragraph
            paragraph_path = os.path.join(output_dir, "paragraph.txt")
            with open(paragraph_path, "w", encoding="utf-8") as f:
                f.write(paragraph)
            
            # Save image descriptions
            descriptions_path = os.path.join(output_dir, "image_descriptions.txt")
            with open(descriptions_path, "w", encoding="utf-8") as f:
                for desc in descriptions:
                    f.write(desc + "\n")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, "text_metadata.json")
            metadata = {
                "prompt_type": prompt_type,
                "paragraph_length": len(paragraph),
                "word_count": len(paragraph.split()),
                "sentence_count": len(self._split_into_sentences(paragraph)),
                "description_count": len(descriptions)
            }
            
            import json
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved texts to {output_dir}")
            
            return {
                "paragraph": paragraph_path,
                "descriptions": descriptions_path,
                "metadata": metadata_path
            }
            
        except Exception as e:
            self.logger.error(f"Error saving text files: {e}")
            raise
    
    def _clean_paragraph(self, response: str) -> str:
        """Clean and format the generated paragraph."""
        # Split into lines
        lines = response.split('\n')
        
        # Remove lines that end with a colon (often headers)
        cleaned_lines = [line for line in lines if not line.strip().endswith(':')]
        
        # Join lines into a single paragraph
        paragraph = ' '.join(cleaned_lines)
        
        # Remove extra whitespace
        paragraph = ' '.join(paragraph.split())
        
        # Remove quotes if the entire paragraph is quoted
        if paragraph.startswith('"') and paragraph.endswith('"'):
            paragraph = paragraph[1:-1]
        
        return paragraph.strip()
    
    def _validate_paragraph(self, paragraph: str) -> bool:
        """Validate that the paragraph is acceptable."""
        # Check minimum length
        if len(paragraph) < 50:
            return False
        
        # Check for refusal messages
        refusal_phrases = [
            "I can't fulfill this request",
            "I cannot generate",
            "I'm unable to",
            "I apologize",
            "I won't be able to"
        ]
        
        paragraph_lower = paragraph.lower()
        for phrase in refusal_phrases:
            if phrase.lower() in paragraph_lower:
                return False
        
        return True
    
    def _split_into_sentences(self, paragraph: str) -> List[str]:
        """Split paragraph into sentences."""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Remove leading numbering
            sentence = re.sub(r'^\d+\.\s*', '', sentence)
            
            # Skip very short sentences
            if len(sentence.split()) >= 2:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_image_count(self, sentence: str) -> int:
        """Calculate number of images for a sentence based on length."""
        word_count = len(sentence.split())
        
        if word_count <= 18:
            return 1
        elif word_count <= 28:
            return 2
        else:
            return 3
    
    def _generate_single_description(
        self,
        prompt_type: str,
        sentence: str,
        paragraph: str,
        topic: str
    ) -> Optional[str]:
        """Generate a single image description."""
        image_prompt_template = prompts.get(prompt_type, {}).get("image_description")
        if not image_prompt_template:
            self.logger.error(f"No image description template for prompt type: {prompt_type}")
            return None
        
        image_prompt = image_prompt_template.format(
            sentence=sentence,
            paragraph=paragraph,
            topic=topic
        )
        
        try:
            # Generate description
            response = self.llm.invoke(image_prompt)
            
            # Clean response
            description = response.replace('\n', ' ').replace('"', '').strip()
            
            # Ensure proper format
            if not description.startswith("An image of"):
                description = f"An image of {description}"
            
            # Add quality modifiers
            description = f"8k HD, realistic image, iphone photo, photo realistic, {description}"
            
            return description
            
        except Exception as e:
            self.logger.warning(f"Error generating image description: {e}")
            return None


# Standalone functions for backward compatibility
def generate_content_local(prompt_type: str, topic: str) -> str:
    """Legacy function for generating content."""
    from config import load_config
    config = load_config()
    generator = TextGenerator(config)
    return generator.generate_content(prompt_type, topic)


def extract_image_descriptions_local(prompt_type: str, paragraph: str, topic: str) -> List[str]:
    """Legacy function for extracting image descriptions."""
    from config import load_config
    config = load_config()
    generator = TextGenerator(config)
    return generator.extract_image_descriptions(prompt_type, paragraph, topic)


def save_texts(prompt_type: str, paragraph: str, descriptions: List[str], output_dir: str = "output/texts"):
    """Legacy function for saving texts."""
    from config import load_config
    config = load_config()
    generator = TextGenerator(config)
    return generator.save_texts(prompt_type, paragraph, descriptions, output_dir)


def main():
    """Main execution for testing."""
    if len(sys.argv) < 3:
        print("Usage: python text_generation.py <topic> <prompt_type>")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    from config import load_config
    config = load_config()
    
    # Extract arguments
    topic = sys.argv[1]
    prompt_type = sys.argv[2]
    
    # Initialize generator
    generator = TextGenerator(config)
    
    # Generate content
    paragraph = generator.generate_content(prompt_type, topic)
    print("Generated Paragraph:")
    print(paragraph)
    
    # Extract descriptions
    descriptions = generator.extract_image_descriptions(prompt_type, paragraph, topic)
    print("\nExtracted Image Descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc[:100]}...")
    
    # Save texts
    generator.save_texts(prompt_type, paragraph, descriptions)
    print("\nTexts saved to output/texts/")


if __name__ == "__main__":
    main()