#!/usr/bin/env python3
"""
batch_example.py - Example script for custom batch video generation

This script demonstrates how to create custom batch runs with specific
configurations and prompt combinations.
"""

import subprocess
import json
import os
from datetime import datetime


def create_custom_batch():
    """Example of creating a custom batch with specific parameters."""
    
    # Define your custom topics
    topics = [
        "Why Monday mornings are actually a government conspiracy",
        "The secret life of office staplers",
        "How to train your houseplant for world domination",
        "Why socks disappear in the dryer - a scientific investigation",
        "The underground society of shopping cart returners"
    ]
    
    # Save topics to a file
    topics_file = "custom_topics.txt"
    with open(topics_file, "w", encoding="utf-8") as f:
        for topic in topics:
            f.write(topic + "\n")
    
    # Define batch configurations
    batch_configs = [
        {
            "name": "humor_flux_batch",
            "args": [
                "--batch",
                "--prompts-file", topics_file,
                "--num-runs", "5",
                "--model", "Flux",
                "--prompt_type", "conspiracy_theory",
                "--voice", "expressive"
            ]
        },
        {
            "name": "educational_sd_batch",
            "args": [
                "--batch",
                "--prompts-file", topics_file,
                "--num-runs", "3",
                "--model", "SD",
                "--prompt_type", "science_explanation",
                "--voice", "trustworthy"
            ]
        }
    ]
    
    # Run each batch configuration
    for config in batch_configs:
        print(f"\n{'='*60}")
        print(f"Running batch: {config['name']}")
        print(f"{'='*60}\n")
        
        cmd = ["python", "main.py"] + config['args']
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ Batch '{config['name']}' completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Batch '{config['name']}' failed with error: {e}")
    
    # Clean up
    if os.path.exists(topics_file):
        os.remove(topics_file)


def create_randomized_batch():
    """Example of creating a fully randomized batch."""
    
    print("\n" + "="*60)
    print("Running fully randomized batch")
    print("="*60 + "\n")
    
    cmd = [
        "python", "main.py",
        "--batch",
        "--num-runs", "10"  # Will use random prompts, models, voices
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Randomized batch completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Randomized batch failed with error: {e}")


def create_overnight_batch():
    """Example of creating a large overnight batch with logging."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"overnight_batch_{timestamp}.log"
    
    print("\n" + "="*60)
    print(f"Starting overnight batch - logging to {log_file}")
    print("This will generate 50 videos with mixed settings")
    print("="*60 + "\n")
    
    # Create varied topics for overnight run
    overnight_topics = [
        # Tech/Internet
        "Why WiFi signals are actually tiny wizards",
        "The truth about why printers smell fear",
        "How to hack reality using HTML",
        
        # Food
        "Why pizza is technically a vegetable salad",
        "The forbidden technique of microwaving ice cream",
        "How ancient civilizations used spoons for time travel",
        
        # Nature
        "Why trees are just really slow giraffes",
        "The secret language of houseplants",
        "How clouds are actually sky sheep",
        
        # Daily Life
        "Why alarm clocks are plotting against humanity",
        "The hidden meaning behind mismatched socks",
        "How elevators are powered by awkward silence",
        
        # Add more topics as needed...
    ]
    
    # Save topics
    topics_file = "overnight_topics.txt"
    with open(topics_file, "w", encoding="utf-8") as f:
        for topic in overnight_topics:
            f.write(topic + "\n")
    
    # Run overnight batch with output redirection
    cmd = [
        "python", "main.py",
        "--batch",
        "--prompts-file", topics_file,
        "--num-runs", "50",
        "--output_dir", f"output/overnight_{timestamp}"
    ]
    
    # Run with output logging
    with open(log_file, "w") as log:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✓ Overnight batch completed successfully")
            else:
                print(f"\n✗ Overnight batch failed with return code: {process.returncode}")
                
        except Exception as e:
            print(f"\n✗ Overnight batch failed with error: {e}")
    
    # Clean up
    if os.path.exists(topics_file):
        os.remove(topics_file)
    
    print(f"\nBatch log saved to: {log_file}")


def main():
    """Main entry point for batch examples."""
    
    print("Video Generator - Batch Examples")
    print("================================\n")
    
    print("Select batch type:")
    print("1. Custom batch with specific settings")
    print("2. Fully randomized batch")
    print("3. Large overnight batch with logging")
    print("4. Run all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        create_custom_batch()
    elif choice == "2":
        create_randomized_batch()
    elif choice == "3":
        response = input("\nThis will generate 50 videos. Continue? (y/n): ").strip().lower()
        if response == 'y':
            create_overnight_batch()
        else:
            print("Overnight batch cancelled.")
    elif choice == "4":
        create_custom_batch()
        create_randomized_batch()
        print("\nSkipping overnight batch in 'all' mode (too many videos)")
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()