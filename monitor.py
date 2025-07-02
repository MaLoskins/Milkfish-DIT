#!/usr/bin/env python3
"""
monitor.py - Monitor and analyze batch video generation runs

This utility helps track progress, identify failures, and analyze
the results of batch video generation runs.
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class BatchMonitor:
    """Monitor batch video generation progress and results."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.batch_dirs = self._find_batch_directories()
    
    def _find_batch_directories(self) -> List[Path]:
        """Find all batch directories in the output folder."""
        batch_pattern = os.path.join(self.output_dir, "batch_*")
        return sorted([Path(d) for d in glob.glob(batch_pattern) if os.path.isdir(d)])
    
    def get_batch_summary(self, batch_dir: Path) -> Optional[Dict]:
        """Load batch summary from a batch directory."""
        summary_file = batch_dir / "batch_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def analyze_batch(self, batch_dir: Path) -> Dict:
        """Analyze a single batch directory."""
        summary = self.get_batch_summary(batch_dir)
        if not summary:
            return {"error": "No summary file found"}
        
        # Count successes and failures
        successful = sum(1 for run in summary["runs"] if run["success"])
        failed = sum(1 for run in summary["runs"] if not run["success"])
        
        # Get failure reasons
        failures = []
        for run in summary["runs"]:
            if not run["success"] and run.get("error"):
                failures.append({
                    "run_id": run["run_id"],
                    "topic": run["topic"],
                    "error": run["error"]
                })
        
        # Calculate timings
        durations = [run.get("duration", 0) for run in summary["runs"] if run.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Get model and voice distribution
        models = {}
        voices = {}
        prompt_types = {}
        
        for run in summary["runs"]:
            model = run.get("model", "Unknown")
            models[model] = models.get(model, 0) + 1
            
            voice = run.get("voice", "Unknown")
            voices[voice] = voices.get(voice, 0) + 1
            
            prompt_type = run.get("prompt_type", "Unknown")
            prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
        
        return {
            "batch_dir": str(batch_dir),
            "timestamp": summary.get("batch_timestamp", "Unknown"),
            "total_runs": summary.get("total_runs", 0),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / summary["total_runs"] * 100) if summary["total_runs"] > 0 else 0,
            "avg_duration": avg_duration,
            "total_duration": sum(durations),
            "models": models,
            "voices": voices,
            "prompt_types": prompt_types,
            "failures": failures
        }
    
    def get_latest_batch(self) -> Optional[Path]:
        """Get the most recent batch directory."""
        if self.batch_dirs:
            return self.batch_dirs[-1]
        return None
    
    def monitor_active_batch(self, batch_dir: Path) -> None:
        """Monitor an active batch run in real-time."""
        import time
        
        print(f"Monitoring batch: {batch_dir.name}")
        print("Press Ctrl+C to stop monitoring\n")
        
        last_update = None
        
        try:
            while True:
                summary = self.get_batch_summary(batch_dir)
                if summary:
                    completed = len(summary.get("runs", []))
                    total = summary.get("total_runs", 0)
                    
                    if last_update != completed:
                        last_update = completed
                        successful = sum(1 for run in summary["runs"] if run["success"])
                        failed = completed - successful
                        
                        print(f"\rProgress: {completed}/{total} "
                              f"(✓ {successful} | ✗ {failed})", end='', flush=True)
                        
                        if completed >= total:
                            print("\n\nBatch completed!")
                            break
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def generate_report(self, batch_dir: Optional[Path] = None) -> str:
        """Generate a detailed report for a batch or all batches."""
        if batch_dir:
            batches = [batch_dir]
        else:
            batches = self.batch_dirs
        
        if not batches:
            return "No batch directories found."
        
        report = []
        report.append("="*60)
        report.append("VIDEO GENERATION BATCH REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_videos = 0
        total_successful = 0
        total_failed = 0
        all_failures = []
        
        for batch in batches:
            analysis = self.analyze_batch(batch)
            
            if "error" in analysis:
                report.append(f"\n{batch.name}: ERROR - {analysis['error']}")
                continue
            
            report.append(f"\nBatch: {batch.name}")
            report.append("-" * 40)
            report.append(f"Timestamp: {analysis['timestamp']}")
            report.append(f"Total Runs: {analysis['total_runs']}")
            report.append(f"Successful: {analysis['successful']} ({analysis['success_rate']:.1f}%)")
            report.append(f"Failed: {analysis['failed']}")
            report.append(f"Avg Duration: {analysis['avg_duration']:.1f}s per video")
            report.append(f"Total Duration: {analysis['total_duration']:.1f}s ({analysis['total_duration']/60:.1f} minutes)")
            
            # Model distribution
            report.append("\nModels Used:")
            for model, count in analysis['models'].items():
                report.append(f"  - {model}: {count}")
            
            # Voice distribution
            report.append("\nVoices Used:")
            for voice, count in analysis['voices'].items():
                report.append(f"  - {voice}: {count}")
            
            # Failures
            if analysis['failures']:
                report.append("\nFailures:")
                for failure in analysis['failures'][:5]:  # Show first 5
                    report.append(f"  - Run {failure['run_id']}: {failure['topic'][:50]}...")
                    report.append(f"    Error: {failure['error'][:100]}...")
                if len(analysis['failures']) > 5:
                    report.append(f"  ... and {len(analysis['failures']) - 5} more failures")
            
            # Update totals
            total_videos += analysis['total_runs']
            total_successful += analysis['successful']
            total_failed += analysis['failed']
            all_failures.extend(analysis['failures'])
        
        # Overall summary
        if len(batches) > 1:
            report.append("\n" + "="*60)
            report.append("OVERALL SUMMARY")
            report.append("="*60)
            report.append(f"Total Batches: {len(batches)}")
            report.append(f"Total Videos: {total_videos}")
            report.append(f"Total Successful: {total_successful} ({total_successful/total_videos*100:.1f}%)")
            report.append(f"Total Failed: {total_failed}")
            
            # Common failure reasons
            if all_failures:
                error_counts = {}
                for failure in all_failures:
                    error_type = failure['error'].split(':')[0]
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                report.append("\nCommon Failure Reasons:")
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"  - {error}: {count} occurrences")
        
        return "\n".join(report)
    
    def find_incomplete_videos(self) -> List[Dict]:
        """Find videos that started but didn't complete."""
        incomplete = []
        
        for batch_dir in self.batch_dirs:
            # Check each run directory in the batch
            run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            
            for run_dir in run_dirs:
                # Check if final video exists
                video_path = run_dir / "video" / "final_video.mp4"
                if not video_path.exists():
                    # Check what stage it reached
                    stages = {
                        "text": (run_dir / "texts" / "paragraph.txt").exists(),
                        "images": bool(list((run_dir / "images").glob("*.png"))),
                        "audio": (run_dir / "audio" / "paragraph.mp3").exists(),
                        "video": video_path.exists()
                    }
                    
                    incomplete.append({
                        "batch": batch_dir.name,
                        "run_dir": run_dir.name,
                        "stages_completed": stages,
                        "last_stage": next((k for k, v in reversed(list(stages.items())) if v), "none")
                    })
        
        return incomplete


def main():
    """Main entry point for the monitor utility."""
    parser = argparse.ArgumentParser(description="Monitor batch video generation")
    parser.add_argument(
        "command",
        choices=["status", "report", "watch", "incomplete"],
        help="Command to execute"
    )
    parser.add_argument(
        "--batch",
        help="Specific batch directory name (e.g., batch_20240101_120000)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory to monitor (default: output)"
    )
    parser.add_argument(
        "--save-report",
        help="Save report to specified file"
    )
    
    args = parser.parse_args()
    
    monitor = BatchMonitor(args.output_dir)
    
    if args.command == "status":
        # Show status of all batches
        if not monitor.batch_dirs:
            print("No batch directories found.")
            return
        
        print("Found batches:")
        for batch_dir in monitor.batch_dirs:
            summary = monitor.get_batch_summary(batch_dir)
            if summary:
                runs = len(summary.get("runs", []))
                total = summary.get("total_runs", 0)
                status = "Complete" if runs >= total else f"In Progress ({runs}/{total})"
                print(f"  - {batch_dir.name}: {status}")
            else:
                print(f"  - {batch_dir.name}: No summary found")
    
    elif args.command == "report":
        # Generate report
        batch_dir = None
        if args.batch:
            batch_path = Path(args.output_dir) / args.batch
            if batch_path.exists():
                batch_dir = batch_path
            else:
                print(f"Batch directory not found: {args.batch}")
                return
        
        report = monitor.generate_report(batch_dir)
        print(report)
        
        if args.save_report:
            with open(args.save_report, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.save_report}")
    
    elif args.command == "watch":
        # Watch active batch
        if args.batch:
            batch_path = Path(args.output_dir) / args.batch
        else:
            batch_path = monitor.get_latest_batch()
        
        if batch_path and batch_path.exists():
            monitor.monitor_active_batch(batch_path)
        else:
            print("No active batch found to monitor.")
    
    elif args.command == "incomplete":
        # Find incomplete videos
        incomplete = monitor.find_incomplete_videos()
        if incomplete:
            print(f"Found {len(incomplete)} incomplete videos:\n")
            for item in incomplete:
                print(f"Batch: {item['batch']}")
                print(f"Run: {item['run_dir']}")
                print(f"Last completed stage: {item['last_stage']}")
                print(f"Stages: {item['stages_completed']}")
                print("-" * 40)
        else:
            print("No incomplete videos found.")


if __name__ == "__main__":
    main()