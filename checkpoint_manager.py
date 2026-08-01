#!/usr/bin/env python3
"""
Checkpoint Manager for Reliquary Miner

Monitors the /state endpoint and downloads updated model checkpoints.
Automatically removes old checkpoints and maintains the model at a specific path.
"""

import os
import shutil
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

from huggingface_hub import snapshot_download
import requests

# Configuration
STATE_ENDPOINT = os.getenv("RELIQUARY_STATE_ENDPOINT", "http://localhost:8000/state")
MODEL_DIR = Path.home() / "reliquary-miner" / "model"
STATE_CACHE_FILE = Path.home() / "reliquary-miner" / ".checkpoint_state"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_cached_checkpoint_state() -> Optional[Dict[str, Any]]:
    """Load the last known checkpoint state from cache."""
    if STATE_CACHE_FILE.exists():
        try:
            with open(STATE_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached state: {e}")
    return None


def save_checkpoint_state(checkpoint_info: Dict[str, Any]) -> None:
    """Save the current checkpoint state to cache."""
    STATE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_CACHE_FILE, "w") as f:
            json.dump(checkpoint_info, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint state: {e}")


def fetch_state() -> Optional[Dict[str, Any]]:
    """
    Fetch the /state endpoint.
    
    Returns:
        Dict with keys: state, window_n, checkpoint_n, checkpoint_repo_id, checkpoint_revision, cooldown_prompts
        None if request fails.
    """
    try:
        response = requests.get(STATE_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch /state: {e}")
        return None


def backup_old_model() -> None:
    """Backup the old model directory before downloading a new one."""
    if MODEL_DIR.exists():
        backup_dir = MODEL_DIR.parent / f"{MODEL_DIR.name}.backup.{int(time.time())}"
        try:
            shutil.move(str(MODEL_DIR), str(backup_dir))
            logger.info(f"Backed up old model to {backup_dir}")
            
            # Optional: remove backup after a delay or keep only last N backups
            # For now, we keep the backup for safety
        except Exception as e:
            logger.error(f"Failed to backup old model: {e}")


def download_checkpoint(repo_id: str, revision: str) -> bool:
    """
    Download the checkpoint using hf.snapshot_download() to a specific local directory.
    
    Args:
        repo_id: The Hugging Face repository ID (e.g., "owner/model-name")
        revision: The specific revision/commit hash
        
    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        logger.info(f"Starting download: repo_id={repo_id}, revision={revision}")
        
        # Ensure the parent directory exists
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        
        # If old model exists, backup it first
        if MODEL_DIR.exists():
            backup_old_model()
        
        # Download to local_dir - this maintains the original file structure
        # and creates a .cache/huggingface/ folder for metadata
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(MODEL_DIR),
            # Optional: filter patterns to exclude large files if needed
            # allow_patterns=["*.json", "*.safetensors", "*.txt", ".gitattributes"],
        )
        
        logger.info(f"Successfully downloaded checkpoint to {downloaded_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        # Attempt to restore from backup if download failed
        restore_from_backup()
        return False


def restore_from_backup() -> None:
    """Restore the most recent backup if download fails."""
    try:
        backup_files = sorted(MODEL_DIR.parent.glob(f"{MODEL_DIR.name}.backup.*"))
        if backup_files:
            latest_backup = backup_files[-1]
            logger.info(f"Restoring from backup: {latest_backup}")
            if MODEL_DIR.exists():
                shutil.rmtree(MODEL_DIR)
            shutil.move(str(latest_backup), str(MODEL_DIR))
            logger.info("Backup restored successfully")
    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")


def cleanup_old_backups(keep_count: int = 2) -> None:
    """Keep only the most recent backups to save disk space."""
    try:
        backup_files = sorted(MODEL_DIR.parent.glob(f"{MODEL_DIR.name}.backup.*"))
        if len(backup_files) > keep_count:
            for old_backup in backup_files[:-keep_count]:
                shutil.rmtree(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
    except Exception as e:
        logger.warning(f"Failed to cleanup old backups: {e}")


def cleanup_hf_cache_metadata() -> None:
    """
    Clean up the .cache/huggingface metadata folder inside MODEL_DIR.
    This is optional - HF recommends keeping it for faster updates, but you can remove it.
    """
    cache_dir = MODEL_DIR / ".cache" / "huggingface"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            logger.info("Cleaned up HuggingFace cache metadata")
        except Exception as e:
            logger.warning(f"Failed to cleanup HF cache: {e}")


def monitor_and_update(poll_interval: int = 60) -> None:
    """
    Main loop: continuously monitor /state and download when checkpoint updates.
    
    Args:
        poll_interval: Seconds between state polls
    """
    cached_state = load_cached_checkpoint_state()
    current_checkpoint_revision = cached_state.get("checkpoint_revision") if cached_state else None
    
    logger.info(f"Starting checkpoint monitor (poll interval: {poll_interval}s)")
    if current_checkpoint_revision:
        logger.info(f"Current checkpoint revision: {current_checkpoint_revision}")
    
    while True:
        try:
            state = fetch_state()
            
            if state is None:
                logger.warning("Failed to fetch state, retrying...")
                time.sleep(poll_interval)
                continue
            
            checkpoint_repo_id = state.get("checkpoint_repo_id")
            checkpoint_revision = state.get("checkpoint_revision")
            checkpoint_n = state.get("checkpoint_n")
            window_n = state.get("window_n")
            
            logger.debug(f"State: window={window_n}, checkpoint_n={checkpoint_n}, revision={checkpoint_revision}")
            
            # Check if checkpoint has updated
            if checkpoint_revision != current_checkpoint_revision:
                logger.info(
                    f"Checkpoint update detected!\n"
                    f"  Old revision: {current_checkpoint_revision}\n"
                    f"  New revision: {checkpoint_revision}\n"
                    f"  Repo: {checkpoint_repo_id}"
                )
                
                # Download the new checkpoint
                if download_checkpoint(checkpoint_repo_id, checkpoint_revision):
                    current_checkpoint_revision = checkpoint_revision
                    
                    # Save the new checkpoint state
                    save_checkpoint_state({
                        "checkpoint_revision": checkpoint_revision,
                        "checkpoint_repo_id": checkpoint_repo_id,
                        "checkpoint_n": checkpoint_n,
                        "last_update": time.time(),
                    })
                    
                    # Optionally clean up HF cache metadata
                    cleanup_hf_cache_metadata()
                    
                    # Clean up old backups
                    cleanup_old_backups(keep_count=2)
                    
                    logger.info("Checkpoint update completed successfully")
                else:
                    logger.error("Checkpoint download failed")
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            logger.info("Checkpoint monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in monitor loop: {e}", exc_info=True)
            time.sleep(poll_interval)


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor /state endpoint and download checkpoint updates"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=STATE_ENDPOINT,
        help="URL of the /state endpoint (default: $RELIQUARY_STATE_ENDPOINT or http://localhost:8000/state)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory to download models to (default: ~/reliquary-miner/model)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between state polls (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Download checkpoint once and exit (instead of continuous monitoring)",
    )
    
    args = parser.parse_args()
    
    # Override globals if provided
    globals()["STATE_ENDPOINT"] = args.endpoint
    globals()["MODEL_DIR"] = args.model_dir
    
    if args.once:
        # One-time download
        logger.info("Running in one-time mode")
        state = fetch_state()
        if state:
            checkpoint_repo_id = state.get("checkpoint_repo_id")
            checkpoint_revision = state.get("checkpoint_revision")
            download_checkpoint(checkpoint_repo_id, checkpoint_revision)
            save_checkpoint_state({
                "checkpoint_revision": checkpoint_revision,
                "checkpoint_repo_id": checkpoint_repo_id,
                "checkpoint_n": state.get("checkpoint_n"),
                "last_update": time.time(),
            })
    else:
        # Continuous monitoring
        monitor_and_update(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
