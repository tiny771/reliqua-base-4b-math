#!/usr/bin/env python3
"""
Example usage of checkpoint_manager functions programmatically.
"""

from pathlib import Path
from checkpoint_manager import (
    fetch_state,
    download_checkpoint,
    load_cached_checkpoint_state,
    save_checkpoint_state,
    cleanup_old_backups,
)

def example_check_current_checkpoint():
    """Example: Check what checkpoint the validator is currently running."""
    print("=== Checking Current Checkpoint ===")
    
    state = fetch_state()
    if state:
        print(f"Validator State: {state.get('state')}")
        print(f"Window: {state.get('window_n')}")
        print(f"Checkpoint N: {state.get('checkpoint_n')}")
        print(f"Repository: {state.get('checkpoint_repo_id')}")
        print(f"Revision: {state.get('checkpoint_revision')}")
    else:
        print("Failed to fetch state")


def example_download_if_newer():
    """Example: Download if newer than cached."""
    print("\n=== Checking for Updates ===")
    
    state = fetch_state()
    if not state:
        print("Failed to fetch state")
        return
    
    cached_state = load_cached_checkpoint_state()
    current_revision = cached_state.get("checkpoint_revision") if cached_state else None
    
    new_revision = state.get("checkpoint_revision")
    repo_id = state.get("checkpoint_repo_id")
    
    print(f"Current cached revision: {current_revision}")
    print(f"Latest revision: {new_revision}")
    
    if new_revision != current_revision:
        print("✓ Update available, downloading...")
        if download_checkpoint(repo_id, new_revision):
            save_checkpoint_state({
                "checkpoint_revision": new_revision,
                "checkpoint_repo_id": repo_id,
                "checkpoint_n": state.get("checkpoint_n"),
            })
            print("✓ Download complete")
        else:
            print("✗ Download failed")
    else:
        print("✓ Already on latest revision")


def example_manual_download():
    """Example: Manually download a specific revision."""
    print("\n=== Manual Download ===")
    
    # Example values - replace with actual repo and revision
    repo_id = "reliquary/model-name"
    revision = "abc123def456"
    
    print(f"Downloading {repo_id}@{revision}...")
    if download_checkpoint(repo_id, revision):
        print("✓ Success")
    else:
        print("✗ Failed")


def example_cleanup_backups():
    """Example: Clean up old backups."""
    print("\n=== Cleaning Backups ===")
    
    print("Keeping last 1 backup, removing older...")
    cleanup_old_backups(keep_count=1)
    print("✓ Done")


if __name__ == "__main__":
    print("Reliquary Checkpoint Manager - Examples\n")
    
    example_check_current_checkpoint()
    example_download_if_newer()
    
    # Uncomment to test:
    # example_manual_download()
    # example_cleanup_backups()
