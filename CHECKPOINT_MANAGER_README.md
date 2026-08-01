# Reliquary Checkpoint Manager

## Overview

`checkpoint_manager.py` is a utility script that:
1. **Monitors the `/state` endpoint** continuously to detect when the checkpoint is updated
2. **Downloads the new model checkpoint** from Hugging Face Hub using `snapshot_download()`
3. **Backs up the old model** before replacing it
4. **Saves to a specific path**: `~/reliquary-miner/model` (customizable)
5. **Cleans up backups** to manage disk space

## How it works

### 1. State Endpoint Polling
The script polls your Reliquary validator's `/state` endpoint (e.g., `http://localhost:8000/state`). The response contains:
```json
{
  "state": "OPEN",
  "window_n": 123,
  "checkpoint_n": 45,
  "checkpoint_repo_id": "owner/model-name",
  "checkpoint_revision": "abc123def456",
  "cooldown_prompts": { ... }
}
```

### 2. Checkpoint Detection
When `checkpoint_revision` changes from the previously cached value, the script recognizes an update.

### 3. Download with `snapshot_download()`
Uses Hugging Face Hub's `snapshot_download()` function:
```python
snapshot_download(
    repo_id="owner/model-name",        # From /state response
    revision="abc123def456",            # Specific commit hash
    local_dir="~/reliquary-miner/model" # Your target directory
)
```

**Key points:**
- Downloads to `local_dir` instead of HF's default cache (useful for training/inference)
- Creates `.cache/huggingface/` metadata folder inside `local_dir` for incremental updates
- Maintains original file structure (safetensors, config.json, tokenizer files, etc.)
- Concurrent downloads for speed

### 4. Old Model Backup
Before downloading, the old model is moved to:
```
~/reliquary-miner/model.backup.<timestamp>
```

If the download fails, the backup is automatically restored.

### 5. Cleanup
- **HuggingFace metadata** (`.cache/huggingface/`) can be optionally removed
- **Old backups** (keeps last 2 by default) are pruned to save disk space

## Installation

### 1. Install dependencies
```bash
pip install -r checkpoint_requirements.txt
```

### 2. Make script executable
```bash
chmod +x checkpoint_manager.py
```

## Usage

### Option A: Run continuously (recommended for production)

```bash
python3 checkpoint_updates.py \
  --endpoint http://209.20.157.231:8080/state \
  --model-dir ~/reliquary-miner/model \
  --poll-interval 60
```

**Parameters:**
- `--endpoint`: URL of the `/state` endpoint (default: `http://localhost:8000/state`)
- `--model-dir`: Where to download models (default: `~/reliquary-miner/model`)
- `--poll-interval`: Seconds between state checks (default: 60)

### Option B: One-time download

```bash
python3 checkpoint_manager.py --once
```

Downloads the current checkpoint once and exits.

### Option C: Run as systemd service

```bash
# Copy service file
sudo cp reliquary-checkpoint-manager.service /etc/systemd/system/

# Enable and start
sudo systemctl enable reliquary-checkpoint-manager.service
sudo systemctl start reliquary-checkpoint-manager.service

# Monitor logs
sudo journalctl -u reliquary-checkpoint-manager.service -f
```

**Customize the service:**
Edit `/etc/systemd/system/reliquary-checkpoint-manager.service` to adjust:
- `Environment="RELIQUARY_STATE_ENDPOINT=http://localhost:8000/state"`
- `--poll-interval 60`
- `User=` (if running as different user)

## Output Structure

After download, your `~/reliquary-miner/model/` directory will look like:
```
model/
├── .gitattributes
├── .cache/
│   └── huggingface/          # Metadata (can be deleted)
├── config.json
├── model.safetensors         # Model weights
├── reliquary_recovery_manifest.json
├── tokenizer.json
├── tokenizer_config.json
└── ...other files...
```

## State Cache

The script maintains a checkpoint state cache at:
```
~/.reliquary-miner/.checkpoint_state
```

This JSON file tracks the last downloaded revision, so the script can detect updates across restarts.

## Logging

The script logs to:
- **stdout/stderr** (when running directly)
- **systemd journal** (when running as service): `sudo journalctl -u reliquary-checkpoint-manager.service`

Log levels: INFO (default), WARNING, ERROR

To enable DEBUG logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Environment Variables

- `RELIQUARY_STATE_ENDPOINT`: Override the default `/state` URL
  ```bash
  export RELIQUARY_STATE_ENDPOINT=http://validator.example.com:8000/state
  python3 checkpoint_manager.py
  ```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `/state` endpoint unreachable | Retries after poll_interval |
| Download fails | Restores from latest backup, logs error |
| Local disk full | Download will fail; keep backups cleanup enabled |
| Multiple instances running | Last-modified metadata prevents conflicts (but avoid this) |

## Disk Space Management

The script keeps:
- **Current model**: `~/reliquary-miner/model/`
- **Last 2 backups**: `model.backup.<timestamp>`
- **HF metadata** (optional): `.cache/huggingface/` inside model directory

### To save disk space:

**Option 1:** Disable backup retention
```python
cleanup_old_backups(keep_count=0)  # Don't keep any backups
```

**Option 2:** Always clean HF metadata after download
```python
cleanup_hf_cache_metadata()  # Called by default in the script
```

**Option 3:** Manually clean old backups
```bash
rm -rf ~/reliquary-miner/model.backup.*
```

## Troubleshooting

### "Failed to fetch /state"
- Check the `--endpoint` URL is correct and accessible
- Ensure your Reliquary validator is running
- Check firewall rules if validator is remote

### "Failed to download checkpoint"
- Verify Hugging Face repo is accessible: `huggingface-cli repo info owner/model-name`
- Set `HF_TOKEN` if repo is private: `export HF_TOKEN=hf_xxxxx`
- Check disk space: `df -h ~/reliquary-miner/`
- Check internet connection

### "Backup restored successfully"
- Download failed; model was restored from backup
- Check logs for the actual error
- Try running with more verbose logging (edit script)

### Checkpoint not updating
- Verify `/state` is returning a new `checkpoint_revision`
- Check poll interval isn't too long
- Ensure script is still running: `ps aux | grep checkpoint_manager`

## Integration with Miner

The miner can reference the downloaded model at:
```python
model_path = Path.home() / "reliquary-miner" / "model"
# Use for inference
```

Or set environment variable:
```bash
export RELIQUARY_MODEL_PATH=~/reliquary-miner/model
```

## Performance Notes

- **First download**: Full model size (~2-5GB depending on model)
- **Subsequent updates**: Only changed files downloaded (incremental)
- **Metadata**: `.cache/huggingface/` enables incremental updates; keep it if possible
- **Bandwidth**: Concurrent downloads from HF (typically 100+ MB/s on good connection)

## Example: Full Setup

```bash
# 1. Create workspace
mkdir -p ~/reliquary-miner
cd ~/reliquary-miner

# 2. Install dependencies
pip install -r checkpoint_requirements.txt

# 3. Test one-time download
python3 checkpoint_manager.py --once

# 4. Set up as systemd service for production
sudo cp reliquary-checkpoint-manager.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable reliquary-checkpoint-manager.service
sudo systemctl start reliquary-checkpoint-manager.service

# 5. Monitor
sudo journalctl -u reliquary-checkpoint-manager.service -f
```

## References

- Hugging Face Hub Download Guide: https://huggingface.co/docs/huggingface_hub/en/guides/download
- `snapshot_download()` API: https://huggingface.co/docs/huggingface_hub/v0.23.0/en/package_reference/file_download#huggingface_hub.snapshot_download
- Reliquary Concepts: [concepts.md](reliquary/docs/concepts.md) §The core loop
