module.exports = {
  apps: [
    {
      name: 'grader-server',
      script: 'reliquary/reliquary/environment/grader/server.py',
      interpreter: 'python3',               // or full path to your venv python
      // If you use a virtual environment:
      // interpreter: '/root/reliquary-miner/.venv/bin/python',
      args: '',
      cwd: '/root/reliquary-miner',         // adjust to your project root
      env: {
        NODE_ENV: 'production',
        PORT: 3000,                         // example
        PYTHONUNBUFFERED: '1',              // real‑time logs
        // Add any other env vars for the server
      },
      out_file: './logs/grader-out.log',
      error_file: './logs/grader-err.log',
      merge_logs: true,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s',
      kill_timeout: 5000,
    },
    {
      name: 'vllm-server',
      script: '/root/reliquary-miner/.venv-vllm/bin/vllm',
      interpreter: '/root/reliquary-miner/.venv-vllm/bin/python',
      args: 'serve ReliquaryForge/qwen3.5-4b-reliquary-v4 --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 --max-model-len 8192 --served-model-name reliquary --language-model-only --logits-processors reliquary.miner_code.logit_processor:ForcedSeedLogitsProcessor',
      cwd: '/root/reliquary-miner',
      out_file: './logs/vllm-out.log',
      error_file: './logs/vllm-err.log',
      merge_logs: true,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '30s',
      env: {
        CUDA_VISIBLE_DEVICES: '0',
        VLLM_SERVER_DEV_MODE: '1',
        PYTHONPATH: '.',
        PYTHONUNBUFFERED: '1',
        RELIQUARY_PROTOCOL_PROFILE:'qwen35-4b-auction-v3',
        // Add any other environment variables needed
      },
      kill_timeout: 5000,
    },
    {
      name: 'reliquary-mine',

      // Path to your Python script (the reliquary binary)
      script: '/root/reliquary-miner/.venv/bin/reliquary',

      // Use the Python interpreter from the same virtual environment
      interpreter: '/root/reliquary-miner/.venv/bin/python',

      // Pass the original arguments
      args: 'mine-code --checkpoint ReliquaryForge/qwen3.5-4b-reliquary-v4 --max-concurrent 200 --environments opencodeinstruct',

      // Working directory (adjust if needed)
      cwd: '/root/reliquary-miner',

      // Logging
      out_file: './logs/out.log',
      error_file: './logs/err.log',
      merge_logs: true,

      // Process behaviour
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s',
      // max_memory_restart: '4G',

      // Environment (optional)
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',      // ensures logs appear in real-time
        GRAIL_ATTN_IMPL: 'sdpa',
        RELIQUARY_PROTOCOL_PROFILE:'qwen35-4b-auction-v3',
      },

      kill_timeout: 5000,
    }
  ]
};