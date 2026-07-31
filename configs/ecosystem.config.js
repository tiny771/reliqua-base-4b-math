module.exports = {
  apps: [
    {
      name: 'reliquary-mine',

      // Path to your Python script (the reliquary binary)
      script: '/root/reliquary-miner/.venv/bin/reliquary',

      // Use the Python interpreter from the same virtual environment
      interpreter: '/root/reliquary-miner/.venv/bin/python',

      // Pass the original arguments
      args: 'mine-v3 --checkpoint ReliquaryForge/qwen3.5-2b-reliquary-v2 --max-concurrent 128',

      // Working directory (adjust if needed)
      cwd: '/root/reliquary-miner',

      // Logging
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
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
        GRAIL_ATTN_IMPL: 'eager',
        // Add any other needed env vars, e.g. CUDA_VISIBLE_DEVICES
      },

      kill_timeout: 5000,
    }
  ]
};