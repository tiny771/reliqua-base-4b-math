module.exports = {
  apps: [
    {
      name: 'reliquary-mine',

      // Path to your Python script (the reliquary binary)
      script: '/home/user/venv/.distil/bin/reliquary',

      // Use the Python interpreter from the same virtual environment
      interpreter: '/home/user/venv/.distil/bin/python',

      // Pass the original arguments
      args: 'mine-math --checkpoint /mnt/d/models/Qwen3.5-2B --max-concurrent 128',

      // Working directory (adjust if needed)
      cwd: '/mnt/d/bigben/reliquary-miner',

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
        GRAIL_ATTN_IMPL: 'eager',
        // Add any other needed env vars, e.g. CUDA_VISIBLE_DEVICES
      },

      kill_timeout: 5000,
    }
  ]
};