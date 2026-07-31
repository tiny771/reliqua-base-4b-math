# Miner Note: Sparse Window Liveness

Recent validator hardening rejects bad terminations and malformed proof
submissions earlier. That protects training quality, but it also means windows
can fill more slowly while miners adapt.

The validator now seals sparse windows instead of waiting for the long safety
timeout:

- Normal path is unchanged for a single environment: a window seals when the
  active environment target reaches 8 distinct valid prompts. In mixed mode,
  each active environment targets 8 groups and partial envs skip GRPO/publish.
- If a window has at least 4 distinct valid prompts and no new valid submission
  lands for 180 seconds, it force-seals partial.
- If a sparse or zero-valid window remains open for 600 seconds after queue and
  proof work are drained, it force-seals partial.
- If bounded proof admission is exhausted and all admitted proof work has
  drained, the window also force-seals partial instead of waiting for the long
  safety timeout.
- Partial windows do not train a GRPO step unless 8 prompts are present. Unused
  slot share burns; it is not redistributed.

For miners, this means the current meta rewards clean, early, valid submissions
more than late retries. Watch `/health` for:

- `distinct_valid_prompt_count`
- `seconds_since_last_valid_submission`
- `sparse_valid_idle_seal_seconds`
- `sparse_valid_max_window_seconds`

If your miner is mostly seeing `bad_termination`, fix local EOS handling before
increasing volume. Spamming late or invalid submissions will not keep the window
alive and will not help once the sparse idle timer has elapsed.
