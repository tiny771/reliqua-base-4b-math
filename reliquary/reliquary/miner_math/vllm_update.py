import asyncio
import logging
import os
from typing import Optional

import requests
from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

MODEL = "ReliquaryForge/qwen3.5-4b-reliquary-v4"
SERVER = "http://127.0.0.1:8000"
CHECK_INTERVAL = 10

MODEL_SNAPSHOT_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "processor_config.json",
    "image_processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


class WeightWatcher:
    def __init__(self):
        self.api = HfApi()
        self.current_sha: Optional[str] = None

    async def latest_sha(self) -> str:
        info = await asyncio.to_thread(
            self.api.model_info,
            MODEL,
        )
        return info.sha

    async def download_snapshot(self, revision: str) -> str:
        logging.info("Downloading revision %s", revision)

        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=MODEL,
            revision=revision,
            allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
        )

        logging.info("Downloaded snapshot:")
        logging.info(path)

        return path

    def reload_vllm(self, weight_path: str):
        payload = {
            "method": "reload_weights",
            "kwargs": {
                "weight_path": weight_path,
            },
        }

        logging.info("Reloading weights...")

        response = requests.post(
            SERVER + "/collective_rpc",
            json=payload,
            timeout=600,
        )

        response.raise_for_status()

        logging.info(response.text)

    async def initialize(self):
        self.current_sha = await self.latest_sha()
        logging.info("Current SHA = %s", self.current_sha)

    async def run(self):
        await self.initialize()

        while True:

            try:
                latest = await self.latest_sha()

                if latest != self.current_sha:

                    logging.info("================================================")
                    logging.info("New model detected!")
                    logging.info("Old SHA: %s", self.current_sha)
                    logging.info("New SHA: %s", latest)

                    # path = await self.download_snapshot(latest)

                    # self.reload_vllm(path)

                    self.current_sha = latest

                    logging.info("Weights successfully reloaded.")
                    logging.info("================================================")

            except Exception:
                logging.exception("Failed to update weights")

            await asyncio.sleep(CHECK_INTERVAL)


async def main():
    watcher = WeightWatcher()
    await watcher.run()


if __name__ == "__main__":
    asyncio.run(main())