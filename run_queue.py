"""
run_queue.py

Runs several pipeline commands one after another while keeping Windows awake
for the WHOLE sequence. (Each runner keeps the PC awake while it runs, but in
the gap between two runs the laptop could drop into Modern Standby and freeze
the next one, which is what happened on 2026-10-02.)

The lid must stay open: closing it forces standby regardless.

Usage:
    python run_queue.py "run_final_studies.py" "run_ciciot.py --stage train"
"""
import shlex
import subprocess
import sys
import time

from run_clean_baselines import keep_awake
from src import config as cfg
from src.utils import setup_logger

logger = setup_logger("run_queue")


def main():
    commands = sys.argv[1:]
    if not commands:
        raise SystemExit(__doc__)
    keep_awake()
    for i, command in enumerate(commands, 1):
        logger.info(f"[queue {i}/{len(commands)}] starting: python {command}")
        t0 = time.time()
        result = subprocess.run([sys.executable, *shlex.split(command)], cwd=cfg.BASE_DIR)
        logger.info(f"[queue {i}/{len(commands)}] exit code {result.returncode} after {(time.time() - t0) / 60:.1f} min")
        if result.returncode != 0:
            logger.error("Stopping the queue because a step failed.")
            raise SystemExit(result.returncode)
    logger.info("Queue finished.")


if __name__ == "__main__":
    main()
