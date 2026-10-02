"""
Step-by-step status of a multi-experiment run, written to run_status.json so
watch_dashboard.py can show what is done, running and still pending.
"""
import json
import os
import time

from src import config as cfg

STATUS_PATH = cfg.BASE_DIR / "run_status.json"


class RunStatus:
    def __init__(self, title: str, steps: list[str]):
        self.data = {
            "title": title,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid(),
            "finished": None,
            "steps": [{"name": s, "state": "pending", "detail": "", "started": None,
                       "finished": None, "result": ""} for s in steps],
        }
        self._save()

    def _step(self, name):
        return next(s for s in self.data["steps"] if s["name"] == name)

    def start(self, name, detail=""):
        step = self._step(name)
        step.update(state="running", detail=detail, started=time.time())
        self._save()

    def done(self, name, result=""):
        step = self._step(name)
        step.update(state="done", result=result, finished=time.time())
        self._save()

    def skip(self, name, result="already done"):
        step = self._step(name)
        step.update(state="skipped", result=result)
        self._save()

    def fail(self, name, error):
        step = self._step(name)
        step.update(state="failed", result=str(error)[:200], finished=time.time())
        self._save()

    def finish(self):
        self.data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def _save(self):
        atomic_write_json(STATUS_PATH, self.data)


def atomic_write_json(path, data, attempts=10):
    """Write JSON via a temp file and rename, retrying while a reader holds the file.

    On Windows the rename fails with "Access is denied" if the dashboard happens to
    be reading the target at that instant; a status update must never crash a run.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    return False
