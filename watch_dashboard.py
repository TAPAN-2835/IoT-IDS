"""
watch_dashboard.py

Live terminal dashboard for long runs (PowerShell / cmd / Windows Terminal).
Reads the files the pipeline already writes, so it can be started or stopped
at any time without affecting the run:

  run_status.json         step list of the current run (run_clean_baselines.py)
  training_progress.json  epoch-level progress of the current deep-learning model
  results/experiment_registry.csv  finished experiments
  pipeline.log            recent log lines

Usage:
    python watch_dashboard.py          # Ctrl+C to close (the run keeps going)
"""
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import pandas as pd
import psutil
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

BASE = Path(__file__).resolve().parent
STATUS = BASE / "run_status.json"
PROGRESS = BASE / "training_progress.json"
REGISTRY = BASE / "results" / "experiment_registry.csv"
LOG = BASE / "pipeline.log"

STATE_STYLE = {
    "done": ("DONE", "bold green"),
    "running": ("RUNNING", "bold yellow"),
    "pending": ("waiting", "dim"),
    "skipped": ("skipped", "cyan"),
    "failed": ("FAILED", "bold red"),
}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_secs(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _run_alive(status):
    try:
        return status and psutil.pid_exists(status["pid"]) and not status.get("finished")
    except Exception:
        return False


def header_panel(status):
    if not status:
        return Panel(Text("No run_status.json yet - start a run with run_clean_baselines.py", style="dim"),
                     title="IoT-IDS run monitor")
    alive = _run_alive(status)
    if status.get("finished"):
        state = Text(f"FINISHED at {status['finished']}", style="bold green")
    elif alive:
        state = Text("RUNNING", style="bold yellow")
    else:
        state = Text("STOPPED (process not running - check the log below)", style="bold red")
    done = sum(s["state"] in ("done", "skipped") for s in status["steps"])
    line = Text.assemble(("Run: ", "bold"), status["title"], "   ", ("Started: ", "bold"), status["started"],
                         "   ", ("Status: ", "bold"), state, f"   ({done}/{len(status['steps'])} steps)")
    return Panel(line, title="IoT-IDS run monitor", border_style="blue")


def steps_panel(status):
    table = Table(expand=True, show_edge=False, header_style="bold")
    table.add_column("State", width=9)
    table.add_column("Step", ratio=2)
    table.add_column("Result / what it is doing", ratio=4)
    table.add_column("Time", justify="right", width=8)
    if status:
        now = time.time()
        for s in status["steps"]:
            label, style = STATE_STYLE.get(s["state"], (s["state"], ""))
            if s["state"] == "running" and s["started"]:
                duration = _fmt_secs(now - s["started"])
            elif s["started"] and s["finished"]:
                duration = _fmt_secs(s["finished"] - s["started"])
            else:
                duration = ""
            info = s["result"] if s["state"] in ("done", "failed", "skipped") else s["detail"]
            table.add_row(Text(label, style=style), s["name"], info, duration)
    return Panel(table, title="Steps", border_style="blue")


def training_panel(status):
    running = next((s for s in (status or {}).get("steps", []) if s["state"] == "running"), None)
    prog = _read_json(PROGRESS)
    if running is None:
        return Panel(Text("Nothing training right now.", style="dim"), title="Current model", border_style="magenta")
    if "xgb" in running["name"]:
        return Panel(Text(f"{running['name']}: XGBoost boosting on the GPU (stops early when validation "
                          "loss stops improving; usually 1-3 minutes).", style="yellow"),
                     title="Current model", border_style="magenta")
    if running["name"].startswith("Preprocess"):
        return Panel(Text("Preprocessing in a child process: stream CSV, canonicalise tokens, encode, "
                          "save parquet, then shortcut scan (about 2-3 minutes).", style="yellow"),
                     title="Current model", border_style="magenta")
    if not prog or prog.get("experiment_id") != running["name"]:
        return Panel(Text(f"{running['name']}: loading data onto the GPU...", style="yellow"),
                     title="Current model", border_style="magenta")

    epoch, total = prog.get("epoch", 0), prog.get("total_epochs", 1) or 1
    bar = ProgressBar(total=total, completed=epoch, width=40)
    lines = [
        Text.assemble(("Experiment: ", "bold"), prog["experiment_id"], "   ", ("Phase: ", "bold"), prog.get("phase", "")),
        Text.assemble(("Epoch ", "bold"), f"{epoch}/{total}  "),
        bar,
    ]
    if "val_macro_f1" in prog:
        lines.append(Text(f"train loss {prog['train_loss']:.4f}   val macro-F1 {prog['val_macro_f1']:.4f}   "
                          f"best {prog['best_val_macro_f1']:.4f}   "
                          f"no-improvement epochs {prog.get('patience_counter', 0)}/{prog.get('patience_limit', '?')}"))
    elif "train_loss" in prog:
        lines.append(Text(f"train loss {prog['train_loss']:.4f}   val loss {prog['val_loss']:.4f}   "
                          f"best val {prog.get('best_val_loss', 0):.4f}   "
                          f"early-stop patience {prog.get('patience_counter', 0)}/{prog.get('patience_limit', '?')}"))
    elapsed = prog.get("elapsed_s", 0)
    if epoch:
        eta = elapsed / epoch * (total - epoch)
        lines.append(Text(f"elapsed {_fmt_secs(elapsed)}   ~{_fmt_secs(elapsed / epoch)}/epoch   "
                          f"ETA {_fmt_secs(eta)}", style="dim"))
    return Panel(Group(*lines), title="Current model", border_style="magenta")


def system_panel():
    lines = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip()
        name, util, used, total, temp, power = [x.strip() for x in out.split(",")]
        lines.append(Text(f"GPU  {name}"))
        lines.append(Text(f"     util {util}%   VRAM {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB   "
                          f"{temp} C   {power} W", style="green" if int(util) > 30 else ""))
    except Exception:
        lines.append(Text("GPU  nvidia-smi not available", style="dim"))
    mem = psutil.virtual_memory()
    ram_style = "red" if mem.percent > 90 else ("yellow" if mem.percent > 80 else "")
    lines.append(Text(f"RAM  {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB used ({mem.percent:.0f}%)   "
                      f"CPU {psutil.cpu_percent():.0f}%", style=ram_style))
    battery = psutil.sensors_battery()
    if battery:
        if battery.power_plugged:
            lines.append(Text(f"PWR  plugged in ({battery.percent:.0f}%)", style="green"))
        else:
            lines.append(Text(f"PWR  ON BATTERY ({battery.percent:.0f}%) - GPU is throttled, plug in for full speed",
                              style="bold red"))
    return Panel(Group(*lines), title="Laptop", border_style="green")


def results_panel():
    table = Table(expand=True, show_edge=False, header_style="bold")
    for col, kw in (("Experiment", {"ratio": 3}), ("Model", {"ratio": 1}), ("Policy", {"ratio": 1}),
                    ("Accuracy", {"justify": "right"}), ("Macro-F1", {"justify": "right"})):
        table.add_column(col, **kw)
    try:
        reg = pd.read_csv(REGISTRY)
        recent = reg[reg["experiment_id"].str.match(r"^([CS]\d|L_)")].tail(10)
        for _, r in recent.iterrows():
            table.add_row(r["experiment_id"], str(r["model"]), str(r["feature_policy"]),
                          f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}")
    except Exception:
        table.add_row("(registry not readable yet)", "", "", "", "")
    return Panel(table, title="Latest results (C = clean, S = strict, L = loss/seed study)", border_style="cyan")


def log_panel(n=8):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            tail = deque((l.rstrip() for l in f if " - " in l), maxlen=n)
        text = Text()
        for line in tail:
            short = line[11:19] + "  " + line.split(" - ", 3)[-1][:150]
            style = "red" if (" ERROR " in line or "Traceback" in line) else ("yellow" if "WARNING" in line else "")
            text.append(short + "\n", style=style)
    except FileNotFoundError:
        text = Text("pipeline.log not found", style="dim")
    return Panel(text, title="Log (pipeline.log)", border_style="white")


def render():
    status = _read_json(STATUS)
    layout = Layout()
    layout.split_column(
        Layout(header_panel(status), size=3),
        Layout(name="middle", ratio=3),
        Layout(log_panel(), size=10),
    )
    layout["middle"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["left"].split_column(Layout(steps_panel(status), ratio=3), Layout(results_panel(), ratio=2))
    layout["right"].split_column(Layout(training_panel(status), ratio=1), Layout(system_panel(), ratio=1))
    return layout


def main():
    try:
        with Live(render(), refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(2)
                live.update(render())
    except KeyboardInterrupt:
        print("Dashboard closed. Any run in progress keeps going.")


if __name__ == "__main__":
    main()
