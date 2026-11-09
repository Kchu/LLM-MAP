"""Console-logging helpers shared by the run scripts."""
import time


def banner(title, **info):
    """Print a block header, e.g. at the start/end of a run."""
    line = "=" * 80
    details = "\n".join(f"  {key:<12}: {value}" for key, value in info.items())
    print(f"\n{line}\n{title}" + (f"\n{details}" if details else "") + f"\n{line}", flush=True)


def log(tag, msg):
    """Print a time-stamped, tagged message, e.g. [12:00:00] [LLM+MAP] Generating plan..."""
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)
