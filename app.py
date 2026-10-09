#!/usr/bin/env python3
"""
BoZo, The Watch Dog - an ambient, local, screen-aware AI companion built on Ollama.

  OCR (RapidOCR / Tesseract) -> reads on-screen text, spots errors
  nimble                     -> fast text-only gate: "is NOW the right moment?"
  qwen3.8:27b                -> vision + reasoning; writes the whisper

Run:  python app.py        (lives in the system tray; chat window bottom-left)
Everything stays on your machine. Screenshots only go to your local Ollama.
"""
import base64
import collections
import hashlib
import io
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

import mss
import psutil
import requests
from PIL import Image, ImageChops, ImageDraw, ImageStat

# ------------------------------------------------------------------ config ---
OLLAMA_HOST = "http://localhost:11434"
GATE_MODEL = "nimble"
VISION_MODEL = "qwen3.8:27b"

# All persistent user data (logs, memory, prefs) lives in one per-user folder.
USER_DATA_DIR = (os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")),
                              "screen_copilot")
                 if os.name == "nt"
                 else os.path.join(os.path.expanduser("~"), ".screen_copilot"))
os.makedirs(USER_DATA_DIR, exist_ok=True)


def _init_logging() -> logging.Logger:
    """File-backed logs (rotating, last ~1.5 MB). A pythonw build has no stdout,
    so this is the only reliable record of what the app did."""
    lg = logging.getLogger("bozo")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        try:
            fh = logging.handlers.RotatingFileHandler(
                os.path.join(USER_DATA_DIR, "bozo.log"),
                maxBytes=750_000, backupCount=2, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            lg.addHandler(fh)
        except OSError:  # unwritable data dir -> in-memory only, never crash
            pass
    return lg


log = _init_logging()

TICK_SECONDS = 2.5          # monitor loop period
SETTLE_TICKS = 2            # screen must be calm this many ticks before we look
MIN_ANALYSIS_GAP = 8        # never analyse more often than this (seconds)
MIN_TIP_GAP = 12            # min seconds between unprompted whispers
SCREEN_CHANGE_PCT = 3.0     # mean pixel change (0-100) that counts as "changed"
# Proactive gate: measured nimble scores in proactive framing are 0.39-0.64 on
# genuinely-useful screens and only drop lower on blank/repetitive ones. 0.32
# passes the useful ones; qwen's NOTHING sentinel remains the final filter.
GATE_THRESHOLD = 0.48       # conservative: require a stronger relevance signal
FLOW_BLOCK = 0.70           # nimble P(user_in_flow) above which we stay quiet
DWELL_SECONDS = 30          # screen static this long = user is reading/stuck
DWELL_GAP = 40              # min seconds between dwell-triggered analyses
DWELL_COMPOSE = 12          # in a compose-style window, this much static = user paused mid-write
DWELL_AGENT = 15            # in an AI-agent window (ChatGPT/Claude/...), this much static = offer a better prompt
OCR_MAX_CHARS = 1800
TOAST_SECONDS = 20          # base display time for a whisper (seconds)

RAM_WARN, SWAP_WARN, VRAM_WARN, DISK_WARN, CPU_WARN = 90, 60, 92, 95, 95

WHISPER_PROMPT = (
    "You are BoZo, The Watch Dog, a quiet ambient assistant that whispers help at the "
    "right moment. You get a screenshot, the OCR text of the screen, recent "
    "activity, and live system stats. Work out what the user is doing and what is "
    "going wrong or could go better. Cover: visible errors/stack traces (likely "
    "cause + concrete fix), research pointers, and RAM/VRAM/CPU/disk problems "
    "(name the process or setting). Rules: max 70 words, plain text, ONE "
    "actionable idea, no greeting, don't narrate the obvious, ignore the Screen "
    "Copilot window itself. Only attribute memory use to a process or model if the "
    "stats show it (e.g. weights counted as in GPU VRAM are not using system RAM). If nothing is genuinely useful, reply exactly: NOTHING"
)
CHAT_PROMPT = (
    "You are BoZo, The Watch Dog, a helpful on-device assistant. You may receive the "
    "user's current screenshot and live system stats; use them when relevant to "
    "debug problems, research topics, or diagnose RAM/VRAM/CPU/disk issues. "
    "Be direct and practical; give concrete steps or code when useful."
)

ERROR_PATTERNS = [re.compile(p, re.I) for p in (
    r"Traceback \(most recent call last\)", r"\b\w+(Error|Exception)\s*:",
    r"\bfatal\b", r"segmentation fault", r"out of memory|\bOOM\b",
    r"npm ERR!", r"command not found", r"permission denied", r"\bpanic:",
    r"build failed", r"cannot find module", r"error\[E\d+\]",
    r"\b(404|500|502|503)\b.{0,24}(error|not found|server)",
    r"exit code [1-9]", r"unhandled (exception|rejection)", r"\bat 0x[0-9a-f]+\b",
)] 

# Screen OCR can include BoZo's own diagnostics or transient developer notices.
# Those must not trigger an error whisper or be repeated as advice.
INTERNAL_NOISE_PATTERNS = [re.compile(p, re.I) for p in (
    r"nimble.{0,100}(classifier|embedding|vector database|cannot generate embeddings)",
    r"(dedicated )?embedding model.{0,100}(vector database|chromadb|qdrant)",
    r"use nimble only for the final classification",
    r"(screen copilot|bozo).{0,60}(memory|watching|vision model)",
    r"negative_prompt is passed but classifier-free guidance",
)]

def strip_internal_noise(text):
    """Remove OCR lines that are internal diagnostics rather than user work."""
    kept = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p.search(line) for p in INTERNAL_NOISE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)

def is_internal_or_duplicate_reply(reply, recent=()):
    """Reject leaked diagnostics and near-verbatim repeats from proactive whispers."""
    s = re.sub(r"\s+", " ", (reply or "")).strip().lower()
    if not s:
        return True, "empty"
    if any(p.search(s) for p in INTERNAL_NOISE_PATTERNS):
        return True, "internal_diagnostic"
    if any(x in s for x in ("nimble is a classifier", "dedicated embedding model", "populate your vector database")):
        return True, "embedding_diagnostic"
    # Avoid exact and substantial substring repeats; don't suppress merely related advice.
    for old in recent:
        old = re.sub(r"\s+", " ", str(old or "")).strip().lower()
        if len(old) >= 55 and (s == old or s in old or old in s):
            return True, "duplicate_recent_tip"
    return False, ""

# ---------------------------------------------------------------- helpers ---
_session = requests.Session()


def ollama_models():
    try:
        r = _session.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return None


def resolve_tag(installed, name):
    """Exact installed tag for a model name (e.g. 'nimble' -> 'nimble:9b').
    Ollama's /v1/systemone and /api/chat require the FULL tag, so a bare name
    in GATE_MODEL / VISION_MODEL would 404. Returns None if not installed."""
    if name in installed:
        return name
    base = name.split(":")[0]
    for tag in sorted(installed):
        if tag.split(":")[0] == base:
            return tag
    return None


AGENT_SYS = (
    "You are a PROMPT-ENGINEERING copilot. The user is working inside an "
    "AI agent / LLM chat (ChatGPT, Claude, OpenCode, Hermes, OpenClaw, Codex, "
    "Cursor, or similar) and has paused. From the conversation on screen, "
    "work out what the user is REALLY trying to achieve. Then write ONE "
    "improved prompt the user can paste into that agent to get a better "
    "result. Make it concrete: name the visible files, errors, constraints, "
    "and expected output; add the missing spec (inputs, edge cases, format, "
    "stop conditions) that the user's current ask is missing. Plain text, "
    "max 120 words, no headings, no meta commentary. If the user is already "
    "iterating with a strong prompt and nothing is clearly missing, or the "
    "screen shows a finished result, reply exactly: NOTHING"
)


def _copy_to_clipboard(text):
    """Copy text to the OS clipboard. Windows: PowerShell Set-Clipboard
    (no external dep). Returns True on success, False on failure — the
    caller decides what to show. Never raises."""
    if platform.system() == "Windows":
        import tempfile
        try:
            with tempfile.NamedTemporaryFile("w", delete=False,
                                             encoding="utf-8", suffix=".txt") as f:
                f.write(text)
                path = f.name
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Content", "-Raw", path, "|", "Set-Clipboard"],
                capture_output=True, timeout=5,
            )
            import os
            try:
                os.unlink(path)
            except OSError:
                pass
            return r.returncode == 0
        except Exception:
            return False
    # Linux/Fallback: xclip / xsel
    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
        try:
            p = subprocess.run(cmd, input=text.encode(), timeout=5,
                               capture_output=True)
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


COMPOSE_SYS = (
    "You are a DRAFTING copilot. The user is mid-composition (email, chat, "
    "LinkedIn post, or similar) and paused. Read: the recipient/subject (if any), "
    "the partial text they started, and — if they mentioned a project or prior "
    "context in their partial text — use that. Do NOT invent recipients, dates, "
    "facts, numbers, or stats the user did not mention. Complete their sentence "
    "in their own voice; match formality and tone from whatever they already "
    "wrote. If it is an email and no Subject line is visible, start with a "
    "Subject: first. Output the full draft, ready to paste — no explanations, "
    "no meta commentary, no quotes around the whole thing. If there is no "
    "actual partial text to compose from (e.g. user is reading someone else's "
    "message), reply exactly: NOTHING TO DRAFT"
)

EXPLAIN_SYSTEM = (
    "You are a senior engineer reading the user's screen. The user hit a bug "
    "or an error and wants to understand WHY it happened and HOW to fix it. "
    "Give: (1) LIKELY ROOT CAUSE — one or two plain sentences; (2) THE FIX — "
    "the concrete change (exact diff / command / env-var). If there are "
    "multiple plausible causes, rank them (most likely first) and name the "
    "smallest next diagnostic step to disambiguate. Do not pad, do not "
    "apologise, do not invent facts about code you can't see. Do not reply "
    "NOTHING."
)


def has_model(installed, name):
    return resolve_tag(installed, name) is not None


_TAG_CACHE = None


def _installed_tags():
    global _TAG_CACHE
    if _TAG_CACHE is None:
        tags = ollama_models()          # set(...) on success, None if Ollama is down
        if tags is not None:
            _TAG_CACHE = tags           # only cache a *live* read; retry if down
    return _TAG_CACHE if _TAG_CACHE is not None else set()


def model_tag(name):
    """Full installed tag for a model name; falls back to the name itself if not
    yet resolved (e.g. before startup check). Keeps bare GATE_MODEL/VISION_MODEL
    working even when Ollama's tag is 'name:xx'."""
    return resolve_tag(_installed_tags(), name) or name


def refresh_models(installed=None):
    """Re-read installed tags (call after a `ollama pull`, or pass a known set to seed it)."""
    global _TAG_CACHE
    if installed is not None:
        _TAG_CACHE = installed
    else:
        _TAG_CACHE = ollama_models()
    return _TAG_CACHE


# The user's chosen vision model (top-right dropdown). Stored in one small JSON
# file so the choice survives a restart; falls back to the built-in default.
_PREFS_FILE = os.path.join(USER_DATA_DIR, "prefs.json")


def _read_prefs():
    try:
        with open(_PREFS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _current_vision_model():
    name = _read_prefs().get("vision_model")
    if not isinstance(name, str) or not name.strip():
        return VISION_MODEL
    return name.strip()


def _set_vision_model(name):
    prefs = _read_prefs()
    prefs["vision_model"] = name
    try:
        with open(_PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except OSError as e:
        log.error("could not persist vision model choice: %s", e)
    refresh_models()                     # re-read tags so model_tag() resolves it


_user32 = None


def _get_user32():
    """Private user32 handle with explicit prototypes. A private WinDLL (not the
    shared ctypes.windll.user32) so we don't alter signatures other libs rely on."""
    global _user32
    if _user32 is None:
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL("user32")
        u.GetForegroundWindow.restype = wintypes.HWND   # pointer-sized, not c_int
        u.GetForegroundWindow.argtypes = []
        u.GetWindowTextW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        _user32 = u
    return _user32


def active_window_title():
    """Best-effort foreground window title (no extra deps)."""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes
            u = _get_user32()
            hwnd = u.GetForegroundWindow()
            if not hwnd:
                return ""
            buf = ctypes.create_unicode_buffer(512)
            n = u.GetWindowTextW(hwnd, buf, 512)
            return buf.value if n > 0 else ""
        if system == "Darwin":
            script = ('tell application "System Events" to get name of first '
                      'application process whose frontmost is true')
            return subprocess.check_output(["osascript", "-e", script], timeout=2,
                                           text=True).strip()
        return subprocess.check_output(["xdotool", "getactivewindow", "getwindowname"],
                                       timeout=2, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def find_errors(text):
    """Return up to 3 OCR lines that look like errors."""
    hits = []
    for line in text.splitlines():
        line = line.strip()
        if 6 <= len(line) and any(p.search(line) for p in ERROR_PATTERNS):
            hits.append(line[:160])
        if len(hits) == 3:
            break
    return hits


def sig(line):
    """Error-line signature for de-dup. Strip dynamic tokens (hex pointers/
    addresses, then digit runs like timestamps, pids, '1234ms') so the same
    logical error collapses to one sig regardless of the changing values."""
    t = line.lower()
    t = re.sub(r"0x[0-9a-f]+", "#ptr", t)   # hex addresses: 0x7f8a1b2c
    t = re.sub(r"[\d:.\-/]+", "#", t)       # digit runs + separators
    return hashlib.md5(t.encode()).hexdigest()[:10]


class ErrorLog:
    """Thread-safe, per-session, TTL-based log of observed error signatures
    (Tier 1 of the intelligence upgrade). Lets the whisper know whether a
    current error is a repeat and how often it has recurred, so the model can
    avoid repeating a prior fix."""

    def __init__(self, capacity=200):
        self.capacity = capacity
        self._log = {}      # sig -> {"count": int, "first_seen": float,
                            #         "last_seen": float, "snippet": str}
        self._lock = threading.Lock()

    def add(self, key, snippet):
        """Registers a sighting. Returns a record dict. Thread-safe."""
        with self._lock:
            now = time.time()
            rec = self._log.get(key)
            if rec is None:
                self._log[key] = {"count": 1, "first_seen": now,
                                  "last_seen": now, "snippet": snippet[:160]}
                if len(self._log) > self.capacity:
                    oldest = min(self._log, key=lambda k: self._log[k]["last_seen"])
                    del self._log[oldest]
                return {"count": 1, "seconds_since_first": 0.0,
                        "seconds_since_last_sighting": 0.0,
                        "is_new_to_session": True, "snippet": snippet[:160]}
            rec["count"] += 1
            prev_last = rec["last_seen"]
            rec["last_seen"] = now
            return {"count": rec["count"],
                    "seconds_since_first": round(now - rec["first_seen"], 1),
                    "seconds_since_last_sighting": round(now - prev_last, 1),
                    "is_new_to_session": False,
                    "snippet": rec["snippet"]}

    def get(self, key):
        with self._lock:
            return dict(self._log.get(key, {})) or None

    def all(self):
        with self._lock:
            return {k: dict(v) for k, v in self._log.items()}


# ------------------------------------------------------------ system stats ---
_nvml = None
GPU_BACKEND = "none"
GPU_ERROR = ""
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml = pynvml
    GPU_BACKEND = "pynvml"
except Exception as e:
    GPU_ERROR = f"{type(e).__name__}: {e}"


def _gpu_pynvml():
    if not _nvml:
        return None
    try:
        out = []
        for i in range(_nvml.nvmlDeviceGetCount()):
            h = _nvml.nvmlDeviceGetHandleByIndex(i)
            mem = _nvml.nvmlDeviceGetMemoryInfo(h)
            util = _nvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = _nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            name = _nvml.nvmlDeviceGetName(h)
            out.append({
                "name": name.decode() if isinstance(name, bytes) else name,
                "vram_used_gb": round(mem.used / 2**30, 2),
                "vram_total_gb": round(mem.total / 2**30, 2),
                "vram_pct": round(mem.used / mem.total * 100, 1),
                "gpu_util_pct": util.gpu, "temp_c": temp,
            })
        return out or None
    except Exception:
        return None


def _gpu_smi():
    """Fallback: parse nvidia-smi (ships with the NVIDIA driver, no pip needed)."""
    try:
        kw = {"creationflags": 0x08000000} if platform.system() == "Windows" else {}
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=5, text=True, stderr=subprocess.DEVNULL, **kw)
        gpus = []
        for line in out.strip().splitlines():
            name, used, total, util, temp = [x.strip() for x in line.split(",")]
            used_gb, total_gb = float(used) / 1024, float(total) / 1024
            gpus.append({
                "name": name, "vram_used_gb": round(used_gb, 2),
                "vram_total_gb": round(total_gb, 2),
                "vram_pct": round(used_gb / total_gb * 100, 1),
                "gpu_util_pct": int(float(util)) if util.replace(".", "").isdigit() else 0,
                "temp_c": int(float(temp)) if temp.replace(".", "").isdigit() else None,
            })
        return gpus or None
    except Exception:
        return None


_gpu_cache = None


def _gpu_sampler():
    """Poll off the UI thread so nvidia-smi never stalls the window."""
    global _gpu_cache, GPU_BACKEND
    while True:
        g = _gpu_pynvml()
        if g is None:
            g = _gpu_smi()
            if g:
                GPU_BACKEND = "nvidia-smi"
        _gpu_cache = g
        time.sleep(2)


def gpu_stats():
    return _gpu_cache


threading.Thread(target=_gpu_sampler, daemon=True).start()


def read_stats(with_procs=False):
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    root = "C:\\" if platform.system() == "Windows" else "/"
    s = {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(vm.used / 2**30, 2),
        "ram_total_gb": round(vm.total / 2**30, 2),
        "ram_pct": vm.percent, "swap_pct": sw.percent,
        "disk_pct": psutil.disk_usage(root).percent, "gpus": gpu_stats(),
    }
    if with_procs:
        procs = []
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                procs.append((p.info["memory_info"].rss, p.info["name"]))
            except Exception:
                pass
        procs.sort(reverse=True)
        s["top_memory_processes"] = [f"{n} ({b / 2**30:.1f} GB)" for b, n in procs[:4]]
        try:
            ps = _session.get(f"{OLLAMA_HOST}/api/ps", timeout=2).json().get("models", [])
            s["ollama_loaded"] = [
                f"{m['name']} ({m.get('size', 0) / 2**30:.1f} GB total, "
                f"{m.get('size_vram', 0) / 2**30:.1f} GB of it in GPU VRAM)" for m in ps]
        except Exception:
            pass
    return s


def find_anomalies(s):
    a = []
    if s["ram_pct"] >= RAM_WARN:
        a.append(f"RAM at {s['ram_pct']:.0f}%")
    if s["swap_pct"] >= SWAP_WARN:
        a.append(f"swap at {s['swap_pct']:.0f}% (system is paging)")
    if s["disk_pct"] >= DISK_WARN:
        a.append(f"disk {s['disk_pct']:.0f}% full")
    if s["cpu_pct"] >= CPU_WARN:
        a.append(f"CPU pegged at {s['cpu_pct']:.0f}%")
    for g in s["gpus"] or []:
        if g["vram_pct"] >= VRAM_WARN:
            a.append(f"VRAM at {g['vram_pct']:.0f}% on {g['name']}")
        if g["temp_c"] and g["temp_c"] >= 85:
            a.append(f"GPU temperature {g['temp_c']}C")
    return a


# -------------------------------------------------------------------- OCR ---
class OCR:
    """RapidOCR (pip-only) with Tesseract fallback. Safe to call from any thread."""

    def __init__(self):
        self.kind, self.engine, self.lock = None, None, threading.Lock()
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine, self.kind = RapidOCR(), "rapidocr"
            return
        except Exception:
            pass
        try:
            from rapidocr import RapidOCR  # newer package layout
            self.engine, self.kind = RapidOCR(), "rapidocr3"
            return
        except Exception:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.engine, self.kind = pytesseract, "tesseract"
        except Exception:
            pass

    def read(self, img):
        if not self.engine:
            return ""
        with self.lock:
            try:
                if self.kind == "rapidocr":
                    import numpy as np
                    res, _ = self.engine(np.array(img))
                    return "\n".join(r[1] for r in (res or []))
                if self.kind == "rapidocr3":
                    import numpy as np
                    res = self.engine(np.array(img))
                    return "\n".join(res.txts or [])
                return self.engine.image_to_string(img)
            except Exception:
                return ""


# ----------------------------------------------------------------- screen ---
class ScreenWatcher:
    """Owned by the monitor thread (mss handles are thread-bound)."""

    def __init__(self):
        self.sct = mss.mss()
        self.prev_small = None
        self.last_full = None
        self._lock = threading.Lock()  # grab() (monitor) vs jpeg_b64() (chat worker)

    def grab(self, mask_rects=()):
        mon = self.sct.monitors[1]  # primary monitor
        raw = self.sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        if mask_rects:  # blank our own windows so we never read ourselves
            d = ImageDraw.Draw(img)
            for x, y, w, h in mask_rects:
                x, y = x - mon["left"], y - mon["top"]
                d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
        small = img.resize((64, 36)).convert("L")  # computed on every frame so both branches bind it
        if self.prev_small is not None:
            change = ImageStat.Stat(ImageChops.difference(small, self.prev_small)).mean[0] / 255 * 100
        else:
            change = 0.0
        with self._lock:
            self.last_full = img
            self.prev_small = small
        return change

    def scaled(self, max_w):
        with self._lock:
            img = self.last_full
        if img is None:
            return None
        if img.width > max_w:
            img = img.resize((max_w, int(img.height * max_w / img.width)))
        return img

    def jpeg_b64(self, max_w=1280):
        img = self.scaled(max_w)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ models ---
def nimble_gate(state):
    """nimble is a text-only classifier. It sees OCR text + context, not pixels.
    Returns dict(speak, flow, topic, urgency) or None if unavailable."""
    body = {
        "model": model_tag(GATE_MODEL), "state": state, "keep_alive": "10m",
        "questions": {
            "worth_speaking": {
                "type": "noul",
                "instructions": ("You are a conservative relevance gate. Speak only when the visible screen "
                                 "contains a specific, timely problem or a clearly useful action. "
                                 "Do not invent generic productivity tips. Treat model diagnostics, "
                                 "classifier/embedding warnings, app logs, and BoZo's own UI as "
                                 "internal noise, not user problems. If uncertain, score below 0.48. "
                                 "Do NOT repeat recent_tips."),
            },
            "user_in_flow": {
                "type": "noul",
                "instructions": ("Is the user actively TYPING or editing code such that an "
                                 "unprompted tip would break their work? A STATIC screen where "
                                 "the user is READING (docs, an error, code, a log) is NOT flow "
                                 "- that is exactly when the copilot should speak. Keep this LOW "
                                 "for static/reading states."),
            },
            "topic": {
                "type": "choice",
                "instructions": ("Pick the single most useful angle for the user. "
                                 "Choose a topic only when it is grounded in the user's actual task. "
                                 "Prefer 'nothing' for generic, repetitive, internal diagnostic, "
                                 "or ambiguous content."),
                "criteria": {
                    "compose": "The user is writing an email, message, post or chat and has paused mid-compose",
                    "explain": "A visible error/stack trace is on screen that the user may be reading",
                    "agent": "The user is inside an AI chat/agent (ChatGPT, Claude, OpenCode, Hermes) and paused mid-task",
                    "research": "User is reading/searching and may want pointers",
                    "resources": "RAM, VRAM, CPU, disk or thermal issue",
                    "workflow": "A productivity or tooling tip fits the task",
                    "nothing": "Truly nothing to say (blank/locked/repeated tip)",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is it to speak?",
                "criteria": ["Can wait", "Soon", "Now"],
            },
        },
    }
    # Two attempts: the first one can hit a cold model load or a request
    # queued behind another in-flight Ollama generation; both are transient.
    last_err = "unknown"
    for attempt in (1, 2):
        try:
            # (connect, read) — the read side is the one that matters: a 9B
            # gate call is 3-5 s warm but 100+ s on a cold load or queue
            # stall. 30 s aborted good decisions; 180 s absorbs it.
            r = _session.post(f"{OLLAMA_HOST}/v1/systemone", json=body,
                              timeout=(10, 180))
            r.raise_for_status()
            a = r.json()["answers"]
            return {"speak": a["worth_speaking"]["noul"], "flow": a["user_in_flow"]["noul"],
                    "topic": a["topic"]["choice"], "urgency": a["urgency"]["score"]}
        except Exception as e:
            # Log the real reason so "gate silently not firing" is diagnosable
            # (timeout, 404/409, bad body, etc.) instead of a bare None.
            last_err = f"{type(e).__name__}: {e}"
            h = getattr(e, "response", None)
            if h is not None:
                last_err = f"HTTP {h.status_code} {h.text[:300]!r}"
    log.warning("nimble_gate failed after 2 attempts: %s", last_err)
    print(f"nimble_gate failed after 2 attempts: {last_err}")
    return None


def prepublish_check(text):
    """Pre-publish triage of a text the user is about to send/post.
    nimble scores 6 dimensions 0-1 (higher = worse). Returns
    (verdict, scores, blurb) where verdict is 'Ship it' | 'Rethink it' |
    'Hard no', scores = {toxicity, tone, inclusivity, misread, virality},
    or (None, None, None) if nimble is unavailable.
    All local, one nimble call (~1s)."""
    body = {
        "model": model_tag(GATE_MODEL), "state": {"text": text[:1500],
           "context": "user about to publish/send this text publicly"},
        "keep_alive": "10m",
        "questions": {
            "toxicity": {"type": "score",
                         "instructions": "How toxic, insulting or abusive is this text?",
                         "criteria": ["completely harmless", "severely abusive"]},
            "tone": {"type": "score",
                     "instructions": ("How negatively worded is it? Score high for "
                                      "aggressive, sarcastic, dismissive, fear-mongering "
                                      "or blame-shifting phrasing; low for calm, "
                                      "professional or neutral wording."),
                     "criteria": ["calm and professional", "aggressive or hostile"]},
            "inclusivity": {"type": "score",
                            "instructions": ("Inclusivity RISK: how much does it exclude, "
                                             "mock, stereotype or discriminate against "
                                             "people or groups? 0 is fully inclusive."),
                            "criteria": ["fully inclusive", "exclusionary or discriminatory"]},
            "misread": {"type": "score",
                        "instructions": ("MISREAD RISK: how likely is a reasonable reader "
                                         "to misinterpret the intent, or find it "
                                         "ambiguous/contradictory? 0 is unambiguous."),
                        "criteria": ["unambiguous", "highly likely to be misread"]},
            "secrets": {"type": "score",
                        "instructions": ("ACCIDENTAL-EXPOSURE RISK: does this contain "
                                         "credentials, API keys, passwords, tokens, "
                                         "internal URLs, or personal data that should "
                                         "never be published? 0 = no sensitive data."),
                        "criteria": ["nothing sensitive", "contains credentials or secrets"]},
            "virality": {"type": "score",
                         "instructions": ("How likely is this to be picked up, "
                                          "screenshot-shared or quoted — especially if a "
                                          "misread or bad-take version? 1 = will travel "
                                          "widely regardless of intent."),
                         "criteria": ["will not travel", "highly shareable"]},
        },
    }
    try:
        r = _session.post(f"{OLLAMA_HOST}/v1/systemone", json=body,
                          timeout=(10, 180))
        r.raise_for_status()
        a = r.json()["answers"]
        scores = {k: float(a[k]["score"]) for k in
                  ("toxicity", "tone", "inclusivity", "secrets", "misread", "virality")}
    except Exception as e:
        return None, None, f"nimble call failed — {e}"
    return _verdict(scores), scores, None


def _verdict(s):
    """Deterministic verdict on the nimble dimension scores (0-1, high = bad).
      - Hard no: any negative dimension (toxicity/tone/inclusivity) >= 0.65,
                 all three >= 0.40, OR accidental-exposure (secrets) >= 0.60
      - Rethink: any negative dimension >= 0.30, OR misread >= 0.50,
                 OR secrets >= 0.45, OR high virality riding on a flagged tone
      - Ship it: everything else. Virality alone is a heads-up, not a block.
    """
    neg = (s["toxicity"], s["tone"], s["inclusivity"])
    if max(neg) >= 0.65 or (min(neg) >= 0.40) or s["secrets"] >= 0.60:
        return "Hard no"
    if max(neg) >= 0.30 or s["secrets"] >= 0.45 or s["misread"] >= 0.50 or \
       (s["virality"] >= 0.60 and (max(neg) >= 0.30 or s["misread"] >= 0.30)):
        return "Rethink it"
    return "Ship it"


# Words that carry no actionable content — a reply made up *entirely* of these
# (plus stopwords) is "there is nothing to say". The moment a concrete noun,
# verb or object appears ("regex", "anchoring", "port", "restart") it is a tip.
# Biased toward KEEPING: a false-positive tip is a harmless toast; discarding a
# real tip (the reported bug) is the failure we must not cause.
_NO_TIP_VOCAB = {
    # stopwords / fillers
    "a", "an", "the", "i", "is", "it", "its", "that", "this", "to", "of", "in",
    "on", "at", "by", "be", "been", "but", "and", "or", "very", "really", "just",
    "only", "for", "with", "any", "anything", "no", "not", "nothing", "none",
    # "there is nothing to say" vocabulary
    "useful", "usefully", "suggestion", "suggestions", "tip", "tips", "advice",
    "actionable", "worth", "worthless", "here", "there", "say", "saying", "add",
    "added", "adding", "offer", "offers", "offering", "report", "reports",
    "reporting", "help", "helpful", "available", "found", "notable",
    "genuinely", "have", "has", "had", "can", "could", "would", "should",
}


def _is_nothing(reply: str) -> bool:
    """True if the model effectively said 'nothing' (handles near-miss
    phrasings like 'Nothing is genuinely useful.' that a naive startswith
    check would either miss or, worse, mis-catch).

    Discard-only rule, so a valid tip is never swallowed: a useful tip always
    contains at least one concrete word — a thing it is about or an action it
    suggests ('...consider anchoring it', '...restart it', '...check the
    port'). A reply made up entirely of absence-vocabulary ('Nothing here,
    but nothing I can add.') is what we drop."""
    words = re.findall(r"[\w'’\-]+", reply.lower())
    if not words:
        return True
    return all(w in _NO_TIP_VOCAB for w in words)


def qwen_chat(messages, image_b64=None):
    if image_b64 and messages:
        messages = [dict(m) for m in messages]
        messages[-1]["images"] = [image_b64]
    body = {"model": model_tag(_current_vision_model()), "messages": messages, "stream": False,
            "think": False, "keep_alive": "15m",
            "options": {"num_ctx": 8192, "temperature": 0.4}}
    r = _session.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ------------------------------------------------------------- tray icon ---
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logo", "boZo.png")
_logo_cache = {}


def _content_box(im):
    """Tight box of non-white pixels (border scan; source art has a solid
    white card background)."""
    w, h = im.size
    px = im.load()
    white = lambda c: c[0] > 240 and c[1] > 240 and c[2] > 240
    top = 0
    while top < h and all(white(px[x, top]) for x in range(0, w, 8)):
        top += 1
    bottom = h - 1
    while bottom > 0 and all(white(px[x, bottom]) for x in range(0, w, 8)):
        bottom -= 1
    left = 0
    while left < w and all(white(px[left, y]) for y in range(0, h, 8)):
        left += 1
    right = w - 1
    while right > 0 and all(white(px[right, y]) for y in range(0, h, 8)):
        right -= 1
    return (left, top, right + 1, bottom + 1)


def _logo_tile(size=64):
    """The BoZo dog logo, content-centred on a size x size white tile."""
    if size in _logo_cache:
        return _logo_cache[size]
    im = Image.open(LOGO_PATH).convert("RGBA")
    box = _content_box(im)
    pad = max(1, int(size * 0.04))          # small margin, like a badge
    inner = size - 2 * pad
    crop = im.crop(box).resize((inner, inner), Image.LANCZOS)
    tile = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    tile.paste(crop, (pad, pad))
    _logo_cache[size] = tile
    return tile


def _fallback_icon(state):
    """Old drawn eye-icon, used if logo/boZo.png is missing."""
    colors = {"on": "#2dd4bf", "paused": "#6b7280", "snoozed": "#f59e0b", "busy": "#818cf8"}
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, 60, 60], fill=colors.get(state, "#2dd4bf"))
    d.ellipse([20, 22, 44, 42], fill="white")
    d.ellipse([27, 27, 37, 37], fill="#14161a")
    return img


def make_icon(state):
    """Tray/window icon: the BoZo logo tile with a state-coloured badge
    (on/paused/snoozed/busy) in the bottom-right corner."""
    try:
        img = _logo_tile(64)
    except Exception:
        return _fallback_icon(state)
    d = ImageDraw.Draw(img)
    col = {"on": "#2dd4bf", "paused": "#6b7280", "snoozed": "#f59e0b",
           "busy": "#818cf8"}.get(state, "#2dd4bf")
    d.ellipse([48, 48, 64, 64], fill=col, outline="#14161a", width=2)
    return img


# ------------------------------------------------------------- pre-publish ---
PRE_COLORS = {"Ship it": "#2dd4bf", "Rethink it": "#ffb86b", "Hard no": "#ff6b6b"}
PRE_DIMS = ("toxicity", "tone", "inclusivity", "secrets", "misread", "virality")
PRE_NAMES = {"toxicity": "TOXICITY", "tone": "TONE", "inclusivity": "INCLUSIVITY",
             "secrets": "EXPOSURE", "misread": "MISREAD-RISK", "virality": "VIRALITY"}


class PrepublishCheck:
    """Paste-in window: signal-check a draft (post/email/DM) across 6 dimensions
    using nimble:9b (~1s, local). Shows Ship it / Rethink it / Hard no as a
    calibrated-signal *estimate* (classifier scores are not calibrated
    correctness) — treat it as a reading, not a ruling.
    Verdict logic is deterministic Python — no 'thinking' step, no big tokens."""
    W, H = 560, 500

    def __init__(self, root):
        self.root = root
        self.win = None
        self._result = None
        self.busy = False

    def open(self):
        if self.win is not None:
            try:
                self.win.lift()
                self.win.focus_force()
                return
            except tk.TclError:
                self.win = None
        w = tk.Toplevel(self.root)
        w.title("Pre-publish check")
        w.geometry(f"{self.W}x{self.H}")
        w.configure(bg="#10131a")
        w.protocol("WM_DELETE_WINDOW", self.close)
        fg, bg2 = "#e6e1cf", "#1a1f2b"
        head = tk.Frame(w, bg=bg2); head.pack(fill="x")
        tk.Label(head, text="Would you post this?", font=("Segoe UI", 14, "bold"),
                 bg=bg2, fg=fg).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Label(head, text="Paste a tweet, message, email or draft. "
                            "Local check in ~1s, zero tokens on the draft.",
                 font=("Segoe UI", 9), bg=bg2, fg="#8b96ab").pack(anchor="w", padx=14, pady=(0, 10))
        wrap = tk.Frame(w, bg="#10131a"); wrap.pack(fill="both", expand=True, padx=10)
        self.text = tk.Text(wrap, bg="#0d1017", fg=fg, bd=0, wrap="word",
                            font=("Consolas", 11), relief="flat")
        sb = tk.Scrollbar(wrap, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # Try to prefill from the clipboard — the natural flow is copy-paste-check.
        self._try_prefill(w)
        bar = tk.Frame(w, bg="#10131a"); bar.pack(fill="x", padx=10, pady=(8, 0))
        tk.Button(bar, text="Clear", command=lambda: self.text.delete("1.0", "end"),
                  bd=0, bg="#2a2f38", fg=fg, padx=12).pack(side="left")
        tk.Button(bar, text="Paste", command=self._paste, bd=0,
                  bg="#2a2f38", fg=fg, padx=12).pack(side="left", padx=6)
        self.check_btn = tk.Button(bar, text="CHECK  (~1s)",
                                   command=self._run, bd=0,
                                   bg="#12334a", fg="#9ecbff",
                                   font=("Segoe UI", 10, "bold"), padx=16)
        self.check_btn.pack(side="right")
        self.panel = tk.Frame(w, bg="#10131a"); self.panel.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        self.win = w
        w.update_idletasks()
        w.lift()
        w.focus_force()

    def _clip(self, w):
        try:
            return (w.clipboard_get() or "").strip()
        except tk.TclError:
            return ""

    def _try_prefill(self, w):
        clip = self._clip(w)
        if clip and len(clip) < 8000:
            self.text.insert("1.0", clip + "\n")
            self.text.see("1.0")

    def _paste(self):
        clip = self._clip(self.win)
        if clip:
            self.text.insert("1.0", clip + "\n")

    def close(self):
        if self.win:
            try:
                self.win.destroy()
            except tk.TclError:
                pass
        self.win = None

    def _clear_panel(self):
        for c in self.panel.winfo_children():
            c.destroy()
        if self._result is not None:
            try:
                self._result.destroy()
            except tk.TclError:
                pass
            self._result = None

    def _run(self):
        if self.busy:
            return
        text = self.text.get("1.0", "end").strip()
        if len(text) < 8:
            self._show_error("Paste something to check — at least a sentence.")
            return
        self._clear_panel()
        self.busy = True
        self.check_btn.configure(state="disabled", text="CHECKING…")
        lbl = tk.Label(self.panel, text="nimble is scoring toxicity, tone, "
                                        "inclusivity, exposure, misread-risk, virality…",
                       font=("Segoe UI", 10), bg="#10131a", fg="#8b96ab", anchor="w")
        lbl.pack(fill="x", pady=24)
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text):
        verdict, scores, err = prepublish_check(text)
        self.root.after(0, lambda: self._finish(verdict, scores, err, text))

    def _finish(self, verdict, scores, err, text):
        self.busy = False
        self.check_btn.configure(state="normal", text="CHECK  (~1s)")
        self._clear_panel()
        if err is not None or verdict is None:
            self._show_error(err or "No verdict — try again.")
            return
        color = PRE_COLORS.get(verdict, "#2dd4bf")
        p = tk.Frame(self.win, bg="#10131a"); p.pack(fill="both", expand=True)
        self._result = p
        head = tk.Frame(p, bg="#10131a"); head.pack(fill="x", pady=(8, 6))
        tk.Label(head, text=verdict.upper(), font=("Segoe UI", 20, "bold"),
                 bg="#10131a", fg=color).pack(anchor="w")
        tk.Label(head, text={
            "Ship it": "Low signal on all six dimensions — likely fine as a draft.",
            "Rethink it": "A couple of dimensions are elevated — worth one read-through.",
            "Hard no": "Strong signal on sensitive dimensions — revise before sending."}.get(verdict, ""),
            font=("Segoe UI", 9), bg="#10131a", fg="#8b96ab").pack(anchor="w")
        # Dimension rows (0-1; higher = more likely to bite back)
        inner = tk.Frame(p, bg="#10131a"); inner.pack(fill="both", expand=True, pady=(0, 4))
        for k in PRE_DIMS:
            v = scores[k]
            row = tk.Frame(inner, bg="#10131a"); row.pack(fill="x", pady=(3, 3))
            tk.Label(row, text=PRE_NAMES[k], font=("Consolas", 9), width=16, anchor="w",
                     bg="#10131a", fg="#cfd8e3").pack(side="left")
            b = tk.Frame(row, bg="#222837", width=240, height=6)
            b.pack(side="left", padx=(0, 6))
            b.pack_propagate(False)
            f = tk.Frame(b, bg=color if v >= 0.30 else "#3a4152")
            f.place(x=0, y=0, relwidth=min(1.0, v), height=6)
            tk.Label(row, text=f"{v:.2f}", font=("Consolas", 9), bg="#10131a", fg="#8b96ab"
                     ).pack(side="left")
        tk.Label(p, text="Signal, not a ruling — a local classifier's rough estimate "
                        "(unverified benchmark). 100% local, nothing went to a cloud.",
                 font=("Segoe UI", 8), bg="#10131a", fg="#5a6478").pack(anchor="w", pady=(6, 0))

    def _show_error(self, msg):
        self._clear_panel()
        tk.Label(self.panel, text=msg, font=("Segoe UI", 10),
                 bg="#10131a", fg="#ff8b8b", anchor="w", wraplength=self.W - 60).pack(fill="x", pady=24)


# ------------------------------------------------------------------ toast ---
class Toast:
    """A small auto-fading 'whisper' above the chat window."""
    W = 380

    def __init__(self, root, on_feedback):
        self.root, self.on_feedback = root, on_feedback
        self.win, self._after, self._copying = None, None, False

    def rect(self):
        if not self.win:
            return None
        try:
            return (self.win.winfo_x(), self.win.winfo_y(),
                    self.win.winfo_width(), self.win.winfo_height())
        except tk.TclError:
            return None

    def close(self):
        if self._after:
            self.root.after_cancel(self._after)
            self._after = None
        if self.win:
            try:
                self.win.destroy()
            except tk.TclError:
                pass
        self.win = None

    def _do_copy(self, text, btn):
        """COPY button: copy the draft/fix to the OS clipboard without
        dismissing the toast (so the user can keep reading it)."""
        if self._copying:
            return
        self._copying = True
        try:
            btn.configure(state="disabled", text="COPYING…")
        except tk.TclError:
            pass
        ok = _copy_to_clipboard(text)
        def _restore():
            self._copying = False
            try:
                if ok:
                    btn.configure(state="normal", text="✓ COPIED", fg="#8be28b")
                else:
                    btn.configure(state="normal", text="COPY (failed)", fg="#ff8b8b")
            except tk.TclError:
                pass
        self.root.after(2500, _restore)

    def show(self, text, topic, anchor_y, copyable=False):
        self.close()
        accent = {"debugging": "#ff6b6b", "resources": "#ffb86b", "research": "#7aa2f7",
                  "workflow": "#2dd4bf", "compose": "#5eead4", "explain": "#ff6b6b"}.get(topic, "#2dd4bf")
        w = tk.Toplevel(self.root)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        try:
            w.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        outer = tk.Frame(w, bg=accent)
        outer.pack(fill="both", expand=True)
        body = tk.Frame(outer, bg="#1b1e25")
        body.pack(fill="both", expand=True, padx=(4, 0))
        tk.Label(body, text=topic.upper(), bg="#1b1e25", fg=accent,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(body, text=text, bg="#1b1e25", fg="#ececec", wraplength=self.W - 40,
                 justify="left", anchor="w", font=("Segoe UI", 10)).pack(fill="x", padx=10, pady=4)
        row = tk.Frame(body, bg="#1b1e25")
        row.pack(fill="x", padx=8, pady=(0, 6))
        if copyable:
            copy_btn = tk.Button(row, text="COPY", bd=0, bg="#1f2b33", fg=accent,
                                 font=("Segoe UI", 8, "bold"),
                                 command=lambda: self._do_copy(text, copy_btn))
            copy_btn.pack(side="left", padx=2)
        for label, val in (("helpful", 1), ("not now", -1)):
            tk.Button(row, text=label, bd=0, bg="#2a2f38", fg="#cfcfcf", font=("Segoe UI", 8),
                      command=lambda v=val: (self.on_feedback(topic, v), self.close())
                      ).pack(side="left", padx=2)
        tk.Button(row, text="dismiss", bd=0, bg="#2a2f38", fg="#cfcfcf", font=("Segoe UI", 8),
                  command=self.close).pack(side="right", padx=2)
        w.update_idletasks()
        h = w.winfo_reqheight()
        w.geometry(f"{self.W}x{h}+12+{max(0, anchor_y - h - 8)}")
        self.win = w
        self._copying = False
        ms = (TOAST_SECONDS + min(len(text) // 25, 10)) * 1000
        self._fade(0.0, True, ms)
        w.bind("<Enter>", lambda e: self._hold())

    def _hold(self):  # hovering keeps it alive
        if self._after:
            self.root.after_cancel(self._after)
        self._after = self.root.after(6000, lambda: self._fade(1.0, False, 0))

    def _fade(self, a, up, hold_ms):
        if not self.win:
            return
        try:
            if up:
                a = min(0.96, a + 0.16)
                self.win.attributes("-alpha", a)
                nxt = (lambda: self._fade(a, True, hold_ms)) if a < 0.96 else \
                      (lambda: self._fade(0.96, False, 0))
                self._after = self.root.after(30 if a < 0.96 else hold_ms, nxt)
            else:
                a -= 0.08
                if a <= 0:
                    self.close()
                    return
                self.win.attributes("-alpha", a)
                self._after = self.root.after(40, lambda: self._fade(a, False, 0))
        except tk.TclError:
            self.close()


# ------------------------------------------------------------------ memory ---
class MemoryWindow:
    """Browse, delete, prune, export and stat BoZo's local memory.

    One self-contained Toplevel opened from the toolbar (idempotent: a second
    click just re-raises the existing window). All store access goes through
    the MemoryStore passed in, so nothing here repeats SQL."""
    BG = "#14161a"
    FG = "#e6e6e6"
    MUT = "#7a8290"
    ROW_FG = {
        "error": "#ffab91", "tip": "#a5d6a7", "chat_me": "#90caf9",
        "chat_bozo": "#80cbc4", "note": "#d7ccc8",
    }

    def __init__(self, parent_root, memstore):
        self.mem = memstore
        self.win = tk.Toplevel(parent_root)
        self.win.title("BoZo \u2014 Memory (local, deletable)")
        self.win.configure(bg=self.BG)
        self.win.geometry("760x540+80+80")
        self.win.minsize(560, 420)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._rows = []          # newest-first, matches listbox row order
        self._flash_until = 0.0
        self._build()
        self.refresh()
        self._update_stat()

    # --------------------------------------------------------------- layout --
    def _build(self):
        w = self.win
        tk.Label(w, text="BoZo memory \u2014 everything stored locally, nothing in the cloud",
                 font=("Segoe UI", 15, "bold"), bg=self.BG, fg=self.FG).pack(anchor="w",
                 padx=12, pady=(10, 2))
        self.stat_lbl = tk.Label(w, text="", anchor="w", bg=self.BG, fg=self.MUT,
                                 font=("Segoe UI", 8, "italic"))
        self.stat_lbl.pack(fill="x", padx=12, pady=(0, 6))

        tools = tk.Frame(w, bg=self.BG)
        tools.pack(fill="x", padx=12, pady=(0, 6))
        self.cat_var = tk.StringVar(value="All")
        cats = ["All", "error", "tip", "chat_me", "chat_bozo", "note"]
        drop = ttk.Combobox(tools, textvariable=self.cat_var, values=cats,
                            state="readonly", width=12)
        drop.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        drop.pack(side="left")
        tk.Label(tools, text="filter:", bg=self.BG, fg=self.MUT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(6, 4))
        self.stat_lbl2 = tk.Label(tools, text="", bg=self.BG, fg="#ffd58a",
                                  font=("Segoe UI", 9))
        self.stat_lbl2.pack(side="left", padx=(10, 0))

        mid = tk.Frame(w, bg=self.BG)
        mid.pack(fill="both", expand=True, padx=12, pady=4)
        self.lb = tk.Listbox(mid, bg="#1d2026", fg=self.FG, bd=0, relief="flat",
                             font=("Consolas", 10), activestyle="none",
                             highlightthickness=0, selectmode="extended")
        sb = ttk.Scrollbar(mid, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.lb.pack(side="left", fill="both", expand=True)
        self.lb.bind("<Double-Button-1>", self._on_double_click)

        self.detail = tk.Label(w, text="Select a row to read it in full.",
                               anchor="nw", justify="left", wraplength=700,
                               bg=self.BG, fg=self.MUT, font=("Segoe UI", 9))
        self.detail.pack(fill="x", padx=12, pady=(2, 0))

        acts = tk.Frame(w, bg=self.BG)
        acts.pack(fill="x", padx=12, pady=(6, 12))
        tk.Button(acts, text="\u2716 Delete selected", command=self._del_selected,
                  bg="#31200f", fg="#ffd58a", bd=0, padx=10,
                  pady=4).pack(side="left", padx=2)
        tk.Button(acts, text="Prune to retention (90d/30d)", command=self._prune,
                  bg="#26292f", fg=self.FG, bd=0, padx=10,
                  pady=4).pack(side="left", padx=2)
        tk.Button(acts, text="Export as JSON", command=self._export,
                  bg="#26292f", fg=self.FG, bd=0, padx=10,
                  pady=4).pack(side="left", padx=2)
        tk.Button(acts, text="DELETE ALL (typed confirm)", command=self._del_all,
                  bg="#2a1414", fg="#ff8a8a", bd=0, padx=10,
                  pady=4).pack(side="left", padx=2)
        tk.Label(acts, text="", bg=self.BG).pack(side="left", expand=True)
        tk.Button(acts, text="Close", command=self.close, bg="#26292f",
                  fg=self.MUT, bd=0, padx=10, pady=4).pack(side="right")

    # ------------------------------------------------------------ operations --
    def refresh(self):
        cat = self.cat_var.get() if self.cat_var.get() != "All" else None
        try:
            self._rows = self.mem.list_memories(category=cat, limit=800)
        except Exception as e:
            self._rows = []
            self.stat_lbl2.config(text=f"store error: {e}")
        self.lb.delete(0, "end")
        for r in self._rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(r["created_at"]))
            self.lb.insert("end", f"   {when}   {r['category']:<8}  {r['text'][:72].strip()}")
            self.lb.itemconfig(self.lb.size() - 1,
                               foreground=self.ROW_FG.get(r["category"], self.FG))

    def _on_double_click(self, _event=None):
        sel = list(self.lb.curselection())
        if sel:
            self._show_detail(self._rows[sel[0]])

    def _show_detail(self, row):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["created_at"]))
        win = ("  \u2022  window: " + row["window_title"]) if row.get("window_title") else ""
        self.detail.config(
            text=f"#{row['id']}   \u2022   {when}   \u2022   {row['category']}{win}\n\n{row['text'].strip()}",
            fg=self.FG)

    def _del_selected(self):
        rows = [self._rows[i] for i in self.lb.curselection()]
        if not rows:
            self._flash("nothing selected")
            return
        if not self._confirm("Delete selected memory?",
                             f"Delete {len(rows)} row(s) from local memory?",
                             yesno=True):
            return
        n = self.mem.delete_ids([r["id"] for r in rows])
        self._flash(f"deleted {n}")
        self.refresh()
        self._update_stat()

    def _prune(self):
        from memory_store import RETENTION_DAYS
        summary = ", ".join(f"{c}>={d}d" for c, d in RETENTION_DAYS.items())
        if not self._confirm("Prune old memory?",
                             f"Delete rows older than their retention "
                             f"({summary}). Chat rows age out fastest.",
                             yesno=True):
            return
        n = self.mem.prune()
        self._flash(f"pruned {n}")
        self.refresh()
        self._update_stat()

    def _export(self):
        import tkinter.filedialog as fd
        initial_dir = os.path.dirname(self.mem.path)
        default_name = "bozo_memory_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
        path = fd.asksaveasfilename(
            title="Export BoZo memory", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
            initialfile=default_name,
            initialdir=initial_dir, parent=self.win)
        if not path:
            return
        try:
            n = self.mem.export_json(path)
            self._flash(f"exported {n} rows to {os.path.basename(path)}")
        except Exception as e:
            self._flash(f"export failed: {e}")
            return
        try:
            import webbrowser as wb
            wb.open("file:///" + path.replace(os.sep, "/"))
        except Exception:
            pass

    def _del_all(self):
        import tkinter.simpledialog as sd
        try:
            st = self.mem.stats()
            n = st.get("total_memories", 0)
        except Exception:
            n = 0
        confirm = sd.askstring(
            "Danger \u2014 erase ALL memory",
            (f"Permanently delete ALL {n} memory rows AND ALL feedback "
             f"stored at:\n\n     {self.mem.path}\n\n"
             "This cannot be undone. Export first if unsure.\n\n"
             "Type exactly `yes` to confirm."),
            parent=self.win)
        if confirm != "yes":
            self._flash("aborted (typed something else)")
            return
        removed = self.mem.delete_all()
        self._flash(f"clean slate \u2014 {removed} rows removed")
        self.refresh()
        self._update_stat()

    # ---------------------------------------------------------------- helpers --
    def _confirm(self, title, body, yesno=True):
        import tkinter.messagebox as mb
        try:
            return mb.askyesno(title, body, parent=self.win) if yesno \
                else mb.showinfo(title, body, parent=self.win)
        except Exception:
            return True  # dialog subsystem broken shouldn't block a delete

    def _flash(self, text):
        self.stat_lbl2.config(text=text)
        if time.time() > self._flash_until:
            self._flash_until = time.time() + 4.0
            self.win.after(4200, lambda: self.stat_lbl2.config(text=""))

    def _update_stat(self):
        try:
            st = self.mem.stats()
            by = st.get("by_category") or {}
            cats = ", ".join(f"{c}:{n}" for c, n in sorted(by.items())) or "empty"
            kb = max(0, st.get("db_bytes", 0)) / 1024
            self.stat_lbl.config(text=(
                f"{st.get('total_memories', 0)} rows   \u00b7   {cats}   \u00b7   "
                f"{kb:,.0f} KB on disk   \u00b7   fts5: {yes if (yes:=st.get('fts5')) else 'fallback'}   \u00b7   "
                f"{self.mem.path}"))
        except Exception as e:
            self.stat_lbl.config(text=f"(stats unavailable: {e})")

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass


# --------------------------------------------------------------------- app ---
class CopilotApp:
    W, H = 400, 500

    def __init__(self):
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        self.root = tk.Tk()
        self.root.title("BoZo, The Watch Dog")
        try:  # window icon = the same logo tile
            from PIL import ImageTk
            self._win_icon = ImageTk.PhotoImage(_logo_tile(32))
            self.root.iconphoto(True, self._win_icon)
        except Exception:
            pass
        self.root.attributes("-topmost", True)
        self.sh = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}+12+{self.sh - self.H - 60}")
        self.root.configure(bg="#14161a")

        self.ui_q, self.cmd_q = queue.Queue(), queue.Queue()
        self.paused, self.look_now, self.stop = threading.Event(), threading.Event(), threading.Event()
        self.busy = threading.Lock()
        self.snooze_until = 0.0
        self.watcher = None
        self.ocr = None
        self.tray = None
        self.mask_rects = []
        self.recent_tips = collections.deque(maxlen=6)
        self.error_log = ErrorLog(capacity=200)   # Tier 1: per-session error recurrence
        self.feedback = collections.defaultdict(int)
        self.dwell_fired_at = 0.0                 # Timestamp of last dwell-trigger (prevents rapid-fire)
        self.dwell_compose_at = 0.0               # Timestamp of last compose-proactive offer
        self.dwell_agent_at = 0.0                 # Timestamp of last agent-proactive offer
        self.dwell_fired_for = set()              # sigs already dwell-triggered (avoid repeat same error)
        self.timeline = collections.deque(maxlen=12)
        self.cur_window, self.window_since = "", time.time()
        self.history = [{"role": "system", "content": CHAT_PROMPT}]
        self.last_anomalies = set()
        self.last_tip_sig = ""             # sig of the most recent spoken fix (feedback hook)

        # Tier 3: local, inspectable, deletable memory. The app must still run
        # if the store cannot be opened (locked file, weird path, ...).
        self.mem = None
        try:
            from memory_store import MemoryStore
            self.mem = MemoryStore()
        except Exception as e:
            log.warning("Memory store unavailable: %s", e)
        if self.mem is not None:
            try:
                recent = self.mem.recent_chat(10)
                if recent:
                    turns = "\n".join(
                        f"{'You' if t['role'] == 'me' else 'BoZo'}: {t['text'][:220]}"
                        for t in recent)
                    self.history[0]["content"] = (
                        CHAT_PROMPT +
                        "\n\nEarlier in your recent sessions (context only — do "
                        "not repeat these):\n" + turns)
            except Exception:
                log.exception("could not seed chat history from memory")

        self.toast = Toast(self.root, self._on_feedback)
        self.prepub = PrepublishCheck(self.root)
        self._build_ui()
        psutil.cpu_percent(interval=None)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._drain_queue)
        self.root.after(1000, self._refresh_status)
        self._start_tray()
        threading.Thread(target=self._startup_check, daemon=True).start()
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    # ---- UI ----
    def _build_ui(self):
        bg, fg = "#14161a", "#e6e6e6"

        # Top bar: the vision-model chooser (top-right). Pack first so it is
        # never clipped by the status bar or the chat log below it.
        top = tk.Frame(self.root, bg=bg)
        top.pack(side="top", fill="x", padx=8, pady=(6, 0))
        tk.Label(top, text="Vision model", bg=bg, fg="#9fd3ff",
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=(8, 0))
        self.vision_combo = ttk.Combobox(top, state="readonly", width=24,
                                         font=("Segoe UI", 10))
        self.vision_combo.pack(side="right")
        self.vision_combo.set(_current_vision_model())
        self.vision_combo.bind("<<ComboboxSelected>>", self._on_vision_model_change)

        self.status = tk.Label(self.root, text="", anchor="w", bg="#1d2026", fg="#9fd3ff",
                               font=("Consolas", 9), padx=8, pady=4, justify="left")
        self.status.pack(side="top", fill="x")

        # Input bar FIRST: a pack(fill=both, expand=True) sibling packed later
        # would starve it of height and clip the buttons out of view on any
        # window shorter than the default. Packed first with fill="x", the bar
        # always gets its full required height and the log takes the rest.
        bar = tk.Frame(self.root, bg=bg)
        bar.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self.entry = tk.Text(bar, bg="#1d2026", fg=fg, insertbackground=fg, bd=0,
                             font=("Segoe UI", 10), height=2, wrap="word")

        def _newline(_event=None):
            self.entry.insert("insert", "\n")
            return "break"

        # Ctrl+Enter inserts a line break; plain Enter sends.
        self.entry.bind("<Return>", self._on_send)
        self.entry.bind("<Control-Return>", _newline)
        self.entry.bind("<Control-KP_Enter>", _newline)

        # The three buttons pack first *inside the bar* so they reserve their
        # space; the entry then fills what's left. (Same starvation rule as
        # the root — a widget packed last is the one that can get clipped.)
        self.pause_btn = tk.Button(bar, text="Pause", command=self._toggle_pause, bd=0,
                                   bg="#2a2f38", fg=fg, padx=8)
        self.look_btn = tk.Button(bar, text="Look now", command=self.look_now.set, bd=0,
                                  bg="#2a2f38", fg=fg, padx=8)
        self.mem_btn = tk.Button(bar, text="Memory", command=self.open_memory, bd=0,
                                 bg="#2a2f38", fg=fg, padx=8)
        self.pause_btn.pack(side="right", padx=2)
        self.mem_btn.pack(side="right", padx=2)
        self.look_btn.pack(side="right", padx=2)
        self.entry.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=4)

        frame = tk.Frame(self.root, bg=bg)
        frame.pack(side="top", fill="both", expand=True)
        self.log = tk.Text(frame, wrap="word", bg=bg, fg=fg, bd=0, padx=10, pady=8,
                           font=("Segoe UI", 10), state="disabled")
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_config("tip", foreground="#e6e6e6")
        self.log.tag_config("warn", foreground="#ffb86b")
        self.log.tag_config("me", foreground="#8be28b")
        self.log.tag_config("sys", foreground="#7a8290", font=("Segoe UI", 9, "italic"))
        self.log.tag_config("stamp", foreground="#5c6370", font=("Segoe UI", 8))


    def post(self, text, tag="tip"):
        self.ui_q.put(("log", text, tag))

    def open_memory(self):
        """Open/close the Memory management window (browse / delete / prune / export)."""
        try:
            if self._memwin and self._memwin.win.winfo_exists():
                self._memwin.win.lift()
                self._memwin.win.attributes("-topmost", True)
                return
        except (tk.TclError, AttributeError):
            pass
        if self.mem is None:
            self.post("Memory store unavailable — see bozo.log for why.", "warn")
            return
        self._memwin = MemoryWindow(self.root, self.mem)

    def _populate_vision_combo(self, installed=None):
        """Refresh the dropdown with every installed Ollama model (sorted by
        name). Pass `installed` (a set of tags) to avoid a second /api/tags
        hit; the currently-selected model is kept if it is still installed."""
        current = self.vision_combo.get() or _current_vision_model()
        if installed is None:
            installed = set(ollama_models() or set())
        installed = sorted(installed)
        self.vision_combo["values"] = installed
        if current in installed:
            self.vision_combo.set(current)
        elif installed:
            self.vision_combo.set(installed[0])
        else:
            self.vision_combo.set("(Ollama offline)")

    def _populate_vision_combo_refresh(self):
        """Forced re-read: pull a fresh model list from Ollama, then repopulate.
        Reachable from the tray 'Refresh models' item (or `/models` in chat)
        after an `ollama pull`. The previous selection is kept if the server
        still lists it."""
        installed = refresh_models()   # hits /api/tags
        self._populate_vision_combo(installed)
        self.post("Model list refreshed from Ollama.", "sys")

    def _on_vision_model_change(self, _event=None):
        chosen = self.vision_combo.get().strip()
        if not chosen or chosen == "(Ollama offline)":
            return
        installed = set(ollama_models() or set())
        if chosen not in installed:
            self.post(
                f"\u26a0 {chosen} is not installed on this machine. "
                f"Ollama would 404 on /api/chat. Install it first `ollama pull {chosen.split(':')[0]}` "
                f"and pick it from the list.",
                "warn",
            )
            return
        current = _current_vision_model()
        if chosen == current:
            return
        _set_vision_model(chosen)
        resolved = model_tag(chosen)
        self.post(
            f"Vision model -> {resolved}. Whispers (auto + Look now) now use it. "
            f"Restart the app for the tray / GPU status line to pick it up.",
            "sys",
        )

    def whisper(self, text, topic, copyable=False):
        """Route a toast (optionally with a copyable payload) through the UI queue."""
        self.ui_q.put(("toast", text, topic, copyable))

    def _anchor_y(self):
        try:
            if self.root.state() == "normal":
                return self.root.winfo_y()
        except tk.TclError:
            pass
        return self.sh - 60

    def _drain_queue(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                # Backward-compat: older producers sent 3-tuples; new whisper
                # (draft/explain) sends a 4-tuple with a copyable flag.
                if len(item) == 4:
                    kind, text, tag, copyable = item
                else:
                    kind, text, tag = item
                    copyable = False
                if kind == "toast":
                    self.toast.show(text, tag, self._anchor_y(), copyable=copyable)
                    text, tag = text, ("warn" if tag in ("resources", "debugging", "explain") else "tip")
                self.log.configure(state="normal")
                self.log.insert("end", time.strftime("%H:%M  "), "stamp")
                self.log.insert("end", text + "\n\n", tag)
                self.log.configure(state="disabled")
                self.log.see("end")
        except queue.Empty:
            pass
        try:
            while True:
                self._handle_cmd(self.cmd_q.get_nowait())
        except queue.Empty:
            pass
        if not self.stop.is_set():
            self.root.after(150, self._drain_queue)

    def _handle_cmd(self, cmd):
        # Background threads may enqueue a *callable* to run on the main
        # thread (e.g. the startup populate, which touches tkinter). String
        # commands keep the existing dispatch below.
        if callable(cmd):
            cmd()
            return
        if cmd == "toggle_window":
            if self.root.state() == "withdrawn":
                self.root.deiconify()
            else:
                self.root.withdraw()
        elif cmd == "toggle_pause":
            self._toggle_pause()
        elif cmd == "snooze":
            if time.time() < self.snooze_until:
                self.snooze_until = 0
                self.post("Snooze cancelled.", "sys")
            else:
                self.snooze_until = time.time() + 30 * 60
                self.post("Snoozed for 30 minutes. Critical alerts still show.", "sys")
            self._sync_tray()
        elif cmd == "look":
            self.look_now.set()
        elif cmd == "prepub":
            self.prepub.open()
        elif cmd == "refresh_models":
            self._populate_vision_combo_refresh()
        elif cmd == "quit":
            self._quit()

    def _refresh_status(self):
        """Status bar tick. MUST never let an exception kill the reschedule
        chain — a single bad read (e.g. the drive behind `psutil` unavailable)
        must not freeze the bar forever."""
        try:
            s = read_stats()
            l1 = f"RAM {s['ram_used_gb']:.1f}/{s['ram_total_gb']:.0f} GB ({s['ram_pct']:.0f}%)   CPU {s['cpu_pct']:.0f}%"
            l2 = "VRAM n/a (see chat for why)"
            if s["gpus"]:
                g = s["gpus"][0]
                l2 = (f"VRAM {g['vram_used_gb']:.1f}/{g['vram_total_gb']:.0f} GB ({g['vram_pct']:.0f}%)"
                      f"   GPU {g['gpu_util_pct']}%")
            l2 += f"   [{self._state_name()}]"
            self.status.configure(text=f"{l1}\n{l2}")
        except Exception:
            log.exception("Status refresh failed")
            self.status.configure(text=f"Status unavailable — see {os.path.join(USER_DATA_DIR, 'bozo.log')}")
        self.mask_rects = [r for r in (self._own_rect(), self.toast.rect()) if r]
        if not self.stop.is_set():
            self.root.after(1000, self._refresh_status)

    def _own_rect(self):
        try:
            if self.root.state() == "normal":
                return (self.root.winfo_x(), self.root.winfo_y(),
                        self.root.winfo_width(), self.root.winfo_height())
        except tk.TclError:
            pass
        return None

    def _state_name(self):
        if self.paused.is_set():
            return "paused"
        if time.time() < self.snooze_until:
            return "snoozed"
        return "watching"

    def _toggle_pause(self):
        if self.paused.is_set():
            self.paused.clear()
            self.pause_btn.configure(text="Pause")
            self.post("Resumed watching your screen.", "sys")
        else:
            self.paused.set()
            self.pause_btn.configure(text="Resume")
            self.post("Paused. Automatic capture stopped — **Look now** still works as a manual override.", "sys")
        self._sync_tray()

    def _on_feedback(self, topic, val):
        self.feedback[topic] += val
        # Tier 3: feed the outcome back into memory so a "not helpful" on a
        # recurring error steers the next explain away from the same fix.
        sig = getattr(self, "last_tip_sig", "") 
        if sig and self.mem is not None:
            try:
                self.mem.record_feedback(sig, 1 if val > 0 else 0, f"toast:{topic}:{val}")
            except Exception:
                log.exception("memory: feedback record failed")
        note = "Got it, I'll whisper about that more." if val > 0 else \
               "Noted, I'll hold back on that unless it's urgent."
        self.post(note, "sys")

    def _on_close(self):
        if self.tray:
            self.root.withdraw()
        else:
            self._quit()

    def _quit(self):
        self.stop.set()
        if self.tray:
            try:
                self.tray.stop()
            except Exception:
                pass
        self.toast.close()
        if self.mem is not None:
            try:
                self.mem.close()
            except Exception:
                pass
        self.root.destroy()

    # ---- tray ----
    def _start_tray(self):
        try:
            import pystray
        except Exception:
            self.post("Tray icon unavailable (pip install pystray). Closing the window quits.", "sys")
            return
        put = self.cmd_q.put
        menu = pystray.Menu(
            pystray.MenuItem("Show / hide chat", lambda: put("toggle_window"), default=True),
            pystray.MenuItem(lambda i: "Resume watching" if self.paused.is_set() else "Pause watching",
                             lambda: put("toggle_pause")),
            pystray.MenuItem("Look now", lambda: put("look")),
            pystray.MenuItem("Pre-publish check", lambda: put("prepub")),
            pystray.MenuItem("Refresh models", lambda: put("refresh_models")),
            pystray.MenuItem(lambda i: "Cancel snooze" if time.time() < self.snooze_until
                             else "Snooze 30 min", lambda: put("snooze")),
            pystray.MenuItem("Quit", lambda: put("quit")),
        )
        try:
            self.tray = pystray.Icon("bozo-watchdog", make_icon("on"), "BoZo, The Watch Dog", menu)
            self.tray.run_detached()
        except Exception as e:
            self.tray = None
            self.post(f"Tray icon failed to start on this platform ({e}).", "sys")

    def _sync_tray(self):
        if not self.tray:
            return
        state = self._state_name()
        try:
            self.tray.icon = make_icon({"watching": "on"}.get(state, state))
            self.tray.title = f"BoZo, The Watch Dog ({state})"
        except Exception:
            pass

    # ---- startup ----
    def _startup_check(self):
        installed = ollama_models()
        def _enqueue_populate(installed=installed):
            # Runs on the UI thread (pumped by _drain_queue). tkinter objects
            # must only be touched from the main thread, so this is enqueued
            # rather than called directly (this runs on a background thread).
            self._populate_vision_combo(installed)
        if installed is None:
            self.post(f"Can't reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`.", "warn")
            self.cmd_q.put(_enqueue_populate)
            return
        refresh_models(installed)  # seed tag cache so model_tag() is ready pre-first-gate
        self.cmd_q.put(_enqueue_populate)
        vision = _current_vision_model()
        missing = [m for m in (GATE_MODEL, vision) if not has_model(installed, m)]
        if missing:
            self.post("Missing models:\n" + "\n".join(f"  ollama pull {m}" for m in missing), "warn")
        else:
            self.post(f"Ready. Gate: {model_tag(GATE_MODEL)}, vision: {model_tag(vision)}. I'll whisper when "
                      "I spot something useful, and you can ask me anything.", "sys")
        time.sleep(3)  # let the GPU sampler take its first reading
        if gpu_stats():
            g = gpu_stats()[0]
            self.post(f"GPU: {g['name']} ({g['vram_total_gb']:.0f} GB) via {GPU_BACKEND}.", "sys")
        else:
            self.post("No GPU stats.\n"
                      f"  pynvml: {GPU_ERROR or 'loaded but returned nothing'}\n"
                      "  nvidia-smi: not found / failed\n"
                      "Fix: run `python -m pip install nvidia-ml-py` using the SAME python "
                      "you launch app.py with, and check that `nvidia-smi` works in a terminal.",
                      "warn")

    # ---- context memory ----
    def _track_window(self, title, now):
        if title and title != self.cur_window:
            if self.cur_window:
                self.timeline.append((self.cur_window, now - self.window_since))
            self.cur_window, self.window_since = title, now

    def _compose_mode(self, text=""):
        """True when the active window looks like an email/message/chat/post
        composer. Uses the window title first; falls back to on-screen OCR
        markers (Gmail/Outlook/LinkedIn/Slack/WhatsApp compose affordances)."""
        win = (self.cur_window or "").lower()
        markers = (
            "compose", "new message", "write message", "draft",
            "send email", "new mail", "start a post", "new post",
            "reply", "message ", "chat with",
        )
        if any(m in win for m in markers):
            return True
        app_names = ("gmail", "outlook", "linkedin", "slack", "whatsapp",
                     "teams", "discord", "mail", "twitter", "facebook")
        if not any(a in win for a in app_names):
            return False
        # In a messaging app, require on-screen compose signals (avoids
        # drafting when the user is only reading someone else's message).
        t = (text or "").lower()
        return any(k in t for k in ("to:", "subject:", "cc:", "bcc:",
                                    "write a message", "add a message",
                                    "message body", "start a post",
                                    "what's it about", "write"))

    def _agent_mode(self, text=""):
        """True when the active window is an AI chat/agent surface and the
        user appears paused (reading the exchange, not actively iterating).
        Uses the window title; falls back to on-screen markers."""
        win = (self.cur_window or "").lower()
        hits = ("chatgpt", "gpt-4", "gpt-5", "chatgpt.com", "claude.ai",
                "claude-", "claude desktop", "openclaw", "opencode",
                "codex", "hermes", "copilot", "gemini", "cursor",
                "chatgpt ", "anthropic")
        if any(h in win for h in hits):
            return True
        # No agent in the title: require a strong on-screen marker (a named
        # model in the UI or a system/assistant prompt bar).
        t = (text or "").lower()
        return any(k in t for k in ("chatgpt", "claude-3", "claude-4", "claude 3.",
                                    "claude 4", "opencode", "openclaw",
                                    "codex", "anthropic"))

    def context_summary(self, now):
        def fmt(t, secs):
            return f"{t[:60]} ({secs / 60:.0f}m)" if secs >= 60 else f"{t[:60]} ({secs:.0f}s)"
        items = [fmt(t, s) for t, s in list(self.timeline)[-4:]]
        items.append(fmt(self.cur_window or "unknown", now - self.window_since) + " [current]")
        return " -> ".join(items)

    # ---- monitor loop ----
    def _monitor_loop(self):
        try:
            self.watcher = ScreenWatcher()
        except Exception as e:
            self.post(f"Screen capture unavailable: {e}", "warn")
            return
        self.ocr = OCR()
        self.post("OCR: " + (self.ocr.kind or "none (pip install rapidocr-onnxruntime)"), "sys")

        last_tip = last_analysis = 0.0
        pending_change, calm_ticks = False, 0
        last_err = 0                       # consecutive loop failures -> backoff
        compose_window_known = None   # window title whose compose state was last OCR-checked
        compose_next_check = 0.0      # don't re-OCR-check the same window more often than this
        while not self.stop.is_set():
            time.sleep(TICK_SECONDS)
            forced = self.look_now.is_set()
            now = time.time()
            snoozed = now < self.snooze_until
            if (self.paused.is_set() or snoozed) and not forced:
                # While snoozed we still watch system health for critical alerts.
                if snoozed and not self.paused.is_set():
                    crit = find_anomalies(read_stats())
                    new = set(crit) - self.last_anomalies
                    self.last_anomalies = set(crit)
                    if new and self.busy.acquire(blocking=False):
                        try:
                            self.watcher.grab(self.mask_rects)
                            self._analyze(read_stats(), crit, True, False, 0.0)
                        except Exception:
                            pass
                        finally:
                            self.busy.release()
                continue
            self.look_now.clear()
            try:
                change = self.watcher.grab(self.mask_rects)
                self._track_window(active_window_title(), now)
                if change >= SCREEN_CHANGE_PCT:
                    pending_change, calm_ticks = True, 0
                else:
                    calm_ticks += 1
                stats = read_stats()
                anomalies = find_anomalies(stats)
                new_anoms = set(anomalies) - self.last_anomalies
                self.last_anomalies = set(anomalies)

                settled = calm_ticks >= SETTLE_TICKS
                dwell_ok = (
                    calm_ticks * TICK_SECONDS >= DWELL_SECONDS
                    and now - self.dwell_fired_at >= DWELL_GAP
                    and now - last_tip >= MIN_TIP_GAP
                )
                # Compose-proactive: user paused mid-compose in a compose window.
                # Cheap check first (window title); OCR confirmation only if it
                # looks plausible and the rate limit allows.
                compose_due = False
                if (not new_anoms and not forced
                        and calm_ticks * TICK_SECONDS >= DWELL_COMPOSE
                        and now - self.dwell_compose_at >= 5 * 60
                        and now - last_tip >= MIN_TIP_GAP
                        and now >= compose_next_check):
                    if self._compose_mode():  # window-title path (free)
                        compose_due = True
                    else:
                        # title inconclusive: ask OCR once for this window.
                        ocr_sample = ""
                        try:
                            if self.ocr and self.ocr.kind and self.watcher.last_full:
                                ocr_sample = self.ocr.read(self.watcher.scaled(1280)) or ""
                                compose_window_known = self.cur_window
                        except Exception:
                            pass
                        if self._compose_mode(ocr_sample):
                            compose_due = True
                        else:
                            compose_next_check = now + 60  # this window: re-check in 1m
                # Agent co-pilot: user paused inside ChatGPT/Claude/agent chat —
                # offer a sharper prompt they can paste back.
                agent_due = False
                if (not new_anoms and not forced and not compose_due
                        and calm_ticks * TICK_SECONDS >= DWELL_AGENT
                        and now - self.dwell_agent_at >= 10 * 60
                        and now - last_tip >= MIN_TIP_GAP):
                    if self._agent_mode():
                        agent_due = True
                due = bool(new_anoms) or compose_due or agent_due or \
                    (pending_change and settled
                       and now - last_analysis >= MIN_ANALYSIS_GAP
                       and now - last_tip >= MIN_TIP_GAP) or \
                    (dwell_ok and now - last_analysis >= MIN_ANALYSIS_GAP
                       and now - last_tip >= MIN_TIP_GAP)
                if not due or not self.busy.acquire(blocking=False):
                    # Throttled "why is it silent?" trace: one line per 30 s of
                    # continuous non-firing so bozo.log explains the silence
                    # without spamming every 2.5 s tick.
                    if (not due and now - getattr(self, "_last_skip_log", 0.0) >= 30):
                        self._last_skip_log = now
                        busy_free = self.busy.acquire(blocking=False)
                        if busy_free:
                            self.busy.release()
                        if not busy_free:
                            reason = "busy_lock"
                        elif now - last_tip < MIN_TIP_GAP:
                            reason = "cooldown_tip"
                        elif now - last_analysis < MIN_ANALYSIS_GAP:
                            reason = "cooldown_anal"
                        elif not settled:
                            reason = "unsettled"
                        elif not new_anoms and not pending_change and not dwell_ok:
                            reason = "no_trigger"
                        else:
                            reason = "other"
                        self._log_decision("loop_skip", reason=reason,
                                           settled=bool(settled),
                                           calm_s=round(calm_ticks * TICK_SECONDS, 1),
                                           since_tip_s=round(now - last_tip, 1),
                                           since_anal_s=round(now - last_analysis, 1),
                                           dwell_s=dwell_ok,
                                           window=self.cur_window)
                    continue
                try:
                    last_analysis, pending_change = now, False
                    self._set_tray_busy(True)
                    if compose_due:
                        self.dwell_compose_at = time.time()
                        eff_intent = "compose"
                    elif agent_due:
                        self.dwell_agent_at = time.time()
                        eff_intent = "agent"
                    else:
                        eff_intent = None
                    spoke = self._analyze(stats, anomalies, bool(new_anoms),
                                          forced, change, calm_ticks * TICK_SECONDS,
                                          cooldown_ok=(now - last_tip >= MIN_TIP_GAP),
                                          forced_intent=eff_intent)
                    if spoke:
                        last_tip = time.time()
                        if dwell_ok:
                            self.dwell_fired_at = time.time()
                finally:
                    self._set_tray_busy(False)
                    self.busy.release()
            except requests.exceptions.ConnectionError:
                last_err += 1
                log.warning("Lost connection to Ollama (backoff %ss)",
                            min(10 * 2 ** last_err, 30))
                self.post("Lost connection to Ollama.", "warn")
                time.sleep(min(10 * 2 ** last_err, 30))
            except Exception as e:
                last_err += 1
                log.exception("Monitor loop error (backoff %ss)",
                              min(10 * 2 ** last_err, 30))
                self.post("Monitoring hit an internal error; see bozo.log for details.", "warn")
                time.sleep(min(10 * 2 ** last_err, 30))
            else:
                # Clean cycle — drop any accumulated backoff so a recovered
                # Ollama returns to the normal ~2.5 s cadence immediately.
                last_err = 0

    def _set_tray_busy(self, busy):
        if self.tray:
            try:
                self.tray.icon = make_icon("busy" if busy else
                                           {"watching": "on"}.get(self._state_name(), self._state_name()))
            except Exception:
                pass

    def _tally_errors(self, text, errors):
        """Tier 1 — split current OCR errors into *first sightings this session*
        (triggers the whisper, same as before) vs *repeats* (no automatic
        re-trigger, but the state dict carries the count so the gate model can
        still choose to speak, and the whisper prompt can avoid repeating a
        previously-tried fix).

        Returns
          (errors, new_errors, recurrence)
            errors       — all current OCR error lines (unchanged from find_errors)
            new_errors   — first-sighting-only; used for trigger semantics (bypass gate)
            recurrence   — {sig: {previous_sightings, seconds_since_first_seen, ...}}
                           for errors that have recurred in this session
        """
        new_errors, recurrence = [], {}
        for e in errors:
            k = sig(e)
            rec = self.error_log.add(k, e)
            if rec["is_new_to_session"]:
                new_errors.append(e)
            prev = max(0, rec["count"] - 1)
            if prev >= 1:
                f = self.error_log.get(k)
                recurrence[k] = {
                    "previous_sightings": prev,
                    "seconds_since_first_seen": round(time.time() - f["first_seen"], 1),
                    "text": e[:160],
                }
        return errors, new_errors, recurrence

    # ---- diagnostics ----
    def _log_decision(self, stage, **kw):
        """Single funnel for every auto-path decision (speak or stay silent).
        Written to bozo.log at INFO so "why did it not speak?" is a grep away
        next time. Format: decision=<stage> outcome=<spoke|silent> k=v ...
        NOTE: first arg is `stage` (not `reason`) so callers may pass
        reason=... in kw without a collision."""
        fields = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
        outcome = "spoke" if stage == "spoke" else "silent"
        log.info("decision=%s outcome=%s %s", stage, outcome, fields.strip())

    def _analyze(self, stats, anomalies, urgent_anomaly, forced, change,
                 calm_seconds=0.0, cooldown_ok=True, forced_intent=None):
        now = time.time()
        # 1. Read the screen. This is what lets text-only nimble "see" errors.
        text = ""
        if self.ocr and self.ocr.kind and self.watcher.last_full is not None:
            text = self.ocr.read(self.watcher.scaled(1920))[:OCR_MAX_CHARS * 3]
        raw_ocr_chars = len(text)
        text = strip_internal_noise(text)
        errors = find_errors(text)
        # Tier 1: split first sightings (trigger) from repeats (context).
        errors, new_errors, recurrence = self._tally_errors(text, errors)
        # Tier 3: remember each first-sighting error (keyed by signature) so a
        # repeat of the *same* bug is recognised across sessions.
        if self.mem is not None and new_errors:
            for e in new_errors:
                try:
                    self.mem.record_error(sig(e), e, self.cur_window)
                except Exception:
                    log.exception("memory: failed to record error")
        state = {
            "active_window": self.cur_window,
            "recent_activity": self.context_summary(now),
            "screen_calm_seconds": round(calm_seconds, 1),
            "error_lines_on_screen": errors or "none",
            "new_error_detected": bool(new_errors),
            "error_recurrence": recurrence or "none this session",
            "screen_text_excerpt": text[:600],
            "ocr_chars_before_noise_filter": raw_ocr_chars,
            "ocr_chars_after_noise_filter": len(text),
            "system_alerts": anomalies or "none",
            "cpu_pct": stats["cpu_pct"], "ram_pct": stats["ram_pct"],
            "vram_pct": [g["vram_pct"] for g in stats["gpus"]] if stats["gpus"] else None,
            "recent_tips": list(self.recent_tips)[-4:],
            "tip_feedback": dict(self.feedback) or "none yet",
        }

        # 2. Is NOW the right moment?
        # forced_intent="explain" (triggered by a fresh error or the forced
        # explain path) or a real error/anomaly skip the gate and force the
        # explain-style delivery.
        explain_intent = (forced_intent == "explain"
                          or (forced and bool(new_errors)))
        agent_intent = forced_intent == "agent"
        topic = "explain" if explain_intent else \
                ("agent" if agent_intent else
                 ("resources" if urgent_anomaly else
                  ("compose" if forced_intent == "compose" else "workflow")))
        bypass = forced or urgent_anomaly or bool(new_errors) \
                or forced_intent in ("explain", "compose", "agent")
        if not bypass:
            if not cooldown_ok:
                self._log_decision("cooldown", calm_s=round(calm_seconds, 1))
                return False
            g = nimble_gate(state)
            if g is not None and forced_intent not in ("compose", "explain", "agent"):
                # Proactive mode: if nimble says "nothing" but its own speak
                # score meets the threshold, trust the score and default to a
                # generic topic instead of balking out.
                if g["topic"] == "nothing":
                    self._log_decision("gate_topic_nothing", speak=round(g["speak"], 3))
                    return False
                topic = g["topic"] if g["topic"] else "workflow"
                if g["speak"] < GATE_THRESHOLD:
                    self._log_decision("gate_below_threshold",
                                       speak=round(g["speak"], 3),
                                       threshold=GATE_THRESHOLD,
                                       topic=g["topic"], errors=len(errors),
                                       window=self.cur_window)
                    return False
                # flow / user-feedback gates skipped for intent-driven calls
                if forced_intent not in ("compose", "explain", "agent"):
                    if g["flow"] > FLOW_BLOCK and g["urgency"] < 1.5:
                        self._log_decision("flow_block",
                                           flow=round(g["flow"], 3),
                                           urgency=round(g["urgency"], 2),
                                           speak=round(g["speak"], 3))
                        return False  # user is in flow; hold the thought
                    if self.feedback[topic] <= -2 and g["urgency"] < 1.5:
                        self._log_decision("user_feedback_waved_off",
                                           topic=topic,
                                           feedback=self.feedback[topic])
                        return False  # user keeps waving this topic off

        # 3. Ask the big model, with pixels + text + context + stats.
        full = read_stats(with_procs=True)
        recurrence_note = ""
        if recurrence:
            lines = [
                f"  • {v['text']}  (seen {v['previous_sightings']}x earlier this session, "
                f"first {v['seconds_since_first_seen']}s ago)"
                for v in recurrence.values()
            ]
            recurrence_note = ("This error has recurred this session. Earlier tips "
                               "about it are in recent_tips — propose the NEXT likely "
                               "cause or a different concrete fix, not the same one:\n"
                               + "\n".join(lines) + "\n")
        # Tier 3: self-healing + learned context for this error.
        memory_note = ""
        if explain_intent and self.mem is not None:
            try:
                seen_sigs = [sig(e) for e in new_errors[:6]] or \
                            [sig(e) for e in errors[:6]]
                ctx = self.mem.format_context(
                    self.mem.recall(" ".join(errors[:6]) or self.cur_window,
                                    sigs=seen_sigs, window_title=self.cur_window, k=4,
                                    categories=["error", "tip", "chat_bozo"]))
                if ctx:
                    memory_note += ("\n" + ctx + "\n")
                for s in seen_sigs[:2]:
                    h = self.mem.history_for_sig(s)
                    if h and h["not_yet"] >= 1:
                        memory_note += (f"NOTE: this exact error already came back after a "
                                        f"previous fix ({h['not_yet']}× marked 'not it'). "
                                        f"Do NOT repeat the obvious first suggestion — "
                                        f"propose a different, more likely root cause.\n")
            except Exception:
                log.exception("memory: recall for explain failed")
        # Pick the right system prompt for the mode.
        if explain_intent:
            sys_prompt = EXPLAIN_SYSTEM
            prompt = (f"ACTIVE WINDOW: {self.cur_window}\n"
                      f"RECENT ACTIVITY: {self.context_summary(now)}\n"
                      f"ERROR-LOOKING LINES (OCR): {errors or 'none found — read the screenshot'}\n"
                      f"{recurrence_note}{memory_note}"
                      f"SCREEN TEXT (OCR, may be noisy):\n{text[:OCR_MAX_CHARS]}\n"
                      f"Live stats: {json.dumps(full)}\n"
                      f"Already said recently (don't repeat): {list(self.recent_tips)[-4:]}\n"
                      "Explain the visible bug: give the LIKELY ROOT CAUSE in one or two "
                      "sentences in plain terms, then the CONCRETE FIX (the exact change "
                      "or command). If several causes are plausible, rank them. Do not "
                      "reply NOTHING.")
        elif topic == "compose" or forced_intent == "compose":
            sys_prompt = COMPOSE_SYS
            prompt = (f"ACTIVE WINDOW: {self.cur_window}\n"
                      f"RECENT ACTIVITY: {self.context_summary(now)}\n"
                      f"SCREEN TEXT (OCR, may be noisy, includes recipient/subject if "
                      f"visible):\n{text[:OCR_MAX_CHARS]}\n"
                      f"Live stats: {json.dumps(full)}\n"
                      f"Already said recently (don't repeat): {list(self.recent_tips)[-4:]}\n"
                      "Draft the message: complete the partial text in the user's own "
                      "voice. Include a Subject if it's an email and none is visible. "
                      "Do not invent recipients, facts, or stats. If the screen has no "
                      "actual partial text to compose from, say 'Nothing to draft yet.'")
        elif agent_intent:
            sys_prompt = AGENT_SYS
            prompt = (f"ACTIVE WINDOW: {self.cur_window}\n"
                      f"RECENT ACTIVITY: {self.context_summary(now)}\n"
                      f"SCREEN TEXT (OCR, may be noisy — this is the conversation/task "
                      f"in the agent, incl. the user's last prompt):\n{text[:OCR_MAX_CHARS]}\n"
                      f"Already offered recently (don't repeat): {list(self.recent_tips)[-4:]}\n"
                      "The user is paused mid-task in an AI agent. Read what they are "
                      "trying to achieve, then offer ONE better prompt they can copy and "
                      "paste to drive the work further. If the screen shows no task "
                      "underway, say 'Nothing to suggest yet.'")
        else:
            sys_prompt = WHISPER_PROMPT
            prompt = (f"Likely focus: {topic}.\n"
                      f"Active window: {self.cur_window}\n"
                      f"Recent activity: {self.context_summary(now)}\n"
                      f"System alerts: {anomalies or 'none'}\n"
                      f"Error-looking lines (OCR): {errors or 'none'}\n"
                      f"{recurrence_note}"
                      f"Screen text (OCR, may be noisy):\n{text[:OCR_MAX_CHARS]}\n"
                      f"Live stats: {json.dumps(full)}\n"
                      f"Already said recently (don't repeat): {list(self.recent_tips)[-4:]}\n")
            if not new_errors and not urgent_anomaly and not forced:
                # Gate-driven proactive tip: nudge the big model out of its
                # default "there's nothing to say" posture (qwen refuses to
                # tip on clean screens unless explicitly told to do so).
                prompt += ("PROACTIVE MODE: this tip was not triggered by an error. "
                           "Offer ONE concrete, specific tip or improvement the user is "
                           "most likely to appreciate (workflow shortcut, code "
                           "improvement, or research pointer). Do not reply NOTHING.")
        # The screenshot itself can contain an internal warning popup. Tell the vision
        # model not to treat BoZo's own diagnostics as the user's task.
        prompt += ("\n\nRELEVANCE / SELF-NOISE RULE: Ignore BoZo/Screen Copilot UI, "
                   "agent diagnostics, classifier or embedding warnings, and model "
                   "configuration notices. Do not repeat or explain them. Base advice "
                   "on the user's underlying app/task. If no specific useful action "
                   "is supported, return NOTHING.")
        reply = qwen_chat([{"role": "system", "content": sys_prompt},
                           {"role": "user", "content": prompt}],
                          image_b64=self.watcher.jpeg_b64())
        reject, reject_reason = is_internal_or_duplicate_reply(
            reply, list(self.recent_tips)[-6:])
        if reject:
            self._log_decision("reply_filtered", filter_reason=reject_reason,
                               reply_len=len(reply or ""), topic=topic)
            return False
        if not reply or _is_nothing(reply):
            self._log_decision("nothing_swallowed",
                               reply_len=len(reply or ""),
                               forced=bool(forced), forced_intent=forced_intent,
                               topic=topic, errors=len(errors),
                               recent_tips=len(self.recent_tips))
            if forced or forced_intent:
                self.post("Nothing notable on screen right now.", "sys")
            return False
        self._log_decision("spoke", topic=topic, reply_len=len(reply),
                           errors=len(errors), forced=bool(forced),
                           recent_tips=len(self.recent_tips))
        self.recent_tips.append(reply[:120])
        # Tier 3: remember what was said, keyed by the error signature if this
        # was an error explain (enables cross-session 'did the fix work?').
        tip_sig = sig(new_errors[0]) if new_errors else ""
        self.last_tip_sig = tip_sig
        if self.mem is not None:
            try:
                self.mem.record_tip(tip_sig, reply[:400], self.cur_window)
            except Exception:
                log.exception("memory: failed to record tip")
        # Drafts, bug explanations and prompt offers are copy-ables: big card
        # with a COPY button so the user can paste them into the app.
        copyable = (explain_intent or agent_intent
                    or topic in ("compose", "agent", "explain")
                    or forced_intent in ("explain", "compose", "agent"))
        self.whisper(reply, topic, copyable=copyable)
        return True

    # ---- user chat ----
    def _on_send(self, _event=None):
        text = self.entry.get("1.0", "end-1c").strip()
        if not text:
            return "break"
        self.entry.delete("1.0", "end")
        self.post(text, "me")
        if self.mem is not None:
            try:
                self.mem.record_chat("me", text, self.cur_window)
            except Exception:
                log.exception("memory: failed to record user chat")
        threading.Thread(target=self._chat_worker, args=(text,), daemon=True).start()
        return "break"  # stop Tk's default newline insertion in the Text widget

    def _chat_worker(self, text):
        low = text.lower().strip()
        if low in ("check", "/check", "prepub", "/prepub", "check my draft"):
            # Must open on the UI thread (worker thread is this one); queue it.
            self.cmd_q.put("prepub")
            return
        if low in ("models", "/models", "refresh models", "refresh-models"):
            # Re-read the installed model list so a fresh `ollama pull` shows up
            # in the top-right dropdown without a restart.
            self.cmd_q.put("refresh_models")
            return
        with self.busy:
            try:
                live = self.watcher is not None and not self.paused.is_set()
                img = self.watcher.jpeg_b64() if live else None  # latest frame from the loop
                stats = json.dumps(read_stats(with_procs=True))
                # Tier 3: inject relevant local memory (past fixes, related chat)
                # as soft context for this question.
                ctx = ""
                if self.mem is not None:
                    try:
                        ctx = self.mem.format_context(
                            self.mem.recall(text, window_title=self.cur_window, k=5))
                    except Exception:
                        log.exception("memory: recall for chat failed")
                user_ctx = f"{text}\n\n{ctx}" if ctx else text
                self.history.append({"role": "user",
                                     "content": f"{user_ctx}\n\n[live stats: {stats}]"})
                reply = qwen_chat(self.history, image_b64=img)
                self.history[-1]["content"] = text  # keep history light
                self.history.append({"role": "assistant", "content": reply})
                if self.mem is not None:
                    try:
                        self.mem.record_chat("bozo", reply, self.cur_window)
                    except Exception:
                        log.exception("memory: failed to record assistant chat")
                # Trim: keep system message + last 12 turns, only if long enough
                # that we actually need trimming (avoids duplicating the
                # system message when history is short).
                if len(self.history) > 13:
                    self.history = [self.history[0]] + self.history[-12:]
                self.post(reply)
            except Exception as e:
                self.post(f"Chat failed: {e}", "warn")

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            # Ctrl+C: don't just unwind out of Tk's mainloop (that skips
            # tray.stop() and can strand a ghost tray icon); go through the
            # same clean shutdown the window's X button uses.
            print("\nShutting down (Ctrl+C)…")
            try:
                self._quit()
            except Exception:
                # Worst case a partially-torn-down Tcl state: cleanup of
                # stop flag / threads / tray has already had its chance.
                pass

if __name__ == "__main__":
    CopilotApp().run()
