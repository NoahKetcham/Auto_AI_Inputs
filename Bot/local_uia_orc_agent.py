"""
Improved Local UIA Agent - Desktop Automation with MiniMax M2.5

Improvements over original:
- Robust API retry logic (handles empty responses)
- Multiple JSON extraction fallbacks
- Environment variable config (no hardcoded keys)
- File logging + console output
- State persistence (resume after crash)
- Better error handling
- Configurable via config.json
"""

import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pyautogui
import mss
from PIL import Image

# Optional deps (agent will still run if missing)
try:
    import yaml
except ImportError:
    yaml = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import uiautomation as auto
except Exception:
    auto = None

from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------

CONFIG_FILE = Path(__file__).parent / "config.json"
STATE_FILE = Path(__file__).parent / "agent_state.json"
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Default config
DEFAULT_CONFIG = {
    "model_name": "MiniMax-M2.5",
    "api_key_env": "OPENAI_API_KEY",  # Set this env var instead of hardcoding
    "api_base_url": "https://api.minimax.io/v1",
    "loop_delay_sec": 0.25,
    "action_pause_sec": 0.06,
    "confirmation_capture_delay_sec": 0.08,
    "max_actions_per_turn": 5,  # More actions per turn for faster task completion
    "max_tokens": 800,  # Allow longer responses for multi-step planning
    "kill_xy_threshold": 8,
    "disable_typing_if_password_likely": True,
    "ocr_enabled": False,  # Set True if tesseract installed (pip install pytesseract + Tesseract binary)
    "uia_enabled": True,  # Auto-enabled when uiautomation available; set False to disable
    "uia_max_lines": 120,  # Richer UI tree for better context
    "uia_max_children": 16,  # More children per node
    "uia_max_depth": 5,  # Deeper tree traversal
    "uia_include_bounds": True,  # Include element coordinates when available
    "confirmation_log_enabled": True,  # Log per-iteration outcomes to reduce looping
    "confirmation_log_max_entries": 8,  # How many confirmation entries to include in prompt
    "loop_warning_threshold": 3,  # Warn if this many consecutive iterations had no state change
    "cursor_overlay_enabled": True,  # Show bright red overlay cursor for visibility
    "cursor_overlay_size": 48,  # Window size in pixels (larger = more visible)
    "cursor_overlay_color": "#FF0000",  # Bright red
    "overlay_init_sec": 0.15,
    "ocr_target_width": 960,  # Smaller = faster OCR (default 1280)
    "reflection_enabled": True,  # Call API to reflect when stuck (loop) and improve next action
    "reflection_events_file": "agent_events.jsonl",  # Log events + reflections for analysis
    "log_level": "INFO",
    "log_file": "agent.log",
}


def load_config() -> Dict[str, Any]:
    """Load config from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save config to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def load_settings() -> Dict[str, Any]:
    """Load API keys and env overrides from settings.yaml. Safe to commit settings.example.yaml."""
    if not SETTINGS_FILE.exists():
        return {}
    if yaml is None:
        logger.warning("PyYAML not installed. Run: pip install pyyaml. Using env vars only.")
        return {}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load settings.yaml: {e}")
        return {}


def get_provider() -> Tuple[str, str, str, str]:
    """Resolve active provider: (api_key, api_base_url, model_name, provider_label)."""
    providers = SETTINGS.get("providers") or {}
    active = SETTINGS.get("active", "minimax")
    provider = providers.get(active)
    
    if provider:
        api_key = provider.get("api_key") or os.environ.get(
            provider.get("api_key_env", f"{active.upper()}_API_KEY")
        )
        if api_key:
            base = provider.get("api_base_url", "")
            model = provider.get("model_name", CONFIG.get("model_name", "MiniMax-M2.5"))
            return (api_key, base, model, active)
    
    # Fallback: legacy flat api_key / api_base_url
    api_key = SETTINGS.get("api_key") or os.environ.get(CONFIG.get("api_key_env", "OPENAI_API_KEY"))
    if api_key:
        base = SETTINGS.get("api_base_url") or CONFIG.get("api_base_url", "https://api.minimax.io/v1")
        model = CONFIG.get("model_name", "MiniMax-M2.5")
        return (api_key, base, model, "legacy")
    
    print("No API key found. Either:")
    print("  1. Copy Bot/settings.example.yaml to Bot/settings.yaml")
    print("  2. Set active provider and add api_key under providers.<name>")
    print("  3. Or set api_key at top level (legacy)")
    sys.exit(1)


# Load config
CONFIG = load_config()
SETTINGS = load_settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------
# State Persistence
# ----------------------------

@dataclass
class AgentState:
    goal: str = ""
    start_time: float = 0
    iteration: int = 0
    actions_executed: List[Dict] = None
    confirmation_log: List[Dict] = None  # Per-iteration outcomes for loop avoidance
    
    def __post_init__(self):
        if self.actions_executed is None:
            self.actions_executed = []
        if self.confirmation_log is None:
            self.confirmation_log = []


def load_state() -> Optional[AgentState]:
    """Load persisted state if exists."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            if "confirmation_log" not in data:
                data["confirmation_log"] = []
            return AgentState(**data)
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
    return None


def save_state(state: AgentState) -> None:
    """Persist state to file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(asdict(state), f, indent=2)
    except Exception as e:
        logger.error(f"Could not save state: {e}")


def clear_state() -> None:
    """Remove state file."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ----------------------------
# Data Models
# ----------------------------

@dataclass
class ScreenState:
    screen_w: int
    screen_h: int
    cursor_x: int
    cursor_y: int
    active_window: str
    uia_text: str
    ocr_text: str


def state_fingerprint(state: ScreenState, max_chars: int = 400) -> str:
    """Compact fingerprint for change detection (window + key UI content)."""
    content = (state.uia_text or "")[:max_chars] + "|" + (state.ocr_text or "")[:max_chars]
    h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
    return f"{state.active_window}|{h}"


def build_confirmation_entry(
    iteration: int,
    actions: List[Dict],
    state_before: ScreenState,
    state_after: ScreenState,
) -> Dict[str, Any]:
    """Build a confirmation log entry showing whether actions produced visible change."""
    fp_before = state_fingerprint(state_before)
    fp_after = state_fingerprint(state_after)
    changed = fp_before != fp_after
    action_summaries = []
    for a in actions:
        t = a.get("type", "?")
        s = t
        if a.get("keys"):
            s += f" {a.get('keys')}"
        if a.get("text"):
            s += f" '{str(a.get('text', ''))[:25]}...'"
        if a.get("x") is not None:
            s += f" @({a.get('x')},{a.get('y')})"
        action_summaries.append(s)
    return {
        "iteration": iteration,
        "actions": action_summaries,
        "window_before": state_before.active_window,
        "window_after": state_after.active_window,
        "state_changed": changed,
        "outcome": "changed" if changed else "no_change",
    }


def detect_loop_warning(confirmation_log: List[Dict], threshold: int = 3) -> Optional[str]:
    """Return warning string if last N iterations had no state change."""
    if not confirmation_log or len(confirmation_log) < threshold:
        return None
    recent = confirmation_log[-threshold:]
    if all(entry.get("state_changed") is False for entry in recent):
        return (
            f"LOOP WARNING: Last {threshold} iterations produced NO visible change. "
            "Try a different approach (e.g. different keys, click elsewhere, or wait longer)."
        )
    return None


# ----------------------------
# Safety
# ----------------------------

def kill_switch_triggered() -> bool:
    x, y = pyautogui.position()
    return x <= CONFIG["kill_xy_threshold"] and y <= CONFIG["kill_xy_threshold"]


# ----------------------------
# Cursor Overlay (visible red indicator)
# ----------------------------

_cursor_overlay_stop = threading.Event()


def _run_cursor_overlay() -> None:
    """Run a bright red overlay circle that follows the cursor. Uses tkinter."""
    try:
        import tkinter as tk
    except ImportError:
        logger.warning("tkinter not available, cursor overlay disabled")
        return
    
    size = CONFIG.get("cursor_overlay_size", 48)
    color = CONFIG.get("cursor_overlay_color", "#FF0000")
    half = size // 2
    # Circle fills most of window with small margin
    pad = 4
    r = half - pad
    
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    # Transparent background so only the red circle shows
    try:
        root.wm_attributes("-transparentcolor", "#010101")
    except Exception:
        pass  # Some platforms don't support this
    
    root.geometry(f"{size}x{size}+0+0")
    root.configure(bg="#010101")
    
    canvas = tk.Canvas(root, width=size, height=size, bg="#010101", highlightthickness=0)
    canvas.pack()
    # Bright red circle, slightly larger than standard cursor
    canvas.create_oval(pad, pad, size - pad, size - pad, fill=color, outline=color, width=2)
    
    def update_pos() -> None:
        if _cursor_overlay_stop.is_set():
            root.quit()
            root.destroy()
            return
        try:
            x, y = pyautogui.position()
            root.geometry(f"+{x - half}+{y - half}")
        except Exception:
            pass
        root.after(25, update_pos)
    
    root.after(25, update_pos)
    root.deiconify()
    
    try:
        root.mainloop()
    except Exception:
        pass


def start_cursor_overlay() -> Optional[threading.Thread]:
    """Start the cursor overlay in a daemon thread. Returns the thread or None."""
    if not CONFIG.get("cursor_overlay_enabled", False):
        return None
    _cursor_overlay_stop.clear()
    t = threading.Thread(target=_run_cursor_overlay, daemon=True)
    t.start()
    time.sleep(CONFIG.get("overlay_init_sec", 0.15))
    return t


def stop_cursor_overlay() -> None:
    """Signal the cursor overlay to stop."""
    _cursor_overlay_stop.set()


def likely_sensitive_context(state: ScreenState) -> bool:
    if not CONFIG["disable_typing_if_password_likely"]:
        return False
    blob = f"{state.active_window}\n{state.uia_text}\n{state.ocr_text}"
    return bool(re.search(r"(password|passcode|2fa|otp|verification)", blob, re.IGNORECASE))


# ----------------------------
# Screen Capture
# ----------------------------

def capture_screen() -> Image.Image:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def get_active_window_title() -> str:
    if auto is None:
        return "Unknown"
    try:
        win = auto.GetForegroundControl()
        name = (win.Name or "").strip()
        cls = (win.ClassName or "").strip()
        return f"{name} [{cls}]".strip()
    except Exception:
        return "Unknown"


def extract_uia_text() -> str:
    if not CONFIG["uia_enabled"] or auto is None:
        return ""
    try:
        root = auto.GetForegroundControl()
        lines: List[str] = []
        q = [(root, 0)]
        max_lines = CONFIG.get("uia_max_lines", 80)
        max_children = CONFIG.get("uia_max_children", 12)
        max_depth = CONFIG.get("uia_max_depth", 4)
        include_bounds = CONFIG.get("uia_include_bounds", True)
        
        while q and len(lines) < max_lines:
            ctrl, depth = q.pop(0)
            name = (ctrl.Name or "").strip()
            ctype = str(ctrl.ControlTypeName or "").strip()
            val = ""
            try:
                val = (ctrl.GetValuePattern().Value or "").strip()
            except Exception:
                pass
            
            # Include bounding rect for clickable elements when available
            bounds_str = ""
            if include_bounds and (name or ctype) and depth > 0:
                try:
                    rect = ctrl.BoundingRectangle
                    if rect is not None:
                        w = rect.right - rect.left
                        h = rect.bottom - rect.top
                        if w > 2 and h > 2:
                            cx = (rect.left + rect.right) // 2
                            cy = (rect.top + rect.bottom) // 2
                            bounds_str = f" @({cx},{cy})"
                except Exception:
                    pass
            
            text_bits = [b for b in [ctype, name, val] if b]
            if text_bits:
                indent = "  " * min(depth, 6)
                lines.append(f"{indent}- " + " | ".join(text_bits) + bounds_str)
            
            if depth < max_depth:
                try:
                    children = ctrl.GetChildren()
                except Exception:
                    children = []
                for ch in children[:max_children]:
                    q.append((ch, depth + 1))
        
        return "\n".join(lines)
    except Exception:
        return ""


def extract_ocr_text(img: Image.Image) -> str:
    if not CONFIG["ocr_enabled"] or pytesseract is None:
        return ""
    try:
        # Set tesseract path if specified
        if "tesseract_cmd" in CONFIG:
            pytesseract.pytesseract.tesseract_cmd = CONFIG["tesseract_cmd"]
        
        # Downscale for speed
        w, h = img.size
        target_w = CONFIG.get("ocr_target_width", 960)
        if w > target_w:
            new_h = int(h * (target_w / w))
            img = img.resize((target_w, new_h))
        
        text = pytesseract.image_to_string(img)
        text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
        return text[:2500]
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ----------------------------
# MiniMax API with Robust Parsing
# ----------------------------

SYSTEM_PROMPT = """You are a persistent computer-control agent. Your job is to COMPLETE the user's goal from start to finish.

CONTEXT you receive:
- Active window title, UIA tree (element names/types), OCR text
- RECENT ACTIONS - what you already did (use this to avoid repeating, and to plan next steps)
- CONFIRMATION LOG - whether your last actions changed the screen (✓ changed / ✗ no_change)
- LOOP WARNING - if shown, your recent actions produced no visible change; try a different approach
- Cursor position, screen size

TASK COMPLETION - Drive to the end:
- Break the goal into steps and execute them sequentially
- Don't stop early - keep going until the goal is achieved
- Use action history to know where you are in the workflow
- Only choose "done" when: (a) goal is fully achieved, or (b) you need user input (e.g. password)

KEYBOARD-FIRST (most reliable when context is limited):
- Win = Start menu, type to search, Enter to open
- Ctrl+L = address bar, Ctrl+T = new tab, Ctrl+W = close tab
- Tab/Shift+Tab = navigate, Enter = click/select, Space = toggle/scroll
- Arrow keys = navigation, Escape = cancel/close
- Alt+Tab = switch windows, Alt+F4 = close window

MOUSE when you have coordinates from UIA/OCR or can infer from layout.

AVAILABLE ACTIONS (return ONLY valid JSON):
{
  "goal_summary": "string",
  "risk_flags": ["string"],
  "actions": [
    {
      "type": "move|click|double_click|right_click|middle_click|drag|scroll|type|hotkey|press|key_down|key_up|wait|done",
      "x": 0, "y": 0,
      "button": "left|right|middle",
      "scroll_amount": -500,
      "text": "string",
      "keys": ["ctrl","shift","s"],
      "key": "shift",
      "wait_ms": 500
    }
  ]
}

Action types:
- move: move cursor to x,y
- click, double_click, right_click, middle_click: move to x,y then click
- drag: click at current pos, drag to x,y (use for drag-and-drop)
- scroll: scroll_amount positive=up, negative=down
- type: type text (no passwords)
- hotkey: press key combo, e.g. ["ctrl","c"]
- press: single key, e.g. "enter", "tab", "escape"
- key_down/key_up: hold/release modifier (use with key: "ctrl" etc)
- wait: pause wait_ms milliseconds
- done: stop (only when goal achieved or user input needed)

SITE-SPECIFIC - Search results & YouTube:
- On YouTube/search results: the FIRST item is often an AD or external link. Do NOT click it.
- Skip ads: (1) scroll down 2-3 items, then click a video, OR (2) press Tab 2-4 times then Enter to select a lower result.
- Real video results show: duration (e.g. "10:25"), channel name, view count. Ads show "Ad" or "Sponsored".
- If UIA/OCR shows "Ad" or "Sponsored" on the first item, scroll down or Tab past it before selecting.
- When playing a YouTube search result: scroll down first, then click a result with a duration (not an ad).

Rules:
- Prefer keyboard over mouse when context is limited
- Never type passwords or sensitive info
- Keep coordinates within screen bounds
- Plan 3-5 actions per turn to make progress efficiently"""


def log_event(event: Dict[str, Any]) -> None:
    """Append a structured event to the events log (JSONL)."""
    events_path = Path(__file__).parent / CONFIG.get("reflection_events_file", "agent_events.jsonl")
    try:
        event["ts"] = datetime.now().isoformat()
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Could not write event log: {e}")


def call_reflection(
    client: OpenAI,
    user_goal: str,
    confirmation_log: List[Dict],
    state: ScreenState,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Call the model to reflect on failed attempts and suggest a better approach."""
    if not confirmation_log:
        return None
    recent = confirmation_log[-6:]
    lines = []
    for e in recent:
        outcome = "changed" if e.get("state_changed") else "NO CHANGE"
        acts = "; ".join(e.get("actions", []))
        lines.append(f"  iter {e.get('iteration')}: {acts} → {outcome}")
    log_str = "\n".join(lines)
    
    prompt = f"""The agent is stuck. It tried these actions and got no visible change (or wrong result):

{log_str}

USER GOAL: {user_goal}

CURRENT SCREEN: {state.active_window}
UIA (first 500 chars): {(state.uia_text or '')[:500]}
OCR (first 500 chars): {(state.ocr_text or '')[:500]}

What likely went wrong? What should the agent try differently? Be specific (1-3 sentences). Examples: "First result may be an ad - scroll down first"; "Try Tab to navigate instead of click"; "Wait longer for page load"."""

    model = model_name or CONFIG.get("model_name", "MiniMax-M2.5")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You analyze why desktop automation failed and suggest a better approach. Be brief and actionable."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            log_event({
                "type": "reflection",
                "goal": user_goal,
                "failed_attempts": len([e for e in recent if not e.get("state_changed")]),
                "insight": content,
            })
            return content
    except Exception as e:
        logger.warning(f"Reflection API call failed: {e}")
    return None


def build_user_prompt(
    state: ScreenState,
    user_goal: str,
    action_history: List[Dict],
    confirmation_log: Optional[List[Dict]] = None,
    loop_warning: Optional[str] = None,
    reflection_insight: Optional[str] = None,
) -> str:
    """Build prompt with full context including action history and confirmation log."""
    history_str = ""
    if action_history:
        recent = action_history[-15:]
        parts = []
        for i, a in enumerate(recent):
            p = f"  {i+1}. {a.get('type', '?')}"
            if a.get('x') is not None:
                p += f" @ ({a.get('x')},{a.get('y')})"
            if a.get('keys'):
                p += f" keys={a.get('keys')}"
            if a.get('text'):
                txt = str(a.get('text', ''))[:40]
                p += f" text='{txt}'" + ("..." if len(str(a.get('text', ''))) > 40 else "")
            parts.append(p)
        history_str = f"\nRECENT ACTIONS (what you already did):\n" + "\n".join(parts) + "\n"
    
    confirmation_str = ""
    if confirmation_log and CONFIG.get("confirmation_log_enabled", True):
        max_entries = CONFIG.get("confirmation_log_max_entries", 8)
        recent = confirmation_log[-max_entries:]
        lines = []
        for e in recent:
            outcome = "✓ changed" if e.get("state_changed") else "✗ no_change"
            acts = "; ".join(e.get("actions", []))
            lines.append(f"  iter {e.get('iteration')}: {acts} → {outcome} (window: {e.get('window_after', '?')})")
        confirmation_str = "\nCONFIRMATION LOG (did your actions change the screen?):\n" + "\n".join(lines) + "\n"
    
    loop_str = ""
    if loop_warning:
        loop_str = f"\n*** {loop_warning} ***\n"
    
    reflection_str = ""
    if reflection_insight:
        reflection_str = f"\nREFLECTION (what to try differently): {reflection_insight}\n"
    
    return f"""USER GOAL: {user_goal}
{history_str}{confirmation_str}{loop_str}{reflection_str}
CURRENT SCREEN STATE:
- Screen: {state.screen_w}x{state.screen_h}
- Cursor: ({state.cursor_x},{state.cursor_y})
- Active window: {state.active_window}

UIA_TEXT (UI element tree - use for coordinates when available):
{state.uia_text if state.uia_text else "(none - enable uia_enabled in config for richer context)"}

OCR_TEXT (visible text on screen):
{state.ocr_text if state.ocr_text else "(none - enable ocr_enabled and install tesseract for richer context)"}

Continue toward the goal. Only choose "done" when the goal is fully achieved or you need user input."""


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from model response with multiple fallback strategies."""
    if not text:
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove code fences
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find first JSON object with regex
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Find any curly brace block and try to fix common issues
    m = re.search(r"\{.+\}", text, re.DOTALL)
    if m:
        candidate = m.group(0)
        # Try fixing common issues
        # Remove trailing commas
        candidate = re.sub(r",(\s*[\]}])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    logger.warning(f"Could not extract JSON from: {text[:200]}...")
    return None


def call_model(
    client: OpenAI,
    user_goal: str,
    state: ScreenState,
    action_history: List[Dict],
    confirmation_log: Optional[List[Dict]] = None,
    loop_warning: Optional[str] = None,
    reflection_insight: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call model with retry logic for empty responses."""
    model = model_name or CONFIG.get("model_name", "MiniMax-M2.5")
    prompt = build_user_prompt(
        state, user_goal, action_history,
        confirmation_log=confirmation_log,
        loop_warning=loop_warning,
        reflection_insight=reflection_insight,
    )
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=CONFIG["max_tokens"],
                temperature=0.2,
            )
            
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            
            if not content:
                logger.warning(f"Attempt {attempt + 1}: Empty response, retrying...")
                time.sleep(1)
                continue
            
            result = extract_json(content)
            if result:
                return result
            
            logger.warning(f"Attempt {attempt + 1}: Could not parse JSON, retrying...")
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    raise ValueError("All API retries failed to return valid JSON")


# ----------------------------
# Action Execution
# ----------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def execute_action(action: Dict[str, Any], screen_w: int, screen_h: int) -> None:
    t = action.get("type")
    
    if t == "move":
        x = clamp(int(action.get("x", 0)), 0, screen_w - 1)
        y = clamp(int(action.get("y", 0)), 0, screen_h - 1)
        pyautogui.moveTo(x, y, duration=CONFIG.get("move_duration_sec", 0.04))
    
    elif t in ("click", "double_click", "right_click", "middle_click"):
        x = clamp(int(action.get("x", 0)), 0, screen_w - 1)
        y = clamp(int(action.get("y", 0)), 0, screen_h - 1)
        pyautogui.moveTo(x, y, duration=CONFIG.get("click_move_duration_sec", 0.03))
        
        if t == "right_click":
            pyautogui.click(button="right")
        elif t == "middle_click":
            pyautogui.click(button="middle")
        elif t == "double_click":
            pyautogui.doubleClick()
        else:
            pyautogui.click(button=action.get("button", "left"))
    
    elif t == "drag":
        x = clamp(int(action.get("x", 0)), 0, screen_w - 1)
        y = clamp(int(action.get("y", 0)), 0, screen_h - 1)
        button = action.get("button", "left")
        pyautogui.dragTo(x, y, duration=CONFIG.get("drag_duration_sec", 0.1), button=button)
    
    elif t == "scroll":
        amt = int(action.get("scroll_amount", 0))
        x = action.get("x")
        y = action.get("y")
        if x is not None and y is not None:
            pyautogui.scroll(amt, x=clamp(int(x), 0, screen_w - 1), y=clamp(int(y), 0, screen_h - 1))
        else:
            pyautogui.scroll(amt)
    
    elif t == "type":
        text = action.get("text", "")
        pyautogui.write(text, interval=CONFIG.get("type_interval_sec", 0.008))
    
    elif t == "hotkey":
        keys = action.get("keys", [])
        if keys:
            pyautogui.hotkey(*keys)
    
    elif t == "press":
        key = action.get("key") or action.get("keys", [None])[0]
        if key:
            pyautogui.press(str(key).lower())
    
    elif t == "key_down":
        key = action.get("key") or (action.get("keys", [None])[0] if action.get("keys") else None)
        if key:
            pyautogui.keyDown(str(key).lower())
    
    elif t == "key_up":
        key = action.get("key") or (action.get("keys", [None])[0] if action.get("keys") else None)
        if key:
            pyautogui.keyUp(str(key).lower())
    
    elif t == "wait":
        ms = int(action.get("wait_ms", 300))
        time.sleep(max(0, ms) / 1000.0)
    
    # "done" and unknown types: do nothing


# ----------------------------
# Main Loop
# ----------------------------

def main():
    api_key, api_base_url, model_name, provider_label = get_provider()
    
    # Check for existing state
    existing_state = load_state()
    if existing_state:
        print(f"Found existing session from {datetime.fromtimestamp(existing_state.start_time)}")
        print(f"Goal: {existing_state.goal}")
        resume = input("Resume? (y/n): ").strip().lower() == "y"
        if not resume:
            clear_state()
            existing_state = None
    
    if not existing_state:
        user_goal = input("What do you want the agent to do? ").strip()
        if not user_goal:
            print("No goal provided. Exiting.")
            return
        
        state = AgentState(
            goal=user_goal,
            start_time=time.time(),
            iteration=0,
            actions_executed=[]
        )
        save_state(state)
    else:
        state = existing_state
        user_goal = state.goal
    
    # Initialize client
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    
    print(f"Running. Provider: {provider_label} | Model: {model_name} | Kill switch: move mouse to top-left corner.")
    print(f"Goal: {user_goal}")
    logger.info(f"Starting agent with goal: {user_goal}")
    
    overlay_thread = start_cursor_overlay()
    if overlay_thread:
        print("Cursor overlay enabled (red circle).")
    
    try:
        while True:
            if kill_switch_triggered():
                print("Kill switch triggered. Stopping.")
                logger.info("Kill switch triggered, stopping.")
                break
            
            # Capture screen state
            screen_img = capture_screen()
            screen_w, screen_h = screen_img.size
            cx, cy = pyautogui.position()
            
            screen_state = ScreenState(
                screen_w=screen_w,
                screen_h=screen_h,
                cursor_x=cx,
                cursor_y=cy,
                active_window=get_active_window_title(),
                uia_text=extract_uia_text(),
                ocr_text=extract_ocr_text(screen_img),
            )
            
            # Build loop warning from previous confirmation log (before this turn)
            loop_warning = None
            if state.confirmation_log:
                loop_warning = detect_loop_warning(
                    state.confirmation_log,
                    CONFIG.get("loop_warning_threshold", 3),
                )
            
            # Reflection: when stuck, call API to get improvement suggestions
            reflection_insight = None
            if loop_warning and CONFIG.get("reflection_enabled", False):
                logger.info("Calling reflection API to improve next action...")
                reflection_insight = call_reflection(
                    client, user_goal, state.confirmation_log, screen_state,
                    model_name=model_name,
                )
                if reflection_insight:
                    print(f"💡 Reflection: {reflection_insight[:120]}{'...' if len(reflection_insight) > 120 else ''}")
            
            # Call model
            logger.debug(f"Iteration {state.iteration}: Calling API...")
            try:
                plan = call_model(
                    client, user_goal, screen_state, state.actions_executed,
                    confirmation_log=state.confirmation_log,
                    loop_warning=loop_warning,
                    reflection_insight=reflection_insight,
                    model_name=model_name,
                )
            except Exception as e:
                logger.error(f"API call failed: {e}")
                time.sleep(2)
                continue
            
            # Execute actions
            actions = plan.get("actions", [])[:CONFIG["max_actions_per_turn"]]
            risk_flags = plan.get("risk_flags", [])
            
            print(f"\n--- Iteration {state.iteration} ---")
            print(f"Goal: {plan.get('goal_summary', '')}")
            if loop_warning:
                print(f"⚠ {loop_warning}")
            if risk_flags:
                print(f"Risks: {risk_flags}")
            print(f"Actions: {actions}")
            logger.info(f"Iteration {state.iteration}: {actions}")
            
            # Check for done
            if not actions or actions[0].get("type") == "done":
                print("Agent indicated done.")
                logger.info("Agent marked task as done.")
                log_event({"type": "done", "iteration": state.iteration, "goal": user_goal})
                break
            
            # Execute each action
            for action in actions:
                if kill_switch_triggered():
                    print("Kill switch triggered during execution.")
                    break
                
                # Block typing if sensitive
                if (action.get("type") == "type" and 
                    likely_sensitive_context(screen_state)):
                    logger.warning("Blocking typing - sensitive context detected")
                    continue
                
                execute_action(action, screen_w, screen_h)
                state.actions_executed.append(action)
                time.sleep(CONFIG.get("action_pause_sec", 0.06))
            
            # Capture state after actions for confirmation logging
            if CONFIG.get("confirmation_log_enabled", True) and actions:
                time.sleep(CONFIG.get("confirmation_capture_delay_sec", 0.08))
                state_after = ScreenState(
                    screen_w=screen_w,
                    screen_h=screen_h,
                    cursor_x=pyautogui.position()[0],
                    cursor_y=pyautogui.position()[1],
                    active_window=get_active_window_title(),
                    uia_text=extract_uia_text(),
                    ocr_text=extract_ocr_text(capture_screen()),
                )
                entry = build_confirmation_entry(
                    state.iteration, actions, screen_state, state_after
                )
                state.confirmation_log.append(entry)
                if entry.get("state_changed"):
                    logger.info(f"Iteration {state.iteration}: state changed")
                else:
                    logger.info(f"Iteration {state.iteration}: no visible change")
                
                # Event logging for analysis
                log_event({
                    "type": "iteration",
                    "iteration": state.iteration,
                    "goal": user_goal,
                    "actions": entry.get("actions", []),
                    "outcome": entry.get("outcome", "?"),
                    "window_before": entry.get("window_before"),
                    "window_after": entry.get("window_after"),
                })
            
            # Update and save state
            state.iteration += 1
            save_state(state)
            
            # Delay between iterations
            time.sleep(CONFIG["loop_delay_sec"])
    
    except KeyboardInterrupt:
        print("\nInterrupted. State saved.")
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}")
    finally:
        stop_cursor_overlay()
        print(f"Completed {state.iteration} iterations.")
        logger.info(f"Finished. Total iterations: {state.iteration}")


if __name__ == "__main__":
    main()
