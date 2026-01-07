#!/usr/bin/env python3
"""
Comfy Batch GUI (PyQt via qtpy) — MULTI-SCENE SUPPORT (FIRST PROMPT = IMAGE ONLY)

CHANGE:
- For EACH scene, the FIRST prompt is used ONLY to generate the initial still (image workflow).
- Video generation starts from the NEXT prompt (index 1) and chains as before.
- This lets you keep speech/dialogue OUT of the image prompt and only in video prompts.

JSON formats supported (same as before):
- Single scene: prompts: ["p0(image)", "p1(video)", ...]
- Multi-scene list-of-lists: prompts: [[...], [...]]
- Explicit scenes: scenes: [{"prompts":[...]} , ...]
"""

import copy
import json
import os
import random
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from qtpy.QtCore import QObject, Signal, QRunnable, QThreadPool
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QComboBox, QGroupBox, QFormLayout,
    QTextEdit, QSpinBox, QCheckBox
)

# ----------------------------
# Requests session (retries)
# ----------------------------

def build_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# ----------------------------
# Comfy API helpers
# ----------------------------

def comfy_post_prompt(session: requests.Session, comfy_base: str, workflow_json: Dict[str, Any]) -> str:
    comfy_base = comfy_base.rstrip("/")
    r = session.post(f"{comfy_base}/prompt", json={"prompt": workflow_json}, timeout=120)
    r.raise_for_status()
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {data}")
    return prompt_id

def comfy_get_history(session: requests.Session, comfy_base: str, prompt_id: str) -> Dict[str, Any]:
    comfy_base = comfy_base.rstrip("/")
    r = session.get(f"{comfy_base}/history/{prompt_id}", timeout=60)
    r.raise_for_status()
    return r.json()

def comfy_download_view(session: requests.Session, comfy_base: str, filename: str, subfolder: str = "", filetype: str = "output") -> bytes:
    comfy_base = comfy_base.rstrip("/")
    params = {"filename": filename, "type": filetype}
    if subfolder:
        params["subfolder"] = subfolder
    url = f"{comfy_base}/view?{urllib.parse.urlencode(params)}"
    r = session.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def save_bytes_to_temp(data: bytes, suffix: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"comfy_{random.randint(0, 9999999)}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path

def parse_best_outputs(history_obj: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not history_obj:
        return [], []
    main_key = next(iter(history_obj.keys()), None)
    if not main_key:
        return [], []
    outputs = history_obj.get(main_key, {}).get("outputs", {}) or {}
    if not outputs:
        return [], []

    images: List[Dict[str, str]] = []
    videos: List[Dict[str, str]] = []

    for _, out in outputs.items():
        for info in (out.get("images", []) or []):
            images.append({"filename": info.get("filename", ""), "subfolder": info.get("subfolder", "")})
        for info in (out.get("gifs", []) or []):
            videos.append({"filename": info.get("filename", ""), "subfolder": info.get("subfolder", "")})
        for info in (out.get("videos", []) or []):
            videos.append({"filename": info.get("filename", ""), "subfolder": info.get("subfolder", "")})

    images = [x for x in images if x.get("filename")]
    videos = [x for x in videos if x.get("filename")]
    return images, videos

# ----------------------------
# ffmpeg helpers
# ----------------------------

def ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is in PATH.") from e

def extract_last_frame_ffmpeg(video_path: str, out_png_path: str) -> str:
    ensure_ffmpeg_available()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{probe.stderr.decode('utf-8', errors='ignore')}")
    dur_s = float(probe.stdout.decode("utf-8", errors="ignore").strip() or "0")
    if dur_s <= 0:
        raise RuntimeError("Could not determine video duration for last-frame extraction.")

    epsilon = 0.05
    seek_time = max(0.0, dur_s - epsilon)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seek_time:.6f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vsync", "0",
        out_png_path
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 0 and os.path.exists(out_png_path) and os.path.getsize(out_png_path) > 0:
        return out_png_path

    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "select='eq(n,prev_selected_n)'",
        "-vsync", "vfr",
        out_png_path
    ]
    r2 = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r2.returncode != 0 or not os.path.exists(out_png_path) or os.path.getsize(out_png_path) == 0:
        raise RuntimeError(
            "ffmpeg last-frame extraction failed.\n"
            f"Seek stderr:\n{r.stderr.decode('utf-8', errors='ignore')}\n"
            f"Fallback stderr:\n{r2.stderr.decode('utf-8', errors='ignore')}"
        )
    return out_png_path

def concat_videos_ffmpeg(video_paths: List[str], out_path: str) -> str:
    ensure_ffmpeg_available()
    if not video_paths:
        raise ValueError("No videos to concat.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_path = os.path.join(os.path.dirname(out_path), "concat_list.txt")

    def esc(p: str) -> str:
        return p.replace("'", "'\\''")

    with open(list_path, "w", encoding="utf-8") as f:
        for p in video_paths:
            f.write(f"file '{esc(p)}'\n")

    cmd_copy = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_path
    ]
    r = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    cmd_enc = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path
    ]
    r2 = subprocess.run(cmd_enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r2.returncode != 0:
        raise RuntimeError(
            "ffmpeg concat failed (copy and re-encode).\n"
            f"Copy stderr:\n{r.stderr.decode('utf-8', errors='ignore')}\n"
            f"Re-encode stderr:\n{r2.stderr.decode('utf-8', errors='ignore')}"
        )
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("ffmpeg concat produced an empty output.")
    return out_path

# ----------------------------
# Workflow helpers
# ----------------------------

def summarize_nodes(workflow_json: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    nodes = []
    for node_id, node in workflow_json.items():
        class_type = str(node.get("class_type", ""))
        title = str(node.get("_meta", {}).get("title", "")).strip()
        label = title if title else class_type if class_type else str(node_id)
        nodes.append((str(node_id), label, class_type))
    nodes.sort(key=lambda x: (x[1].lower(), x[0]))
    return nodes

def list_node_inputs(workflow_json: Dict[str, Any], node_id: str) -> List[str]:
    node = workflow_json.get(str(node_id), {})
    inputs = node.get("inputs", {}) or {}
    return sorted([str(k) for k in inputs.keys()], key=lambda s: s.lower())

def set_node_input(workflow_json: Dict[str, Any], node_id: str, input_key: str, value: Any) -> None:
    node_id = str(node_id)
    if node_id not in workflow_json:
        raise KeyError(f"Node id {node_id} not in workflow JSON")
    workflow_json[node_id].setdefault("inputs", {})
    workflow_json[node_id]["inputs"][str(input_key)] = value

# ----------------------------
# Prompt expansion + scene parsing
# ----------------------------

def _calc_target_segments(cfg: Dict[str, Any]) -> int:
    seg = int(cfg.get("segment_sec", 5))
    if "total_segments" in cfg:
        return int(cfg["total_segments"])
    if "total_seconds" in cfg:
        return int(cfg["total_seconds"]) // seg
    if "total_hours" in cfg:
        return int(float(cfg["total_hours"]) * 3600) // seg
    return max(1, len(cfg.get("prompts", [])))

def expand_cycle_prompts(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ordered cycling:
    - Always traverse prompts in the given order for the first full pass.
    - Only if target_segments > len(prompts) do we start a new cycle.
    - shuffle_each_cycle applies ONLY to cycles after the first pass.
    """
    prompts = cfg.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Config must include a non-empty 'prompts' list.")

    mode = str(cfg.get("mode", "cycle")).lower().strip()
    if mode != "cycle":
        raise ValueError("Only mode='cycle' is supported by the simple format.")

    seg_sec = int(cfg.get("segment_sec", 5))
    shuffle_each_cycle = bool(cfg.get("shuffle_each_cycle", False))
    seed_base = int(cfg.get("seed_base", 1000))

    target_segments = _calc_target_segments(cfg)
    n_prompts = len(prompts)

    rng = random.Random(seed_base)

    def normalize_prompt(p: Any) -> Tuple[str, str]:
        if isinstance(p, str):
            return p.strip(), ""
        if isinstance(p, dict):
            return str(p.get("positive", "")).strip(), str(p.get("negative", "")).strip()
        return "", ""

    expanded: List[Dict[str, Any]] = []
    cycle_index = 0
    prompt_order = list(range(n_prompts))  # first cycle is always in-order

    while len(expanded) < target_segments:
        if cycle_index > 0 and shuffle_each_cycle:
            prompt_order = list(range(n_prompts))
            rng.shuffle(prompt_order)
        else:
            prompt_order = list(range(n_prompts))

        for pi in prompt_order:
            if len(expanded) >= target_segments:
                break

            positive, negative = normalize_prompt(prompts[pi])
            if not positive:
                continue

            seg_i = len(expanded)
            expanded.append({
                "name": f"seg_{seg_i:05d}",
                "positive": positive,
                "prompts": positive,
                "negative": negative,
                "segment_sec": seg_sec,
                "seed": seed_base + seg_i,
                "prompt_index": int(pi),
                "cycle_index": int(cycle_index),
            })

        cycle_index += 1

    return expanded

def parse_scenes_from_json(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Important behavior:
    - If 'prompts' is a list-of-lists, each sub-list is a scene and generates
      exactly len(sublist) segments (no duration expansion).
    - If you want duration-based expansion, use explicit 'scenes' objects where each
      scene can specify total_hours/total_seconds/total_segments.
    """
    if "scenes" in data:
        scenes_cfg = data["scenes"]
        if not isinstance(scenes_cfg, list) or not scenes_cfg:
            raise ValueError("'scenes' must be a non-empty list.")
        scenes: List[List[Dict[str, Any]]] = []
        for i, sc in enumerate(scenes_cfg):
            if not isinstance(sc, dict):
                raise ValueError(f"Scene {i} must be an object.")
            merged = dict(data)
            merged.pop("scenes", None)
            merged.update(sc)
            scenes.append(expand_cycle_prompts(merged))
        return scenes

    prompts = data.get("prompts")

    if isinstance(prompts, list) and prompts and isinstance(prompts[0], list):
        scenes: List[List[Dict[str, Any]]] = []
        for plist in prompts:
            sc_cfg = dict(data)
            sc_cfg["prompts"] = plist
            sc_cfg.pop("total_hours", None)
            sc_cfg.pop("total_seconds", None)
            sc_cfg.pop("total_segments", None)
            scenes.append(expand_cycle_prompts(sc_cfg))
        return scenes

    if isinstance(data, dict) and ("prompts" in data):
        return [expand_cycle_prompts(data)]

    raise ValueError("Prompts JSON must be an object with 'prompts' or 'scenes'.")

# ----------------------------
# Binding model
# ----------------------------

@dataclass
class Binding:
    workflow: str       # "image" or "video"
    prompt_field: str   # e.g. "positive", "prompts", "seed", "start_image_path"
    node_id: str
    input_key: str

# ----------------------------
# Project save/load
# ----------------------------

PROJECT_VERSION = 1

@dataclass
class ProjectState:
    version: int
    # UI/settings
    comfy_base: str
    out_dir: str
    poll_interval_s: float
    assemble_final: bool
    generate_fresh_scene_image: bool
    randomize_seeds: bool
    randomize_scope: str

    # data/bindings
    bindings: List[Dict[str, Any]]

    # filenames inside the project folder
    prompts_filename: str
    image_workflow_filename: str
    video_workflow_filename: str

    # serialized JSON blobs (for robustness / portability)
    prompts_json: Dict[str, Any]
    image_workflow_json: Dict[str, Any]
    video_workflow_json: Dict[str, Any]


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project_folder(
    folder: str,
    *,
    comfy_base: str,
    out_dir: str,
    poll_interval_s: float,
    assemble_final: bool,
    generate_fresh_scene_image: bool,
    randomize_seeds: bool,
    randomize_scope: str,
    bindings: List[Binding],
    prompts_data: Dict[str, Any],
    image_wf: Dict[str, Any],
    video_wf: Dict[str, Any],
    prompts_filename: str = "prompts.json",
    image_workflow_filename: str = "workflow_image.json",
    video_workflow_filename: str = "workflow_video.json",
) -> str:
    """
    Writes a self-contained project folder:
      - project.json (settings + serialized JSON blobs)
      - prompts.json
      - workflow_image.json
      - workflow_video.json
    Returns path to project.json
    """
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    # Export standalone JSONs (the actual "used" inputs)
    _safe_write_json(os.path.join(folder, prompts_filename), prompts_data)
    _safe_write_json(os.path.join(folder, image_workflow_filename), image_wf)
    _safe_write_json(os.path.join(folder, video_workflow_filename), video_wf)

    state = ProjectState(
        version=PROJECT_VERSION,
        comfy_base=comfy_base,
        out_dir=out_dir,
        poll_interval_s=float(poll_interval_s),
        assemble_final=bool(assemble_final),
        generate_fresh_scene_image=bool(generate_fresh_scene_image),
        randomize_seeds=bool(randomize_seeds),
        randomize_scope=str(randomize_scope),

        bindings=[asdict(b) for b in bindings],

        prompts_filename=prompts_filename,
        image_workflow_filename=image_workflow_filename,
        video_workflow_filename=video_workflow_filename,

        # serialized copies (lets you recover even if user deletes the files)
        prompts_json=prompts_data,
        image_workflow_json=image_wf,
        video_workflow_json=video_wf,
    )

    project_path = os.path.join(folder, "project.json")
    _safe_write_json(project_path, asdict(state))
    return project_path


def load_project_folder(folder: str) -> ProjectState:
    """
    Loads project.json, then tries to load JSON files from the folder.
    If files are missing, falls back to serialized blobs in project.json.
    """
    folder = os.path.abspath(folder)
    project_path = os.path.join(folder, "project.json")
    raw = _safe_read_json(project_path)

    # basic validation / defaults
    if int(raw.get("version", 0)) != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {raw.get('version')}")

    # Try reading exported JSONs; fallback to embedded JSON blobs.
    def read_or_fallback(filename_key: str, blob_key: str) -> Dict[str, Any]:
        fn = raw.get(filename_key, "")
        if fn:
            p = os.path.join(folder, fn)
            if os.path.exists(p):
                data = _safe_read_json(p)
                if isinstance(data, dict):
                    return data
        blob = raw.get(blob_key)
        if isinstance(blob, dict):
            return blob
        raise ValueError(f"Missing required JSON: {filename_key} / {blob_key}")

    prompts_data = read_or_fallback("prompts_filename", "prompts_json")
    image_wf = read_or_fallback("image_workflow_filename", "image_workflow_json")
    video_wf = read_or_fallback("video_workflow_filename", "video_workflow_json")

    # Put resolved json back into state
    raw["prompts_json"] = prompts_data
    raw["image_workflow_json"] = image_wf
    raw["video_workflow_json"] = video_wf

    return ProjectState(**raw)


# ----------------------------
# Worker
# ----------------------------

class BatchWorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

class BatchWorker(QRunnable):
    def __init__(
        self,
        comfy_base: str,
        scenes: List[List[Dict[str, Any]]],
        image_workflow: Dict[str, Any],
        video_workflow: Dict[str, Any],
        bindings: List[Binding],
        out_dir: str,
        poll_interval_s: float = 1.0,
        max_wait_s: int = 60 * 60,
        assemble_final: bool = True,
        generate_fresh_scene_image: bool = True,
        randomize_seeds: bool = False,
        randomize_scope: str = "per_generation",  # "per_generation" or "per_scene"
    ):
        super().__init__()
        self.signals = BatchWorkerSignals()
        self.session = build_retry_session()

        self.comfy_base = comfy_base
        self.scenes = scenes
        self.image_workflow = image_workflow
        self.video_workflow = video_workflow
        self.bindings = bindings
        self.out_dir = out_dir
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s
        self.assemble_final = assemble_final
        self.generate_fresh_scene_image = generate_fresh_scene_image
        self.randomize_seeds = randomize_seeds
        self.randomize_scope = randomize_scope

        self._stop = False

    def stop(self):
        self._stop = True

    def _resolve_field(self, prompt_item: Dict[str, Any], field: str, extra: Dict[str, Any]) -> Tuple[bool, Any]:
        if field in extra:
            return True, extra[field]
        if field in prompt_item:
            return True, prompt_item[field]
        if field == "prompts" and "positive" in prompt_item:
            return True, prompt_item["positive"]
        if field == "positive" and "prompts" in prompt_item:
            return True, prompt_item["prompts"]
        return False, None

    def _apply_bindings(self, wf_kind: str, wf_json: Dict[str, Any], prompt_item: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        wf = copy.deepcopy(wf_json)
        for b in self.bindings:
            if b.workflow != wf_kind:
                continue
            ok, val = self._resolve_field(prompt_item, b.prompt_field, extra)
            if not ok:
                continue
            set_node_input(wf, b.node_id, b.input_key, val)
        return wf

    def _wait_for_result(self, prompt_id: str) -> Dict[str, Any]:
        start = time.time()
        while not self._stop:
            hist = comfy_get_history(self.session, self.comfy_base, prompt_id)
            if hist and prompt_id in hist and hist[prompt_id].get("outputs"):
                return hist
            if time.time() - start > self.max_wait_s:
                raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")
            time.sleep(self.poll_interval_s)
        raise RuntimeError("Stopped")

    def _download_media_to(self, media: Dict[str, str], target_folder: str) -> str:
        filename = media.get("filename", "")
        subfolder = media.get("subfolder", "") or ""
        if not filename:
            raise RuntimeError(f"Bad media info: {media}")

        data = comfy_download_view(self.session, self.comfy_base, filename=filename, subfolder=subfolder, filetype="output")
        suffix = os.path.splitext(filename)[1] or ".bin"
        temp_path = save_bytes_to_temp(data, suffix)

        os.makedirs(target_folder, exist_ok=True)
        out_path = os.path.join(target_folder, filename)
        if os.path.exists(out_path):
            base, ext = os.path.splitext(filename)
            out_path = os.path.join(target_folder, f"{base}_{int(time.time())}{ext}")

        with open(temp_path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())

        return os.path.abspath(out_path)

    def run(self):
        try:
            per_scene_video_counts = [max(0, len(scene) - 1) for scene in self.scenes]
            total_videos = sum(per_scene_video_counts)

            seg_counter = 0
            seg_sec = self.scenes[0][0].get("segment_sec", "?") if self.scenes and self.scenes[0] else "?"
            self.signals.log.emit(f"[PLAN] scenes={len(self.scenes)} total_videos={total_videos} segment_sec={seg_sec}")
            for si, cnt in enumerate(per_scene_video_counts):
                self.signals.log.emit(f"[PLAN] scene {si + 1}: videos={cnt} (image_prompt=1, video_prompts={cnt})")

            os.makedirs(self.out_dir, exist_ok=True)
            stills_dir = os.path.join(self.out_dir, "stills")
            videos_dir = os.path.join(self.out_dir, "videos")
            chain_dir = os.path.join(self.out_dir, "chain_frames")
            os.makedirs(stills_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(chain_dir, exist_ok=True)

            all_video_paths: List[str] = []
            scene_manifests: List[Dict[str, Any]] = []

            # ----------------------------
            # PASS 1: Generate ALL stills
            # ----------------------------
            if not self.generate_fresh_scene_image:
                raise RuntimeError("Two-pass mode requires 'Generate fresh image per scene' to be enabled.")

            scene_initial_stills: List[Optional[str]] = [None] * len(self.scenes)
            scene_seeds: List[int] = [random.randint(0, 2 ** 31 - 1) for _ in range(len(self.scenes))]

            self.signals.log.emit("\n=== PASS 1/2: GENERATE SCENE STILLS ===")
            for scene_idx, scene_items in enumerate(self.scenes):
                if self._stop:
                    raise RuntimeError("Stopped")
                if not scene_items:
                    continue

                image_item = scene_items[0]
                self.signals.log.emit(f"\n[STILL] SCENE {scene_idx + 1}/{len(self.scenes)} (first prompt only)")

                extra_img: Dict[str, Any] = {}
                if self.randomize_seeds and self.randomize_scope == "per_scene":
                    extra_img["seed"] = scene_seeds[scene_idx]
                elif self.randomize_seeds and self.randomize_scope == "per_generation":
                    extra_img["seed"] = random.randint(0, 2 ** 31 - 1)

                still_wf = self._apply_bindings("image", self.image_workflow, image_item, extra=extra_img)
                still_prompt_id = comfy_post_prompt(self.session, self.comfy_base, still_wf)
                still_hist = self._wait_for_result(still_prompt_id)
                still_images, _ = parse_best_outputs(still_hist)
                if not still_images:
                    raise RuntimeError(f"No image output found from IMAGE workflow for scene {scene_idx + 1}.")

                initial_still = self._download_media_to(still_images[-1], stills_dir)
                if not os.path.exists(initial_still):
                    raise RuntimeError(f"Initial still missing on disk: {initial_still}")

                scene_initial_stills[scene_idx] = os.path.abspath(initial_still)
                self.signals.log.emit(f"[STILL] ✔ scene still: {scene_initial_stills[scene_idx]}")

            if any(s is None for s in scene_initial_stills if len(self.scenes) > 0):
                raise RuntimeError("Failed to generate one or more scene stills.")

            # ----------------------------
            # PASS 2: Generate ALL videos
            # ----------------------------
            self.signals.log.emit("\n=== PASS 2/2: GENERATE VIDEOS ===")

            for scene_idx, scene_items in enumerate(self.scenes):
                if self._stop:
                    raise RuntimeError("Stopped")
                if not scene_items:
                    continue

                image_item = scene_items[0]
                video_items = scene_items[1:]
                initial_still = scene_initial_stills[scene_idx]
                assert initial_still is not None

                self.signals.log.emit(f"\n=== SCENE {scene_idx + 1}/{len(self.scenes)} ===")
                self.signals.log.emit(
                    f"[PLAN] SCENE {scene_idx + 1}: 1 still (already done) + {len(video_items)} video segments")
                self.signals.log.emit(f"[INIT] Using pre-generated still: {initial_still}")

                current_start_png = os.path.abspath(initial_still)
                scene_video_paths: List[str] = []

                for item_idx, item in enumerate(video_items):
                    if self._stop:
                        raise RuntimeError("Stopped")

                    seg_counter += 1
                    self.signals.progress.emit(seg_counter, max(1, total_videos))

                    name = str(item.get("name", f"scene{scene_idx:02d}_seg{item_idx:05d}"))
                    if not os.path.exists(current_start_png):
                        raise RuntimeError(f"Start PNG missing: {current_start_png}")

                    self.signals.log.emit(f"[{seg_counter}/{total_videos}] VIDEO (scene {scene_idx + 1}) for: {name}")

                    extra_vid: Dict[str, Any] = {"start_image_path": os.path.abspath(current_start_png)}
                    if self.randomize_seeds and self.randomize_scope == "per_scene":
                        extra_vid["seed"] = scene_seeds[scene_idx]
                    elif self.randomize_seeds and self.randomize_scope == "per_generation":
                        extra_vid["seed"] = random.randint(0, 2 ** 31 - 1)

                    video_wf = self._apply_bindings("video", self.video_workflow, item, extra=extra_vid)
                    video_prompt_id = comfy_post_prompt(self.session, self.comfy_base, video_wf)
                    vhist = self._wait_for_result(video_prompt_id)

                    _, v_videos = parse_best_outputs(vhist)
                    if not v_videos:
                        raise RuntimeError(f"No video/gif output found for scene {scene_idx + 1}, segment {name}.")

                    video_path = self._download_media_to(v_videos[-1], videos_dir)
                    if not os.path.exists(video_path):
                        raise RuntimeError(f"Video file missing: {video_path}")

                    scene_video_paths.append(video_path)
                    all_video_paths.append(video_path)
                    self.signals.log.emit(f"  ✔ video: {video_path}")

                    out_png = os.path.abspath(os.path.join(chain_dir, f"{Path(video_path).stem}_last.png"))
                    next_png = extract_last_frame_ffmpeg(video_path, out_png)
                    self.signals.log.emit(f"  ✔ next start (ffmpeg last): {next_png}")
                    current_start_png = next_png

                    if not os.path.exists(current_start_png):
                        raise RuntimeError("Chaining failed: next start frame PNG was not created/found.")

                scene_manifests.append({
                    "scene_index": scene_idx,
                    "image_prompt_item": image_item,
                    "initial_still": initial_still,
                    "segments": scene_video_paths
                })

            # 3) Assemble final
            final_path = None
            if (not self._stop) and self.assemble_final and all_video_paths:
                final_path = os.path.abspath(os.path.join(self.out_dir, "final.mp4"))
                self.signals.log.emit(f"\nAssembling {len(all_video_paths)} total segments -> {final_path}")
                concat_videos_ffmpeg(all_video_paths, final_path)
                self.signals.log.emit(f"  ✔ final: {final_path}")

            manifest = {
                "scenes": scene_manifests,
                "all_segments": all_video_paths,
                "final": final_path,
            }
            with open(os.path.join(self.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(repr(e))


# ----------------------------
# Main UI
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comfy Batch: Multi-Scene (First Prompt = Image Only) -> Chained Segments -> Concat")

        self.threadpool = QThreadPool.globalInstance()

        self.prompts_path: Optional[str] = None
        self.scenes: List[List[Dict[str, Any]]] = []

        self.image_wf_path: Optional[str] = None
        self.video_wf_path: Optional[str] = None
        self.image_wf: Optional[Dict[str, Any]] = None
        self.video_wf: Optional[Dict[str, Any]] = None

        self.bindings: List[Binding] = []
        self.worker: Optional[BatchWorker] = None

        self._build_ui()

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    def set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        endpoint_box = QGroupBox("ComfyUI Endpoint")
        endpoint_form = QFormLayout(endpoint_box)
        self.comfy_base_edit = QLineEdit("http://localhost:8188")
        endpoint_form.addRow("Base URL", self.comfy_base_edit)
        layout.addWidget(endpoint_box)

        files_box = QGroupBox("Load Files")
        files_layout = QHBoxLayout(files_box)
        self.btn_load_prompts = QPushButton("Load Prompts JSON")
        self.btn_load_image_wf = QPushButton("Load Image Workflow JSON")
        self.btn_load_video_wf = QPushButton("Load Video Workflow JSON")
        self.btn_save_project = QPushButton("Save Project Folder")
        self.btn_load_project = QPushButton("Load Project Folder")
        files_layout.addWidget(self.btn_save_project)
        files_layout.addWidget(self.btn_load_project)
        files_layout.addWidget(self.btn_load_prompts)
        files_layout.addWidget(self.btn_load_image_wf)
        files_layout.addWidget(self.btn_load_video_wf)
        layout.addWidget(files_box)

        self.lbl_status = QLabel("Ready.")
        layout.addWidget(self.lbl_status)

        bind_box = QGroupBox("Bindings (bind once)")
        bind_layout = QVBoxLayout(bind_box)

        bind_row = QHBoxLayout()
        self.workflow_kind_combo = QComboBox()
        self.workflow_kind_combo.addItems(["image", "video"])

        self.prompt_field_edit = QLineEdit()
        self.prompt_field_edit.setPlaceholderText("prompt field (prompts/positive, negative, seed, start_image_path, ...)")

        self.node_combo = QComboBox()
        self.input_combo = QComboBox()
        self.btn_add_binding = QPushButton("Add Binding")

        bind_row.addWidget(QLabel("Workflow:"))
        bind_row.addWidget(self.workflow_kind_combo, 1)
        bind_row.addWidget(QLabel("Prompt field:"))
        bind_row.addWidget(self.prompt_field_edit, 2)
        bind_row.addWidget(QLabel("Node:"))
        bind_row.addWidget(self.node_combo, 2)
        bind_row.addWidget(QLabel("Input:"))
        bind_row.addWidget(self.input_combo, 1)
        bind_row.addWidget(self.btn_add_binding)
        bind_layout.addLayout(bind_row)

        self.bind_table = QTableWidget(0, 4)
        self.bind_table.setHorizontalHeaderLabels(["workflow", "prompt_field", "node_id", "input_key"])
        self.bind_table.horizontalHeader().setStretchLastSection(True)
        bind_layout.addWidget(self.bind_table)

        btns_row = QHBoxLayout()
        self.btn_remove_binding = QPushButton("Remove Selected Binding")
        btns_row.addWidget(self.btn_remove_binding)
        btns_row.addStretch(1)
        bind_layout.addLayout(btns_row)

        layout.addWidget(bind_box)

        opts_box = QGroupBox("Options")
        opts_layout = QHBoxLayout(opts_box)

        self.chk_scene_image = QCheckBox("Generate fresh image per scene")
        self.chk_scene_image.setChecked(True)

        self.chk_rand_seed = QCheckBox("Randomize seed bindings")
        self.chk_rand_seed.setChecked(False)

        self.seed_scope_combo = QComboBox()
        self.seed_scope_combo.addItems(["per_generation", "per_scene"])
        self.seed_scope_combo.setCurrentText("per_generation")

        opts_layout.addWidget(self.chk_scene_image)
        opts_layout.addWidget(self.chk_rand_seed)
        opts_layout.addWidget(QLabel("Seed scope:"))
        opts_layout.addWidget(self.seed_scope_combo)
        opts_layout.addStretch(1)

        layout.addWidget(opts_box)

        run_box = QGroupBox("Batch Run")
        run_layout = QHBoxLayout(run_box)

        self.out_dir_edit = QLineEdit(os.path.join(os.getcwd(), "comfy_batch_outputs"))
        self.btn_choose_out = QPushButton("Choose Output Dir")

        self.poll_spin = QSpinBox()
        self.poll_spin.setMinimum(1)
        self.poll_spin.setMaximum(60)
        self.poll_spin.setValue(1)

        self.chk_assemble = QCheckBox("Assemble final.mp4")
        self.chk_assemble.setChecked(True)

        self.btn_run = QPushButton("RUN")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)

        run_layout.addWidget(QLabel("Output:"))
        run_layout.addWidget(self.out_dir_edit, 3)
        run_layout.addWidget(self.btn_choose_out)
        run_layout.addWidget(QLabel("Poll (s):"))
        run_layout.addWidget(self.poll_spin)
        run_layout.addWidget(self.chk_assemble)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_stop)

        layout.addWidget(run_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        # Wiring
        self.btn_load_prompts.clicked.connect(self.load_prompts)
        self.btn_load_image_wf.clicked.connect(self.load_image_workflow)
        self.btn_load_video_wf.clicked.connect(self.load_video_workflow)
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_load_project.clicked.connect(self.load_project)

        self.workflow_kind_combo.currentTextChanged.connect(self.refresh_node_list)
        self.node_combo.currentIndexChanged.connect(self.refresh_input_list)

        self.btn_add_binding.clicked.connect(self.add_binding)
        self.btn_remove_binding.clicked.connect(self.remove_selected_binding)

        self.btn_choose_out.clicked.connect(self.choose_out_dir)
        self.btn_run.clicked.connect(self.run_batch)
        self.btn_stop.clicked.connect(self.stop_batch)

        self.refresh_node_list()

        self.append_log("✅ First prompt per scene is IMAGE ONLY; video prompts start from the second prompt.")
        self.append_log("Bind image prompts/positive -> your IMAGE text node.")
        self.append_log("Bind video prompts/positive -> your VIDEO positive text node.")
        self.append_log("Bind video start_image_path -> your Load Image (Path) node input (absolute paths).")




    # ---- Loading ----

    def load_prompts(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Prompts JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and ("prompts" in data or "scenes" in data):
                self._prompts_data = data  # <-- add this
                self.scenes = parse_scenes_from_json(data)
                self.prompts_path = path

                total_videos = sum(max(0, len(s) - 1) for s in self.scenes)
                self.append_log(f"Loaded prompts: {path} | scenes={len(self.scenes)} | videos={total_videos}")
                for i, s in enumerate(self.scenes):
                    self.append_log(f"  scene {i+1}: prompts={len(s)} => videos={max(0, len(s)-1)}")
                self.set_status("Prompts loaded (scenes).")
                return

            QMessageBox.critical(self, "Error", "Prompts JSON must be an object with 'prompts' or 'scenes'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load prompts JSON:\n{repr(e)}")

    def load_image_workflow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open IMAGE Workflow JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                wf = json.load(f)
            if not isinstance(wf, dict):
                raise ValueError("Workflow must be a dict keyed by node ids.")
            self.image_wf = wf
            self.image_wf_path = path
            self.append_log(f"Loaded IMAGE workflow: {path} ({len(wf)} nodes)")
            self.set_status("Image workflow loaded.")
            self.refresh_node_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image workflow:\n{repr(e)}")

    def load_video_workflow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open VIDEO Workflow JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                wf = json.load(f)
            if not isinstance(wf, dict):
                raise ValueError("Workflow must be a dict keyed by node ids.")
            self.video_wf = wf
            self.video_wf_path = path
            self.append_log(f"Loaded VIDEO workflow: {path} ({len(wf)} nodes)")
            self.set_status("Video workflow loaded.")
            self.refresh_node_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video workflow:\n{repr(e)}")

    # ---- Bindings UI ----

    def refresh_node_list(self):
        wf_kind = self.workflow_kind_combo.currentText()
        wf = self.image_wf if wf_kind == "image" else self.video_wf
        self.node_combo.clear()
        self.input_combo.clear()

        if not wf:
            self.node_combo.addItem("(load workflow first)", "")
            return

        for node_id, label, class_type in summarize_nodes(wf):
            self.node_combo.addItem(f"{label}  [id={node_id}]  ({class_type})", node_id)
        self.refresh_input_list()

    def refresh_input_list(self):
        wf_kind = self.workflow_kind_combo.currentText()
        wf = self.image_wf if wf_kind == "image" else self.video_wf
        self.input_combo.clear()
        if not wf:
            return
        node_id = self.node_combo.currentData()
        if not node_id:
            return
        keys = list_node_inputs(wf, node_id)
        if not keys:
            self.input_combo.addItem("(no inputs found)", "")
            return
        for k in keys:
            self.input_combo.addItem(k, k)

    def add_binding(self):
        wf_kind = self.workflow_kind_combo.currentText().strip()
        prompt_field = self.prompt_field_edit.text().strip()
        node_id = self.node_combo.currentData()
        input_key = self.input_combo.currentData()

        if not prompt_field:
            QMessageBox.warning(self, "Missing", "Enter a prompt field.")
            return
        if not node_id or not input_key:
            QMessageBox.warning(self, "Missing", "Select a node and an input key.")
            return

        b = Binding(workflow=wf_kind, prompt_field=prompt_field, node_id=str(node_id), input_key=str(input_key))
        self.bindings.append(b)
        self._refresh_bind_table()
        self.append_log(f"Added binding: {asdict(b)}")

    def remove_selected_binding(self):
        row = self.bind_table.currentRow()
        if row < 0 or row >= len(self.bindings):
            return
        removed = self.bindings.pop(row)
        self._refresh_bind_table()
        self.append_log(f"Removed binding: {asdict(removed)}")

    def _refresh_bind_table(self):
        self.bind_table.setRowCount(0)
        for b in self.bindings:
            r = self.bind_table.rowCount()
            self.bind_table.insertRow(r)
            self.bind_table.setItem(r, 0, QTableWidgetItem(b.workflow))
            self.bind_table.setItem(r, 1, QTableWidgetItem(b.prompt_field))
            self.bind_table.setItem(r, 2, QTableWidgetItem(b.node_id))
            self.bind_table.setItem(r, 3, QTableWidgetItem(b.input_key))
    def _require_loaded_everything(self) -> bool:
        if not self.scenes:
            QMessageBox.warning(self, "Missing", "Load prompts JSON first (or load a project).")
            return False
        if not self.image_wf:
            QMessageBox.warning(self, "Missing", "Load image workflow JSON first (or load a project).")
            return False
        if not self.video_wf:
            QMessageBox.warning(self, "Missing", "Load video workflow JSON first (or load a project).")
            return False
        if not self.bindings:
            QMessageBox.warning(self, "Missing", "Add bindings (or load a project).")
            return False
        return True


    def save_project(self):
        if not self._require_loaded_everything():
            return

        folder = QFileDialog.getExistingDirectory(self, "Choose project folder to save into", os.getcwd())
        if not folder:
            return

        try:
            # We want the original prompt/workflow JSON objects, not the expanded scenes.
            # So re-read from prompts_path if present; otherwise derive from current scenes is NOT safe.
            if self.prompts_path and os.path.exists(self.prompts_path):
                prompts_data = _safe_read_json(self.prompts_path)
            else:
                # If you loaded via project, we keep a copy in memory by storing it on self._prompts_data
                prompts_data = getattr(self, "_prompts_data", None)
                if not isinstance(prompts_data, dict):
                    raise RuntimeError("No source prompts JSON available to save. Load prompts from file or project first.")

            image_wf = self.image_wf
            video_wf = self.video_wf
            assert image_wf is not None and video_wf is not None

            project_path = save_project_folder(
                folder,
                comfy_base=self.comfy_base_edit.text().strip(),
                out_dir=self.out_dir_edit.text().strip(),
                poll_interval_s=float(self.poll_spin.value()),
                assemble_final=bool(self.chk_assemble.isChecked()),
                generate_fresh_scene_image=bool(self.chk_scene_image.isChecked()),
                randomize_seeds=bool(self.chk_rand_seed.isChecked()),
                randomize_scope=str(self.seed_scope_combo.currentText()),
                bindings=self.bindings,
                prompts_data=prompts_data,
                image_wf=image_wf,
                video_wf=video_wf,
            )

            self.append_log(f"✅ Project saved: {project_path}")
            self.set_status("Project saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Project Error", repr(e))


    def load_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose project folder to load", os.getcwd())
        if not folder:
            return

        try:
            state = load_project_folder(folder)

            # Restore settings
            self.comfy_base_edit.setText(state.comfy_base)
            self.out_dir_edit.setText(state.out_dir)
            self.poll_spin.setValue(max(1, int(round(state.poll_interval_s))))
            self.chk_assemble.setChecked(bool(state.assemble_final))
            self.chk_scene_image.setChecked(bool(state.generate_fresh_scene_image))
            self.chk_rand_seed.setChecked(bool(state.randomize_seeds))
            self.seed_scope_combo.setCurrentText(state.randomize_scope)

            # Restore JSONs
            self._prompts_data = state.prompts_json  # keep the raw source prompts JSON in memory
            self.image_wf = state.image_workflow_json
            self.video_wf = state.video_workflow_json

            # Restore "paths" as inside-project pseudo paths (optional, for display/logging only)
            self.prompts_path = os.path.join(folder, state.prompts_filename)
            self.image_wf_path = os.path.join(folder, state.image_workflow_filename)
            self.video_wf_path = os.path.join(folder, state.video_workflow_filename)

            # Restore scenes parsed from raw prompts JSON
            self.scenes = parse_scenes_from_json(self._prompts_data)

            # Restore bindings
            self.bindings = [
                Binding(
                    workflow=str(b.get("workflow", "")),
                    prompt_field=str(b.get("prompt_field", "")),
                    node_id=str(b.get("node_id", "")),
                    input_key=str(b.get("input_key", "")),
                )
                for b in (state.bindings or [])
            ]
            self._refresh_bind_table()

            # Refresh node list in UI now that workflows exist
            self.refresh_node_list()

            total_videos = sum(max(0, len(s) - 1) for s in self.scenes)
            self.append_log(f"✅ Project loaded from: {folder}")
            self.append_log(f"  scenes={len(self.scenes)} | videos={total_videos}")
            self.append_log(f"  prompts={self.prompts_path}")
            self.append_log(f"  image_wf={self.image_wf_path}")
            self.append_log(f"  video_wf={self.video_wf_path}")
            self.set_status("Project loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Load Project Error", repr(e))

    # ---- Run ----

    def choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output directory", self.out_dir_edit.text().strip() or os.getcwd())
        if d:
            self.out_dir_edit.setText(d)

    def run_batch(self):
        if not self.scenes:
            QMessageBox.warning(self, "Missing", "Load prompts JSON first.")
            return
        if not self.image_wf:
            QMessageBox.warning(self, "Missing", "Load image workflow JSON first.")
            return
        if not self.video_wf:
            QMessageBox.warning(self, "Missing", "Load video workflow JSON first.")
            return
        if not self.bindings:
            QMessageBox.warning(self, "Missing", "Add bindings.")
            return

        comfy_base = self.comfy_base_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()
        poll_s = float(self.poll_spin.value())
        assemble = bool(self.chk_assemble.isChecked())

        gen_scene_image = bool(self.chk_scene_image.isChecked())
        rand_seed = bool(self.chk_rand_seed.isChecked())
        seed_scope = str(self.seed_scope_combo.currentText())

        self.worker = BatchWorker(
            comfy_base=comfy_base,
            scenes=self.scenes,
            image_workflow=self.image_wf,
            video_workflow=self.video_wf,
            bindings=self.bindings,
            out_dir=out_dir,
            poll_interval_s=poll_s,
            assemble_final=assemble,
            generate_fresh_scene_image=gen_scene_image,
            randomize_seeds=rand_seed,
            randomize_scope=seed_scope,
        )
        self.worker.signals.log.connect(self.append_log)
        total_videos = sum(max(0, len(s) - 1) for s in self.scenes)
        self.worker.signals.progress.connect(lambda i, t: self.set_status(f"Running... {i}/{t} (videos)"))
        self.worker.signals.error.connect(self.on_worker_error)
        self.worker.signals.finished.connect(self.on_worker_finished)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.set_status(f"Running... 0/{max(1, total_videos)} (videos)")
        self.append_log("=== START ===")
        self.threadpool.start(self.worker)

    def stop_batch(self):
        if self.worker:
            self.worker.stop()
        self.append_log("Stop requested.")
        self.btn_stop.setEnabled(False)

    def on_worker_error(self, msg: str):
        self.append_log(f"ERROR: {msg}")
        self.set_status("Error.")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Batch Error", msg)

    def on_worker_finished(self):
        self.append_log("=== DONE ===")
        self.set_status("Done.")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    app.exec_()

if __name__ == "__main__":
    main()
