# ✨ Comfy Batch GUI — Multi-Scene Chained Generation (Image → Video → Concat)

A tiny, practical PyQt (qtpy) GUI for running **ComfyUI** in **batch mode** with **multi-scene support**, **prompt chaining**, and a **clean “project folder” workflow**.

This tool is built for one job:

> Generate a **still image per scene** (clean, no dialogue), then generate **video segments from the next prompts**, chaining each segment from the **last frame** of the previous one — and optionally **concatenate** everything into a final MP4.

---

## 🌟 What This Does

### ✅ Multi-Scene, Two-Pass Pipeline
For each scene:

1. **PASS 1:** generate the scene’s *initial still image* from the **first prompt only**
2. **PASS 2:** generate video segments from prompts **starting at index 1**
3. After each segment, the tool extracts the **last frame** (ffmpeg) and uses it as the **next start image**
4. Optionally **concats** all segments into `final.mp4`

### ✅ “First Prompt = Image Only”
This is the core idea.

- Prompt 0 → **IMAGE workflow**
- Prompt 1..N → **VIDEO workflow**

So you can keep your still prompt clean and art-directed, and put dialogue / action / timing in the video prompts.

### ✅ Project Save / Load (Portable)
You can save a **project folder** that contains:
- your **prompt JSON**
- your **image workflow JSON**
- your **video workflow JSON**
- your **settings + bindings** (`project.json`)

Loading that folder restores the entire run configuration in one click.

---

## 🧠 Why This Exists

When doing iterative AI animation, you often want:
- one strong establishing frame per scene
- a controlled motion chain without “resetting” between segments
- a way to preserve the **exact workflows and prompts used** for reproducibility
- “session management” (project folders) instead of hunting for file paths

This repo is a simple, opinionated solution.

---

## 🧩 Features

- 🧷 **Bindings system** to map prompt fields → ComfyUI node inputs  
- 🎬 **Chained video segments** using last-frame extraction (ffmpeg)
- 🧱 **Multi-scene support**
- 🎲 **Seed randomization** (per scene or per generation)
- 📦 **Project folder export/import** (everything included)
- 🧾 `manifest.json` output for auditability + reruns

---

## 📁 Output Structure

A typical run produces:

```

comfy_batch_outputs/
stills/
...initial still images...
videos/
...segment videos...
chain_frames/
...last-frame PNGs used for chaining...
manifest.json
final.mp4   (optional)

````

`manifest.json` includes:
- each scene’s still
- each segment path
- final concat path (if created)

---

## 🧾 Prompt JSON Formats

### 1) Single Scene
```json
{
  "segment_sec": 5,
  "prompts": [
    "p0 (image only)",
    "p1 (video)",
    "p2 (video)"
  ]
}
````

### 2) Multi-Scene (list of lists)

```json
{
  "segment_sec": 5,
  "prompts": [
    ["scene1 p0 (image)", "scene1 p1 (video)", "scene1 p2 (video)"],
    ["scene2 p0 (image)", "scene2 p1 (video)"]
  ]
}
```

### 3) Explicit Scenes (recommended for duration expansion)

```json
{
  "segment_sec": 5,
  "scenes": [
    { "prompts": ["s1 p0", "s1 p1", "s1 p2"], "total_seconds": 30 },
    { "prompts": ["s2 p0", "s2 p1"], "total_segments": 8 }
  ]
}
```

---

## 🔗 Bindings: The Secret Sauce

Bindings tell the tool how to inject values into your ComfyUI workflow JSON.

Examples you’ll typically bind:

### IMAGE workflow

* `positive` / `prompts` → the image prompt text node
* `negative` → negative prompt field
* `seed` → sampler seed input

### VIDEO workflow

* `positive` / `prompts` → video prompt node
* `negative` → negative prompt node
* `seed` → sampler seed input
* `start_image_path` → your “Load Image (Path)” node input (**absolute paths**)

> Tip: make sure your ComfyUI workflow uses a node that accepts a file path string for the starting image.

---

## 🧙 Project Save / Load

### Saving a project

Creates a folder containing:

```
my_project/
  project.json
  prompts.json
  workflow_image.json
  workflow_video.json
```

This means:

* the repo can be used like a “mini production system”
* every run can be archived and restored later
* teams can share project folders without missing dependencies

### Loading a project

Restores:

* prompts (and scenes)
* workflows
* bindings
* all UI settings

No re-selecting files. No guessing which workflow version you used.

---

## ⚙️ Requirements

* Python 3.9+
* ComfyUI running (local or remote)
* `ffmpeg` available in PATH

### Python deps

Install the basics:

```bash
pip install -r requirements.txt
```

If you don’t have one yet, you likely need:

* `qtpy`
* a Qt backend (`PyQt6`)
* `requests`
* `urllib3`

Example:

```bash
pip install qtpy PySide6 requests urllib3
```

### ffmpeg

Confirm it works:

```bash
ffmpeg -version
ffprobe -version
```

---

## 🚀 Running

```bash
python comfy_batch_gui.py
```

Then in the GUI:

1. Set your ComfyUI base URL (default: `http://localhost:8188`)
2. Load your **Prompts JSON**
3. Load your **Image Workflow JSON**
4. Load your **Video Workflow JSON**
5. Add bindings (image + video)
6. Choose output directory
7. Run ✨

---

## 🪄 Best Practices (Highly Recommended)

* Keep prompt 0 per scene *purely visual* (no dialogue)
* Put speech/intent/action in the subsequent prompts
* Make sure your video workflow:

  * reads `start_image_path`
  * outputs video (gif/mp4) consistently
* Always save a project folder when a run “works” — it becomes a reproducible asset

---

## 🧯 Troubleshooting

### “No video output found”

Your workflow probably outputs under a different field than `videos/gifs`.
Check `parse_best_outputs()` and confirm your Comfy nodes output into:

* `outputs -> videos` or `outputs -> gifs`

### “ffmpeg not found”

Install ffmpeg and ensure it’s in your PATH.

### “Timed out waiting for prompt”

Increase `max_wait_s` or check your ComfyUI server isn’t stalled.

---

## 🗺️ Roadmap Ideas (if you want to expand)

* per-scene concat + crossfade transitions
* per-segment overrides (CFG, steps, sampler, model switch)
* auto-binding presets for popular workflows
* render queue visualizer (thumbnails + progress per segment)

---

## ❤️ Credits / Philosophy

This repo intentionally stays small and readable.
It’s not a framework — it’s a **tool**.

If you’re building long-form or episodic AI animation, this is meant to be the reliable little workhorse that keeps runs reproducible and chaining stable.

---

## 📜 License

Choose your vibe:

* MIT for maximum openness
* or keep it private if it’s pipeline code

(If you tell me your preference, I can generate the `LICENSE` file too.)

```
