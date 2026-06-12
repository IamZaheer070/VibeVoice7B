"""
VibeVoice 7B - ipywidgets GUI for Colab / Jupyter.

Why ipywidgets (not Gradio)?
    It renders inline in the notebook with no server, no request queue, and no
    AudioStreamer — which is exactly what caused the earlier deadlocks. The model
    runs in a worker thread; a tiny monitor thread only reads GPU stats and updates
    labels, so it cannot deadlock generation.

Usage (Colab cell)
-------------------
    !pip install -q ipywidgets
    import sys; sys.path.insert(0, "/content/VibeVoice")              # so `import vibevoice` works
    sys.path.insert(0, "/content/VibeVoice/demo")                    # location of these files
    from gui_runner import launch_gui
    launch_gui(model_path="/content/VibeVoice/VibeVoice-7B",
               voices_dir="/content/VibeVoice/demo/voices",
               output_dir="/content/VibeVoice/outputs",
               scripts_dir="/content/VibeVoice/demo/text_examples")

The GUI provides:
    * Live GPU usage banner at the top.
    * Quality preset picker (fast / balanced / high / studio) + advanced overrides
      (cfg_scale, inference_steps, seed) for high-quality vs reliable trade-offs.
    * Per-speaker reference-voice dropdowns, each with a ▶ Preview player so you can
      listen before choosing.
    * Script source: pick a .txt file, upload one, or type inline.
    * Progress bar + status while generating; the result plays inline and is saved
      to outputs/<script_stem>/<script_stem>.wav.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Optional

import ipywidgets as widgets
from IPython.display import Audio, display, clear_output

from cli_runner_7b import (
    QUALITY_PRESETS, VibeVoiceEngine, format_gpu_stats, get_gpu_stats,
    parse_script,
)


def launch_gui(model_path: str,
               voices_dir: Optional[str] = None,
               output_dir: str = "./outputs",
               scripts_dir: Optional[str] = None,
               device: Optional[str] = None,
               checkpoint_path: Optional[str] = None) -> None:
    """Build and display the GUI. Loads the model once (may take a minute)."""

    # ── Header / GPU banner ──────────────────────────────────────────────────
    gpu_banner = widgets.HTML()
    title = widgets.HTML("<h3 style='margin:0'>🎙️ VibeVoice 7B</h3>")

    def render_gpu(stats=None):
        stats = stats if stats is not None else get_gpu_stats()
        text = format_gpu_stats(stats)
        gpu_banner.value = (
            "<div style='font-family:monospace;font-size:12px;padding:6px 10px;"
            "border-radius:6px;background:#0f172a;color:#7dd3fc;'>🖥️ " + text + "</div>"
        )

    render_gpu()
    refresh_gpu_btn = widgets.Button(description="↻ GPU", layout=widgets.Layout(width="80px"))
    refresh_gpu_btn.on_click(lambda _b: render_gpu())

    status = widgets.HTML("<i>Loading model …</i>")
    display(widgets.VBox([widgets.HBox([title, refresh_gpu_btn]), gpu_banner, status]))

    # ── Load engine (blocking, once) ─────────────────────────────────────────
    engine = VibeVoiceEngine(model_path=model_path, device=device,
                             voices_dir=voices_dir, checkpoint_path=checkpoint_path)
    voice_names = engine.voice_mapper.display_names()
    voice_paths = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in engine.voice_mapper.list_voices()}
    status.value = f"<b style='color:#16a34a'>Model ready</b> · {len(voice_names)} reference voices · device={engine.device}"

    # ── Quality controls ─────────────────────────────────────────────────────
    quality = widgets.Dropdown(
        options=[("⚡ Fast / most reliable (5 steps, cfg 1.3)", "fast"),
                 ("⚖️ Balanced — default (10 steps, cfg 1.3)", "balanced"),
                 ("✨ High quality (20 steps, cfg 1.5)", "high"),
                 ("💎 Studio — slow (30 steps, cfg 1.5)", "studio")],
        value="balanced", description="Quality:", style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"))

    cfg_slider = widgets.FloatSlider(value=1.3, min=1.0, max=3.0, step=0.05,
                                     description="CFG scale:", readout_format=".2f",
                                     style={"description_width": "initial"},
                                     layout=widgets.Layout(width="360px"))
    steps_slider = widgets.IntSlider(value=10, min=1, max=50, step=1,
                                     description="Inference steps:",
                                     style={"description_width": "initial"},
                                     layout=widgets.Layout(width="360px"))
    seed_box = widgets.IntText(value=42, description="Seed (-1=random):",
                               style={"description_width": "initial"},
                               layout=widgets.Layout(width="240px"))
    clone_chk = widgets.Checkbox(value=True, description="Voice cloning (recommended)")

    def apply_preset(*_):
        preset = QUALITY_PRESETS[quality.value]
        cfg_slider.value = preset["cfg_scale"]
        steps_slider.value = int(preset["inference_steps"])
    quality.observe(apply_preset, names="value")
    apply_preset()

    advanced = widgets.Accordion(children=[widgets.VBox([cfg_slider, steps_slider, seed_box, clone_chk])])
    advanced.set_title(0, "⚙️ Advanced (override preset)")
    advanced.selected_index = None

    # ── Speaker / voice pickers (with preview) ───────────────────────────────
    num_speakers = widgets.IntSlider(value=2, min=1, max=4, step=1, description="Speakers:",
                                     style={"description_width": "initial"})
    speaker_rows = widgets.VBox()
    preview_out = widgets.Output()

    def make_speaker_row(i: int):
        default = voice_names[i] if i < len(voice_names) else (voice_names[0] if voice_names else None)
        dd = widgets.Dropdown(options=voice_names, value=default,
                              description=f"Speaker {i}:",
                              style={"description_width": "initial"},
                              layout=widgets.Layout(width="320px"))
        btn = widgets.Button(description="▶ Preview", layout=widgets.Layout(width="100px"))

        def _preview(_b, _dd=dd):
            with preview_out:
                clear_output(wait=True)
                path = voice_paths.get(_dd.value)
                if path and os.path.exists(path):
                    print(f"Preview: {_dd.value}")
                    display(Audio(filename=path))
                else:
                    print("Voice file not found.")
        btn.on_click(_preview)
        return widgets.HBox([dd, btn]), dd

    speaker_dropdowns = []

    def rebuild_speakers(*_):
        nonlocal speaker_dropdowns
        rows, dds = [], []
        for i in range(num_speakers.value):
            row, dd = make_speaker_row(i)
            rows.append(row)
            dds.append(dd)
        speaker_dropdowns = dds
        speaker_rows.children = rows
    num_speakers.observe(rebuild_speakers, names="value")
    rebuild_speakers()

    # ── Script source ────────────────────────────────────────────────────────
    script_options = []
    if scripts_dir and os.path.isdir(scripts_dir):
        script_options = sorted(f for f in os.listdir(scripts_dir) if f.lower().endswith(".txt"))
    script_dd = widgets.Dropdown(options=["(type below / upload)"] + script_options,
                                 value="(type below / upload)", description="Script file:",
                                 style={"description_width": "initial"},
                                 layout=widgets.Layout(width="420px"))
    uploader = widgets.FileUpload(accept=".txt", multiple=False, description="Upload .txt")
    script_box = widgets.Textarea(
        placeholder="Speaker 0: Hello, welcome to the show.\nSpeaker 1: Thanks for having me!",
        layout=widgets.Layout(width="100%", height="160px"))
    name_box = widgets.Text(value="", description="Output name:",
                            placeholder="auto from file / 'inline'",
                            style={"description_width": "initial"},
                            layout=widgets.Layout(width="420px"))

    def on_pick_file(*_):
        if script_dd.value in script_options and scripts_dir:
            path = os.path.join(scripts_dir, script_dd.value)
            try:
                script_box.value = open(path, encoding="utf-8").read()
                name_box.value = os.path.splitext(script_dd.value)[0]
            except Exception as e:
                script_box.value = f"<could not read file: {e}>"
    script_dd.observe(on_pick_file, names="value")

    def on_upload(*_):
        if not uploader.value:
            return
        item = list(uploader.value.values())[0] if isinstance(uploader.value, dict) else uploader.value[0]
        content = item["content"] if isinstance(item, dict) else item.content
        fname = item["metadata"]["name"] if isinstance(item, dict) else item.name
        script_box.value = content.decode("utf-8", errors="replace")
        name_box.value = os.path.splitext(fname)[0]
    uploader.observe(on_upload, names="value")

    # ── Generate + output ────────────────────────────────────────────────────
    generate_btn = widgets.Button(description="🚀 Generate", button_style="primary",
                                  layout=widgets.Layout(width="180px", height="40px"))
    progress = widgets.FloatProgress(value=0, min=0, max=1.0, description="Progress:",
                                     bar_style="info", style={"description_width": "initial"},
                                     layout=widgets.Layout(width="100%"))
    gen_status = widgets.HTML("")
    result_out = widgets.Output()

    def progress_cb(info):
        # Called from the monitor thread; widget updates from threads are fine in Colab.
        progress.value = max(progress.value, float(info.get("fraction", 0.0)))
        render_gpu(info.get("gpu"))
        gen_status.value = f"<span style='font-family:monospace'>elapsed {info.get('elapsed',0):.0f}s · {progress.value*100:.0f}%</span>"

    def do_generate(_b):
        generate_btn.disabled = True
        progress.value = 0.0
        progress.bar_style = "info"
        with result_out:
            clear_output()
        text = script_box.value.strip()
        if not text:
            with result_out:
                print("❌ Please provide a script (pick a file, upload, or type).")
            generate_btn.disabled = False
            return

        names = [dd.value for dd in speaker_dropdowns]
        seed = None if seed_box.value is None or seed_box.value < 0 else int(seed_box.value)
        stem = (name_box.value.strip()
                or "inline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

        def worker():
            try:
                result = engine.generate(
                    script=text, speaker_names=names,
                    cfg_scale=cfg_slider.value, inference_steps=steps_slider.value,
                    seed=seed, disable_voice_cloning=not clone_chk.value,
                    num_speakers_hint=num_speakers.value, progress_callback=progress_cb)
                path = engine.save(result, output_dir, stem)
                progress.value = 1.0
                progress.bar_style = "success"
                with result_out:
                    clear_output()
                    print(f"✅ Saved → {path}")
                    print(f"   duration {result.duration_s:.2f}s · gen {result.generation_s:.1f}s · RTF {result.rtf:.2f}x")
                    display(Audio(filename=path, autoplay=False))
            except Exception as e:
                import traceback
                progress.bar_style = "danger"
                with result_out:
                    clear_output()
                    print(f"❌ Generation failed: {e}\n")
                    traceback.print_exc()
            finally:
                generate_btn.disabled = False

        threading.Thread(target=worker, daemon=True).start()

    generate_btn.on_click(do_generate)

    # ── Layout ───────────────────────────────────────────────────────────────
    settings_panel = widgets.VBox([quality, advanced])
    voices_panel = widgets.VBox([num_speakers, speaker_rows,
                                 widgets.HTML("<small>Tip: preview a voice before generating.</small>"),
                                 preview_out])
    script_panel = widgets.VBox([widgets.HBox([script_dd, uploader]), name_box, script_box])

    tabs = widgets.Tab(children=[settings_panel, voices_panel, script_panel])
    tabs.set_title(0, "🎛️ Quality")
    tabs.set_title(1, "🎭 Voices")
    tabs.set_title(2, "📝 Script")

    display(widgets.VBox([tabs, generate_btn, progress, gen_status, result_out]))


if __name__ == "__main__":
    print("This module is meant to be imported in a notebook:\n"
          "    from gui_runner import launch_gui\n"
          "    launch_gui(model_path='/content/VibeVoice/VibeVoice-7B')")