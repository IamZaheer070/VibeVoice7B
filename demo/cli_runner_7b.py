"""
VibeVoice 7B - Stable runner (core engine + CLI).

Highlights
----------
* Voice cloning ON by default (requirement 1).
* Organized output: outputs/<script_stem>/<script_stem>.wav + <script_stem>.settings.json.
  Generated audio is named after the input text file.
* Live GPU usage + an (estimate-based) progress bar during generation.
* Robust error handling: attn fallback, OOM hints, empty/NaN-audio detection.
* Quality presets so you can trade speed vs fidelity without remembering numbers.
* Importable `VibeVoiceEngine` used by the ipywidgets GUI (gui_runner.py).

CLI examples
------------
# Single script, voice cloning on, "balanced" quality, named after the txt file
python cli_runner_7b.py \
    --model_path /content/VibeVoice/VibeVoice-7B \
    --txt_path demo/text_examples/2p_short.txt \
    --speaker_names Alice Frank \
    --quality balanced

# High quality, explicit overrides win over the preset
python cli_runner_7b.py --model_path ... --txt_path script.txt \
    --speaker_names Alice --quality high --cfg_scale 1.6 --inference_steps 24

# Batch: every .txt in a folder -> its own output sub-folder
python cli_runner_7b.py --model_path ... --batch_dir demo/text_examples \
    --speaker_names Alice Frank --quality balanced

# List the reference voices the model can clone
python cli_runner_7b.py --model_path ... --list_voices
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Quality presets  (heuristic starting points — tune to taste)
# ──────────────────────────────────────────────────────────────────────────────
# cfg_scale : higher = stronger adherence to text/voice, but too high can sound
#             tense/over-articulated. 1.3 is the project default.
# steps     : DDPM denoising steps. More steps = cleaner audio but slower.
QUALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "fast":     {"cfg_scale": 1.3, "inference_steps": 5},    # quickest / most reliable
    "balanced": {"cfg_scale": 1.3, "inference_steps": 10},   # project default
    "high":     {"cfg_scale": 1.5, "inference_steps": 20},   # cleaner, slower
    "studio":   {"cfg_scale": 1.5, "inference_steps": 30},   # slowest, short clips only
}

SAMPLE_RATE = 24000


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    icon = {"INFO": "✅", "WARN": "⚠️ ", "ERROR": "❌", "STEP": "🔹", "GPU": "🖥️ "}.get(level, "  ")
    print(f"[{ts}] {icon} {msg}", flush=True)


def rule(title: str = "") -> None:
    width = 64
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("─" * pad + f" {title} " + "─" * pad, flush=True)
    else:
        print("─" * width, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# GPU monitoring
# ──────────────────────────────────────────────────────────────────────────────
def get_gpu_stats() -> Optional[dict]:
    """Return {name, util, mem_used, mem_total, temp} (MiB) or None if no GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3, stderr=subprocess.DEVNULL,
        )
        name, util, used, total, temp = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"name": name, "util": float(util), "mem_used": float(used),
                "mem_total": float(total), "temp": float(temp)}
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return {"name": props.name, "util": float("nan"),
                    "mem_used": torch.cuda.memory_reserved(0) / (1024 ** 2),
                    "mem_total": props.total_memory / (1024 ** 2), "temp": float("nan")}
        except Exception:
            return None
    return None


def format_gpu_stats(stats: Optional[dict]) -> str:
    if stats is None:
        return "GPU: (not available)"
    util = "n/a" if stats["util"] != stats["util"] else f"{stats['util']:.0f}%"
    temp = "" if stats["temp"] != stats["temp"] else f"  {stats['temp']:.0f}°C"
    mem_pct = 100.0 * stats["mem_used"] / max(1.0, stats["mem_total"])
    return (f"{stats['name']} | util {util} | "
            f"mem {stats['mem_used']/1024:.1f}/{stats['mem_total']/1024:.1f} GiB ({mem_pct:.0f}%){temp}")


class GenerationMonitor:
    """
    Background thread reporting elapsed time, an estimate-based progress fraction,
    and live GPU stats while a generation runs. It only READS gpu stats and calls
    a callback — it never touches the model, so it cannot deadlock the run.
    """
    def __init__(self, expected_seconds: float,
                 callback: Optional[Callable[[dict], None]] = None, interval: float = 1.0):
        self.expected_seconds = max(8.0, expected_seconds)
        self.callback = callback
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.start_time = 0.0

    def _run(self) -> None:
        tau = self.expected_seconds / 2.0
        while not self._stop.is_set():
            elapsed = time.time() - self.start_time
            frac = 0.95 * (1.0 - math.exp(-elapsed / tau))   # eases toward 95%
            gpu = get_gpu_stats()
            if self.callback is not None:
                self.callback({"elapsed": elapsed, "fraction": frac, "gpu": gpu})
            else:
                log(f"... {elapsed:5.1f}s  ~{frac*100:4.1f}%  |  {format_gpu_stats(gpu)}", "GPU")
            self._stop.wait(self.interval)

    def __enter__(self) -> "GenerationMonitor":
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.callback is not None:
            self.callback({"elapsed": time.time() - self.start_time, "fraction": 1.0,
                           "gpu": get_gpu_stats()})


# ──────────────────────────────────────────────────────────────────────────────
# Voice mapping
# ──────────────────────────────────────────────────────────────────────────────
class VoiceMapper:
    """Maps speaker names to reference voice files for cloning."""
    SUPPORTED = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

    def __init__(self, voices_dir: Optional[str] = None) -> None:
        if voices_dir is None:
            voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
        self.voices_dir = voices_dir
        self.voice_presets: Dict[str, str] = {}
        self._setup()

    def _setup(self) -> None:
        if not os.path.exists(self.voices_dir):
            log(f"Voices directory not found at {self.voices_dir}", "WARN")
            return
        files = sorted(f for f in os.listdir(self.voices_dir)
                       if f.lower().endswith(self.SUPPORTED)
                       and os.path.isfile(os.path.join(self.voices_dir, f)))
        for f in files:
            self.voice_presets[os.path.splitext(f)[0]] = os.path.join(self.voices_dir, f)
        # Short aliases: "en-Alice_woman" -> "Alice"
        aliases: Dict[str, str] = {}
        for name, path in self.voice_presets.items():
            short = name.split("_")[0]
            if "-" in short:
                short = short.split("-")[-1]
            aliases.setdefault(short, path)
        for k, v in aliases.items():
            self.voice_presets.setdefault(k, v)
        if self.voice_presets:
            uniq = self.list_voices()
            log(f"Found {len(uniq)} reference voices in {self.voices_dir}")
            log("Available voices: " + ", ".join(os.path.splitext(os.path.basename(p))[0] for p in uniq))

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        low = speaker_name.lower()
        for name, path in self.voice_presets.items():
            if name.lower() in low or low in name.lower():
                log(f"Fuzzy-matched speaker '{speaker_name}' -> '{name}'", "WARN")
                return path
        if self.voice_presets:
            default = list(self.voice_presets.values())[0]
            log(f"No voice preset for '{speaker_name}', using default {os.path.basename(default)}", "WARN")
            return default
        raise ValueError(f"No voice presets available in {self.voices_dir}")

    def list_voices(self) -> List[str]:
        seen = {}
        for path in self.voice_presets.values():
            seen[path] = os.path.splitext(os.path.basename(path))[0]
        return [p for p, _ in sorted(seen.items(), key=lambda kv: kv[1].lower())]

    def display_names(self) -> List[str]:
        return [os.path.splitext(os.path.basename(p))[0] for p in self.list_voices()]


# ──────────────────────────────────────────────────────────────────────────────
# Script parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_script(script: str, fallback_speakers: int) -> Tuple[str, int]:
    """Normalize a script to 'Speaker 0..N' format. Returns (formatted, num_speakers)."""
    lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
    pat = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
    has_explicit = any(pat.match(ln) for ln in lines)
    parsed: List[Tuple[int, str]] = []

    if has_explicit:
        cur_id, cur_text = None, ""
        for ln in lines:
            m = pat.match(ln)
            if m:
                if cur_id is not None and cur_text:
                    parsed.append((cur_id, cur_text.strip()))
                cur_id = int(m.group(1))
                cur_text = m.group(2).strip()
            else:
                cur_text = f"{cur_text} {ln}".strip() if cur_text else ln
        if cur_id is not None and cur_text:
            parsed.append((cur_id, cur_text.strip()))
        ids = [sid for sid, _ in parsed]
        if ids and min(ids) > 0:
            shift = min(ids)
            parsed = [(sid - shift, t) for sid, t in parsed]
    else:
        for i, ln in enumerate(lines):
            parsed.append((i % max(1, fallback_speakers), ln))

    if not parsed:
        return "", 0
    num = max(sid for sid, _ in parsed) + 1
    formatted = "\n".join(f"Speaker {sid}: {t}" for sid, t in parsed)
    return formatted, num


# ──────────────────────────────────────────────────────────────────────────────
# Output path handling (requirement 1: name audio after the input txt file)
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize(value: str, max_len: int = 80) -> str:
    value = re.sub(r"\s+", "_", (value or "").strip())
    value = re.sub(r"[^A-Za-z0-9_\-\.]+", "", value)
    return value[:max_len] or "audio"


def build_output_paths(output_root: str, script_stem: str) -> Tuple[str, str]:
    """outputs/<stem>/<stem>.wav (+ _v2,_v3 if it exists) and a settings sidecar."""
    stem = _sanitize(script_stem)
    folder = os.path.join(output_root, stem)
    os.makedirs(folder, exist_ok=True)
    wav = os.path.join(folder, f"{stem}.wav")
    if os.path.exists(wav):
        n = 2
        while os.path.exists(os.path.join(folder, f"{stem}_v{n}.wav")):
            n += 1
        wav = os.path.join(folder, f"{stem}_v{n}.wav")
    settings = os.path.splitext(wav)[0] + ".settings.json"
    return wav, settings


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GenerationResult:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    generation_s: float
    rtf: float
    settings: dict = field(default_factory=dict)
    output_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────
class VibeVoiceEngine:
    """Loads the model once; generate() can be called many times (GUI-friendly)."""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 voices_dir: Optional[str] = None, checkpoint_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.voice_mapper = VoiceMapper(voices_dir)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None
        self._load()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is None:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        device = device.lower()
        if device == "mpx":
            device = "mps"
        if device == "mps" and not torch.backends.mps.is_available():
            log("MPS not available; falling back to CPU.", "WARN")
            return "cpu"
        return device

    def _load(self) -> None:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        rule("Model Loading")
        log(f"Loading processor from {self.model_path}", "STEP")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        if self.device == "mps":
            dtype, attn = torch.float32, "sdpa"
        elif self.device == "cuda":
            dtype, attn = torch.bfloat16, "flash_attention_2"
        else:
            dtype, attn = torch.float32, "sdpa"

        log(f"Loading model  device={self.device}  dtype={dtype}  attn={attn}", "STEP")
        Model = VibeVoiceForConditionalGenerationInference
        try:
            if self.device == "mps":
                self.model = Model.from_pretrained(
                    self.model_path, torch_dtype=dtype, attn_implementation=attn, device_map=None)
                self.model.to("mps")
            else:
                self.model = Model.from_pretrained(
                    self.model_path, torch_dtype=dtype, device_map=self.device, attn_implementation=attn)
        except Exception as exc:
            if attn == "flash_attention_2":
                log(f"flash_attention_2 unavailable ({type(exc).__name__}: {exc}). Falling back to sdpa.", "WARN")
                log("Tip: `pip install flash-attn --no-build-isolation` for faster runs.", "INFO")
                self.model = Model.from_pretrained(
                    self.model_path, torch_dtype=dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation="sdpa")
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise

        if self.checkpoint_path:
            from vibevoice.modular.lora_loading import load_lora_assets
            log(f"Loading LoRA adapter from {self.checkpoint_path}", "STEP")
            report = load_lora_assets(self.model, self.checkpoint_path)
            loaded = [n for n, ok in (
                ("language LoRA", report.language_model),
                ("diffusion head LoRA", report.diffusion_head_lora),
                ("diffusion head weights", report.diffusion_head_full),
                ("acoustic connector", report.acoustic_connector),
                ("semantic connector", report.semantic_connector),
            ) if ok]
            log("Adapter components: " + (", ".join(loaded) if loaded else "none (check path)"),
                "INFO" if loaded else "WARN")

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++", beta_schedule="squaredcos_cap_v2")
        if hasattr(self.model.model, "language_model"):
            log(f"LM attention: {self.model.model.language_model.config._attn_implementation}")
        log("Model ready.", "INFO")

    def generate(self, script: str, speaker_names: List[str], cfg_scale: float = 1.3,
                 inference_steps: int = 10, seed: Optional[int] = None,
                 disable_voice_cloning: bool = False, num_speakers_hint: Optional[int] = None,
                 progress_callback: Optional[Callable[[dict], None]] = None) -> GenerationResult:
        fallback = num_speakers_hint or (len(speaker_names) if speaker_names else 1)
        formatted, num_speakers = parse_script(script, fallback)
        if not formatted:
            raise ValueError("No valid script content found.")
        formatted = formatted.replace("\u2019", "'").replace("\u2018", "'")

        if not speaker_names:
            speaker_names = self.voice_mapper.display_names()[:num_speakers]
        if len(speaker_names) < num_speakers:
            raise ValueError(f"Script needs {num_speakers} speakers but only {len(speaker_names)} name(s) given.")
        speaker_names = speaker_names[:num_speakers]

        voice_samples = None
        if not disable_voice_cloning:
            paths = [self.voice_mapper.get_voice_path(n) for n in speaker_names]
            voice_samples = [self._read_audio(p) for p in paths]
            for n, p in zip(speaker_names, paths):
                log(f"  voice: {n} -> {os.path.basename(p)}")

        self.model.set_ddpm_inference_steps(num_steps=int(inference_steps))

        torch_device = self.device if self.device in ("cuda", "mps") else "cpu"
        generator = None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                generator = torch.Generator(device=torch_device)
                generator.manual_seed(int(seed))
            except Exception:
                generator = None

        kwargs = dict(text=[formatted], padding=True, return_tensors="pt", return_attention_mask=True)
        if voice_samples is not None:
            kwargs["voice_samples"] = [voice_samples]
        inputs = self.processor(**kwargs)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(torch_device)

        expected = max(15.0, len(formatted.split()) * 0.6)   # cosmetic estimate

        log("Starting generation ...", "STEP")
        t0 = time.time()
        try:
            with GenerationMonitor(expected_seconds=expected, callback=progress_callback):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": False},
                        generator=generator, verbose=False, refresh_negative=True,
                        is_prefill=not disable_voice_cloning)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            torch.cuda.empty_cache()
            raise RuntimeError("CUDA out of memory. Try a shorter script, fewer inference "
                               "steps, or the 'fast' preset. (cache cleared)") from exc
        gen_s = time.time() - t0

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("Model returned no audio. Check script formatting and speaker count.")
        audio = self._to_numpy(outputs.speech_outputs[0])
        if audio.size == 0:
            raise RuntimeError("Generated audio is empty.")
        if not np.isfinite(audio).all():
            log("Non-finite samples detected; sanitizing.", "WARN")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        dur = len(audio) / SAMPLE_RATE
        rtf = gen_s / dur if dur > 0 else float("inf")
        log(f"Done in {gen_s:.1f}s | audio {dur:.2f}s | RTF {rtf:.2f}x")

        return GenerationResult(
            audio=audio, sample_rate=SAMPLE_RATE, duration_s=dur, generation_s=gen_s, rtf=rtf,
            settings={"model_path": self.model_path, "speaker_names": speaker_names,
                      "num_speakers": num_speakers, "cfg_scale": cfg_scale,
                      "inference_steps": int(inference_steps), "seed": seed,
                      "voice_cloning": not disable_voice_cloning, "device": self.device})

    @staticmethod
    def _read_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
        import soundfile as sf
        import librosa
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)

    @staticmethod
    def _to_numpy(speech) -> np.ndarray:
        if torch.is_tensor(speech):
            if speech.dtype == torch.bfloat16:
                speech = speech.float()
            speech = speech.detach().cpu().numpy()
        return np.asarray(speech, dtype=np.float32).squeeze()

    def save(self, result: GenerationResult, output_root: str, script_stem: str) -> str:
        import soundfile as sf
        wav_path, settings_path = build_output_paths(output_root, script_stem)
        audio = result.audio
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        sf.write(wav_path, (audio * 32767).astype(np.int16), result.sample_rate, subtype="PCM_16")
        result.output_path = wav_path
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump({**result.settings, "duration_s": round(result.duration_s, 3),
                       "generation_s": round(result.generation_s, 3), "rtf": round(result.rtf, 3),
                       "saved_at": datetime.now().isoformat(timespec="seconds"),
                       "wav": os.path.basename(wav_path)}, fh, indent=2)
        log(f"Saved audio → {wav_path}")
        log(f"Saved settings → {settings_path}")
        return wav_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def resolve_quality(args: argparse.Namespace) -> Tuple[float, int]:
    """Preset gives the base; explicit --cfg_scale / --inference_steps override."""
    preset = QUALITY_PRESETS.get(args.quality, QUALITY_PRESETS["balanced"])
    cfg = args.cfg_scale if args.cfg_scale is not None else preset["cfg_scale"]
    steps = args.inference_steps if args.inference_steps is not None else int(preset["inference_steps"])
    return float(cfg), int(steps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VibeVoice 7B – stable runner (CLI).",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="Local VibeVoice 7B model directory")
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--txt_path", help="Path to a script .txt file")
    src.add_argument("--script", help="Inline script text (overrides --txt_path)")
    src.add_argument("--batch_dir", help="Process every .txt in this folder")
    p.add_argument("--speaker_names", nargs="+", default=None,
                   help="Speaker names in order, e.g. --speaker_names Alice Frank")
    p.add_argument("--num_speakers", type=int, default=None,
                   help="Speaker count when the script has no 'Speaker N:' tags")
    p.add_argument("--voices_dir", default=None, help="Reference voices directory (default: ./voices)")
    p.add_argument("--output_dir", default="./outputs", help="Root output directory")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detected if omitted)")
    p.add_argument("--quality", choices=list(QUALITY_PRESETS), default="balanced",
                   help="Quality preset: fast | balanced | high | studio")
    p.add_argument("--cfg_scale", type=float, default=None, help="Override preset CFG scale")
    p.add_argument("--inference_steps", type=int, default=None, help="Override preset DDPM steps")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--checkpoint_path", default=None, help="LoRA adapter directory (optional)")
    p.add_argument("--disable_voice_cloning", action="store_true",
                   help="Turn OFF voice cloning (cloning is ON by default)")
    p.add_argument("--list_voices", action="store_true", help="List reference voices and exit")
    return p.parse_args()


def _run_one(engine: VibeVoiceEngine, script_text: str, script_stem: str, args, cfg, steps) -> None:
    rule(f"Generate: {script_stem}")
    result = engine.generate(
        script=script_text, speaker_names=args.speaker_names or [], cfg_scale=cfg,
        inference_steps=steps, seed=args.seed, disable_voice_cloning=args.disable_voice_cloning,
        num_speakers_hint=args.num_speakers)
    engine.save(result, args.output_dir, script_stem)


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.model_path):
        raise SystemExit(f"Model path not found: {args.model_path}")

    if args.list_voices:
        vm = VoiceMapper(args.voices_dir)
        rule("Reference Voices")
        for path in vm.list_voices():
            print(f"  {os.path.splitext(os.path.basename(path))[0]:24s} {path}")
        return

    cfg, steps = resolve_quality(args)
    log(f"Quality '{args.quality}'  ->  cfg_scale={cfg}  inference_steps={steps}")

    engine = VibeVoiceEngine(model_path=args.model_path, device=args.device,
                             voices_dir=args.voices_dir, checkpoint_path=args.checkpoint_path)
    try:
        if args.batch_dir:
            files = sorted(f for f in os.listdir(args.batch_dir) if f.lower().endswith(".txt"))
            if not files:
                raise SystemExit(f"No .txt files in {args.batch_dir}")
            log(f"Batch: {len(files)} script(s). (Single GPU runs these sequentially.)")
            for i, f in enumerate(files, 1):
                rule(f"[{i}/{len(files)}] {f}")
                text = open(os.path.join(args.batch_dir, f), encoding="utf-8").read()
                _run_one(engine, text, os.path.splitext(f)[0], args, cfg, steps)
        else:
            if args.script:
                text, stem = args.script, "inline_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            elif args.txt_path:
                if not os.path.exists(args.txt_path):
                    raise SystemExit(f"Script file not found: {args.txt_path}")
                text = open(args.txt_path, encoding="utf-8").read()
                stem = os.path.splitext(os.path.basename(args.txt_path))[0]
            else:
                raise SystemExit("Provide --txt_path, --script, or --batch_dir.")
            _run_one(engine, text, stem, args, cfg, steps)
    except Exception:
        import traceback
        log("Generation failed:", "ERROR")
        traceback.print_exc()
        raise SystemExit(1)

    rule("All done 🎙️")


if __name__ == "__main__":
    main()