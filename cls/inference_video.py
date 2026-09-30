"""EdgeCVS inference video renderer.

Runs dual-head (CLS + SEG) inference on a surgical video and renders a
medical-device style overlay. Produces TWO variants per run, side by side in
the same output folder:

  {video_name}_seg.mp4    — anatomy segmentation overlay + CVS panel
  {video_name}_plain.mp4  — CVS panel only (clean feed)

Output directory and CLI/config interface are unchanged from the previous
version; only the rendering differs. Videos are re-encoded to H.264
(yuv420p + faststart) via ffmpeg when available, so they are web/GitHub-Pages
ready as-is. Frames are upscaled to ``render_height`` (default 1080) BEFORE
the UI is drawn, so panel text stays crisp even for low-resolution sources.
"""

import os
import shutil
import subprocess
import time
from collections import deque

import cv2
import hydra
import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from models.classifier import build_model
from utils.hub import download_run, load_checkpoint
from utils.utils import load_experiment_config

# Normalization constants
ENDOSCAPES_MEAN = [0.347791, 0.224614, 0.232119]
ENDOSCAPES_STD = [0.230785, 0.178361, 0.178082]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------------
# UI theme (RGB) — calm surgical-monitor palette, single teal accent
# ----------------------------------------------------------------------------
INK_PANEL = (9, 17, 19)          # panel ground
TEAL = (47, 214, 163)            # accent / criterion met
RED = (232, 90, 90)              # criterion not met
SLATE = (114, 134, 131)          # inactive
TXT_HI = (236, 243, 241)         # primary text
TXT_LO = (146, 163, 160)         # secondary text
HAIRLINE = (255, 255, 255, 26)   # 10% white separators

# Segmentation classes: index -> (display name, RGB fill)
SEG_CLASSES = {
    1: ("Cystic plate", (167, 139, 250)),
    2: ("Hepatocystic triangle", (245, 196, 81)),
    3: ("Cystic artery", (244, 113, 116)),
    4: ("Cystic duct", (74, 222, 128)),
    5: ("Gallbladder", (96, 165, 250)),
    6: ("Instrument", (148, 187, 183)),
}
MASK_ALPHA = 0.38                # fill opacity
TOOL_CLASS = 6                   # instruments: outline only, lighter fill

CRITERIA = [
    ("C1", "Two structures"),
    ("C2", "Hepatocystic triangle"),
    ("C3", "Cystic plate"),
]
THRESH_ON, THRESH_OFF = 0.55, 0.45   # hysteresis around the 0.5 decision line


def _load_font(size, bold=False, mono=False):
    """DejaVu ships with matplotlib and most Linux distros; fall back safely."""
    names = (["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"] if mono else
             ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    search = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]
    try:
        import matplotlib
        search.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except ImportError:
        pass
    for d in search:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class HudRenderer:
    """Draws the medical-device interface on top of (already upscaled) frames."""

    def __init__(self, width, height, device_label, seg_variant):
        self.W, self.H = width, height
        self.s = height / 1080.0                     # UI scale factor
        s = self.s
        self.seg_variant = seg_variant
        self.device_label = device_label

        self.f_title = _load_font(round(25 * s), bold=True)
        self.f_small = _load_font(round(16 * s))
        self.f_tiny = _load_font(round(14 * s))
        self.f_mono = _load_font(round(18 * s), mono=True)
        self.f_mono_sm = _load_font(round(15 * s), mono=True)

        # CVS panel fonts — deliberately large for demo readability
        self.f_p_head = _load_font(round(23 * s))
        self.f_p_chip = _load_font(round(22 * s), bold=True)
        self.f_p_label = _load_font(round(30 * s))
        self.f_p_code = _load_font(round(20 * s), mono=True)
        self.f_p_val = _load_font(round(22 * s), mono=True)

        # geometry
        self.margin = round(28 * s)
        self.panel_w = round(742 * s)
        self.row_h = round(76 * s)
        self.header_h = round(72 * s)
        self.panel_h = self.header_h + 3 * self.row_h + round(22 * s)
        self.radius = round(16 * s)

        self._legend_rows = list(SEG_CLASSES.items())

    # -- helpers -------------------------------------------------------------
    def _panel(self, draw, xy, alpha=216):
        draw.rounded_rectangle(xy, radius=self.radius,
                               fill=(*INK_PANEL, alpha),
                               outline=(255, 255, 255, 34), width=1)

    def _text_w(self, font, text):
        return font.getbbox(text)[2]

    # -- main entry ----------------------------------------------------------
    def render(self, frame_bgr, probs, met, latency_ms, present_classes):
        """probs: smoothed [3] floats; met: [3] bools (hysteresis)."""
        base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        hud = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(hud)
        self._draw_brand(d)
        self._draw_perf_chip(d, latency_ms)
        self._draw_cvs_panel(d, probs, met)
        if self.seg_variant:
            self._draw_legend(d, present_classes)
        if all(met):
            self._draw_achieved_frame(d)

        out = Image.alpha_composite(base, hud).convert("RGB")
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

    # -- components ----------------------------------------------------------
    def _draw_brand(self, d):
        s, m = self.s, self.margin
        sub = "DUAL-HEAD INFERENCE" if self.seg_variant else "REAL-TIME CVS ASSESSMENT"
        name = "EdgeCVS-5M"
        pad = round(14 * s)
        w_name = self._text_w(self.f_title, name)
        w_sub = self._text_w(self.f_tiny, sub)
        w = pad * 2 + round(16 * s) + max(w_name, w_sub)
        h = round(64 * s)
        self._panel(d, (m, m, m + w, m + h), alpha=200)
        # status dot
        cx, cy = m + pad + round(4 * s), m + h // 2
        d.ellipse((cx - 4 * s, cy - 4 * s, cx + 4 * s, cy + 4 * s), fill=(*TEAL, 255))
        tx = m + pad + round(16 * s)
        d.text((tx, m + round(9 * s)), name, font=self.f_title, fill=(*TXT_HI, 255))
        d.text((tx, m + round(40 * s)), sub, font=self.f_tiny, fill=(*TXT_LO, 255))

    def _draw_perf_chip(self, d, latency_ms):
        s, m = self.s, self.margin
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        l1 = f"{latency_ms:5.1f} ms  ·  {fps:5.1f} FPS"
        l2 = self.device_label
        pad = round(14 * s)
        w = pad * 2 + max(self._text_w(self.f_mono, l1), self._text_w(self.f_mono_sm, l2))
        h = round(64 * s)
        x0 = self.W - m - w
        self._panel(d, (x0, m, self.W - m, m + h), alpha=200)
        d.text((x0 + pad, m + round(9 * s)), l1, font=self.f_mono, fill=(*TEAL, 255))
        d.text((x0 + pad, m + round(38 * s)), l2, font=self.f_mono_sm, fill=(*TXT_LO, 255))

    def _draw_cvs_panel(self, d, probs, met):
        s, m = self.s, self.margin
        x0 = m
        y0 = self.H - m - self.panel_h
        x1, y1 = x0 + self.panel_w, self.H - m
        self._panel(d, (x0, y0, x1, y1))

        pad = round(24 * s)
        # header ------------------------------------------------------------
        d.text((x0 + pad, y0 + round(22 * s)), "CRITICAL VIEW OF SAFETY",
               font=self.f_p_head, fill=(*TXT_LO, 255))
        n_met = sum(met)
        achieved = n_met == 3
        chip_txt = "ACHIEVED" if achieved else f"{n_met} / 3"
        cw = self._text_w(self.f_p_chip, chip_txt) + round(34 * s)
        ch = round(40 * s)
        cx1 = x1 - pad
        cy0 = y0 + round(16 * s)
        chip_fill = (*TEAL, 235) if achieved else (255, 255, 255, 42)
        chip_txt_col = (7, 32, 25, 255) if achieved else (*TXT_HI, 255)
        d.rounded_rectangle((cx1 - cw, cy0, cx1, cy0 + ch), radius=ch // 2, fill=chip_fill)
        d.text((cx1 - cw + round(17 * s), cy0 + round(7 * s)), chip_txt,
               font=self.f_p_chip, fill=chip_txt_col)
        d.line((x0 + pad, y0 + self.header_h, x1 - pad, y0 + self.header_h),
               fill=HAIRLINE, width=1)

        # criterion rows ------------------------------------------------------
        bar_w = round(172 * s)
        bar_h = round(11 * s)
        val_w = self._text_w(self.f_p_val, "100%")
        for i, ((code, label), p, ok) in enumerate(zip(CRITERIA, probs, met)):
            ry = y0 + self.header_h + round(12 * s) + i * self.row_h
            cyc = ry + self.row_h // 2 - round(4 * s)

            state_col = TEAL if ok else RED

            # status icon: filled circle — check when met, cross when not
            r = round(16 * s)
            icx = x0 + pad + r
            d.ellipse((icx - r, cyc - r, icx + r, cyc + r), fill=(*state_col, 255))
            lw = max(3, round(3.2 * s))
            glyph_col = (9, 30, 24, 255) if ok else (46, 12, 12, 255)
            if ok:
                d.line((icx - r * 0.42, cyc + r * 0.05, icx - r * 0.10, cyc + r * 0.42),
                       fill=glyph_col, width=lw)
                d.line((icx - r * 0.10, cyc + r * 0.42, icx + r * 0.46, cyc - r * 0.34),
                       fill=glyph_col, width=lw)
            else:
                d.line((icx - r * 0.38, cyc - r * 0.38, icx + r * 0.38, cyc + r * 0.38),
                       fill=glyph_col, width=lw)
                d.line((icx + r * 0.38, cyc - r * 0.38, icx - r * 0.38, cyc + r * 0.38),
                       fill=glyph_col, width=lw)

            # code + label
            tx = x0 + pad + 2 * r + round(18 * s)
            d.text((tx, cyc - round(21 * s)), code, font=self.f_p_code,
                   fill=(*state_col, 255))
            d.text((tx + round(52 * s), cyc - round(19 * s)), label,
                   font=self.f_p_label, fill=(*TXT_HI, 255))

            # probability bar + value, right-aligned
            vx1 = x1 - pad
            bx1 = vx1 - val_w - round(14 * s)
            bx0 = bx1 - bar_w
            by0 = cyc - bar_h // 2
            d.rounded_rectangle((bx0, by0, bx1, by0 + bar_h), radius=bar_h // 2,
                                fill=(255, 255, 255, 26))
            fill_w = max(bar_h, round(bar_w * float(np.clip(p, 0, 1))))
            d.rounded_rectangle((bx0, by0, bx0 + fill_w, by0 + bar_h),
                                radius=bar_h // 2, fill=(*state_col, 255))
            # 0.5 threshold tick
            tx05 = bx0 + round(bar_w * 0.5)
            d.line((tx05, by0 - round(4 * s), tx05, by0 + bar_h + round(4 * s)),
                   fill=(255, 255, 255, 110), width=max(1, round(1.5 * s)))
            d.text((bx1 + round(14 * s), cyc - round(13 * s)), f"{p * 100:3.0f}%",
                   font=self.f_p_val, fill=(*TXT_HI, 255))

    def _draw_legend(self, d, present_classes):
        s, m = self.s, self.margin
        pad = round(16 * s)
        row = round(31 * s)
        names_w = max(self._text_w(self.f_small, n) for n, _ in SEG_CLASSES.values())
        w = pad * 2 + round(26 * s) + names_w
        h = pad * 2 + round(24 * s) + row * len(SEG_CLASSES)
        x0 = self.W - m - w
        y0 = self.H - m - h
        self._panel(d, (x0, y0, self.W - m, self.H - m), alpha=200)
        d.text((x0 + pad, y0 + round(12 * s)), "STRUCTURES",
               font=self.f_tiny, fill=(*TXT_LO, 255))
        yy = y0 + round(12 * s) + round(24 * s)
        for cls_id, (name, col) in self._legend_rows:
            on = cls_id in present_classes
            a_chip = 255 if on else 70
            a_txt = 255 if on else 110
            cs = round(13 * s)
            ccy = yy + row // 2
            d.rounded_rectangle((x0 + pad, ccy - cs // 2, x0 + pad + cs, ccy + cs // 2),
                                radius=round(4 * s), fill=(*col, a_chip))
            d.text((x0 + pad + cs + round(12 * s), ccy - round(10 * s)), name,
                   font=self.f_small,
                   fill=(*(TXT_HI if on else TXT_LO), a_txt))
            yy += row

    def _draw_achieved_frame(self, d):
        """Subtle full-frame teal confirmation when all criteria are met."""
        w = max(3, round(4 * self.s))
        d.rounded_rectangle((w // 2, w // 2, self.W - w // 2, self.H - w // 2),
                            radius=round(10 * self.s),
                            outline=(*TEAL, 150), width=w)


class VideoInferenceRunner:
    def __init__(self, CFG):
        self.CFG = CFG
        self.device = torch.device(getattr(CFG, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.fold = getattr(CFG, "fold", 0)

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        print(f"[Info] Building model for Exp: {self.CFG.exp}, Fold: {self.fold}")
        self.model = build_model(self.CFG)

        state_dict = load_checkpoint(self.CFG, self.fold, self.device)
        if state_dict is None:
            raise FileNotFoundError(f"No checkpoint found for {self.CFG.exp} fold {self.fold}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        self.input_size = getattr(self.CFG, "img_size", (448, 448))
        if isinstance(self.input_size, int):
            self.input_size = (self.input_size, self.input_size)

        norm_type = getattr(self.CFG, "normalization", "imagenet")
        mean, std = (ENDOSCAPES_MEAN, ENDOSCAPES_STD) if norm_type == "endoscapes" else (IMAGENET_MEAN, IMAGENET_STD)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(self.input_size, antialias=False),
            T.Normalize(mean=mean, std=std)
        ])

    def _device_label(self):
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            name = name.replace("NVIDIA ", "").replace("GeForce ", "")
            return name.upper()
        return "CPU INFERENCE"

    # -- segmentation overlay --------------------------------------------------
    def _apply_masks(self, frame, pred_mask):
        """Curated per-class fills + crisp contours; small specks removed."""
        h, w = frame.shape[:2]
        min_area = int(0.0002 * h * w)
        out = frame.copy()
        present = set()

        for cls_id, (_, rgb) in SEG_CLASSES.items():
            m = (pred_mask == cls_id).astype(np.uint8)
            if m.sum() < min_area:
                continue
            # drop tiny disconnected specks (flicker noise)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            keep = np.zeros_like(m)
            for j in range(1, n):
                if stats[j, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == j] = 1
            if not keep.any():
                continue
            present.add(cls_id)

            bgr = np.array(rgb[::-1], dtype=np.float32)
            alpha = MASK_ALPHA * (0.6 if cls_id == TOOL_CLASS else 1.0)
            idx = keep.astype(bool)
            out[idx] = (out[idx].astype(np.float32) * (1 - alpha) + bgr * alpha).astype(np.uint8)

            contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, tuple(int(c) for c in bgr), 2, cv2.LINE_AA)

        return out, present

    # -- H.264 encode ------------------------------------------------------------
    @staticmethod
    def _encode_h264(tmp_path, final_path, crf=19):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[Warn] ffmpeg not found — keeping raw mp4v output.")
            os.replace(tmp_path, final_path)
            return
        cmd = [ffmpeg, "-y", "-v", "error", "-i", tmp_path,
               "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", final_path]
        subprocess.run(cmd, check=True)
        os.remove(tmp_path)

    @torch.no_grad()
    def process_video(self, input_video_path, output_video_path, start_time=None, end_time=None):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Time calculation logic
        start_frame = int(start_time * 60 * fps) if start_time else 0
        start_frame = min(start_frame, total_video_frames - 1)

        end_frame = int(end_time * 60 * fps) if end_time else total_video_frames
        end_frame = min(end_frame, total_video_frames)

        if start_frame >= end_frame:
            raise ValueError(f"Start time ({start_time} min) is greater than or equal to end time ({end_time} min).")

        frames_to_process = end_frame - start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"[Info] Seeking to start frame {start_frame} ({start_time} minutes)...")

        # Render at >= render_height so UI text is crisp on low-res sources
        render_h = int(getattr(self.CFG, "render_height", 1080))
        if orig_h < render_h:
            scale = render_h / orig_h
            out_w, out_h = int(round(orig_w * scale / 2) * 2), render_h
        else:
            out_w, out_h = orig_w - orig_w % 2, orig_h - orig_h % 2

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        stem, _ = os.path.splitext(output_video_path)
        paths = {"seg": f"{stem}_seg.mp4", "plain": f"{stem}_plain.mp4"}
        tmps = {k: f"{stem}_{k}_tmp.mp4" for k in paths}
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers = {k: cv2.VideoWriter(tmps[k], fourcc, fps, (out_w, out_h)) for k in paths}

        huds = {
            "seg": HudRenderer(out_w, out_h, self._device_label(), seg_variant=True),
            "plain": HudRenderer(out_w, out_h, self._device_label(), seg_variant=False),
        }

        print(f"[Info] Processing video: {input_video_path}")
        print(f"       Source {orig_w}x{orig_h} @ {fps:.2f} FPS -> render {out_w}x{out_h} | {frames_to_process} frames")

        # Temporal smoothing: rolling mean + display EMA + met/not-met hysteresis
        smoothing_window = getattr(self.CFG, "smoothing_window", 15)
        prob_history = deque(maxlen=smoothing_window)
        disp_probs = None
        met = [False, False, False]
        lat_hist = deque(maxlen=30)
        use_cuda = self.device.type == "cuda"

        for _ in tqdm(range(frames_to_process), desc="Inferencing Video Segment"):
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Preprocess
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

            # 2. Inference (timed: model forward only)
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = self.model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()
            lat_hist.append((time.perf_counter() - t0) * 1000.0)
            latency_ms = float(np.mean(lat_hist))

            class_logits, mask_logits = outputs["logits"], outputs["seg_output"]

            # 3. Upscale frame to render size
            frame_hi = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4) \
                if (out_w, out_h) != (orig_w, orig_h) else frame

            # 4. Segmentation overlay (seg variant only)
            mask_logits_resized = torch.nn.functional.interpolate(
                mask_logits, size=(out_h, out_w), mode='bilinear', align_corners=False
            )
            pred_mask = torch.argmax(mask_logits_resized, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            seg_frame, present = self._apply_masks(frame_hi, pred_mask)

            # 5. Classification with smoothing + hysteresis
            class_probs = torch.sigmoid(class_logits).squeeze(0).cpu().numpy()
            if len(class_probs) < 3:
                class_probs = np.pad(class_probs, (0, 3 - len(class_probs)), constant_values=0.0)
            prob_history.append(class_probs[:3])
            smoothed = np.mean(prob_history, axis=0)
            disp_probs = smoothed if disp_probs is None else 0.85 * disp_probs + 0.15 * smoothed
            for i in range(3):
                if met[i] and disp_probs[i] < THRESH_OFF:
                    met[i] = False
                elif not met[i] and disp_probs[i] >= THRESH_ON:
                    met[i] = True

            # 6. Render both variants
            writers["seg"].write(huds["seg"].render(seg_frame, disp_probs, met, latency_ms, present))
            writers["plain"].write(huds["plain"].render(frame_hi, disp_probs, met, latency_ms, set()))

        cap.release()
        for w in writers.values():
            w.release()

        crf = int(getattr(self.CFG, "encode_crf", 19))
        for k in paths:
            print(f"[Info] Encoding {k} variant to H.264 (crf {crf})...")
            self._encode_h264(tmps[k], paths[k], crf=crf)
            print(f"[Success] Saved: {paths[k]}")


def parse_time(value):
    """Seconds from a timestamp: "mm:ss", "hh:mm:ss" or plain seconds. None stays None."""
    if value is None:
        return None
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


@hydra.main(config_path="../configs/cls", config_name="config", version_base=None)
def main(CFG: DictConfig):
    # Fetch the released weights from the Hugging Face Hub (skipped if hf_repo is null)
    if CFG.hf_repo:
        download_run(CFG)

    # Load the saved experiment config, merged with runtime overrides.
    final_cfg = load_experiment_config(CFG)
    # The overlay needs the segmentation head
    final_cfg.seg_head = True

    input_video = to_absolute_path(final_cfg.input_video)
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = os.path.join(to_absolute_path(final_cfg.output_video_dir), f"{video_name}.mp4")

    # process_video takes minutes
    start, end = parse_time(final_cfg.start), parse_time(final_cfg.end)
    start_time = start / 60 if start is not None else None
    end_time = end / 60 if end is not None else None

    runner = VideoInferenceRunner(final_cfg)
    runner.process_video(input_video, output_video, start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    main()
