---
title: BG Remover Pro
emoji: ✂️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Studio-grade AI background removal API + web UI
---

# BG Remover Pro

Studio-grade AI background removal powered by **rembg**, **ONNX Runtime**, and **pymatting**.
Edge-preserving consensus engine — never distorts subject pixels.

## ✨ Features

- **Parallel multi-engine consensus** — 2-3 ML models vote on each pixel
- **Edge-preserving refinement** — original image quality is never compromised
- **Smart model routing** — auto-selects best model for portraits vs objects
- **5 quality tiers**: Fast / Balanced / Premium / Ultra / Portrait
- **Background replacement** — transparent, solid color, or custom image
- **Batch processing** — up to 20 images at once

## 🚀 Usage

### Web UI
Open the Space URL and use the drag-and-drop interface.

### API

```bash
curl -X POST "https://YOUR-USERNAME-bg-remover-pro.hf.space/remove-bg?quality=premium" \
     -F "file=@photo.jpg" \
     --output result.png