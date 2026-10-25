# 🖼️ FIBO 2 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/eduardoabreu81/fibo2-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate and edit with **[FIBO 2.0](https://huggingface.co/briaai/Fibo-2.0-dev)** by Bria inside Forge Neo, using the checkpoint, preset and Generate button you already know - text-to-image from a structured JSON caption or plain text, and image editing with Edit Turbo in img2img.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual. If your Forge Neo learns FIBO 2.0 by itself, the extension steps aside and says so.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.2.0 - Fused Kernels

- **Faster steps** - the DiT runs on comfy-kitchen's fused kernels, as ComfyUI does for FIBO 2.0: norms, RoPE and gated residuals in single passes, and SwiGLU folded into the INT8 GEMM - see [Fused Kernels](#-fused-kernels)
- **[FIBO 2] Fused kernels** setting - off runs the previous path, for comparisons

### v0.1.0 - First Release

- **FIBO 2.0 in Forge Neo** - Bria's Dev and Edit Turbo models, with their Qwen3-VL 4B text encoder, in INT8 or GGUF
- **Structured captions** - Bria's JSON captions (VGL), cleaned as in training, or plain text; things go where the caption says
- **Text that renders** - each `text_render` entry is shown to the text encoder as a picture of its words, accents included, as upstream does
- **Editing in img2img** - Edit Turbo changes the img2img picture following your instruction; ImageStitch adds more pictures, Inpaint limits the change to a painted area
- **Upstream's schedule** - the shift follows the picture's size, as Bria's pipeline sets it
- **fibo2** UI preset - Dev's settings, ready to generate

---

## 🎯 Features

### 🖼️ Native Generation

<p align="center"><img src=".github/images/text-to-image.jpg" alt="A Japanese woman in a cream trench coat on a Kyoto street at dusk, made with Dev, and Hatsune Miku singing on a concert stage, made with Edit Turbo from a VGL caption" width="720"></p>

- Runs inside txt2img and img2img - no separate tab, no extra environment
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- Uses the same autoencoder as FLUX.2 - if you have `flux2-vae`, you already have it
- On an RTX 2070 (8 GB) with the INT8 files: about 6 seconds a step for a 1024x1024 picture with Dev (5 minutes at its 50 steps), about 40 seconds with Edit Turbo at 4 steps

### 📝 Structured Captions

- FIBO 2.0 was trained on long JSON captions: `short_description`, `objects` (each with its description, location, size, pose...), `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `text_render` and more - see [Bria's VGL docs](https://docs.bria.ai/vgl)
- Paste a JSON caption into the prompt; empty fields and scores are dropped, as in training
- Plain sentences work too
- Works with sd-dynamic-prompts on: it takes the `{ }` out of the caption before the model reads it, and FIBO 2 puts them back - the infotext shows the caption without its `{ }`, but the model gets it whole
- An empty negative prompt is the "null caption" the model was trained with - leave it empty

### 📍 Placing Things

<p align="center"><img src=".github/images/locations.jpg" alt="A red apple top-left and a blue mug bottom-right on a white table, and the same caption with the two locations swapped" width="720"></p>

- Each object of a caption has a `location` in words - `"top-left"`, `"center"`, `"bottom-right foreground"` - and a `relative_size` - `"small"`, `"medium"`, `"large within frame"`; the model puts things there
- Swap two locations and the things swap places

### 🔤 Text Rendering

<p align="center"><img src=".github/images/text-rendering.jpg" alt="A bakery sign that reads FIBO 2 ON FORGE NEO with a FRESH BREAD chalkboard, and a Brazilian bakery sign that reads PÃO DE QUEIJO with a CAFÉ COM LEITE chalkboard" width="720"></p>

- Put the words in `text_render`: `"text_render": [{"text": "OPEN 24H", "location": "center", ...}]`
- The text encoder sees a picture of each text from the middle of the generation on, as Bria's pipeline does
- Accented letters are drawn with the fonts Bria's training used (Aileron, DejaVu Sans), shipped with the extension: `CAFÉ COM LEITE` comes out right; a tilde can still land on the wrong letter now and then

### ✏️ Editing with Edit Turbo

<p align="center"><img src=".github/images/references.jpg" alt="A sleeping fox in the snow and a bakery front, and the fox from the first picture asleep on the bakery's sidewalk, made from both" width="720"></p>

- Put the picture in img2img, write what to change (`Add a red knitted scarf around the neck of the fox`) and set **Denoising strength** to 1 - everything you did not ask to change stays as it was
- Adding, removing and swapping things works best; changing the light of the whole scene (day to night) barely changes it
- Or in txt2img: add the picture through ImageStitch; with a single picture the result comes out at about 1 megapixel in its proportions, whatever the size sliders say, as Wan2GP does (the console says which size it used). Turn this off under **Settings → Stable Diffusion → [FIBO 2] Reference Sets the Size** to keep the sliders' size
- More pictures through ImageStitch: the img2img picture is `<image_1>`, the ones you add are `<image_2>`, `<image_3>`... - for example `Put the sleeping fox from <image_1> on the sidewalk in front of the bakery in <image_2>`
- Each extra picture is used at about 1 megapixel, as upstream does, so it costs time: on an RTX 2070 (8 GB) an edit with two pictures took about 10 minutes, against 40 seconds with one
- A JSON prompt can carry the instruction too: `{"edit_instruction": "...", "short_description": "...", ...}`
- Turn editing off under **Settings → Stable Diffusion → [FIBO 2] Enable Reference** for plain image-to-image
- Edit Turbo also makes pictures from text alone, in 4 steps without CFG

### 🖌️ Masked Edits

<p align="center"><img src=".github/images/masked-edit.jpg" alt="The woman in the cream trench coat, and the same picture with only the painted coat changed to a red wool coat" width="720"></p>

- Change only part of the picture with **Inpaint**: paint the area, write the change (`Change her trench coat to a bright red wool coat`) and set Denoising strength to 1
- The painted area is greyed out for the model, as in Bria's pipeline, and the rest of the picture stays as it was

### ⚡ Turbo LoRA

- Comfy-Org's turbo LoRA turns Dev into a 4-step model: download `fibo_2.0_turbo_lora_avg_rank_270_bf16` from the `loras` folder of [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0), put it in `models/Lora` and add it to the prompt, for example `<lora:fibo_2.0_turbo_lora_avg_rank_270_bf16:1>`
- Set **Steps** to 4 and **CFG** to 1, as for Edit Turbo
- With the same seed, Dev with the LoRA gives nearly the same picture as Edit Turbo

### ⚡ Fused Kernels

- The DiT block runs on [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)'s fused kernels, the ones ComfyUI uses for FIBO 2.0: each modulated norm is one pass, the q/k norms and RoPE are one pass, gated residuals are updated in place, and SwiGLU is folded into the INT8 GEMM that follows it
- **INT8 ConvRot safetensors** (`fibo2_*_int8_convrot`) get all of it, the SwiGLU fold included; **bf16 safetensors** get the norm, RoPE and residual fusions; **GGUF** loads as before and is not tuned for
- Runs on CUDA when Forge Neo computes in bf16 or float32; a card Forge Neo runs in float16 (RTX 20 series) keeps the previous path, unchanged
- The console says which path is active when the model first runs: `[FIBO 2] fused comfy-kitchen kernels: on (bfloat16), SwiGLU folded into the INT8 ConvRot GEMM: yes`
- **Settings → Stable Diffusion → [FIBO 2] Fused kernels** turns it off, for A/B pictures and timings, and for bug reports (the two paths give the same picture up to bf16 rounding)

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Tells you which file is missing under **VAE / Text Encoder** instead of failing with a generic error
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/fibo2-forge-neo`
4. Click **Install** and restart the WebUI
5. Download the files from [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0) and place them in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| FIBO 2.0 checkpoint | `fibo2_dev_int8_convrot` - or `fibo2_edit_turbo_int8_convrot` for Edit Turbo | `models/Stable-diffusion` |
| Qwen3-VL 4B text encoder | `qwen3vl_4b_int8_convrot` | `models/text_encoder` |
| FLUX.2 autoencoder | `flux2-vae` | `models/VAE` |

   GGUF checkpoints load too, in the same folder: [vantagewithai](https://huggingface.co/vantagewithai/Fibo-2.0-Dev-and-Edit-Turbo-ComfyUI-GGUF) (Dev and Edit Turbo) or [Abiray](https://huggingface.co/Abiray/Fibo-2.0-edit-turbo-GGUF) (Edit Turbo). Q4_K_M is about 5-6 GB instead of 8.4 GB.

6. Restart Forge Neo, so the new text encoder shows up in the list
7. Pick the **fibo2** UI preset, the checkpoint, and the text encoder and autoencoder under **VAE / Text Encoder**

> [!Note]
> Tested with the INT8 files of Dev and Edit Turbo, Abiray's Edit Turbo Q4_K_M GGUF and vantagewithai's Dev Q4_K_M GGUF, on an RTX 2070 (8 GB), Forge Neo 2.31: text-to-image at 1024x1024 and 1424x800, text rendering with accents, edits of one picture, of two (ImageStitch) and of a painted area (Inpaint), and the turbo LoRA with Dev in INT8 and in GGUF. The BF16 files have not been tested yet. The fused kernels of v0.2.0 match the previous path in the CPU tests; their live check on a bf16-capable card is pending.

---

## ⚙️ Recommended Settings

| Model | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| Dev | Euler | Simple | 50 | 5 |
| Edit Turbo | Euler | Simple | 4 | 1 |

The **fibo2** preset sets Dev's values; for Edit Turbo, or Dev with the turbo LoRA, set 4 steps and CFG 1 - or change the preset's values on the **FIBO2** page of **Settings**, and it keeps yours.

The shift follows the picture's size, as in Bria's pipeline: 1.15 at 1024x1024, more for bigger pictures, less for smaller ones - there is no Shift slider to set. FIBO 2.0 was trained at about 1 megapixel: 1024x1024, 1424x800, 800x1424... Use sides that are a multiple of 16.

---

## 💡 Tips

- Long, detailed captions are what the model knows best; a language model can turn a sentence into a VGL caption
- FIBO 2.0 does not know characters by name - "Hatsune Miku" gives an unrelated girl with the name written on the screens - but describe her (long turquoise twin tails, grey top with a turquoise tie, headset microphone...) and she comes out right
- For edits, use **Crop and resize** in img2img and set its size to about 1 megapixel in the picture's proportions (1024x1024 for a square picture, 1216x896 for a 4:3 one); at other sizes edits can zoom in on what changes or lose detail, and the console says which size to use
- New text encoder files appear in the list after a restart of Forge Neo
- Before uninstalling, switch the UI preset to another one

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[FIBO 2.0](https://huggingface.co/briaai/Fibo-2.0-dev)** by Bria - the model and its reference pipeline
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** by comfyanonymous and Comfy-Org - the FIBO 2.0 implementation this port follows ([PR #17240](https://github.com/Comfy-Org/ComfyUI/pull/17240)), and the INT8 files
- **[Qwen3-VL](https://huggingface.co/Qwen)** by the Qwen team - the text encoder
- **[comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)** by Comfy-Org - the fused kernels, and ComfyUI's FIBO 2.0 model for the way it uses them
- **[moss-image-forge-neo](https://github.com/eduardoabreu81/moss-image-forge-neo)** PR #3 by RichyRich515 - the Forge-side pattern for the fused block

The FIBO 2.0 weights are © BRIA AI under the [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license: non-commercial use only. This extension does not include them.

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/fibo2-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/fibo2-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
