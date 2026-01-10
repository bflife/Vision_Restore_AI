# Vision-Restore AI 🔮

**The Ultimate Image Restoration & Upscaling Suite**

![Version 2.1.0](https://img.shields.io/badge/version-2.1.0-blue.svg) ![License MIT](https://img.shields.io/badge/license-MIT-green.svg) ![Python 3.10](https://img.shields.io/badge/python-3.10-yellow.svg)

Vision-Restore AI is a professional-grade, local AI application designed to transform low-quality, blurry, or damaged images into high-resolution masterpieces. It unifies state-of-the-art restoration models into a single, easy-to-use interface with powerful pro features.

---

## 🌟 Key Features

### 🚀 Next-Gen AI Engines
*   **HAT (Hybrid Attention Transformer)**: The latest breakthrough in super-resolution, recovering intricate textures and details.
*   **SwinIR (Image Restoration)**: Transformer-based model excellent for reducing compression artifacts.
*   **Real-ESRGAN**: The industry standard for robust, fast, and general-purpose upscaling.

### 💎 Advanced Face Restoration
*   **CodeFormer**: World-class face restoration that repairs heavily degraded faces with adjustable fidelity controls.
*   **GFPGAN**: High-quality alternative for natural enhancement.
*   **Smart Selection**: Automatically detects faces and applies the best model.

### 🛡️ Full Cross-Platform Acceleration (New in v2.1)
*   **Universal GPU Support**: Runs natively on **NVIDIA (CUDA)**, **AMD (DirectML)**, and **Intel** GPUs.
*   **DirectML Pulse Mode™**: A revolutionary thermal management system that allows laptops to run heavy AI workloads without overheating or shutting down.
*   **Smart Safety Clamps**: Intelligent resolution limiting prevents crashes on low-VRAM devices.

### 🎛️ Professional Workflow
*   **Interactive Comparison**: Real-time "Before/After" slider with **Zoom & Pan** for pixel-perfect inspection.
*   **Re-enhance ("Lazy Mode")**: Instantly use your result as the next input for recursive upscaling (e.g., 4x -> 16x).
*   **Preset Manager**: Save/Load your favorite settings (e.g., "Old Photo Restore", "Anime 4x").
*   **Quality Metrics**: Built-in analysis tools (BRISQUE, Noise Estimation) to guide your settings.

---

## 🖥️ Supported Hardware
*   **NVIDIA GPUs**: Full CUDA acceleration (Fastest).
*   **AMD / Intel ARC**: Full DirectML acceleration via `torch-directml`.
*   **CPU**: Supported (Slow but accurate) via Safe Mode.

---

## 📥 Installation

### Prerequisites
*   **Windows 10 / 11**
*   **Python 3.10** (Crucial: Python 3.11+ is not yet fully compatible with all AI libraries).

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/vision-restore-ai.git
    cd vision-restore-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install AMD Support (If using AMD/Intel GPU)**
    Double-click `install_amd_support.bat` or run:
    ```bash
    pip install torch-directml
    ```

4.  **Launch the App**
    Double-click `run_gui.py` or run:
    ```bash
    py -3.10 run_gui.py
    ```

---

## 🎮 User Guide

### The Interface
*   **Sidebar**: All your control settings.
*   **Canvas**: Drag & Drop images here. use Mouse Wheel to zoom, Click & Drag to pan.
*   **Comparison**: Toggle the "Eye" icon or dragging the slider to compare.

### Settings Explained
| Setting | Description |
| :--- | :--- |
| **Upscale Model** | `HAT` (Max Detail), `SwinIR` (Clean), `Real-ESRGAN` (Fast/Balanced). |
| **Scale** | Target upscale factor (2x, 4x, 8x). |
| **Face Enhance** | Enable to fix faces. Select `CodeFormer` for damaged faces. |
| **Fidelity** | (CodeFormer only) 0.0 = Max Restoration (AI Hallucination), 1.0 = Max Fidelity (Realism). Recommendation: 0.5-0.7. |
| **Denoise** | Pre-processing step to remove grain before upscaling. |

### Recursive Enhancement
Want more than 4x?
1.  Upscale your image.
2.  Click **"Use as Input"** (below the Save buttons).
3.  Click **Enhance** again.

---

## 🛠️ Troubleshooting

### Laptop Shutting Down?
If your laptop turns off during processing, it's a hardware thermal protection trigger.
*   **Solution 1**: The app now has "Pulse Mode" enabled by default for AMD/DirectML to prevent this. Ensure you are on v2.1.
*   **Solution 2**: Use **Safe Mode**. Double-click `run_safe_mode.bat`. This forces CPU usage, which is 100% safe (though slower).

### "Module not found" Error?
Ensure you are using Python 3.10. Check by running `py --list` in your terminal.

---

## 🤖 Command Line Interface (CLI)

For batch processing or scripts:

```bash
# Basic Upscale
py -3.10 -m vision_restore.main --input image.png --output result.png --scale 4

# Advanced Restore (SwinIR + CodeFormer)
py -3.10 -m vision_restore.main -i photos/ -o output/ --model swinir --face-model codeformer --fidelity 0.6
```

---

## 🤝 Contributing
Contributions are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
