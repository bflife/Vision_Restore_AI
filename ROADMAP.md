# Vision-Restore AI Roadmap

This document outlines the planned development milestones and future improvements for Vision-Restore AI.

---

## 🎯 Vision

To become the leading open-source solution for local, privacy-focused image enhancement, empowering users with professional-grade AI upscaling and restoration without cloud dependencies.

---

## Current Status: v2.1.0 ✅ (AMD Stability Release)

The latest release includes revolutionary "Pulse Mode" technology for laptop stability, full AMD/Intel support, and a professional recreation of the GUI with pro features.

---

## Milestones

### Phase 1: Foundation ✅ (Complete)
**Target: v1.0.0**
- [x] Core upscaling engine with Real-ESRGAN
- [x] GFPGAN face restoration integration
- [x] CustomTkinter GUI with dark theme
- [x] Before/After comparison slider
- [x] Batch processing support
- [x] CLI interface

### Phase 2: Professional Restoration Suite ✅ (Complete)
**Target: v2.0.0**
- [x] **HAT (Hybrid Attention Transformer)** Integration
- [x] **SwinIR** Integration
- [x] **CodeFormer** (Advanced Face Restoration)
- [x] **Interactive Pro GUI** (Zoom, Pan, Slider)
- [x] **Preset Manager**
- [x] **Quality Metrics Engine** (BRISQUE, Noise Estimation)

### Phase 3: Hardware Universality & Stability ✅ (Complete)
**Target: v2.1.0 (Current)**
- [x] **Full AMD / Intel GPU Support** via DirectML
- [x] **"Pulse Mode"** Thermal Throttling for Laptops
- [x] **Safe Mode** Launcher (CPU Fallback)
- [x] **Smart Tile Clamping** for low-VRAM safety
- [x] **Recursive Enhancement** ("Use as Input")

---

### Phase 4: Next-Gen Features (In Progress) 🔄
**Target: v3.0.0 | Q1 2026**

#### Video
- [ ] **Video Upscaling**: Frame-by-frame processing with temporal consistency
- [ ] **Scene Detection**: Auto-split scenes for better handling

#### Workflow
- [ ] **Undo/Redo History**: Full state management
- [ ] **Plugin Architecture**: Support custom PyTorch models via drag-drop
- [ ] **Cloud Sync**: Optional settings sync across devices

#### Models
- [ ] **DAT (Dual Aggregation Transformer)**
- [ ] **RestoreFormer**
- [ ] **Colorization**: DeOldify integration

---

### Phase 5: Enterprise & Scale
**Target: v4.0.0 | Q3 2026**
- [ ] Multi-GPU processing
- [ ] Distributed processing cluster
- [ ] Web Interface / API Server

---

## Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In Progress |
| 📋 | Planned |

---

*Last updated: January 2026*
