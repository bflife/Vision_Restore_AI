"""
Vision-Restore AI - Main Application

The main CustomTkinter application window for Vision-Restore AI.
"""

import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, List
import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

from vision_restore import __version__
from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger
from vision_restore.core.history import HistoryManager
from vision_restore.core.model_manager import ModelManager
from vision_restore.engine.orchestrator import EnhancementOrchestrator
from vision_restore.gui.components.before_after import BeforeAfterCanvas
from vision_restore.gui.components.settings_panel import SettingsPanel
from vision_restore.gui.components.batch_panel import BatchPanel
from vision_restore.gui.components.progress_bar import ProcessingProgress
from vision_restore.utils.image_io import load_image, save_image, numpy_to_pil, pil_to_numpy

logger = get_logger(__name__)


class VisionRestoreApp(ctk.CTk):
    """Main Vision-Restore AI Application Window."""

    def __init__(self, gpu_id: Optional[int] = None):
        """Initialize the application.
        
        Args:
            gpu_id: Optional GPU ID override (e.g. -1 for Safe Mode)
        """
        super().__init__()
        self.cli_gpu_id = gpu_id
        
        # Configuration
        # Configuration
        self.config = get_config()
        self.config.load()
        
        # Override config if CLI args provided
        if hasattr(self, 'cli_gpu_id') and self.cli_gpu_id is not None:
             self.config.gpu_id = self.cli_gpu_id
             logger.info(f"CLI Override: Forced GPU ID to {self.config.gpu_id}")
        
        # Window setup
        self.title(f"Vision-Restore AI v{__version__}")
        self.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.minsize(1000, 700)
        
        # Theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # State
        self.current_image_path: Optional[Path] = None
        self.original_image: Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Components
        self.orchestrator: Optional[EnhancementOrchestrator] = None
        self.model_manager = ModelManager(self.config)
        self.history = HistoryManager()
        
        # Build UI
        self._setup_layout()
        self._create_menu()
        self._bind_shortcuts()
        self._bind_drag_drop()
        
        # Check models on startup
        self.after(500, self._check_models_on_startup)

    def _setup_layout(self):
        """Create the main application layout."""
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Left Sidebar - Settings
        # ═══════════════════════════════════════════════════════════════════════
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        self.sidebar_frame.grid_propagate(False)
        
        # Logo/Title
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🔮 Vision-Restore AI",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Professional Image Upscaling",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Settings panel
        self.settings_panel = SettingsPanel(
            self.sidebar_frame,
            config=self.config,
            on_settings_changed=self._on_settings_changed,
            on_interaction_end=self._push_history,
        )
        self.settings_panel.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.button_frame.grid(row=4, column=0, padx=20, pady=20, sticky="sew")
        
        self.open_button = ctk.CTkButton(
            self.button_frame,
            text="📂 Open Image",
            command=self._open_image,
            height=40,
        )
        self.open_button.pack(fill="x", pady=5)
        
        self.enhance_button = ctk.CTkButton(
            self.button_frame,
            text="✨ Enhance",
            command=self._start_enhancement,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#4CAF50",
            hover_color="#45a049",
            state="disabled",
        )
        self.enhance_button.pack(fill="x", pady=10)
        
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="💾 Save Result",
            command=self._save_result,
            height=40,
            state="disabled",
        )
        self.save_button.pack(fill="x", pady=5)
        
        self.save_comp_button = ctk.CTkButton(
            self.button_frame,
            text="📑 Save Comparison",
            command=self._save_comparison,
            height=40,
            state="disabled",
            fg_color="#454545",
        )
        self.save_comp_button.pack(fill="x", pady=5)

        self.reuse_button = ctk.CTkButton(
            self.button_frame,
            text="🔄 Use as Input",
            command=self._use_result_as_input,
            height=40,
            state="disabled",
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
        )
        self.reuse_button.pack(fill="x", pady=5)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Center - Image Preview
        # ═══════════════════════════════════════════════════════════════════════
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Before/After canvas
        self.preview = BeforeAfterCanvas(
            self.main_frame,
            on_drop=self._on_image_drop,
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Progress bar
        self.progress = ProcessingProgress(self.main_frame)
        self.progress.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.progress.grid_remove()  # Hidden until processing
        
        # ═══════════════════════════════════════════════════════════════════════
        # Bottom - Batch Panel (collapsible)
        # ═══════════════════════════════════════════════════════════════════════
        self.batch_panel = BatchPanel(
            self,
            config=self.config,
            on_process_batch=self._process_batch,
        )
        self.batch_panel.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 10))

    def _create_menu(self):
        """Create the application menu bar."""
        # Note: CustomTkinter doesn't have built-in menu support
        # We use the sidebar buttons instead
        pass

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.bind("<Control-o>", lambda e: self._open_image())
        self.bind("<Control-s>", lambda e: self._save_result())
        self.bind("<Control-e>", lambda e: self._start_enhancement())
        self.bind("<Escape>", lambda e: self._cancel_processing())
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())
        self.bind("<Control-Shift-z>", lambda e: self._redo())

    def _bind_drag_drop(self):
        """Setup drag and drop support."""
        # Note: Drag-drop is handled by BeforeAfterCanvas
        pass

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _check_models_on_startup(self):
        """Check if required models are available."""
        missing = self.model_manager.get_missing_models(
            self.config.enable_face_enhance
        )
        
        if missing:
            size_mb = self.model_manager.get_total_download_size(
                self.config.enable_face_enhance
            )
            
            result = messagebox.askyesno(
                "Download Models",
                f"Vision-Restore AI needs to download {len(missing)} model(s) "
                f"({size_mb}MB) for first-time setup.\n\n"
                f"Missing models:\n• " + "\n• ".join(missing) + "\n\n"
                "Download now?",
            )
            
            if result:
                self._download_models()
            else:
                logger.warning("Models not downloaded, some features may not work")

    def _download_models(self):
        """Download required models with progress."""
        self.progress.grid()
        self.progress.set_status("Downloading models...")
        
        def download_thread():
            def progress_callback(progress: float, message: str):
                self.after(0, lambda: self.progress.set_progress(progress, message))
            
            success = self.model_manager.download_all_required(
                self.config.enable_face_enhance,
                progress_callback,
            )
            
            self.after(0, lambda: self._on_download_complete(success))
        
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def _on_download_complete(self, success: bool):
        """Handle model download completion."""
        self.progress.grid_remove()
        
        if success:
            messagebox.showinfo("Success", "Models downloaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to download some models. Check your internet connection.")

    def _on_settings_changed(self):
        """Handle settings change."""
        logger.debug("Settings changed")
        self.config.save()

    def _push_history(self):
        """Push current state to history."""
        state = self.config.to_dict()
        self.history.push_state(state)

    def _undo(self):
        """Undo last change."""
        state = self.history.undo()
        if state:
            self._apply_state(state)
            logger.info("Undo performed")

    def _redo(self):
        """Redo last change."""
        state = self.history.redo()
        if state:
            self._apply_state(state)
            logger.info("Redo performed")

    def _apply_state(self, state):
        """Apply a configuration state."""
        # Update config object
        for key, value in state.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update UI
        self.settings_panel._sync_ui_from_config()
        self.config.save()
        
        # If we have an enhanced image, re-run enhancement (instant cache!)
        if self.enhanced_image is not None and not self.is_processing:
             self._start_enhancement()

    def _open_image(self):
        """Open an image file."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
            ("All files", "*.*"),
        ]
        
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self._load_image(Path(filepath))

    def _load_image(self, path: Path):
        """Load an image from path."""
        try:
            image = load_image(path)
            if image is None:
                messagebox.showerror("Error", f"Failed to load image: {path}")
                return
            
            self.current_image_path = path
            self.original_image = image
            self.enhanced_image = None
            
            # Convert to PIL for display
            pil_image = numpy_to_pil(image)
            self.preview.set_before_image(pil_image)
            self.preview.set_after_image(None)
            
            # Enable enhance button
            self.enhance_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            
            # Update window title
            self.title(f"Vision-Restore AI - {path.name}")
            
            logger.info(f"Loaded image: {path} ({image.shape[1]}x{image.shape[0]})")
            
            # Reset history on new image load
            self.history.set_initial_state(self.config.to_dict())
            
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _on_image_drop(self, path: str):
        """Handle dropped image file."""
        self._load_image(Path(path))

    def _start_enhancement(self):
        """Start the enhancement process."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.enhance_button.configure(state="disabled", text="⏳ Processing...")
        self.open_button.configure(state="disabled")
        self.progress.grid()
        self.progress.set_progress(0, "Starting enhancement...")
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._enhancement_thread,
            daemon=True,
        )
        self.processing_thread.start()

    def _enhancement_thread(self):
        """Enhancement processing thread."""
        try:
            # Create orchestrator if needed
            if self.orchestrator is None:
                self.orchestrator = EnhancementOrchestrator(self.config)
            
            # Progress callback
            def progress_callback(progress: float, message: str):
                self.after(0, lambda: self.progress.set_progress(progress, message))
            
            # Run enhancement
            result = self.orchestrator.enhance(
                self.original_image,
                scale=self.config.scale,
                face_enhance=self.config.enable_face_enhance,
                preprocess=self.config.enable_denoise,
                postprocess=self.config.enable_sharpen,
                progress_callback=progress_callback,
            )
            
            self.after(0, lambda: self._on_enhancement_complete(result))
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            error_msg = str(e)
            self.after(0, lambda: self._on_enhancement_error(error_msg))

    def _on_enhancement_complete(self, result: np.ndarray):
        """Handle enhancement completion."""
        self.enhanced_image = result
        self.is_processing = False
        
        # Update UI
        pil_result = numpy_to_pil(result)
        self.preview.set_after_image(pil_result)
        
        self.enhance_button.configure(state="normal", text="✨ Enhance")
        self.open_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.save_comp_button.configure(state="normal")
        self.reuse_button.configure(state="normal")
        self.progress.set_progress(1.0, "Enhancement complete!")
        
        # Hide progress after delay
        self.after(2000, self.progress.grid_remove)
        
        logger.info(f"Enhancement complete: {result.shape[1]}x{result.shape[0]}")

    def _on_enhancement_error(self, error: str):
        """Handle enhancement error."""
        self.is_processing = False
        
        self.enhance_button.configure(state="normal", text="✨ Enhance")
        self.open_button.configure(state="normal")
        self.progress.grid_remove()
        
        messagebox.showerror("Error", f"Enhancement failed:\n{error}")

    def _cancel_processing(self):
        """Cancel current processing."""
        if self.is_processing:
            # Note: Full cancellation would require cooperative cancellation
            # in the orchestrator. For now, we just update the UI
            self.is_processing = False
            self.enhance_button.configure(state="normal", text="✨ Enhance")
            self.open_button.configure(state="normal")
            self.progress.grid_remove()
            logger.info("Processing cancelled by user")

    def _save_result(self):
        """Save the enhanced image."""
        if self.enhanced_image is None:
            messagebox.showwarning("Warning", "No enhanced image to save.")
            return
        
        # Get output filename
        if self.current_image_path:
            default_name = f"{self.current_image_path.stem}_{self.config.scale}x"
        else:
            default_name = "enhanced"
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("WebP files", "*.webp"),
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{self.config.output_format}",
            initialfile=default_name,
            filetypes=filetypes,
        )
        
        if filepath:
            try:
                save_image(
                    self.enhanced_image,
                    Path(filepath),
                    quality=self.config.output_quality,
                )
                logger.info(f"Saved result to: {filepath}")
                messagebox.showinfo("Success", f"Image saved to:\n{filepath}")
            except Exception as e:
                logger.error(f"Failed to save: {e}")
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def _save_comparison(self):
        """Save the comparison image."""
        if not self.preview.after_image:
            messagebox.showwarning("Warning", "No comparison to save.")
            return
            
        comp_img = self.preview.get_comparison_full_res()
        if comp_img is None:
            return
            
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("WebP files", "*.webp"),
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="comparison_view",
            filetypes=filetypes,
            title="Save Comparison Image"
        )
        
        if filepath:
            try:
                comp_img.save(filepath)
                messagebox.showinfo("Success", f"Comparison saved successfully!")
            except Exception as e:
                logger.error(f"Failed to save comparison: {e}")
                messagebox.showerror("Error", f"Failed to save comparison: {e}")

    def _use_result_as_input(self):
        """Use the enhanced result as the new input image."""
        if self.enhanced_image is None:
            return
            
        # Cycle image
        self.original_image = self.enhanced_image.copy()
        self.enhanced_image = None
        
        # Reset UI
        pil_image = numpy_to_pil(self.original_image)
        self.preview.set_before_image(pil_image)
        self.preview.set_after_image(None)
        
        # Reset buttons (Re-enhance disable until processed again)
        self.reuse_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.save_comp_button.configure(state="disabled")
        
        # Reset history for new base
        self.history.set_initial_state(self.config.to_dict())
        
        # Update Info
        h, w = self.original_image.shape[:2]
        logger.info(f"Re-using result as input: {w}x{h}")
        self.title(f"Vision-Restore AI - Re-enhanced Input ({w}x{h})")
        
        messagebox.showinfo("Re-enhance", "Result is now the input image. You can enhance it again!")

    def _process_batch(self, files: List[Path], output_dir: Path):
        """Process batch of images."""
        if not files:
            return
        
        # TODO: Implement full batch processing with BatchPanel integration
        logger.info(f"Batch processing {len(files)} files to {output_dir}")
        messagebox.showinfo(
            "Batch Processing",
            f"Batch processing {len(files)} images...\n"
            "This feature will be available in the next update.",
        )

    def on_closing(self):
        """Handle window close."""
        # Save config
        self.config.save()
        
        # Cleanup resources
        if self.orchestrator:
            self.orchestrator.cleanup()
        
        self.destroy()


def main():
    """Main entry point for the GUI application."""
    try:
        # Configure high DPI
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        
        # Parse Safe Mode arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-id", type=int, help="Force specific GPU ID (-1 for CPU Safe Mode)", default=None)
        args, _ = parser.parse_known_args()
        
        # Create and run app
        app = VisionRestoreApp(gpu_id=args.gpu_id)
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
