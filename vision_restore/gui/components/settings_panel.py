"""
Vision-Restore AI - Settings Panel v2.0

Comprehensive settings panel for image enhancement options.
v2.0 includes model selection, CodeFormer fidelity, and new presets.
"""

from typing import Callable, Optional
import customtkinter as ctk

from vision_restore.core.config import Config
from vision_restore.core.logger import get_logger
from vision_restore.core.presets import preset_manager

logger = get_logger(__name__)


class SettingsPanel(ctk.CTkFrame):
    """Settings panel with all enhancement options."""

    def __init__(
        self,
        parent,
        config: Config,
        on_settings_changed: Optional[Callable[[], None]] = None,
        on_interaction_end: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """Initialize the settings panel.
        
        Args:
            parent: Parent widget
            config: Application configuration
            on_settings_changed: Callback when settings change (live)
            on_interaction_end: Callback when interaction finishes (for history)
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.config = config
        self.on_settings_changed = on_settings_changed
        self.on_interaction_end = on_interaction_end
        
        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create all setting widgets."""
        row = 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # Scale Factor
        # ═══════════════════════════════════════════════════════════════════════
        self.scale_label = ctk.CTkLabel(
            self,
            text="🔍 Upscale Factor",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.scale_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        self.scale_var = ctk.StringVar(value=f"{self.config.scale}x")
        self.scale_selector = ctk.CTkSegmentedButton(
            self,
            values=["2x", "4x", "8x"],
            variable=self.scale_var,
            command=self._on_scale_changed,
        )
        self.scale_selector.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # Quality Preset (Pro)
        # ═══════════════════════════════════════════════════════════════════════
        self.preset_label = ctk.CTkLabel(
            self,
            text="⚡ Presets",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.preset_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        self.preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preset_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        
        self.preset_var = ctk.StringVar(value="Balanced (Default)")
        self.preset_menu = ctk.CTkOptionMenu(
            self.preset_frame,
            values=preset_manager.get_preset_names(),
            variable=self.preset_var,
            command=self._on_preset_changed,
        )
        self.preset_menu.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.save_preset_btn = ctk.CTkButton(
            self.preset_frame,
            text="💾",
            width=30,
            command=self._save_preset_dialog,
            fg_color="#444",
        )
        self.save_preset_btn.pack(side="left", padx=2)
        
        self.del_preset_btn = ctk.CTkButton(
            self.preset_frame,
            text="🗑️",
            width=30,
            command=self._delete_current_preset,
            fg_color="#C62828",
            hover_color="#B71C1C",
        )
        self.del_preset_btn.pack(side="left", padx=2)
        row += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # v2.0: Model Selection
        # ═══════════════════════════════════════════════════════════════════════
        self.model_label = ctk.CTkLabel(
            self,
            text="🧠 AI Model",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.model_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        upscale_model = getattr(self.config, 'upscale_model', 'auto')
        self.upscale_model_var = ctk.StringVar(value=upscale_model.title())
        self.upscale_model_menu = ctk.CTkOptionMenu(
            self,
            values=["Auto (Best)", "Real-ESRGAN (Fast/Anime)", "SwinIR (Nature/General)", "HAT (Details/Text)"],
            variable=self.upscale_model_var,
            command=self._on_upscale_model_changed,
        )
        self.upscale_model_menu.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # Enhancement Options
        # ═══════════════════════════════════════════════════════════════════════
        self.options_label = ctk.CTkLabel(
            self,
            text="✨ Enhancement Options",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.options_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # Face enhancement
        self.face_var = ctk.BooleanVar(value=self.config.enable_face_enhance)
        self.face_switch = ctk.CTkSwitch(
            self,
            text="Face Enhancement",
            variable=self.face_var,
            command=self._on_face_changed,
        )
        self.face_switch.grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        
        # v2.0: Face model selection
        face_model = getattr(self.config, 'face_model', 'auto')
        self.face_model_var = ctk.StringVar(value=face_model.title())
        self.face_model_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.face_model_frame.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        
        self.face_model_label = ctk.CTkLabel(
            self.face_model_frame,
            text="  Face Model:",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self.face_model_label.pack(side="left")
        
        self.face_model_menu = ctk.CTkOptionMenu(
            self.face_model_frame,
            values=["Auto", "GFPGAN", "CodeFormer"],
            variable=self.face_model_var,
            command=self._on_face_model_changed,
            width=100,
            height=24,
        )
        self.face_model_menu.pack(side="right")
        row += 1
        
        # Denoising
        self.denoise_var = ctk.BooleanVar(value=self.config.enable_denoise)
        self.denoise_switch = ctk.CTkSwitch(
            self,
            text="Denoising",
            variable=self.denoise_var,
            command=self._on_denoise_changed,
        )
        self.denoise_switch.grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        
        # Sharpening
        self.sharpen_var = ctk.BooleanVar(value=self.config.enable_sharpen)
        self.sharpen_switch = ctk.CTkSwitch(
            self,
            text="Sharpening",
            variable=self.sharpen_var,
            command=self._on_sharpen_changed,
        )
        self.sharpen_switch.grid(row=row, column=0, sticky="w", pady=(2, 15))
        row += 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # Advanced Settings (Collapsible)
        # ═══════════════════════════════════════════════════════════════════════
        self.advanced_visible = ctk.BooleanVar(value=False)
        self.advanced_button = ctk.CTkButton(
            self,
            text="🔧 Advanced Settings ▼",
            command=self._toggle_advanced,
            fg_color="transparent",
            hover_color=("#e0e0e0", "#333333"),
            anchor="w",
            height=30,
        )
        self.advanced_button.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        row += 1
        
        # Advanced settings frame
        self.advanced_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"))
        self.advanced_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        self.advanced_frame.grid_remove()  # Initially hidden
        row += 1
        
        self._create_advanced_settings()
        
        # ═══════════════════════════════════════════════════════════════════════
        # Output Format
        # ═══════════════════════════════════════════════════════════════════════
        self.format_label = ctk.CTkLabel(
            self,
            text="💾 Output Format",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.format_label.grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        self.format_var = ctk.StringVar(value=self.config.output_format.upper())
        self.format_selector = ctk.CTkSegmentedButton(
            self,
            values=["PNG", "JPG", "WEBP"],
            variable=self.format_var,
            command=self._on_format_changed,
        )
        self.format_selector.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

    def _create_advanced_settings(self):
        """Create advanced settings widgets."""
        adv_row = 0
        
        # v2.0: CodeFormer Fidelity
        fidelity_label = ctk.CTkLabel(
            self.advanced_frame,
            text="CodeFormer Fidelity:",
            font=ctk.CTkFont(size=11),
        )
        fidelity_label.grid(row=adv_row, column=0, sticky="w", padx=10, pady=5)
        
        fidelity = getattr(self.config, 'codeformer_fidelity', 0.7)
        self.fidelity_slider = ctk.CTkSlider(
            self.advanced_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self._on_fidelity_changed,
        )
        self.fidelity_slider.set(fidelity * 100)
        self.fidelity_slider.grid(row=adv_row, column=1, sticky="ew", padx=10, pady=5)
        self.fidelity_slider.bind("<ButtonRelease-1>", self._on_interaction_end)
        adv_row += 1
        
        # Tile size
        tile_label = ctk.CTkLabel(
            self.advanced_frame,
            text="Tile Size:",
            font=ctk.CTkFont(size=11),
        )
        tile_label.grid(row=adv_row, column=0, sticky="w", padx=10, pady=5)
        
        self.tile_var = ctk.StringVar(value=str(self.config.tile_size))
        tile_menu = ctk.CTkOptionMenu(
            self.advanced_frame,
            values=["256", "384", "512", "768"],
            variable=self.tile_var,
            command=self._on_tile_changed,
            width=80,
        )
        tile_menu.grid(row=adv_row, column=1, sticky="e", padx=10, pady=5)
        adv_row += 1
        
        # Denoise strength
        denoise_strength_label = ctk.CTkLabel(
            self.advanced_frame,
            text="Denoise Strength:",
            font=ctk.CTkFont(size=11),
        )
        denoise_strength_label.grid(row=adv_row, column=0, sticky="w", padx=10, pady=5)
        
        self.denoise_strength_slider = ctk.CTkSlider(
            self.advanced_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self._on_denoise_strength_changed,
        )
        self.denoise_strength_slider.set(self.config.denoise_strength * 100)
        self.denoise_strength_slider.grid(row=adv_row, column=1, sticky="ew", padx=10, pady=5)
        self.denoise_strength_slider.bind("<ButtonRelease-1>", self._on_interaction_end)
        adv_row += 1
        
        # Sharpen amount
        sharpen_label = ctk.CTkLabel(
            self.advanced_frame,
            text="Sharpen Amount:",
            font=ctk.CTkFont(size=11),
        )
        sharpen_label.grid(row=adv_row, column=0, sticky="w", padx=10, pady=5)
        
        self.sharpen_slider = ctk.CTkSlider(
            self.advanced_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self._on_sharpen_amount_changed,
        )
        self.sharpen_slider.set(self.config.sharpen_amount * 100)
        self.sharpen_slider.grid(row=adv_row, column=1, sticky="ew", padx=10, pady=5)
        self.sharpen_slider.bind("<ButtonRelease-1>", self._on_interaction_end)
        adv_row += 1
        
        # GPU selection
        gpu_label = ctk.CTkLabel(
            self.advanced_frame,
            text="GPU:",
            font=ctk.CTkFont(size=11),
        )
        gpu_label.grid(row=adv_row, column=0, sticky="w", padx=10, pady=5)
        
        self.gpu_var = ctk.StringVar(value="Auto" if self.config.gpu_id >= 0 else "CPU")
        gpu_menu = ctk.CTkOptionMenu(
            self.advanced_frame,
            values=["Auto", "CPU"],
            variable=self.gpu_var,
            command=self._on_gpu_changed,
            width=80,
        )
        gpu_menu.grid(row=adv_row, column=1, sticky="e", padx=10, pady=5)
        adv_row += 1
        
        # v2.0: Quality metrics toggle
        qm_enabled = getattr(self.config, 'enable_quality_metrics', True)
        self.qm_var = ctk.BooleanVar(value=qm_enabled)
        self.qm_switch = ctk.CTkSwitch(
            self.advanced_frame,
            text="Quality Analysis",
            variable=self.qm_var,
            command=self._on_qm_changed,
        )
        self.qm_switch.grid(row=adv_row, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        adv_row += 1
        
        # Configure grid
        self.advanced_frame.grid_columnconfigure(1, weight=1)

    def _toggle_advanced(self):
        """Toggle advanced settings visibility."""
        if self.advanced_frame.winfo_ismapped():
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text="🔧 Advanced Settings ▼")
        else:
            self.advanced_frame.grid()
            self.advanced_button.configure(text="🔧 Advanced Settings ▲")

    def _on_scale_changed(self, value: str):
        """Handle scale factor change."""
        self.config.scale = int(value.replace("x", ""))
        logger.debug(f"Scale changed to: {self.config.scale}")
        self._notify_change()
        self._on_interaction_end()

    def _on_preset_changed(self, value: str):
        """Handle preset change."""
        preset = preset_manager.get_preset(value)
        if not preset:
            return
            
        # Apply settings
        if "scale" in preset: self.config.scale = preset["scale"]
        if "model_name" in preset: self.config.upscale_model = preset["model_name"]
        if "face_enhance" in preset: self.config.enable_face_enhance = preset["face_enhance"]
        if "denoise" in preset: self.config.enable_denoise = preset["denoise"]
        if "sharpen" in preset: self.config.enable_sharpen = preset["sharpen"]
        
        # Apply config updates
        self._sync_ui_from_config()
        self._notify_change()
        self._on_interaction_end()

    def _save_preset_dialog(self):
        """Show dialog to save new preset."""
        dialog = ctk.CTkInputDialog(text="Enter preset name:", title="Save Preset")
        name = dialog.get_input()
        if name:
            settings = {
                "scale": self.config.scale,
                "model_name": self.config.upscale_model,
                "face_enhance": self.config.enable_face_enhance,
                "denoise": self.config.enable_denoise,
                "sharpen": self.config.enable_sharpen,
                "output_format": self.config.output_format,
            }
            preset_manager.save_user_preset(name, settings)
            self._refresh_presets(selected=name)

    def _delete_current_preset(self):
        """Delete current preset."""
        name = self.preset_var.get()
        if preset_manager.delete_preset(name):
             self._refresh_presets()

    def _refresh_presets(self, selected=None):
        """Refresh preset menu values."""
        names = preset_manager.get_preset_names()
        self.preset_menu.configure(values=names)
        if selected:
            self.preset_var.set(selected)
        elif names:
            self.preset_var.set(names[0])

    def _on_upscale_model_changed(self, value: str):
        """Handle upscale model selection."""
        model_map = {
            "Auto (Best)": "auto", 
            "Real-ESRGAN (Fast/Anime)": "realesrgan", 
            "SwinIR (Nature/General)": "swinir", 
            "HAT (Details/Text)": "hat"
        }
        self.config.upscale_model = model_map.get(value, "auto")
        logger.debug(f"Upscale model: {self.config.upscale_model}")
        self._notify_change()
        self._on_interaction_end()

    def _on_face_model_changed(self, value: str):
        """Handle face model selection."""
        model_map = {"Auto": "auto", "GFPGAN": "gfpgan", "CodeFormer": "codeformer"}
        self.config.face_model = model_map.get(value, "auto")
        logger.debug(f"Face model: {self.config.face_model}")
        self._notify_change()
        self._on_interaction_end()

    def _on_fidelity_changed(self, value: float):
        """Handle CodeFormer fidelity slider."""
        self.config.codeformer_fidelity = value / 100
        self._notify_change()

    def _on_qm_changed(self):
        """Handle quality metrics toggle."""
        self.config.enable_quality_metrics = self.qm_var.get()
        logger.debug(f"Quality metrics: {self.config.enable_quality_metrics}")
        self._notify_change()
        self._on_interaction_end()

    def _on_face_changed(self):
        """Handle face enhancement toggle."""
        self.config.enable_face_enhance = self.face_var.get()
        logger.debug(f"Face enhancement: {self.config.enable_face_enhance}")
        self._notify_change()
        self._on_interaction_end()

    def _on_denoise_changed(self):
        """Handle denoise toggle."""
        self.config.enable_denoise = self.denoise_var.get()
        logger.debug(f"Denoise: {self.config.enable_denoise}")
        self._notify_change()
        self._on_interaction_end()

    def _on_sharpen_changed(self):
        """Handle sharpen toggle."""
        self.config.enable_sharpen = self.sharpen_var.get()
        logger.debug(f"Sharpen: {self.config.enable_sharpen}")
        self._notify_change()
        self._on_interaction_end()

    def _on_format_changed(self, value: str):
        """Handle output format change."""
        self.config.output_format = value.lower()
        logger.debug(f"Output format: {self.config.output_format}")
        self._notify_change()
        self._on_interaction_end()

    def _on_tile_changed(self, value: str):
        """Handle tile size change."""
        self.config.tile_size = int(value)
        logger.debug(f"Tile size: {self.config.tile_size}")
        self._notify_change()
        self._on_interaction_end()

    def _on_denoise_strength_changed(self, value: float):
        """Handle denoise strength slider."""
        self.config.denoise_strength = value / 100
        self._notify_change()

    def _on_sharpen_amount_changed(self, value: float):
        """Handle sharpen amount slider."""
        self.config.sharpen_amount = value / 100
        self._notify_change()

    def _on_gpu_changed(self, value: str):
        """Handle GPU selection change."""
        if value == "CPU":
            self.config.gpu_id = -1
        else:
            self.config.gpu_id = 0
        logger.debug(f"GPU ID: {self.config.gpu_id}")
        self._notify_change()
        self._on_interaction_end()

    def _sync_ui_from_config(self):
        """Update UI to match current config."""
        self.scale_var.set(f"{self.config.scale}x")
        self.face_var.set(self.config.enable_face_enhance)
        self.denoise_var.set(self.config.enable_denoise)
        self.sharpen_var.set(self.config.enable_sharpen)
        self.format_var.set(self.config.output_format.upper())
        self.tile_var.set(str(self.config.tile_size))
        self.denoise_strength_slider.set(self.config.denoise_strength * 100)
        self.sharpen_slider.set(self.config.sharpen_amount * 100)
        
        # v2.0 settings
        if hasattr(self.config, 'upscale_model'):
            model_map = {
                "auto": "Auto (Best)", 
                "realesrgan": "Real-ESRGAN (Fast/Anime)", 
                "swinir": "SwinIR (Nature/General)", 
                "hat": "HAT (Details/Text)"
            }
            self.upscale_model_var.set(model_map.get(self.config.upscale_model, "Auto (Best)"))
        
        if hasattr(self.config, 'face_model'):
            face_map = {"auto": "Auto", "gfpgan": "GFPGAN", "codeformer": "CodeFormer"}
            self.face_model_var.set(face_map.get(self.config.face_model, "Auto"))
        
        if hasattr(self.config, 'codeformer_fidelity'):
            self.fidelity_slider.set(self.config.codeformer_fidelity * 100)

    def _notify_change(self):
        """Notify about settings change."""
        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_interaction_end(self, event=None):
        """Notify about interaction end (for history)."""
        if self.on_interaction_end:
            self.on_interaction_end()

    def get_settings_summary(self) -> str:
        """Get a summary of current settings.
        
        Returns:
            Human-readable settings summary
        """
        features = []
        if self.config.enable_face_enhance:
            face_model = getattr(self.config, 'face_model', 'gfpgan').upper()
            features.append(f"Face ({face_model})")
        if self.config.enable_denoise:
            features.append("Denoise")
        if self.config.enable_sharpen:
            features.append("Sharpen")
        
        upscale_model = getattr(self.config, 'upscale_model', 'realesrgan')
        model_str = {
            "auto": "Auto", 
            "realesrgan": "R-ESRGAN", 
            "swinir": "SwinIR", 
            "hat": "HAT"
        }.get(upscale_model, "R-ESRGAN")
        
        return (
            f"Scale: {self.config.scale}x | "
            f"Model: {model_str} | "
            f"Features: {', '.join(features) if features else 'None'}"
        )

