"""
Vision-Restore AI - Before/After Canvas

Interactive before/after image comparison widget with draggable slider.
"""

import tkinter as tk
from typing import Callable, Optional, Tuple
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class BeforeAfterCanvas(ctk.CTkFrame):
    """Interactive before/after image comparison widget with Pro features.
    
    Features:
    - Draggable slider divider
    - Synchronized zoom/pan (mouse-centered zoom)
    - Pan with Middle/Right click or Space+Drag
    - Pixel-perfect inspection
    """

    def __init__(
        self,
        parent,
        on_drop: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.on_drop = on_drop
        
        # State
        self.before_image: Optional[Image.Image] = None
        self.after_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        
        self.slider_pos = 0.5  # 0.0 to 1.0 relative to canvas width
        
        # Viewport State
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Interaction State
        self.is_dragging_slider = False
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Create canvas
        self.canvas = tk.Canvas(
            self,
            bg="#1a1a1a",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        
        # Bindings
        self.canvas.bind("<Configure>", self._on_resize)
        
        # Slider Interaction (Left Click)
        self.canvas.bind("<Button-1>", self._start_slider_drag)
        self.canvas.bind("<B1-Motion>", self._do_slider_drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_slider_drag)
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_view())
        
        # Pan Interaction (Middle/Right Click)
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        
        self.canvas.bind("<Button-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        
        # Zoom Interaction
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        
        # Make focusable
        self.canvas.config(takefocus=True)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        
        # Keyboard shortcuts
        self.bind("<space>", self._toggle_comparison)
        
        # Draw placeholder
        self.after(100, self._draw_placeholder)

    def set_before_image(self, image: Optional[Image.Image]):
        self.before_image = image
        self.reset_view()

    def set_after_image(self, image: Optional[Image.Image]):
        self.after_image = image
        self._render()

    def reset_view(self):
        """Fit image to canvas."""
        if not self.before_image:
            return
            
        # Reset pan/zoom
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.slider_pos = 0.5
        
        # Fit logic is handled dynamically in render
        self._render()

    def _get_fit_scale(self, img_w, img_h, canvas_w, canvas_h):
        if img_w == 0 or img_h == 0: return 1.0
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        return min(scale_w, scale_h)

    def _render(self):
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10 or h < 10 or not self.before_image:
            if not self.before_image:
                self._draw_placeholder()
            return

        img_w, img_h = self.before_image.size
        
        # Calculate base scale to fit image in canvas
        base_scale = self._get_fit_scale(img_w, img_h, w, h)
        
        # Effective scale including user zoom
        final_scale = base_scale * self.zoom
        
        # Calculate displayed dimensions
        disp_w = int(img_w * final_scale)
        disp_h = int(img_h * final_scale)
        
        # Center the image relative to canvas, apply pan
        # Center = (w/2, h/2)
        # Image TopLeft should be at Center - (disp_w/2, disp_h/2) + Pan
        pad_x = (w - disp_w) / 2 + self.pan_x
        pad_y = (h - disp_h) / 2 + self.pan_y
        
        # 1. Resize/Crop logic
        # For simplicity and performance, we'll resize the whole image for now
        # unless it gets huge. 
        # TODO: Optimization for huge scaling (viewport crop)
        
        # Use Nearest Neighbor for crisp pixels when over-zoomed (e.g. > 200% original size)
        resample_method = Image.Resampling.NEAREST if final_scale > 2.0 else Image.Resampling.LANCZOS
        
        try:
            # Resize "Before"
            before_disp = self.before_image.resize((disp_w, disp_h), resample_method)
            
            final_comp = Image.new("RGB", (w, h), "#1a1a1a")
            
            # Helper to paste centered
            # Paste position needs to be integer
            paste_x = int(pad_x)
            paste_y = int(pad_y)
            
            final_comp.paste(before_disp, (paste_x, paste_y))
            
            # If "After" image exists, overlay it
            split_x = int(w * self.slider_pos)
            
            if self.after_image:
                after_disp = self.after_image.resize((disp_w, disp_h), resample_method)
                
                # Create a temporary composite for After
                after_layer = Image.new("RGB", (w, h), "#1a1a1a")
                after_layer.paste(after_disp, (paste_x, paste_y))
                
                # Crop the Right side of After layer
                # (split_x, 0, w, h)
                if split_x < w:
                    right_side = after_layer.crop((split_x, 0, w, h))
                    final_comp.paste(right_side, (split_x, 0))

            self.display_image = ImageTk.PhotoImage(final_comp)
            self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
            
            # Draw UI Overlays
            if self.after_image:
                self._draw_slider(split_x, h)
                self._draw_labels(split_x, w, h)
                
        except Exception as e:
            logger.error(f"Render error: {e}")

    def _draw_slider(self, x, h):
        self.canvas.create_line(x, 0, x, h, fill="white", width=2)
        # Handle
        center_y = h // 2
        r = 16
        self.canvas.create_oval(x-r, center_y-r, x+r, center_y+r, fill="#2196F3", outline="white", width=2)
        self.canvas.create_text(x, center_y, text="↔", fill="white", font=("Arial", 14, "bold"))

    def _draw_labels(self, split_x, w, h):
        # Before Label
        if split_x > 60:
            self.canvas.create_text(30, h-30, text="Before", fill="#ff6b6b", font=("Arial", 12, "bold"), anchor="w")
        
        # After Label
        if w - split_x > 60:
            self.canvas.create_text(w-30, h-30, text="After", fill="#4CAF50", font=("Arial", 12, "bold"), anchor="e")

    # --- Interaction Handlers ---

    def _on_resize(self, event):
        self._render()

    def _start_slider_drag(self, event):
        # Double click check (handled separately via binding if needed, but manual check here is safer for some tk versions)
        # standard binding <Double-Button-1> is better.
        
        # Only grab slider if near the line AND after image exists
        is_near_slider = False
        if self.after_image:
            w = self.canvas.winfo_width()
            slider_pixel = int(w * self.slider_pos)
            if abs(event.x - slider_pixel) < 30:
                is_near_slider = True
        
        if is_near_slider:
            self.is_dragging_slider = True
            self.canvas.config(cursor="sb_h_double_arrow")
        else:
            # If not slider, start Panning
            self._start_pan(event)

    def _do_slider_drag(self, event):
        if self.is_dragging_slider:
            w = self.canvas.winfo_width()
            self.slider_pos = max(0.0, min(1.0, event.x / w))
            self._render()
        elif self.is_panning:
            self._do_pan(event)

    def _end_slider_drag(self, event):
        if self.is_dragging_slider:
            self.is_dragging_slider = False
        elif self.is_panning:
            self._end_pan(event)
        self.canvas.config(cursor="crosshair")

    def _start_pan(self, event):
        self.is_panning = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.canvas.config(cursor="fleur")

    def _do_pan(self, event):
        if self.is_panning:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.pan_x += dx
            self.pan_y += dy
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self._render()

    def _end_pan(self, event):
        self.is_panning = False
        self.canvas.config(cursor="crosshair")

    def _on_zoom(self, event):
        # Zoom towards mouse cursor
        factor = 1.1 if event.delta > 0 else 0.9
        
        # Limit zoom
        new_zoom = self.zoom * factor
        if new_zoom < 0.1 or new_zoom > 20.0:
            return
            
        # Adjust pan to keep pixel under mouse stationary
        # Math: P_new = Mouse - (Mouse - P_old) * factor
        # P is the offset of image top-left from canvas origin
        # Actually easier implementation:
        
        mouse_x = event.x
        mouse_y = event.y
        
        # Translate mouse to be relative to centered image (before pan)
        # Note: current pad_x = (w - disp_w)/2 + pan_x
        # We need to adjust pan_x so that the point under mouse stays effectively same
        
        # Simple relative offset approach
        # Offset from center of screen
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Mouse offset from center (screen space)
        mx_centered = mouse_x - w/2
        my_centered = mouse_y - h/2
        
        # Adjust pan: Pan moves away from mouse when zooming in
        self.pan_x = mx_centered - (mx_centered - self.pan_x) * factor
        self.pan_y = my_centered - (my_centered - self.pan_y) * factor
        
        self.zoom = new_zoom
        self._render()

    def _toggle_comparison(self, event=None):
        if self.slider_pos > 0.5:
            self.slider_pos = 0.0
        else:
            self.slider_pos = 1.0
        self._render()
        
    def _draw_placeholder(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_text(w//2, h//2, text="Open an image to start", fill="gray", font=("Arial", 16))

    def get_images(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Get the before and after images.
        
        Returns:
            Tuple of (before_image, after_image)
        """
        return self.before_image, self.after_image

    def get_comparison_full_res(self) -> Optional[Image.Image]:
        """Generate high-resolution composite image."""
        if not self.before_image or not self.after_image:
            return None
        
        width, height = self.before_image.size
        # Ensure sizes match
        if self.after_image.size != (width, height):
            img_after = self.after_image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            img_after = self.after_image
            
        composite = self.before_image.copy()
        split_x = int(width * self.slider_pos)
        
        # Paste right side of After image
        if split_x < width:
            after_crop = img_after.crop((split_x, 0, width, height))
            composite.paste(after_crop, (split_x, 0))
            
        # Draw separator line
        draw = ImageDraw.Draw(composite)
        line_width = max(2, int(width * 0.002))
        draw.line([(split_x, 0), (split_x, height)], fill="white", width=line_width)
        
        return composite
