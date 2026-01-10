"""
Vision-Restore AI - Image Canvas

Simple image display canvas with zoom and pan capabilities.
"""

import tkinter as tk
from typing import Optional, Tuple
import customtkinter as ctk
from PIL import Image, ImageTk

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class ImageCanvas(ctk.CTkFrame):
    """Simple image display canvas with zoom/pan capabilities."""

    def __init__(self, parent, **kwargs):
        """Initialize the image canvas.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, **kwargs)
        
        # State
        self.image: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Create canvas
        self.canvas = tk.Canvas(
            self,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        
        # Bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.canvas.bind("<Button-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._do_pan)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        
        # Double-click to reset
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_view())

    def set_image(self, image: Optional[Image.Image]):
        """Set the image to display.
        
        Args:
            image: PIL Image or None to clear
        """
        self.image = image
        self.reset_view()

    def _on_resize(self, event):
        """Handle canvas resize."""
        self._render()

    def _render(self):
        """Render the image on canvas."""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        if self.image is None:
            self._draw_empty()
            return
        
        # Calculate display size
        img_width, img_height = self.image.size
        
        # Scale to fit with zoom
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h) * self.zoom_level
        
        display_width = max(1, int(img_width * scale))
        display_height = max(1, int(img_height * scale))
        
        # Resize image
        displayed = self.image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(displayed)
        
        # Calculate position (centered with pan offset)
        x = (canvas_width - display_width) // 2 + self.pan_x
        y = (canvas_height - display_height) // 2 + self.pan_y
        
        # Draw image
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        
        # Draw info overlay
        self._draw_info(canvas_width, canvas_height, img_width, img_height)

    def _draw_empty(self):
        """Draw empty state."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text="No image",
            font=("Segoe UI", 14),
            fill="#666666",
        )

    def _draw_info(
        self,
        canvas_width: int,
        canvas_height: int,
        img_width: int,
        img_height: int,
    ):
        """Draw image info overlay."""
        info_text = f"{img_width}×{img_height}  |  {int(self.zoom_level * 100)}%"
        
        # Background
        text_width = len(info_text) * 8
        self.canvas.create_rectangle(
            canvas_width - text_width - 20,
            5,
            canvas_width - 5,
            30,
            fill="#00000080",
            outline="",
        )
        
        # Text
        self.canvas.create_text(
            canvas_width - 12,
            17,
            anchor="e",
            text=info_text,
            font=("Consolas", 10),
            fill="#ffffff",
        )

    def _on_scroll(self, event):
        """Handle scroll for zoom."""
        if event.delta > 0:
            self.zoom_level = min(10.0, self.zoom_level * 1.1)
        else:
            self.zoom_level = max(0.1, self.zoom_level / 1.1)
        
        self._render()

    def _start_pan(self, event):
        """Start panning."""
        self.is_panning = True
        self.pan_start_x = event.x - self.pan_x
        self.pan_start_y = event.y - self.pan_y
        self.canvas.config(cursor="fleur")

    def _do_pan(self, event):
        """Perform panning."""
        if self.is_panning:
            self.pan_x = event.x - self.pan_start_x
            self.pan_y = event.y - self.pan_start_y
            self._render()

    def _end_pan(self, event):
        """End panning."""
        self.is_panning = False
        self.canvas.config(cursor="")

    def reset_view(self):
        """Reset zoom and pan to defaults."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render()

    def zoom_to_fit(self):
        """Zoom to fit image in canvas."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render()

    def zoom_actual(self):
        """Zoom to actual size (100%)."""
        if self.image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.image.size
        
        # Calculate scale to fit
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        fit_scale = min(scale_w, scale_h)
        
        # Set zoom to show at 100%
        self.zoom_level = 1.0 / fit_scale
        self.pan_x = 0
        self.pan_y = 0
        self._render()

    def get_image(self) -> Optional[Image.Image]:
        """Get the current image."""
        return self.image
