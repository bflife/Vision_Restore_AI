"""
Vision-Restore AI - Batch Processing Panel

Interface for batch processing multiple images.
"""

from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional
import customtkinter as ctk

from vision_restore.core.config import Config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class BatchPanel(ctk.CTkFrame):
    """Batch processing queue interface."""

    def __init__(
        self,
        parent,
        config: Config,
        on_process_batch: Optional[Callable[[List[Path], Path], None]] = None,
        **kwargs
    ):
        """Initialize the batch panel.
        
        Args:
            parent: Parent widget
            config: Application configuration
            on_process_batch: Callback for batch processing
        """
        super().__init__(parent, height=60, **kwargs)
        
        self.config = config
        self.on_process_batch = on_process_batch
        
        # State
        self.queue: List[Path] = []
        self.output_dir: Optional[Path] = None
        
        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create batch panel widgets."""
        # Configure grid
        self.grid_columnconfigure(2, weight=1)
        
        # Toggle button / Header
        self.toggle_btn = ctk.CTkButton(
            self,
            text="📦 Batch Processing",
            command=self._toggle_expanded,
            fg_color="transparent",
            hover_color=("#e0e0e0", "#333333"),
            anchor="w",
            height=30,
            width=180,
        )
        self.toggle_btn.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        # Queue count
        self.count_label = ctk.CTkLabel(
            self,
            text="0 images",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.count_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # Quick actions (always visible)
        self.quick_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.quick_frame.grid(row=0, column=3, sticky="e", padx=10, pady=5)
        
        self.add_btn = ctk.CTkButton(
            self.quick_frame,
            text="➕ Add",
            command=self._add_images,
            width=70,
            height=28,
        )
        self.add_btn.pack(side="left", padx=2)
        
        self.add_folder_btn = ctk.CTkButton(
            self.quick_frame,
            text="📁 Folder",
            command=self._add_folder,
            width=70,
            height=28,
        )
        self.add_folder_btn.pack(side="left", padx=2)
        
        # Expanded content frame
        self.expanded_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        self.expanded_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        self.expanded_frame.grid_remove()  # Initially hidden
        
        self._create_expanded_content()

    def _create_expanded_content(self):
        """Create the expanded batch panel content."""
        self.expanded_frame.grid_columnconfigure(0, weight=1)
        
        # Queue list
        self.queue_frame = ctk.CTkScrollableFrame(
            self.expanded_frame,
            height=120,
            fg_color=("gray85", "gray20"),
        )
        self.queue_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        
        self.queue_label = ctk.CTkLabel(
            self.queue_frame,
            text="No images in queue",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.queue_label.pack(pady=20)
        
        # Output directory
        output_frame = ctk.CTkFrame(self.expanded_frame, fg_color="transparent")
        output_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        output_frame.grid_columnconfigure(1, weight=1)
        
        output_label = ctk.CTkLabel(
            output_frame,
            text="Output:",
            font=ctk.CTkFont(size=11),
        )
        output_label.grid(row=0, column=0, padx=(0, 10))
        
        self.output_entry = ctk.CTkEntry(
            output_frame,
            placeholder_text="Select output folder...",
            state="readonly",
        )
        self.output_entry.grid(row=0, column=1, sticky="ew")
        
        self.output_btn = ctk.CTkButton(
            output_frame,
            text="📂",
            command=self._select_output,
            width=40,
        )
        self.output_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Action buttons
        action_frame = ctk.CTkFrame(self.expanded_frame, fg_color="transparent")
        action_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        
        self.clear_btn = ctk.CTkButton(
            action_frame,
            text="🗑️ Clear Queue",
            command=self._clear_queue,
            width=120,
            fg_color="transparent",
            border_color=("gray70", "gray30"),
            border_width=1,
            hover_color=("#e0e0e0", "#333333"),
        )
        self.clear_btn.pack(side="left")
        
        self.start_btn = ctk.CTkButton(
            action_frame,
            text="🚀 Start Batch",
            command=self._start_batch,
            width=120,
            fg_color="#4CAF50",
            hover_color="#45a049",
            state="disabled",
        )
        self.start_btn.pack(side="right")

    def _toggle_expanded(self):
        """Toggle expanded view."""
        if self.expanded_frame.winfo_ismapped():
            self.expanded_frame.grid_remove()
            self.toggle_btn.configure(text="📦 Batch Processing ▼")
        else:
            self.expanded_frame.grid()
            self.toggle_btn.configure(text="📦 Batch Processing ▲")

    def _add_images(self):
        """Add images to the queue."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
            ("All files", "*.*"),
        ]
        
        files = filedialog.askopenfilenames(filetypes=filetypes)
        if files:
            for f in files:
                path = Path(f)
                if path not in self.queue:
                    self.queue.append(path)
            
            self._update_queue_display()
            logger.info(f"Added {len(files)} images to queue")

    def _add_folder(self):
        """Add all images from a folder."""
        folder = filedialog.askdirectory()
        if folder:
            folder_path = Path(folder)
            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            
            added = 0
            for file in folder_path.iterdir():
                if file.suffix.lower() in extensions and file not in self.queue:
                    self.queue.append(file)
                    added += 1
            
            self._update_queue_display()
            
            # Auto-set output directory
            if self.output_dir is None:
                self.output_dir = folder_path / "upscaled"
                self._update_output_display()
            
            logger.info(f"Added {added} images from folder")

    def _select_output(self):
        """Select output directory."""
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = Path(folder)
            self._update_output_display()

    def _update_queue_display(self):
        """Update the queue display."""
        # Clear current display
        for widget in self.queue_frame.winfo_children():
            widget.destroy()
        
        if not self.queue:
            self.queue_label = ctk.CTkLabel(
                self.queue_frame,
                text="No images in queue",
                font=ctk.CTkFont(size=11),
                text_color="gray",
            )
            self.queue_label.pack(pady=20)
            self.count_label.configure(text="0 images")
            self.start_btn.configure(state="disabled")
            return
        
        # Display queue items
        for i, path in enumerate(self.queue[:20]):  # Show max 20
            item_frame = ctk.CTkFrame(self.queue_frame, fg_color="transparent")
            item_frame.pack(fill="x", pady=1)
            
            # Filename
            name_label = ctk.CTkLabel(
                item_frame,
                text=f"{i+1}. {path.name}",
                font=ctk.CTkFont(size=10),
                anchor="w",
            )
            name_label.pack(side="left", padx=5)
            
            # Remove button
            remove_btn = ctk.CTkButton(
                item_frame,
                text="×",
                width=20,
                height=20,
                command=lambda p=path: self._remove_from_queue(p),
                fg_color="transparent",
                hover_color=("#ff6666", "#cc4444"),
            )
            remove_btn.pack(side="right", padx=5)
        
        if len(self.queue) > 20:
            more_label = ctk.CTkLabel(
                self.queue_frame,
                text=f"... and {len(self.queue) - 20} more",
                font=ctk.CTkFont(size=10),
                text_color="gray",
            )
            more_label.pack(pady=5)
        
        self.count_label.configure(text=f"{len(self.queue)} images")
        self._check_start_enabled()

    def _update_output_display(self):
        """Update output directory display."""
        if self.output_dir:
            self.output_entry.configure(state="normal")
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, str(self.output_dir))
            self.output_entry.configure(state="readonly")
        
        self._check_start_enabled()

    def _check_start_enabled(self):
        """Check if Start Batch button should be enabled."""
        if self.queue and self.output_dir:
            self.start_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="disabled")

    def _remove_from_queue(self, path: Path):
        """Remove an item from the queue."""
        if path in self.queue:
            self.queue.remove(path)
            self._update_queue_display()

    def _clear_queue(self):
        """Clear the entire queue."""
        self.queue.clear()
        self._update_queue_display()

    def _start_batch(self):
        """Start batch processing."""
        if not self.queue or not self.output_dir:
            return
        
        if self.on_process_batch:
            self.on_process_batch(self.queue.copy(), self.output_dir)

    def add_to_queue(self, files: List[Path]):
        """Add files to the queue programmatically.
        
        Args:
            files: List of file paths
        """
        for f in files:
            if f not in self.queue:
                self.queue.append(f)
        
        self._update_queue_display()

    def get_queue(self) -> List[Path]:
        """Get the current queue.
        
        Returns:
            List of file paths in queue
        """
        return self.queue.copy()

    def get_output_dir(self) -> Optional[Path]:
        """Get the output directory.
        
        Returns:
            Output directory path or None
        """
        return self.output_dir
