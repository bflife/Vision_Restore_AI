"""
Vision-Restore AI - Progress Bar Component

Animated progress bar for displaying processing status.
"""

import customtkinter as ctk

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class ProcessingProgress(ctk.CTkFrame):
    """Animated progress bar with status text."""

    def __init__(self, parent, **kwargs):
        """Initialize the progress component.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, fg_color=("gray90", "gray17"), **kwargs)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Status text
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self,
            orientation="horizontal",
            mode="determinate",
            height=10,
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.progress_bar.set(0)
        
        # Percentage label
        self.percent_label = ctk.CTkLabel(
            self,
            text="0%",
            font=ctk.CTkFont(size=10),
            width=40,
        )
        self.percent_label.grid(row=1, column=1, padx=(0, 10))

    def set_progress(self, progress: float, status: str = ""):
        """Set the progress value and status.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            status: Status message to display
        """
        # Clamp progress
        progress = max(0.0, min(1.0, progress))
        
        # Update progress bar
        self.progress_bar.set(progress)
        
        # Update percentage
        percent = int(progress * 100)
        self.percent_label.configure(text=f"{percent}%")
        
        # Update status
        if status:
            self.status_label.configure(text=status)
        
        # Force update
        self.update_idletasks()

    def set_status(self, status: str):
        """Set only the status text.
        
        Args:
            status: Status message
        """
        self.status_label.configure(text=status)
        self.update_idletasks()

    def set_indeterminate(self, enabled: bool = True):
        """Set indeterminate mode.
        
        Args:
            enabled: Enable or disable indeterminate mode
        """
        if enabled:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
            self.percent_label.configure(text="")
        else:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)
            self.percent_label.configure(text="0%")

    def reset(self):
        """Reset progress to initial state."""
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self.status_label.configure(text="Ready")
        self.progress_bar.configure(mode="determinate")


class MultiStepProgress(ctk.CTkFrame):
    """Multi-step progress indicator with stage visualization."""

    def __init__(self, parent, steps: list[str], **kwargs):
        """Initialize multi-step progress.
        
        Args:
            parent: Parent widget
            steps: List of step names
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.steps = steps
        self.current_step = 0
        self.step_labels = []
        self.step_dots = []
        
        self._create_widgets()

    def _create_widgets(self):
        """Create step indicator widgets."""
        for i, step_name in enumerate(self.steps):
            # Step frame
            step_frame = ctk.CTkFrame(self, fg_color="transparent")
            step_frame.pack(side="left", expand=True, fill="x")
            
            # Dot indicator
            dot = ctk.CTkLabel(
                step_frame,
                text="○",
                font=ctk.CTkFont(size=16),
                text_color="gray",
            )
            dot.pack()
            self.step_dots.append(dot)
            
            # Step name
            label = ctk.CTkLabel(
                step_frame,
                text=step_name,
                font=ctk.CTkFont(size=9),
                text_color="gray",
            )
            label.pack()
            self.step_labels.append(label)

    def set_step(self, step_index: int):
        """Set the current step.
        
        Args:
            step_index: Index of current step
        """
        self.current_step = step_index
        
        for i, (dot, label) in enumerate(zip(self.step_dots, self.step_labels)):
            if i < step_index:
                # Completed
                dot.configure(text="●", text_color="#4CAF50")
                label.configure(text_color="#4CAF50")
            elif i == step_index:
                # Current
                dot.configure(text="◐", text_color="#2196F3")
                label.configure(text_color="#2196F3")
            else:
                # Pending
                dot.configure(text="○", text_color="gray")
                label.configure(text_color="gray")

    def complete_all(self):
        """Mark all steps as complete."""
        for dot, label in zip(self.step_dots, self.step_labels):
            dot.configure(text="●", text_color="#4CAF50")
            label.configure(text_color="#4CAF50")

    def reset(self):
        """Reset all steps to pending."""
        for dot, label in zip(self.step_dots, self.step_labels):
            dot.configure(text="○", text_color="gray")
            label.configure(text_color="gray")
        self.current_step = 0
