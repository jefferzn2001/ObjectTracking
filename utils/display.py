"""
Tkinter-based display for tracking visualization.

Replaces cv2.imshow which segfaults on Linux+NVIDIA due to Qt5/CUDA
GPU context conflicts. Uses Tkinter which has no GPU dependencies.
"""

import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk


class TkDisplay:
    """
    Simple Tkinter window for displaying OpenCV BGR frames.

    Args:
        title (str): Window title.
    """

    def __init__(self, title: str = "Viewer"):
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._closed = False
        self._last_key: Optional[str] = None
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Key>", self._on_key)

    def _on_close(self) -> None:
        """Handle window close."""
        self._closed = True
        self._root.destroy()

    def _on_key(self, event: tk.Event) -> None:
        """Capture key press."""
        self._last_key = event.keysym

    def show(self, bgr_frame: np.ndarray) -> Optional[str]:
        """
        Display a BGR frame and return any key pressed.

        Args:
            bgr_frame (np.ndarray): BGR image to display.

        Returns:
            Optional[str]: Key name if pressed ('q', 'r', 'Escape', etc.), or None.
        """
        if self._closed:
            return "q"

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self._label.configure(image=self._tk_image)

        self._last_key = None
        self._root.update_idletasks()
        self._root.update()

        return self._last_key

    @property
    def closed(self) -> bool:
        """Whether the window has been closed."""
        return self._closed

    def destroy(self) -> None:
        """Close the window."""
        if not self._closed:
            self._closed = True
            try:
                self._root.destroy()
            except tk.TclError:
                pass
