#!/usr/bin/env python3
"""
Vision-Restore AI - Quick Launcher

Run this script to launch the Vision-Restore AI GUI application.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vision_restore.gui.app import main

if __name__ == "__main__":
    main()
