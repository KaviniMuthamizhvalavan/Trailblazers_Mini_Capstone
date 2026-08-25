"""
Pytest configuration and path fix.
Adds project root to sys.path so imports work with plain `pytest`.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
