# domain/point.py
"""
Point - Simple data class representing a geometric point in a channel
"""

import numpy as np


class Point:
    """Represents a geometric point with a name and coordinates."""

    def __init__(self, name, x, y, channel=None):
        """
        Initialize a point.
        
        Args:
            name (str): Point identifier (e.g., 'a', 'b', 'A', 'B')
            x (float): X coordinate
            y (float): Y coordinate
            channel: Reference to parent channel (optional)
        """
        self.name = name
        self.x = x
        self.y = y
        self.channel = channel

    def xy(self):
        """Return coordinates as tuple for geometric calculations."""
        return (self.x, self.y)

    def __str__(self):
        return f"{self.name}({self.x:.2f}, {self.y:.2f})"

    def __repr__(self):
        return f"Point('{self.name}', {self.x:.2f}, {self.y:.2f})"










