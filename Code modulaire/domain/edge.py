# domain/edge.py
"""
Edge - Represents a parabolic or linear edge of a channel
"""


class Edge:
    """
    Represents a channel boundary (parabola or line).
    Stores equation coefficients y = ax² + bx + c (a=0 for lines).
    """

    def __init__(self, edge_type, a, b, c, channel=None):
        """
        Initialize an edge.
        
        Args:
            edge_type (str): Type identifier (e.g., 'parabola_left', 'line_top')
            a (float): Quadratic coefficient (0 for lines)
            b (float): Linear coefficient
            c (float): Constant coefficient
            channel: Reference to parent channel (optional)
        """
        self.type = edge_type
        self.a = a
        self.b = b
        self.c = c
        self.channel = channel

    def coefficients(self):
        """Return coefficients as tuple."""
        return (self.a, self.b, self.c)

    def is_parabola(self):
        """Check if edge is parabolic (vs linear)."""
        return abs(self.a) > 1e-6

    def __str__(self):
        if self.is_parabola():
            return f"{self.type}: y = {self.a:.3f}x² + {self.b:.3f}x + {self.c:.3f}"
        else:
            return f"{self.type}: y = {self.b:.3f}x + {self.c:.3f}"
