# domain/channel.py
"""
Channel - Represents a single MSDP channel in CCD coordinates
Pure data class - all processing logic moved to services
"""


class Channel:
    """
    Represents a single spectral channel in CCD coordinates.
    Contains detected points, computed edges, and corner positions.
    
    All geometric processing is delegated to GeometryService.
    """

    def __init__(self, channel_id, index):
        """
        Initialize a channel.
        
        Args:
            channel_id (int): Channel identifier (1 to nm)
            index (int): Zero-based index in channel list
        """
        self.id = channel_id
        self.index = index

        # Detected points (a, b, c, d, e, f, k, l, m, n)
        self.points = {}

        # Computed edges (parabolas and lines)
        self.edges = []

        # Computed final corners (A, B, C, D, E, F)
        self.points_final = {}

    def set_points(self, points_dict):
        """
        Set detected points for this channel.
        
        Args:
            points_dict (dict): Dictionary with Point objects for each position
        """
        self.points = points_dict

    def set_edges(self, edges):
        """
        Set computed edges for this channel.
        
        Args:
            edges (list): List of Edge objects
        """
        self.edges = edges

    def set_final_points(self, final_points):
        """
        Set computed corner points (ABCDEF).
        
        Args:
            final_points (dict): Dictionary with Point objects for corners
        """
        self.points_final = final_points

    def __str__(self):
        return f"Channel {self.id} (index={self.index})"

    def __repr__(self):
        return f"Channel(id={self.id}, points={len(self.points)}, final={len(self.points_final)})"
