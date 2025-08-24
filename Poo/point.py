class Point:
    def __init__(self, nom, x, y, canal):
        self.nom = nom
        self.x = x
        self.y = y
        self.canal = canal

    def xy(self):
        """Renvoie le tuple (x, y), utile pour les calculs géométriques."""
        return (self.x, self.y)

    def __str__(self):
        return f"{self.nom}({self.x}, {self.y})"
