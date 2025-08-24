class Edge:
    def __init__(self, edge_type, a, b, c, canal):
        """
        Représente une bordure (parabole ou droite) d'un canal.

        :param edge_type: Type du bord (ex: 'parabole_gauche', 'droite_bas')
        :param a: Coefficient a (0 si droite)
        :param b: Coefficient b
        :param c: Coefficient c
        :param canal: Référence vers le canal auquel ce bord appartient
        """
        self.type = edge_type
        self.a = a
        self.b = b
        self.c = c
        self.canal = canal

    def coefficients(self):
        """Retourne les coefficients de l'équation."""
        return (self.a, self.b, self.c)

    def __str__(self):
        if "parabole" in self.type:
            eq = f"{self.a:.2f}x² + {self.b:.2f}x + {self.c:.2f}"
        else:
            # Pour les droites : y = ax + b
            eq = f"{self.b:.2f}x + {self.c:.2f}"
        return f"{self.type} : y = {eq}"
