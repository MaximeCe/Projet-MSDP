from point import Point
from edge import Edge
from tools.computation import Computation
import numpy as np


class Channel:
    def __init__(self, id, image, index, points):
        """
        :param id: identifiant unique du canal (1 à 9)
        :param image: instance de la classe Image
        :param index: position du canal dans la liste
        :param data: dictionnaire des points détectés, ex: {"as_": [...], "bs": [...], ..., "ks": [...], etc.}
        """
        self.id = id
        self.index = index
        self.image = image
        self.points = {}
        self.edges = []
        self.points_final = {}

        self.load_points(points)
        self.build_edges()
        self.compute_corners()

    def load_points(self, points, display = False):
        """Charge les points a-f, k-n depuis les structures as_, bs, etc."""
        noms = ['a', 'b', 'c', 'd', 'e', 'f', 'k', 'l', 'm', 'n']
        for nom in noms:
            liste = points.get(f"{nom}s") or points.get(f"{nom}s_")
            if liste and len(liste) > self.index:
                x, y = liste[self.index]
                self.points[nom] = Point(nom, x, y, self)
        
        if display == True:
            import matplotlib.pyplot as plt
            fig, (ax2) = plt.subplots(1, 1, figsize=(12, 5))

            # Image + points
            ax2.set_title(f"Canal {self.id} - Points détectés")
            ax2.imshow(self.image.data, cmap='gray')
            for point in self.points.values():
                ax2.scatter(point.x, point.y, color='red', s=10)
                ax2.text(point.x, point.y, point.nom, color='yellow', fontsize=8)
            ax2.axis('off')

            plt.tight_layout()
            plt.show()
        

    def build_edges(self):
        """Construit les bords : 2 paraboles (gauche/droite) et 2 droites (haut/bas)"""
        try:
            # Paraboles : c,e,a (gauche) et f,d,b (droite)
            parabole_gauche = Computation.parabolic_interpolation(
                self.points['a'].xy(), self.points['b'].xy(), self.points['c'].xy())
            parabole_droite = Computation.parabolic_interpolation(
                self.points['d'].xy(), self.points['e'].xy(), self.points['f'].xy())
            self.edges.append(Edge("parabole_gauche", *parabole_gauche, canal=self))
            self.edges.append(Edge("parabole_droite", *parabole_droite, canal=self))

            # Droites : c-f (haut), a-d (bas)
            droite_haut = Computation.line_coefficients(
                self.points['l'].xy(), self.points['n'].xy())
            droite_bas = Computation.line_coefficients(
                self.points['k'].xy(), self.points['m'].xy())
            self.edges.append(
                Edge("droite_haut", a=0,b=droite_haut[0], c = droite_haut[1] , canal=self))
            self.edges.append(
                Edge("droite_bas", a=0, b=droite_bas[0], c=droite_bas[1], canal=self))

        except Exception as e:
            print(
                f"❌ Erreur lors de la construction des bords du canal {self.id} : {e}")

    def compute_corners(self, display = False):
        """Calcule les sommets A, B, C, D, E, F du canal à partir des intersections."""

        parabolas = [edge.coefficients()
                        for edge in self.edges if "parabole" in edge.type]
        lines = [edge.coefficients()
                    for edge in self.edges if "parabole" not in edge.type]


        # Mock pour near_points → basé sur les points existants
        near_points = [
            self.points['c'].xy(),
            self.points['f'].xy(),
            self.points['a'].xy(),
            self.points['d'].xy()
        ]
        results = Computation.find_quadrilateral_corners(
            parabolas,
            lines,
            near_points
        )

        noms = ['C', 'F', 'D', 'A']
        for nom, (x, y) in zip(noms, results): # type: ignore
            self.points_final[nom] = Point(nom, x, y, self)
        self.points_final["B"] = Point("B", self.points['b'].x, self.points['b'].y, self)
        self.points_final["E"] = Point("E", self.points['e'].x, self.points['e'].y, self)

        if display:
            import matplotlib.pyplot as plt
            fig, (ax2) = plt.subplots(1, 1, figsize=(12, 5))

            # Image + points

            # Image + paraboles/droites
            ax2.set_title(f"Canal {self.id} - Paraboles et droites")
            ax2.imshow(self.image.data, cmap='gray')
            x = np.arange(self.image.data.shape[1])
            
            for point in self.points_final.values():
                ax2.scatter(point.x, point.y, color='blue', s=1)
                ax2.text(point.x, point.y, point.nom, color='yellow', fontsize=8)

            # Tracé des paraboles
            for p in parabolas:
                y = p[0]*x**2 + p[1]*x + p[2]
                mask = (y >= 0) & (y < self.image.data.shape[0])
                ax2.plot(x[mask], y[mask], 'r-', linewidth=0.5, label='Parabole')
    
            # Tracé des droites
            for l in lines:
                # l = (a=0, b, c) donc y = b*x + c
                y = l[1]*x + l[2]
                mask = (y >= 0) & (y < self.image.data.shape[0])
                ax2.plot(x[mask], y[mask], 'y-', linewidth=0.5, label='Droite')
            ax2.axis('off')

            plt.tight_layout()
            plt.show()


        def __str__(self):
            txt = f"Canal {self.id} : {len(self.points)} pts détectés"
            if self.points_final:
                txt += f", coins A-F calculés : {', '.join(self.points_final.keys())}"
            return txt
