from point import Point
from edge import Edge
from tools import (
    parabolic_interpolation, line_coefficients, get_parallelogram_equations,
    find_quadrilateral_corners
)


class Channel:
    def __init__(self, id, image, index, data):
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

        self.load_points_from_data(data)
        self.build_edges()
        self.compute_corners()

    def load_points_from_data(self, data):
        """Charge les points a-f, k-n depuis les structures as_, bs, etc."""
        noms = ['a', 'b', 'c', 'd', 'e', 'f', 'k', 'l', 'm', 'n']
        for nom in noms:
            liste = data.get(f"{nom}s") or data.get(f"{nom}s_")
            if liste and len(liste) > self.index:
                x, y = liste[self.index]
                self.points[nom] = Point(nom, x, y, self)

    def build_edges(self):
        """Construit les bords : 2 paraboles (gauche/droite) et 2 droites (haut/bas)"""
        try:
            # Paraboles : c,e,a (gauche) et f,d,b (droite)
            pg = parabolic_interpolation(self.points['c'].xy(), self.points['e'].xy(), self.points['a'].xy())
            pd = parabolic_interpolation(self.points['f'].xy(), self.points['d'].xy(), self.points['b'].xy())
            self.edges.append(Edge("parabole_gauche", *pg, self))
            self.edges.append(Edge("parabole_droite", *pd, self))

            # Droites : c-f (haut), a-d (bas)
            dh = line_coefficients(self.points['c'].xy(), self.points['f'].xy())
            db = line_coefficients(self.points['a'].xy(), self.points['d'].xy())
            self.edges.append(Edge("droite_haut", *dh, c=None, canal=self))
            self.edges.append(Edge("droite_bas", *db, c=None, canal=self))

        except Exception as e:
            print(
                f"❌ Erreur lors de la construction des bords du canal {self.id} : {e}")

    def compute_corners(self):
        """Calcule les sommets A, B, C, D, E, F du canal à partir des intersections."""
        try:
            parabolas = [e.coefficients()
                         for e in self.edges if "parabole" in e.type]
            lines = [e.coefficients()
                     for e in self.edges if "droite" in e.type]

            # Mock pour near_points → basé sur les points existants
            near_points = [
                self.points['c'].xy(),
                self.points['f'].xy(),
                self.points['a'].xy(),
                self.points['d'].xy()
            ]

            results = find_quadrilateral_corners(
                parabolas * 9,  # on simule la liste complète de 18 pour éviter les erreurs
                lines * 9,
                self.index + 1,
                near_points
            )

            noms = ['C', 'F', 'D', 'A']
            for nom, (x, y) in zip(noms, results): # type: ignore
                self.points_final[nom] = Point(nom, x, y, self)
            self.points_final["B"] = Point("B", self.points['b'].x, self.points['b'].y, self)
            self.points_final["E"] = Point("E", self.points['e'].x, self.points['e'].y, self)

        except Exception as e:
            print(
                f"❌ Erreur lors du calcul des coins du canal {self.id} : {e}")

    def __str__(self):
        txt = f"Canal {self.id} : {len(self.points)} pts détectés"
        if self.points_final:
            txt += f", coins A-F calculés : {', '.join(self.points_final.keys())}"
        return txt
