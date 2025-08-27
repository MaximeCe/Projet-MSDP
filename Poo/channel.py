from point import Point
from edge import Edge
from tools.computation import Computation


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

    def load_points(self, points):
        """Charge les points a-f, k-n depuis les structures as_, bs, etc."""
        noms = ['a', 'b', 'c', 'd', 'e', 'f', 'k', 'l', 'm', 'n']
        for nom in noms:
            liste = points.get(f"{nom}s") or points.get(f"{nom}s_")
            if liste and len(liste) > self.index:
                x, y = liste[self.index]
                self.points[nom] = Point(nom, x, y, self)

    def build_edges(self):
        """Construit les bords : 2 paraboles (gauche/droite) et 2 droites (haut/bas)"""
        try:
            # Paraboles : c,e,a (gauche) et f,d,b (droite)
            parabole_gauche = Computation.parabolic_interpolation(
                self.points['c'].xy(), self.points['e'].xy(), self.points['a'].xy())
            parabole_droite = Computation.parabolic_interpolation(
                self.points['f'].xy(), self.points['d'].xy(), self.points['b'].xy())
            self.edges.append(Edge("parabole_gauche", *parabole_gauche, canal=self))
            self.edges.append(Edge("parabole_droite", *parabole_droite, canal=self))

            # Droites : c-f (haut), a-d (bas)
            droite_haut = Computation.line_coefficients(
                self.points['l'].xy(), self.points['n'].xy())
            droite_bas = Computation.line_coefficients(
                self.points['k'].xy(), self.points['m'].xy())
            self.edges.append(
                Edge("droite_haut", *droite_haut, c=None, canal=self))
            self.edges.append(
                Edge("droite_bas", *droite_bas, c=None, canal=self))

        except Exception as e:
            print(
                f"❌ Erreur lors de la construction des bords du canal {self.id} : {e}")

    def compute_corners(self):
        """Calcule les sommets A, B, C, D, E, F du canal à partir des intersections."""
        # try:
        # print(
        #     f"edges coefficients = {[edge.coefficients() for edge in self.edges]},edges types = {[edge.type for edge in self.edges]}")
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

        # except Exception as e:
        #     print(
        #         f"❌ Erreur lors du calcul des coins du canal {self.id} : {e}")

    def __str__(self):
        txt = f"Canal {self.id} : {len(self.points)} pts détectés"
        if self.points_final:
            txt += f", coins A-F calculés : {', '.join(self.points_final.keys())}"
        return txt
