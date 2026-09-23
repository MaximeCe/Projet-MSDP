"""
Configuration MSDP — chargeur du fichier ``ms.par`` (format Fortran ``a8,i8``)
ou fichier YAML ``config.yml``.

La source de vérité des paramètres est le fichier ``ms.par`` du pipeline Fortran
(``src/fortran/new/ms.par``) ou un fichier YAML moderne ``config.yml``.
Chaque ligne ms.par porte **exactement** un nom sur 8
caractères (colonnes 1-8, droit-justifié, préfixé d'espaces) et une valeur
entière sur 8 caractères (colonnes 9-16).

Richesses héritées du Fortran que ce parseur préserve :
- les **espaces de tête** sont significatifs côté nom (``par1('      is',...)``
  cherche ``"      is"``, pas ``"is"``) — le nom est donc extrait en ``colonne 1-8``
  puis ``.strip()`` pour l'accès Pythonique, mais aussi conservé brut pour rester
  fidèle aux appels ``par1``.
- le terminateur est le nom ``end`` **à partir de la colonne 1** (le Fortran
  compare ``nom == 'end     '``).
- les lignes de commentaire / en-tête (sans valeur entière en colonne 9-16)
  sont ignorées silencieusement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class MsPar:
    """Un paramètre brut lu depuis ``ms.par``.

    ``name_raw`` conserve la colonne 1-8 intacte (8 caractères, droit-justifiée) —
    c'est ce que compare ``par1`` côté Fortran.
    """

    name_raw: str
    value: int

    @property
    def name(self) -> str:
        """Nom nettoyé (sans espaces de tête/traîne), utilisable en Python."""
        return self.name_raw.strip()


def parse_ms_par(path: str | Path) -> list[MsPar]:
    """Parse un fichier ``ms.par`` et renvoie les paramètres dans l'ordre du fichier.

    Format d'une ligne valide : ``<8 car. nom><8 car. valeur>`` puis éventuellement
    un commentaire. Une ligne est ignorée si la plage 9-16 ne contient pas
    d'entier. La lecture s'arrête au premier nom ``end`` (colonne 1).

    Parameters
    ----------
    path : str | Path
        Chemin vers ``ms.par``.

    Returns
    -------
    list[MsPar]
        Les paramètres, dans l'ordre de déclaration du fichier.
    """
    params: list[MsPar] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if len(line) < 8:
                continue
            name_raw = line[0:8]
            if name_raw == "end     " or name_raw.strip() == "end":
                break
            if len(line) < 16:
                continue
            value_field = line[8:16].strip()
            if not value_field.isdigit():
                # ligne d'en-tête / commentaire (pas un entier en col 9-16)
                continue
            params.append(MsPar(name_raw=name_raw, value=int(value_field)))
    return params


def _to_dict(pars: Iterator[MsPar]) -> dict[str, int]:
    """Convertit une séquence de :class:`MsPar` en dict nom -> valeur.

    Attention : le Fortran peut déclarer le même nom deux fois (ex. ``iobs``
    apparaît 2× dans ms.par). La **dernière** occurrence gagne, cohérent avec
    `par1`/`readpar` qui relisent le fichier et retournent la dernière valeur.
    """
    out: dict[str, int] = {}
    for p in pars:
        out[p.name] = p.value
    return out


@dataclass(frozen=True)
class Config:
    """Configuration typée du pipeline MSDP.

    Résultat de chargement : une table ``params`` (nom -> valeur) + des accès
    typés fréquentissimes. En cas de paramètre absent, ``get`` renvoie le défaut.

    Parameters
    ----------
    ms_par_path : str | Path
        Chemin du fichier ``ms.par`` ou ``config.yml``.
    params : dict[str, int] | None
        Table pré-chargée (utile pour les tests / surcharge). Si fournie, elle
        fait foi sans re-lecture du fichier.
    """

    ms_par_path: str | Path
    params: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Charge une configuration depuis un fichier ``ms.par`` ou ``config.yml``.

        Si l'extension est .yml ou .yaml, charge comme YAML.
        Sinon, charge comme ms.par (format Fortran a8,i8).
        """
        path = Path(path)
        if path.suffix.lower() in (".yml", ".yaml"):
            return cls.from_yaml(path)
        return cls(ms_par_path=path, params=_to_dict(parse_ms_par(path)))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Charge une configuration depuis un fichier YAML."""
        if yaml is None:
            raise RuntimeError("PyYAML n'est pas installé (pip install pyyaml)")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Fichier YAML {path} doit contenir un mapping")
        # Convertir toutes les valeurs en int (le pipeline attend des int)
        params = {k: int(v) for k, v in data.items()}
        return cls(ms_par_path=path, params=params)

    @classmethod
    def from_dict(cls, params: dict[str, int]) -> "Config":
        """Construit une config de toutes pièces (tests, surcharge)."""
        return cls(ms_par_path="<dict>", params=params)

    # -- accès génériques ----------------------------------------------------
    def get(self, key: str, default: int | None = None) -> int | None:
        """Retourne la valeur d'un paramètre (None si absent et pas de défaut)."""
        return self.params.get(key, default)

    def require(self, key: str) -> int:
        """Retourne la valeur d'un paramètre, lève une exception si absent."""
        if key not in self.params:
            raise KeyError(f"Paramètre '{key}' absent de la configuration")
        return self.params[key]

    def __getitem__(self, key: str) -> int:
        return self.require(key)

    def __contains__(self, key: str) -> bool:
        return key in self.params

    # -- accès typés (dimensions CCD) ----------------------------------------
    @property
    def ccd_x(self) -> int:
        """``is`` — dimension X du CCD (1536)."""
        return self.get("is", 1536)

    @property
    def ccd_y(self) -> int:
        """``js`` — dimension Y du CCD (1024)."""
        return self.get("js", 1024)

    @property
    def nm(self) -> int:
        """``nm`` — nombre de canaux (9)."""
        return self.get("nm", 9)

    @property
    def ipermute(self) -> bool:
        """``ipermu`` — permutation X/Y avant géométrie (1 = activée)."""
        return self.get("ipermu", 0) == 1

    @property
    def iswapped(self) -> bool:
        """``iswap`` — échange d'octets (endianness) à l'ouverture (1 = oui)."""
        return self.get("iswap", 0) == 1

    # dimensions du champ permuté
    @property
    def isp(self) -> int:
        """Dimension X après permutation (1024 si ipermu)."""
        return self.ccd_y if self.ipermute else self.ccd_x

    @property
    def jsp(self) -> int:
        """Dimension Y après permutation (1536 si ipermu)."""
        return self.ccd_x if self.ipermute else self.ccd_y

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(from={self.ms_par_path}, nparams={len(self.params)})"