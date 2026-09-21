"""msdp — Portage moderne Python du pipeline MSDP (legacy Fortran 77).

Modules :
- ``config``         : chargeur ms.par (format a8,i8)
- ``io_fits``        : lecture / écriture FITS, échange d'octets
- ``step1_average``  : moyennes dark/flat (ét. 1)
- ``step2_geometry`` : géométrie des canaux / SRECT-newgeom (ét. 2)
- ``step3_*``        : canaux + calibration (ét. 3)
- ``step4_*``        : observations + profils I/V + vitesses (ét. 4)
- ``plotting``       : sorties Matplotlib (remplace PGPLOT)
- ``pipeline``       : orchestrateur de bout en bout
"""

from msdp.config import Config, MsPar, parse_ms_par

__all__ = ["Config", "MsPar", "parse_ms_par"]
__version__ = "0.1.0"