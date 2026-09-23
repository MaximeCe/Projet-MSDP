#!/usr/bin/env python3
"""Convertit ms.par (format a8,i8 Fortran) -> config.yml (YAML moderne).

Même logique de résolution que le Fortran (dernière occurrence gagne).
Les clefs sont les noms propres (strip), valeurs entières."""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msdp.config import parse_ms_par, _to_dict

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/fortran/new/ms.par")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("new_python/config.yml")

params = _to_dict(parse_ms_par(src))
with open(dst, "w", encoding="utf-8") as fh:
    yaml.safe_dump({k: params[k] for k in sorted(params)}, fh,
                   sort_keys=False, default_flow_style=False)
print(f"config.yml écrit: {dst} ({len(params)} paramètres), source={src}")