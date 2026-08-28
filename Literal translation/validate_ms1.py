"""Validation reproductible de l'équivalence ms1.py vs ms1.f.

Compare la sortie brute produite par ms1.py (en-tête 512xint32 + 1536 lignes
de 1024 int16, sans marqueurs de record) à la référence Fortran (record
séquentiel gfortran : 1 record en-tête + 1536 records de 1024 int16, chaque
record préfixé/suffixé par un int32 de longueur).

Usage:
    python validate_ms1.py <sortie_ms1.py> <ref_fortran>
    python validate_ms1.py x170330_09564585_00000 DPSM-Fortran/x170330_09564585_00000
"""
import sys
import struct

import numpy as np

ISP, JSP = 1024, 1536  # dims après permutation (isp=js, jsp=is)


def read_ms1py_raw(path):
    """Lit la sortie brute de ms1.py (tofile, sans marqueurs)."""
    b = open(path, 'rb').read()
    # en-tête : 512 x int32 (2048 o), puis JSP records de ISP int16
    data = np.frombuffer(b[2048:], dtype='<i2').reshape(JSP, ISP)
    return data


def read_fortran_ref(path):
    """Lit la référence Fortran (record séquentiel gfortran)."""
    f = open(path, 'rb')
    recs = []
    while True:
        pre = f.read(4)
        if len(pre) < 4:
            break
        n = struct.unpack('<i', pre)[0]
        recs.append((n, f.read(n)))
        f.read(4)  # suffixe de longueur
    f.close()
    data = np.array([np.frombuffer(d, dtype='<i2') for n, d in recs if n == 2048])
    return data[1:]  # le 1er record est l'en-tête (512 x int32), pas une ligne data


def main():
    py_path, ref_path = sys.argv[1], sys.argv[2]
    py = read_ms1py_raw(py_path)
    ref = read_fortran_ref(ref_path)
    assert py.shape == ref.shape, f"formes différentes {py.shape} vs {ref.shape}"

    diff = np.abs(py.astype(int) - ref.astype(int))
    ok = np.array_equal(py, ref)
    print(f"formes : {py.shape}")
    print(f"coins : py={tuple(int(v) for v in (py[0,0],py[0,-1],py[-1,0],py[-1,-1]))} "
          f"ref={tuple(int(v) for v in (ref[0,0],ref[0,-1],ref[-1,0],ref[-1,-1]))}")
    print(f"IDENTIQUE valeur: {ok}")
    print(f"max abs diff: {int(diff.max())}  nb diff: {int(np.count_nonzero(diff))} / {py.size}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())