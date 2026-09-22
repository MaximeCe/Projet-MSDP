import sys, os
sys.path.insert(0, '.')
from msdp.config import Config
c = Config.from_file('../src/fortran/new/ms.par')
print('lbdvel1/2/3 =', c.get('lbdvel1'), c.get('lbdvel2'), c.get('lbdvel3'))
print('nr, ntrans, iliss, jparab =', c.get('nr'), c.get('ntrans'), c.get('iliss'), c.get('jparab'))