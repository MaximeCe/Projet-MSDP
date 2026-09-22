import sys, time, os
sys.path.insert(0, '.')
from msdp.config import Config
from msdp.pipeline import run_pipeline

c = Config.from_file('../src/fortran/new/ms.par')
t0 = time.time()
res = run_pipeline(c, '/home/max/nextcloud/Workspace/Projet-MSDP/data/input',
                   work_dir='/tmp/msdp_py_run', run_label='001')
dt = time.time() - t0
print('=== RUN COMPLET OK — %.1f s ===' % dt)
out = res.outputs['run_dir']
print('sorties (%d fichiers):' % len([f for f in os.listdir(out) if not f.endswith('.lis')]))
for f in sorted(os.listdir(out)):
    p = os.path.join(out, f)
    print('   %-14s %9d o' % (f, os.path.getsize(p)))
print('vitesses:', len(res.vitesses))