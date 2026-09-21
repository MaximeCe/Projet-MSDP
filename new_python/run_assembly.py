import sys, time, os
sys.path.insert(0, '.')
from msdp.config import Config
from msdp.pipeline import run_pipeline

c = Config.from_file('../src/fortran/new/ms.par')
t0 = time.time()
res = run_pipeline(c, '/home/max/nextcloud/Workspace/Projet-MSDP/data/input',
                   work_dir='/tmp/msdp_py_run', run_label='001')
dt = time.time() - t0
print('=== RUN ASSEMBLAGE OK ===')
print('temps total: %.1f s' % dt)
print('cymx   :', res.cymx.shape)
print('cal    :', res.cal.shape, ' range [%.2f, %.2f]' % (res.cal.min(), res.cal.max()))
print('xr/yr  :', res.xr.shape, res.yr.shape)
print('sobstot:', res.sobstot.shape, ' range [%.1f, %.1f]' % (res.sobstot.min(), res.sobstot.max()))
print('profnf3:', res.profnf3.shape)
print('vitesses (n):', len(res.vitesses), res.vitesses[:4])
print('outputs:', list(res.outputs.keys()))
print('run dir:', res.outputs['run_dir'])
print('fichiers produit:')
for d in sorted(os.listdir(res.outputs['run_dir'])):
    p = os.path.join(res.outputs['run_dir'], d)
    print('   %-16s %8d o' % (d, os.path.getsize(p)))