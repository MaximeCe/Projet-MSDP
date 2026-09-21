"""Smoke-test d'assemblage : pipeline + plotting, small pipeline via données réelles
mais limité à la partie chaîne (sans le run obs complet) pour rester rapide."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from msdp.config import Config
from msdp import step1_average as s1
from msdp import step2_geometry as s2
from msdp import step3_channels as s3c
from msdp import step3_calib as s3b
from msdp import step4_solarobs as s4s
from msdp import step4_ivmaps as s4i
from msdp import plotting

c = Config.from_file('../src/fortran/new/ms.par')
work = '/tmp/msdp_smoke'
os.makedirs(work, exist_ok=True)
avg = s1.run_step1('/home/max/nextcloud/Workspace/Projet-MSDP/data/input', c, work)
dark, flat = work+'/'+avg['dark'].filename, work+'/'+avg['flat'].filename
mf = s2._load_meanflat(dark, flat, c.ccd_y, c.ccd_x)
geom = s2.detect_geometry(mf, c)
xr, yr = geom.to_xr_yr()
iim, jjm = s3c.channels_dims(c.get('li'), c.get('lj'), c.get('milsec'))
cymx = s3c.extract_channels(mf.T, xr, yr, c.get('li'), c.get('lj'), c.get('milsec'), c.nm)

# plotting
plotting.plot_geo(geom.xx, geom.yy, (iim, jjm), work+'/geo2.pdf')
plotting.plot_calmap(np.ones_like(cymx), work+'/cal.pdf')
ob = np.random.default_rng(0).normal(500, 200, (iim, jjm, c.nm)).clip(1, None)
plotting.plot_obs(ob, work+'/obs.pdf')
profnf = np.random.default_rng(1).normal(500, 100, (jjm, 81, 3)).clip(1, None)
plotting.plot_ivprof(profnf, work+'/ivprof1.pdf')
plotting.plot_edges(np.array([151,501,851]), mf[:,500], np.gradient(mf[:,500]), (iim,jjm), work+'/geo1.pdf')

print('SMOKE OK — dims cymx', cymx.shape)
print('pdfs:', sorted(f for f in os.listdir(work) if f.endswith('.pdf')))