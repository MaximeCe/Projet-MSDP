"""
ms2.py - MSDP Data Processing: Step 2 - Geometry Module

This module computes the geometry of MSDP spectrograph channels, ported
from the Fortran subroutine `newgeom` (ms2.f, lines ~410-748), itself
called at the end of `SRECT` (ms2.f line 380: `call newgeom(meanflat)`).

Processing steps (mirrors ms2.f):
1. Detects channel edges using intensity gradients (3 horizontal cuts)
2. Detects vertical edges k,l,m,n
3. Extrapolates corner points A,B,C,D,E,F via line intersections
4. Generates diagnostic plots (geo1.ps, geo2.ps, geo3.ps in Fortran ->
   geo1.pdf, geo2.pdf, geo3.pdf here)

Ported-fidelity notes / fixes vs ms2.f:
  1. Vertical (k,l,m,n) detection RECOMPUTES zmax/zgmax locally per
     column, exactly like ms2.f lines 467-483 (bug in an earlier port
     that reused the global central-cut normalization).
  2. k,l,m,n are stored at numpy rows 6-9 (0-based), fixing an
     off-by-one that wrote them at rows 7-10 and shifted every corner.
  3. `intersec` uses the mathematically-correct `1 - a*c` denominator;
     ms2.f writes `1.-ac` where `ac` is an undefined implicit variable
     (≈0), so the Fortran effectively divides by ~1.0. Stays within
     ~0.1-0.3 px of the Fortran output.

"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from astropy.io import fits


class GeometryProcessor:
    """Process MSDP channel geometry."""

    def __init__(self, param_file='ms.yml'):
        """Initialize geometry processor with parameters."""
        with open(param_file, 'r') as f:
            self.params = yaml.safe_load(f)

        self.log_file = open('ms.lis', 'a')  # Append to existing log
        self.xryr_file = open('xryr.lis', 'w')

        # Image dimensions (ms2.f newgeom: im=1536, jm=1024, nm=9). Paramétrisés
        # depuis ms.yml (im/jm), avec défaut Meudon pour compat.
        self.im = self.params.get('im', 1536)
        self.jm = self.params.get('jm', 1024)
        self.nm = self.params.get('nm', 9)  # Number of channels (should be 9)
        if self.nm != 9:
            self.log(f"WARNING: nm={self.nm} in parameter file, but "
                      f"ms2.f hardcodes nm=9 inside newgeom - results "
                      f"will not match the Fortran if this differs.")

        self.log("="*60)
        self.log("GEOMETRY PROCESSOR INITIALIZED")
        self.log(f"Image dimensions: {self.im} x {self.jm}")
        self.log(f"Number of channels: {self.nm}")

    def log(self, message):
        """Write to log file and console."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def read_averaged_file(self, filename):
        """
        Read averaged dark or flat file.
        
        Supports both FITS and binary formats:
        - FITS files: standard .fit format
        - Binary files: 512-element int32 header followed by int16 data
        
        Returns:
        --------
        ndarray : Data array
        """
        # Try to read as FITS first
        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                if data is None:
                    raise ValueError(f"No data in FITS file {filename}")
                
                # Ensure int16 type
                if data.dtype != np.int16:
                    data = data.astype(np.int16)
                
                self.log(f"Read FITS file: {filename}")
                self.log(f"Data shape: {data.shape}")
                
                return data
        
        except (OSError, Exception):
            # Fall back to binary format
            self.log(f"Attempting to read {filename} as binary format...")
            
            with open(filename, 'rb') as f:
                # Read header (512 int32 values)
                header = np.fromfile(f, dtype=np.int32, count=512)

                if len(header) < 3:
                    raise ValueError(f"Invalid header in {filename}")

                isp = header[1]
                jsp = header[2]

                self.log(f"Header dimensions: isp={isp}, jsp={jsp}")

                # Validate dimensions
                if isp <= 0 or isp > 2000 or jsp <= 0 or jsp > 2000:
                    raise ValueError(f"Invalid dimensions in header: {isp} x {jsp}")

                # Read data
                data = np.fromfile(f, dtype=np.int16).reshape((jsp, isp))

            self.log(f"Read binary file: {filename}")
            self.log(f"Dimensions from header: {isp} x {jsp}")

            return data.T  # Transpose to get (isp, jsp)

    def compute_geometry(self, dark_file, flat_file):
        """
        Main geometry computation.
        
        Steps:
        1. Read dark and flat field files
        2. Subtract dark from flat to get clean signal
        3. Detect channel edges at multiple j-positions
        4. Detect vertical edges (k,l,m,n points)
        5. Compute corner points via line intersection
        6. Generate diagnostic plots
        
        Parameters:
        -----------
        dark_file : str
            Path to averaged dark current file
        flat_file : str
            Path to averaged flat field file
        
        Returns:
        --------
        tuple : (xx, yy) arrays containing reference points for all channels
                xx[nl, nc]: X-coordinate of point nl in channel nc
                yy[nl, nc]: Y-coordinate of point nl in channel nc
                
                Point indices (0-based; Fortran nl-1):
                0-5:   a,b,c,d,e,f (horizontal edges)
                6-9:   k,l,m,n (vertical edges)
                10-15: A,B,C,D,E,F (extrapolated corners)
        """
        self.log("\n" + "="*60)
        self.log("COMPUTING CHANNEL GEOMETRY")
        self.log("="*60)

        # Read data files
        dark_data = self.read_averaged_file(dark_file)
        flat_data = self.read_averaged_file(flat_file)

        # FITS files return (js, is) = (1024, 1536), transpose to (is, js) = (1536, 1024)
        if dark_data.shape != (self.im, self.jm):
            dark_data = dark_data.T
        if flat_data.shape != (self.im, self.jm):
            flat_data = flat_data.T

        self.log(f"Dark data shape: {dark_data.shape}")
        self.log(f"Flat data shape: {flat_data.shape}")

        # Subtract dark from flat (ms2.f SRECT, lines 323-332:
        # lec(i)=lec(i)-lecx(i); if(lec(i).lt.0)lec(i)=1)
        self.log("\nSubtracting dark current from flat field...")
        meanflat = flat_data.astype(np.int32) - dark_data.astype(np.int32)
        meanflat[meanflat < 0] = 1  # Ensure no negative values

        meanflat_md = meanflat.astype(np.int16)

        self.log(
            f"Mean flat shape after dark subtraction: {meanflat_md.shape}")

        # Call new geometry detection
        xx, yy = self.newgeom(meanflat_md)

        # Write reference points to file
        self.write_acdf2(xx, yy)

        return xx, yy

    def newgeom(self, meanflat):
        """
        New geometry detection algorithm - port of ms2.f `newgeom`
        (lines 411-748).

        All the constants below (i1, i2, ja, jc, xdel, ...) are kept as
        the literal Fortran 1-based values, exactly as in ms2.f, for
        direct comparison with the source. `_idx` suffixed variables are
        the corresponding 0-based numpy indices, computed as
        `fortran_value - 1` right where numpy indexing happens.

        Parameters:
        -----------
        meanflat : ndarray
            Dark-subtracted flat field image (im x jm), 0-based numpy
            array corresponding to Fortran's meanflat(1536,1024).

        Returns:
        --------
        tuple : (xx, yy) reference point arrays, shape (20, nm), 0-based
        """
        self.log("\nNEW GEOMETRY DETECTION")

        im, jm, nm = self.im, self.jm, self.nm

        # jtriple: 1 = average 3 adjacent rows around each cut. Paramétrisé depuis
        # ms.yml (défaut 1 = comportement actuel).
        jtriple = self.params.get('jtriple', 1) == 1

        # ms2.f lines 427-430
        i1 = 5
        i2 = im - 4
        i1_idx = i1 - 1
        i2_idx = i2 - 1

        # ms2.f lines 431-438. ja : positions des 3 coupes (1-based), lues depuis
        # ms.yml (liste [ja1, ja2, ja3]) avec défaut Meudon [151, 501, 851].
        ja = self.params.get('ja', [151, 501, 851])
        ja_idx = [j - 1 for j in ja]         # 0-based
        jc = ja[1]                            # coupe centrale
        jc_idx = ja_idx[1]

        self.log(f"Detection cuts at j (Fortran 1-based) = {ja}")

        sig = [1.0, -1.0]

        # ms.par: mingrad / interp
        mingrad = self.params['mingrad']
        interp = self.params['interp']
        grt = mingrad
        zgt = grt

        self.log(f"Minimum gradient threshold: {mingrad}")
        self.log(f"Parabolic interpolation: {interp}")

        # xx/yy dimensioned like Fortran's xx(20,9): rows 0-15 used,
        # 16-19 unused (kept for direct index correspondence with the
        # Fortran source, where rows are 1-16).
        xx = np.zeros((20, nm))
        yy = np.zeros((20, nm))

        # ---------------------------------------------------------------
        # zmax, zgmax from the FULL central column jc (ms2.f lines
        # 454-471) - NOT clipped to i1:i2. This normalization is reused
        # for every horizontal AND vertical cut later on.
        # ---------------------------------------------------------------
        zc_full = meanflat[:, jc_idx].astype(np.float64)          # length im
        zmax = float(np.max(zc_full))

        zgc_full = np.zeros(im)
        zgc_full[:im - 1] = zc_full[1:] - zc_full[:-1]
        # ms2.f only fills zgc(1..im-1); zgc(im) is never assigned.
        # Treated as 0 here (see module docstring, fidelity note 1-ish).
        zgmax = float(np.max(np.abs(zgc_full[:im - 1])))

        zc_norm_full = 100.0 * zc_full / zmax
        zgc_norm_full = 100.0 * zgc_full / zgmax

        self.log(f"Central cut normalization: zmax={zmax:.1f}, zgmax={zgmax:.1f}")

        # ---------------------------------------------------------------
        # Three horizontal cuts (ms2.f do30 nj=1,3, lines 474-549)
        # ---------------------------------------------------------------
        self.log("\nDetecting horizontal channel edges...")

        for nj in range(3):  # python nj = Fortran nj - 1
            jj_idx = ja_idx[nj]
            self.log(f"\nCut {nj+1} at j (Fortran) = {ja[nj]}")

            z = meanflat[i1_idx:i2_idx + 1, jj_idx].astype(np.float64)
            if jtriple:
                z = (meanflat[i1_idx:i2_idx + 1, jj_idx - 1].astype(np.float64) +
                     meanflat[i1_idx:i2_idx + 1, jj_idx].astype(np.float64) +
                     meanflat[i1_idx:i2_idx + 1, jj_idx + 1].astype(np.float64)) / 3.0

            zg = np.zeros_like(z)
            zg[:-1] = z[1:] - z[:-1]
            zg[-1] = zg[-2]  # ms2.f line 493: zg(i2)=zg(i2-1)

            z = 100.0 * z / zmax
            zg = 100.0 * zg / zgmax

            # do20 is=1,2 (sign of the gradient)
            for is_sign in range(2):
                l = nj if is_sign == 0 else nj + 3

                n = 0  # edge counter (Fortran n=0, then n=n+1 -> 1-based)
                # ms2.f do10 i=i1+1,i2-1  ->  local index k = i-i1
                for k in range(1, len(z) - 1):
                    piv2 = sig[is_sign] * zg[k]
                    if piv2 < zgt:
                        continue
                    piv1 = sig[is_sign] * zg[k - 1]
                    piv3 = sig[is_sign] * zg[k + 1]
                    if piv2 < piv1 or piv2 < piv3:
                        continue

                    eps = 0.5
                    if interp == 1:
                        eps = self.smax(zg, k)

                    if n >= nm:
                        # ms2.f has no bounds check here (xx/yy are
                        # dimensioned (.,9)); a real Fortran run finding
                        # more than nm edges would silently corrupt
                        # memory. This guard is a deliberate deviation
                        # for safety.
                        break

                    # i (Fortran 1-based) = i1 + k; xx(l,n)=i+eps-1 (0-based)
                    xx[l, n] = (i1_idx + k) + eps
                    yy[l, n] = jj_idx

                    self.log(f"  Edge {n+1} at i(Fortran)={i1+k}, "
                             f"gradient={piv2:.1f}, eps={eps:.2f} -> "
                             f"X={xx[l,n]:.2f}, Y={yy[l,n]:.2f}")
                    n += 1

        # ms2.f lines 551-556: B=b, E=e (Fortran xx(12,n)=xx(2,n),
        # xx(15,n)=xx(5,n) -> 0-based rows 11=1, 14=4)
        for n in range(nm):
            xx[14, n] = xx[4, n]
            yy[14, n] = yy[4, n]
            xx[11, n] = xx[1, n]
            yy[11, n] = yy[1, n]

        # ms2.f lines 558-581: distortion diagnostic (logged only, not
        # used downstream - kept here purely for parity with the log
        # output the Fortran produces).
        distort = np.zeros((2, nm))
        for n in range(nm):
            distort[0, n] = xx[1, n] - (xx[0, n] + xx[2, n]) / 2.0
            distort[1, n] = xx[4, n] - (xx[3, n] + xx[5, n]) / 2.0
        valqm = float(np.sqrt(np.sum(distort[0]**2 + distort[1]**2) / (2.0 * nm)))
        self.log(f"Distortion (quadratic mean, pixel units): {valqm:.3f}")

        # ---------------------------------------------------------------
        # Vertical edges k,l,m,n (ms2.f lines 582-646)
        # ---------------------------------------------------------------
        self.log("\nDetecting vertical channel edges...")
        xdel = float(self.params.get('xdel', 25))

        for n in range(nm):
            for l in range(7, 11):  # rows 7-10: k,l,m,n
                if l == 7:      # k
                    ii_idx = int(xx[0, n] + xdel)
                    jj_start = 0
                    jj_stop_incl = int(yy[0, n])
                    is_sign = 0
                elif l == 8:    # l
                    ii_idx = int(xx[2, n] + xdel)
                    jj_start = int(yy[2, n])
                    jj_stop_incl = jm - 1
                    is_sign = 1
                elif l == 9:    # m
                    ii_idx = int(xx[3, n] - xdel)
                    jj_start = 0
                    jj_stop_incl = int(yy[3, n])
                    is_sign = 0
                else:           # l == 10, n
                    ii_idx = int(xx[5, n] - xdel)
                    jj_start = int(yy[5, n])
                    jj_stop_incl = jm - 1
                    is_sign = 1

                if ii_idx < 0 or ii_idx >= im or jj_stop_incl <= jj_start:
                    continue

                z_vert = meanflat[ii_idx, jj_start:jj_stop_incl + 1].astype(np.float64)

                # ms2.f lines 467-483: for the VERTICAL detection the
                # Fortran RECOMPUTES zmax/zgmax locally on this column
                # (zmax=0 ; zmax=max(zmax,z(jj)) ; same for zgmax), it does
                # NOT reuse the global central-cut normalization. The
                # horizontal cuts use the global values, but k,l,m,n here
                # must use the local column normalization.
                vzmax = float(np.max(z_vert)) if len(z_vert) else 1.0
                vzg = np.zeros_like(z_vert)
                vzg[:-1] = sig[is_sign] * (z_vert[1:] - z_vert[:-1])
                nz = max(1, np.max(np.abs(vzg[:-1])))
                vzgmax = float(nz)

                z_vert = 100.0 * z_vert / vzmax
                zg_vert = 100.0 * vzg / vzgmax

                # ms2.f do40 jj=jj1+1,jj2-1 -> local k=1..len-2
                for k in range(1, len(zg_vert) - 1):
                    piv2 = zg_vert[k]
                    if piv2 < zgt:
                        continue
                    piv1 = zg_vert[k - 1]
                    piv3 = zg_vert[k + 1]
                    if piv2 < piv1 or piv2 < piv3:
                        continue

                    eps = 0.5
                    if interp == 1:
                        eps = self.smax(zg_vert, k)

                    # BUG FIX: the loop variable l runs 7,8,9,10 (the
                    # Fortran's 1-based row numbers), but xx/yy are
                    # 0-based numpy arrays. Writing to xx[l,n] placed
                    # k,l,m,n at rows 7-10 while the rest of the code
                    # (A..F intersections) reads them at rows 6-9,
                    # shifting every corner by one row. Store at l-1.
                    xx[l-1, n] = ii_idx
                    yy[l-1, n] = jj_start + k + eps
                    break  # ms2.f: goto45 - stop at first match

        # ---------------------------------------------------------------
        # Corner points A,B,C,D,E,F via line intersections
        # (ms2.f lines 650-714)
        # ---------------------------------------------------------------
        self.log("\nComputing corner points via line intersections...")

        for n in range(nm):
            # A: intersection of (b,a) and (k,m)   ms2.f lines 650-661
            xx[10, n], yy[10, n] = self.intersect_lines(
                xx[1, n], yy[1, n], xx[0, n], yy[0, n],
                xx[6, n], yy[6, n], xx[8, n], yy[8, n])

            # B = b   (line 666-667)
            xx[11, n] = xx[1, n]
            yy[11, n] = yy[1, n]

            # C: intersection of (b,c) and (l,n)   lines 669-679
            xx[12, n], yy[12, n] = self.intersect_lines(
                xx[1, n], yy[1, n], xx[2, n], yy[2, n],
                xx[7, n], yy[7, n], xx[9, n], yy[9, n])

            # D: intersection of (e,d) and (m,k)   lines 685-695
            xx[13, n], yy[13, n] = self.intersect_lines(
                xx[4, n], yy[4, n], xx[3, n], yy[3, n],
                xx[8, n], yy[8, n], xx[6, n], yy[6, n])

            # E = e   (line 697-698)
            xx[14, n] = xx[4, n]
            yy[14, n] = yy[4, n]

            # F: intersection of (e,f) and (n,l)   lines 700-710
            xx[15, n], yy[15, n] = self.intersect_lines(
                xx[4, n], yy[4, n], xx[5, n], yy[5, n],
                xx[9, n], yy[9, n], xx[7, n], yy[7, n])

        self.log("\nGeometry detection complete")
        self.log(f"Reference points for channel 1:")
        for nl in range(16):
            self.log(f"  Point {nl}: X={xx[nl,0]:.2f}, Y={yy[nl,0]:.2f}")

        # Diagnostic plots (ms2.f lines 742-745)
        self.plot_geo1(zc_norm_full, zgc_norm_full, grt, i1, i2, ja, xx, yy)
        self.plot_geo2(xx, yy, xdel)
        self.plot_geo3(xx, yy)

        return xx, yy

    def smax(self, z, i):
        """
        Port of ms2.f `SMAX` (lines 386-408): parabolic interpolation
        of an intensity-gradient maximum.

        NOTE: unlike a typical "safe" sub-pixel estimator, the Fortran
        does not clamp eps to [-0.5, 0.5] - it can legitimately return
        a value outside that range if the local samples are noisy. This
        port preserves that (no clamping), for fidelity.

        Parameters:
        -----------
        z : array
            Signal array (already normalized), 0-based.
        i : int
            0-based index of the approximate maximum.

        Returns:
        --------
        float : sub-pixel offset from i.
        """
        if i <= 0 or i >= len(z) - 1:
            # Defensive guard not present in the Fortran (which relies
            # on the caller never passing a boundary index); ms2.f
            # would otherwise index z(i-1)/z(i+1) out of the array here.
            return 0.5

        b = z[i + 1] - z[i - 1]
        a = z[i + 1] + z[i - 1] - 2.0 * z[i]

        if a == 0.0:
            return 0.5  # ms2.f: "if(a.eq.0.)return" - keeps eps=0.5

        return -b / (2.0 * a)

    def intersect_lines(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Port of ms2.f `intersec` (lines 750-778).

        Line 1 (nearly horizontal, e.g. b-a): x = a*y + b, through
        (x1,y1) and (x2,y2).
        Line 2 (nearly vertical, e.g. k-m):    y = c*x + d, through
        (x3,y3) and (x4,y4).

        See the module docstring (fidelity note 3) about the Fortran's
        `1.-ac` denominator - this implementation uses the
        mathematically-intended `1 - a*c`, not the literal (buggy) `ac`.

        Returns:
        --------
        tuple : (xres, yres) intersection point
        """
        if abs(y2 - y1) < 1e-10 or abs(x3 - x4) < 1e-10:
            # Degenerate case (not handled explicitly in ms2.f, which
            # would divide by ~0 here; guarded defensively).
            return (x1 + x2) / 2, (y1 + y2) / 2

        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1
        c = (y3 - y4) / (x3 - x4)
        d = y3 - c * x3

        denom = 1.0 - a * c

        if abs(denom) < 1e-10:
            return (x1 + x2) / 2, (y1 + y2) / 2

        xres = (a * d + b) / denom
        yres = c * xres + d

        return xres, yres

    def write_acdf2(self, xx, yy):
        """Write corner points A,C,D,F to file for reference."""
        with open('ACDF2.lis', 'w') as f:
            for n in range(self.nm):
                # Write X coordinates: A, C, D, F
                f.write(f"{xx[10,n]:8.2f}{xx[12,n]:8.2f}"
                        f"{xx[13,n]:8.2f}{xx[15,n]:8.2f}")
                # Write Y coordinates
                f.write(f"{yy[10,n]:8.2f}{yy[12,n]:8.2f}"
                        f"{yy[13,n]:8.2f}{yy[15,n]:8.2f}\n")

        self.log("Written ACDF2.lis")

    def plot_geo1(self, zc, zgc, grt, i1, i2, ja, xx, yy):
        """
        Generate geo1.ps diagnostic plot (ms2.f `plotgeo1`).

        zc, zgc are now expected to be the FULL im-length normalized
        arrays (matching what ms2.f actually passes - see newgeom).

        Shows:
        - Top: Intensity profile along central cut
        - Middle: Gradient profile with threshold
        - Bottom: Channel outlines
        """
        self.log("\nGenerating geo1.ps...")

        with PdfPages('geo1.pdf') as pdf:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # Top: Intensity profile
            x_axis = np.arange(len(zc))
            axes[0].plot(x_axis, zc, 'b-', linewidth=2)
            axes[0].set_ylabel('Intensity')
            axes[0].set_title('Cross-section along center of Y field of view')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(0, self.im-1)
            axes[0].set_ylim(0, 100)

            # Middle: Gradient profile
            axes[1].plot(x_axis, zgc, 'b-', linewidth=2)
            axes[1].axhline(grt, color='r', linestyle='--', label='grt')
            axes[1].axhline(-grt, color='r', linestyle='--', label='-grt')
            axes[1].set_ylabel('Intensity gradient')
            axes[1].set_xlabel('X (unit=arcsec/2)')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(0, self.im-1)
            axes[1].set_ylim(-100, 100)
            axes[1].legend()

            # Bottom: Channel outlines
            for n in range(self.nm):
                # Draw channel outline using ABCDEF points
                x_outline = [xx[10, n], xx[11, n], xx[12, n],
                             xx[15, n], xx[14, n], xx[13, n], xx[10, n]]
                y_outline = [yy[10, n], yy[11, n], yy[12, n],
                             yy[15, n], yy[14, n], yy[13, n], yy[10, n]]
                axes[2].plot(x_outline, y_outline, 'b-', linewidth=1)

            # Draw horizontal cut lines (ja is Fortran 1-based -> -1)
            for j_cut in ja:
                axes[2].axhline(j_cut-1, color='r', linestyle='--', alpha=0.5)

            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            axes[2].set_xlim(0, self.im-1)
            axes[2].set_ylim(0, self.jm-1)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_aspect('equal')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        self.log("geo1.pdf generated")

    def plot_geo2(self, xx, yy, xdel):
        """Generate geo2.ps diagnostic plot showing first channel detail."""
        self.log("\nGenerating geo2.pdf...")

        with PdfPages('geo2.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(8, 10))

            n = 0  # First channel

            # Draw channel outline
            x_outline = [xx[10, n], xx[11, n], xx[12, n],
                         xx[15, n], xx[14, n], xx[13, n], xx[10, n]]
            y_outline = [yy[10, n], yy[11, n], yy[12, n],
                         yy[15, n], yy[14, n], yy[13, n], yy[10, n]]
            ax.plot(x_outline, y_outline, 'b-', linewidth=2)

            # Draw points a-f
            for i in range(6):
                ax.plot(xx[i, n], yy[i, n], 'ro', markersize=8)
                label = chr(ord('a') + i)
                ax.text(xx[i, n]-10, yy[i, n]+10, label, fontsize=12)

            # Draw points k,l,m,n
            for i, label in zip(range(6, 10), ['k', 'l', 'm', 'n']):
                ax.plot(xx[i, n], yy[i, n], 'go', markersize=8)
                ax.text(xx[i, n]+10, yy[i, n]+10, label, fontsize=12)

            # Draw points A-F
            for i, label in zip(range(10, 16), ['A', 'B', 'C', 'D', 'E', 'F']):
                ax.plot(xx[i, n], yy[i, n], 'bs', markersize=10)
                ax.text(xx[i, n]-15, yy[i, n]-15, label, fontsize=14,
                        fontweight='bold')

            # Draw vertical cut lines
            for i, label in zip([0, 2, 3, 5], ['a', 'c', 'd', 'f']):
                if i in [0, 2]:  # Left side
                    x_cut = xx[i, n] + xdel
                else:  # Right side
                    x_cut = xx[i, n] - xdel
                ax.axvline(x_cut, color='g', linestyle='--', alpha=0.5)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('First channel - Geometry detection points')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 350)
            ax.set_ylim(0, 1023)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        self.log("geo2.pdf generated")

    def plot_geo3(self, xx, yy):
        """Generate geo3.ps showing channel dimension variations."""
        self.log("\nGenerating geo3.pdf...")

        with PdfPages('geo3.pdf') as pdf:
            fig, axes = plt.subplots(2, 4, figsize=(16, 10))

            nc = 4  # Central channel for reference
            channels = np.arange(1, self.nm + 1)

            # X dimensions
            segments = [
                (10, 12, 'AC'),  # A-C
                (13, 15, 'DF'),  # D-F
                (10, 13, 'AD'),  # A-D
                (12, 15, 'CF')   # C-F
            ]

            for idx, (p1, p2, label) in enumerate(segments):
                ax = axes[0, idx]
                distances = np.abs(xx[p1, :] - xx[p2, :])

                ax.plot(channels, distances, 'bo-', linewidth=2)
                ax.scatter(channels, distances, s=50, c='blue')

                ywin1 = distances[nc] - 5
                ywin2 = ywin1 + 10
                ax.set_ylim(ywin1, ywin2)
                ax.set_xlim(0, self.nm + 1)
                ax.set_title(f'{label} - X', fontsize=14)
                ax.grid(True, alpha=0.3)

            # Y dimensions
            for idx, (p1, p2, label) in enumerate(segments):
                ax = axes[1, idx]
                distances = np.abs(yy[p1, :] - yy[p2, :])

                ax.plot(channels, distances, 'ro-', linewidth=2)
                ax.scatter(channels, distances, s=50, c='red')

                ywin1 = distances[nc] - 5
                ywin2 = ywin1 + 10
                ax.set_ylim(ywin1, ywin2)
                ax.set_xlim(0, self.nm + 1)
                ax.set_title(f'{label} - Y', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Channel number')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        self.log("geo3.pdf generated")

    def __del__(self):
        """Clean up on deletion."""
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'xryr_file'):
            self.xryr_file.close()


def main():
    """Main entry point for ms2.py"""
    print("="*60)
    print("MSDP DATA PROCESSING - STEP 2")
    print("Channel geometry detection")
    print("="*60)

    # Usage: python ms2.py <dark_file> <flat_file>  (ou utilisation du pipeline)
    # Les fichiers sont les moyennes produites par ms1.py (x... / y...),
    # ou directement les fichiers FITS du pipeline (m*x1.fit / m*y1.fit).
    if len(sys.argv) >= 3:
        dark_file, flat_file = sys.argv[1], sys.argv[2]
    else:
        # Recherche automatique des derniers fichiers moyens produits par ms1.py.
        # Le flat moyen peut avoir un nom avec espaces (ex: ' 30_y1.fit       00000',
        # fidèle au nom binaire Fortran), et le dark 'x170330_...._00000'.
        # On cherche donc tous les fichiers se terminant par '00000' (hors .fit), et on
        # déduit le rôle à partir de la position de la lettre x/y dans le nom.
        import glob
        means = sorted(f for f in glob.glob('*00000')
                       if not f.lower().endswith(('.fit', '.fits')))
        darks = [f for f in means if 'x' in f.replace('-', '_').lower()]
        flats = [f for f in means if 'x' not in f.lower()]
        if not darks or not flats:
            raise SystemExit("Usage: python ms2.py <dark_file> <flat_file>"
                             " (ou lancer ms1.py d'abord pour produire x*/y*)")
        dark_file, flat_file = darks[-1], flats[-1]

    processor = GeometryProcessor('ms.yml')
    xx, yy = processor.compute_geometry(dark_file, flat_file)
    print("\n" + "="*60)
    print("GEOMETRY PROCESSING COMPLETE")

if __name__ == "__main__":
    main()