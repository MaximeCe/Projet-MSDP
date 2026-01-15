"""
ms2.py - MSDP Data Processing: Step 2 - Geometry Module

This module computes the geometry of MSDP spectrograph channels:
1. Detects channel edges using intensity gradients
2. Computes reference points (a,b,c,d,e,f and k,l,m,n)
3. Extrapolates corner points (A,B,C,D,E,F) via line intersections
4. Generates diagnostic plots (geo1.ps, geo2.ps, geo3.ps)

The geometry defines how to map CCD pixels to solar coordinates (X,Y) 
within each spectral channel, accounting for optical distortions.
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

        # Image dimensions
        self.im = 1536
        self.jm = 1024
        self.nm = self.params['nm']  # Number of channels (9)

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
                
                Point indices:
                1-6: a,b,c,d,e,f (horizontal edges)
                7-10: k,l,m,n (vertical edges)
                11-16: A,B,C,D,E,F (extrapolated corners)
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

        # Subtract dark from flat
        self.log("\nSubtracting dark current from flat field...")
        meanflat = flat_data - dark_data
        meanflat[meanflat < 0] = 1  # Ensure no negative values

        # Permute to get correct orientation for geometry detection
        # meanflat is already (1536, 1024), just use it directly
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
        New geometry detection algorithm.
        
        Detects channel edges using intensity gradient maxima.
        
        Parameters:
        -----------
        meanflat : ndarray
            Dark-subtracted flat field image (im x jm)
        
        Returns:
        --------
        tuple : (xx, yy) reference point arrays
        """
        self.log("\nNEW GEOMETRY DETECTION")

        # Parameters
        i1 = 5
        i2 = self.im - 4
        j1 = 1
        j2 = self.jm

        # Three horizontal cuts for edge detection
        ja = np.array([151, 501, 851])  # 1-based: [1+150, 1+500, 1+850]
        jc = ja[1]  # Central cut

        self.log(f"Detection cuts at j = {ja}")

        # Get gradient threshold
        mingrad = self.params['mingrad']
        interp = self.params['interp']

        self.log(f"Minimum gradient threshold: {mingrad}")
        self.log(f"Parabolic interpolation: {interp}")

        # Initialize arrays for reference points
        # nl: point index (1-16), nc: channel index (1-9)
        xx = np.zeros((20, self.nm))  # Using 20 to match Fortran indexing
        yy = np.zeros((20, self.nm))

        # Compute normalization for central cut
        zc = meanflat[i1:i2+1, jc].astype(float)
        zmax = np.max(zc)

        zgc = np.diff(zc)
        zgmax = np.max(np.abs(zgc))

        self.log(
            f"Central cut normalization: zmax={zmax:.1f}, zgmax={zgmax:.1f}")

        # Normalize for detection
        zc = 100.0 * zc / zmax
        zgc_norm = np.zeros(len(zc))
        zgc_norm[:-1] = 100.0 * zgc / zgmax
        zgc_norm[-1] = zgc_norm[-2]

        grt = mingrad  # Gradient threshold
        sig = np.array([1.0, -1.0])  # Positive and negative gradients

        # Detect edges at three j positions (a,b,c and d,e,f)
        self.log("\nDetecting horizontal channel edges...")

        for nj in range(3):  # Three cuts
            jj = ja[nj]
            self.log(f"\nCut {nj+1} at j = {jj}")

            # Extract and normalize intensity profile
            z = meanflat[i1:i2+1, jj].astype(float)

            # Option: average over 3 adjacent rows
            if jj > 0 and jj < self.jm - 1:
                z = (meanflat[i1:i2+1, jj-1] +
                     meanflat[i1:i2+1, jj] +
                     meanflat[i1:i2+1, jj+1]) / 3.0

            # Compute gradient
            zg = np.diff(z)
            zg_extended = np.zeros(len(z))
            zg_extended[:-1] = zg
            zg_extended[-1] = zg[-1]

            # Normalize
            z = 100.0 * z / zmax
            zg_extended = 100.0 * zg_extended / zgmax

            # Detect edges for both gradient signs
            for is_sign in range(2):  # 0: positive, 1: negative
                if is_sign == 0:
                    l = nj  # Index for xx, yy
                else:
                    l = nj + 3

                n = 0  # Edge counter

                # Scan for gradient maxima
                for i in range(1, len(zg_extended) - 1):
                    piv2 = sig[is_sign] * zg_extended[i]

                    # Check if above threshold
                    if piv2 < grt:
                        continue

                    # Check if local maximum
                    piv1 = sig[is_sign] * zg_extended[i-1]
                    piv3 = sig[is_sign] * zg_extended[i+1]

                    if piv2 < piv1 or piv2 < piv3:
                        continue

                    # Found edge
                    eps = 0.5  # Default sub-pixel position

                    # Parabolic interpolation for sub-pixel accuracy
                    if interp == 1:
                        eps = self.parabolic_max(zg_extended, i)

                    # Store edge position (0-based coordinates)
                    xx[l, n] = i + i1 + eps - 1
                    yy[l, n] = jj - 1

                    self.log(f"  Edge {n+1} at i={i+i1}, gradient={piv2:.1f}, "
                             f"eps={eps:.2f} -> X={xx[l,n]:.2f}, Y={yy[l,n]:.2f}")

                    n += 1

                    if n >= self.nm:
                        break

        # Copy some points for convenience
        for n in range(self.nm):
            xx[14, n] = xx[4, n]  # E = e
            yy[14, n] = yy[4, n]
            xx[11, n] = xx[1, n]  # B = b
            yy[11, n] = yy[1, n]

        # Detect vertical edges (k,l,m,n points)
        self.log("\nDetecting vertical channel edges...")
        xdel = 25.0  # Horizontal offset for vertical cuts

        for n in range(self.nm):
            for l in range(7, 11):  # k,l,m,n
                # Determine position and range for vertical cut
                if l == 7:  # k
                    ii = int(xx[0, n] + 1 + xdel)
                    jj1 = 1
                    jj2 = int(yy[0, n] + 1)
                    is_sign = 0
                elif l == 8:  # l
                    ii = int(xx[2, n] + 1 + xdel)
                    jj1 = int(yy[2, n] + 1)
                    jj2 = self.jm
                    is_sign = 1
                elif l == 9:  # m
                    ii = int(xx[3, n] + 1 - xdel)
                    jj1 = 1
                    jj2 = int(yy[3, n] + 1)
                    is_sign = 0
                else:  # l == 10, n
                    ii = int(xx[5, n] + 1 - xdel)
                    jj1 = int(yy[5, n] + 1)
                    jj2 = self.jm
                    is_sign = 1

                # Extract vertical profile
                if ii < 0 or ii >= self.im:
                    continue

                z_vert = meanflat[ii, jj1:jj2].astype(float)

                # Compute gradient (with sign)
                zg_vert = sig[is_sign] * np.diff(z_vert)
                zg_vert_extended = np.zeros(len(z_vert))
                zg_vert_extended[:-1] = zg_vert
                zg_vert_extended[-1] = zg_vert[-1]

                # Normalize
                z_vert = 100.0 * z_vert / zmax
                zg_vert_extended = 100.0 * zg_vert_extended / zgmax

                # Find maximum gradient
                for jj in range(1, len(zg_vert_extended) - 1):
                    piv2 = zg_vert_extended[jj]

                    if piv2 < grt:
                        continue

                    piv1 = zg_vert_extended[jj-1]
                    piv3 = zg_vert_extended[jj+1]

                    if piv2 < piv1 or piv2 < piv3:
                        continue

                    # Found edge
                    eps = 0.5
                    if interp == 1:
                        eps = self.parabolic_max(zg_vert_extended, jj)

                    xx[l, n] = ii - 1
                    yy[l, n] = jj + jj1 - 1 + eps

                    break

        # Compute corner points A,B,C,D,E,F via line intersections
        self.log("\nComputing corner points via line intersections...")

        for n in range(self.nm):
            # Point A: intersection of lines (b,a) and (k,m)
            x1, y1 = xx[1, n], yy[1, n]  # b
            x2, y2 = xx[0, n], yy[0, n]  # a
            x3, y3 = xx[6, n], yy[6, n]  # k
            x4, y4 = xx[8, n], yy[8, n]  # m
            xx[10, n], yy[10, n] = self.intersect_lines(x1, y1, x2, y2,
                                                        x3, y3, x4, y4)

            # Point B = b
            xx[11, n] = xx[1, n]
            yy[11, n] = yy[1, n]

            # Point C: intersection of lines (b,c) and (l,n)
            x1, y1 = xx[1, n], yy[1, n]  # b
            x2, y2 = xx[2, n], yy[2, n]  # c
            x3, y3 = xx[7, n], yy[7, n]  # l
            x4, y4 = xx[9, n], yy[9, n]  # n
            xx[12, n], yy[12, n] = self.intersect_lines(x1, y1, x2, y2,
                                                        x3, y3, x4, y4)

            # Point D: intersection of lines (e,d) and (k,m)
            x1, y1 = xx[4, n], yy[4, n]  # e
            x2, y2 = xx[3, n], yy[3, n]  # d
            x3, y3 = xx[8, n], yy[8, n]  # m
            x4, y4 = xx[6, n], yy[6, n]  # k
            xx[13, n], yy[13, n] = self.intersect_lines(x1, y1, x2, y2,
                                                        x3, y3, x4, y4)

            # Point E = e
            xx[14, n] = xx[4, n]
            yy[14, n] = yy[4, n]

            # Point F: intersection of lines (e,f) and (l,n)
            x1, y1 = xx[4, n], yy[4, n]  # e
            x2, y2 = xx[5, n], yy[5, n]  # f
            x3, y3 = xx[9, n], yy[9, n]  # n
            x4, y4 = xx[7, n], yy[7, n]  # l
            xx[15, n], yy[15, n] = self.intersect_lines(x1, y1, x2, y2,
                                                        x3, y3, x4, y4)

        # Log results
        self.log("\nGeometry detection complete")
        self.log(f"Reference points for channel 1:")
        for nl in range(16):
            self.log(f"  Point {nl}: X={xx[nl,0]:.2f}, Y={yy[nl,0]:.2f}")

        # Generate plots
        self.plot_geo1(zc, zgc_norm, grt, i1, i2, ja, xx, yy)
        self.plot_geo2(xx, yy, xdel)
        self.plot_geo3(xx, yy)

        return xx, yy

    def parabolic_max(self, z, i):
        """
        Compute sub-pixel position of maximum using parabolic interpolation.
        
        Fits a parabola through points (i-1, i, i+1) and finds maximum.
        
        Parameters:
        -----------
        z : array
            Signal array
        i : int
            Index of approximate maximum
        
        Returns:
        --------
        float : Sub-pixel offset from i (range: -0.5 to +0.5)
        """
        if i <= 0 or i >= len(z) - 1:
            return 0.5

        # Parabolic fit: z = a*x^2 + b*x + c
        # Maximum at x = -b/(2*a)

        b = z[i+1] - z[i-1]
        a = z[i+1] + z[i-1] - 2*z[i]

        if abs(a) < 1e-10:
            return 0.5

        eps = -b / (2.0 * a)

        # Limit to reasonable range
        eps = max(-0.5, min(0.5, eps))

        return eps

    def intersect_lines(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Compute intersection of two lines.
        
        Line 1: passes through (x1,y1) and (x2,y2)
        Line 2: passes through (x3,y3) and (x4,y4)
        
        Returns:
        --------
        tuple : (xres, yres) intersection point
        """
        # Line 1: x = a*y + b (nearly horizontal)
        # Line 2: y = c*x + d (nearly vertical)

        if abs(y2 - y1) < 1e-10 or abs(x3 - x4) < 1e-10:
            # Degenerate case
            return (x1 + x2) / 2, (y1 + y2) / 2

        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1
        c = (y3 - y4) / (x3 - x4)
        d = y3 - c * x3

        denom = 1.0 - a * c

        if abs(denom) < 1e-10:
            # Lines nearly parallel
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
        Generate geo1.ps diagnostic plot.
        
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

            # Draw horizontal cut lines
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

    # Create processor and run
    processor = GeometryProcessor('Literal translation/ms.yml')
    results = processor.compute_geometry('Literal translation/m010_b0101_ms_20170330_09564585_x1.fit',
                                         "Literal translation/m011_b0101_ms_20170330_10013140_y1.fit")
    print("\n" + "="*60)
    print("GEOMETRY PROCESSING COMPLETE")

if __name__ == "__main__":
    main()