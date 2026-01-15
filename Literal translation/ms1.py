"""
ms1.py - MSDP Data Processing: Step 1 - Averaging Module

This module handles the first step of MSDP (Multichannel Subtractive Double Pass)
data processing: averaging of dark current and flat field sequences.

Processing steps:
1. Read parameter file (ms.yml)
2. List and load dark current files (x files)
3. List and load flat field files (y files)
4. Compute averages for each sequence
5. Write averaged files for use in geometry computation

File naming convention:
- x files: dark current sequences
- y files: flat field sequences
- z files: field-stop geometry (if applicable)
- b files: target observations for scientific analysis
"""

import numpy as np
import yaml
import glob
import sys
from astropy.io import fits
from pathlib import Path


class MSDPProcessor:
    """Main class for MSDP data processing."""

    def __init__(self, param_file='ms.yml'):
        """
        Initialize MSDP processor with parameters from YAML file.
        
        Parameters:
        -----------
        param_file : str
            Path to parameter file (default: 'ms.yml')
        """
        self.params = self.read_parameters(param_file)
        self.log_file = open('ms.lis', 'w')
        self.channel_file = open('channel.lis', 'w')

        # Initialize dimensions
        self.is_ccd = self.params['is']  # CCD X-dimension (1536)
        self.js_ccd = self.params['js']  # CCD Y-dimension (1024)

        # After permutation (swapping X and Y)
        self.isp = self.js_ccd  # 1024
        self.jsp = self.is_ccd  # 1536

        self.log(f"MSDP Processor initialized")
        self.log(f"CCD dimensions: {self.is_ccd} x {self.js_ccd}")
        self.log(f"After permutation: {self.isp} x {self.jsp}")

    def read_parameters(self, param_file):
        """Read processing parameters from YAML file."""
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        return params

    def log(self, message):
        """Write message to log file and print to console."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def list_files(self, pattern):
        """
        List files matching the pattern.
        
        Parameters:
        -----------
        pattern : str
            File pattern (e.g., 'm*x1.fit')
        
        Returns:
        --------
        list : Sorted list of matching files
        """
        files = sorted(glob.glob(pattern))
        return files

    def read_fits_file(self, filename, swap_bytes=True):
        """
        Read FITS file and optionally swap bytes for LINUX compatibility.
        
        Parameters:
        -----------
        filename : str
            Path to FITS file
        swap_bytes : bool
            Whether to perform byte swapping (default: True)
        
        Returns:
        --------
        tuple : (header, data array)
        """
        with fits.open(filename) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            if data is None:
                raise ValueError(f"No data found in {filename}")

            # Ensure data is int16 type
            if data.dtype != np.int16:
                data = data.astype(np.int16)

        return header, data

    def permute_data(self, data):
        """
        Permute CCD coordinates: swap X and Y axes and flip.
        
        This converts from CCD coordinates (1536 x 1024) to 
        processing coordinates (1024 x 1536).
        
        Parameters:
        -----------
        data : ndarray
            Input data array (is x js)
        
        Returns:
        --------
        ndarray : Permuted data array (isp x jsp)
        """
        # Original Fortran logic:
        # do i=1,is (1536)
        #   jp = i
        #   do j=1,js (1024)
        #     ip = js+1-j
        #     tabpermu(ip,jp) = tab2(i,j)

        permuted = np.zeros((self.isp, self.jsp), dtype=np.int32)

        for i in range(self.is_ccd):
            jp = i
            for j in range(self.js_ccd):
                ip = self.js_ccd - 1 - j
                permuted[ip, jp] = data[i, j]

        return permuted

    def process_sequence(self, file_type, files, nfa, nfb):
        """
        Process a sequence of files (dark current or flat field).
        
        Parameters:
        -----------
        file_type : str
            Type of file ('dark' or 'flat')
        files : list
            List of file paths
        nfa : int
            First file index to use (1-based)
        nfb : int
            Last file index to use (1-based)
        
        Returns:
        --------
        tuple : (averaged_data, output_filename)
        """
        self.log(f"\n{'='*60}")
        self.log(f"Processing {file_type.upper()} sequence")
        self.log(f"Files {nfa} to {nfb}")

        # Initialize accumulator for averaging
        accumulator = np.zeros((self.isp, self.jsp), dtype=np.int64)

        # Process files in range
        count = 0
        for idx in range(nfa-1, nfb):  # Convert to 0-based indexing
            if idx >= len(files):
                break

            filename = files[idx]
            self.log(f"\nReading file {idx+1}: {filename}")

            # Read FITS file
            header, data = self.read_fits_file(filename,
                                               swap_bytes=self.params['iswap'] == 1)

            # Permute data (swap and flip axes)
            data_permuted = self.permute_data(data)

            # Add to accumulator
            accumulator += data_permuted
            count += 1

            # Log corner values for verification
            if idx == nfa-1:  # First file
                self.log(f"Data shape: {data.shape}")
                self.log(f"Permuted shape: {data_permuted.shape}")
                self.log(f"Corner values (before permutation): "
                         f"{data[0,0]}, {data[-1,0]}, {data[-1,-1]}, {data[0,-1]}")
                self.log(f"Corner values (after permutation): "
                         f"{data_permuted[-1,0]}, {data_permuted[-1,-1]}, "
                         f"{data_permuted[0,-1]}, {data_permuted[0,0]}")

        # Compute average
        if count == 0:
            raise ValueError(f"No files processed for {file_type}")

        averaged = (accumulator / count + 0.5).astype(np.int16)

        self.log(f"\nAveraged {count} files")
        self.log(f"Average extremes: {averaged[0,0]}, {averaged[-1,0]}, "
                 f"{averaged[-1,-1]}, {averaged[0,-1]}")

        # Generate output filename based on last processed file
        last_file = files[nfb-1]
        if file_type == 'dark':
            output_name = self._generate_output_name(last_file, 'x')
        else:  # flat
            output_name = self._generate_output_name(last_file, 'y')

        return averaged, output_name, count

    def _generate_output_name(self, input_file, file_type):
        """
        Generate output filename from input filename.
        
        Format: x170330_00000000_00000 or y170330_00000000_00000
        """
        # Extract date/time portion from input filename
        # Expected format: m*x1.fit or m*y1.fit
        basename = Path(input_file).stem

        # Simple approach: create name from file_type and extract middle portion
        # This mimics the Fortran string manipulation
        if len(basename) >= 15:
            middle_part = basename[-15:]  # Last 15 characters
        else:
            middle_part = "000000_00000000_00000"

        output_name = f"{file_type}{middle_part}"

        return output_name

    def write_averaged_file(self, data, filename):
        """
        Write averaged data to binary file.
        
        Format matches Fortran unformatted output:
        - 512-element header (int32)
        - Data rows (int16)
        
        Parameters:
        -----------
        data : ndarray
            Data to write (isp x jsp)
        filename : str
            Output filename
        """
        self.log(f"\nWriting averaged file: {filename}")

        # Create header (512 int32 values)
        header = np.zeros(512, dtype=np.int32)
        header[0] = 3  # Format identifier
        header[1] = self.isp  # 1024
        header[2] = self.jsp  # 1536
        header[3] = 1

        # Write to binary file
        with open(filename, 'wb') as f:
            # Write header
            header.tofile(f)

            # Write data row by row
            for j in range(self.jsp):
                data[:, j].tofile(f)

        self.log(f"File written: {filename}")
        self.log(f"Dimensions: {self.isp} x {self.jsp}")

        # Log sample values for verification
        i1t = int(self.isp * 0.1)
        i2t = int(self.isp * 0.9)
        self.log(f"\nSample values at row {i1t}: "
                 f"{data[i1t, 75:126:5].tolist()}")
        self.log(f"Sample values at row {i2t}: "
                 f"{data[i2t, 75:126:5].tolist()}")

    def run(self):
        """
        Main processing routine.
        
        Steps:
        1. Process dark current files (x files)
        2. Process flat field files (y files)
        3. Write averaged files
        4. Prepare for geometry computation (ms2.py)
        """
        self.log("\n" + "="*60)
        self.log("MSDP DATA PROCESSING - STEP 1: AVERAGING")
        self.log("="*60)

        # Get file ranges from parameters
        nfx1 = self.params['nfx1']
        nfx2 = self.params['nfx2']
        nfy1 = self.params['nfy1']
        nfy2 = self.params['nfy2']

        # List files
        self.log("\nListing files...")

        # Dark current files
        x_files = self.list_files('m*x1.fit')
        self.log(f"Found {len(x_files)} dark current files (x)")
        if x_files:
            with open('xtab.lis', 'w') as f:
                for xf in x_files:
                    f.write(xf + '\n')

        # Flat field files
        y_files = self.list_files('m*y1.fit')
        self.log(f"Found {len(y_files)} flat field files (y)")
        if y_files:
            with open('ytab.lis', 'w') as f:
                for yf in y_files:
                    f.write(yf + '\n')

        # Process dark current sequence
        if x_files and nfx1 > 0 and nfx2 > 0:
            dark_avg, dark_name, dark_count = self.process_sequence(
                'dark', x_files, nfx1, nfx2
            )
            self.write_averaged_file(dark_avg, dark_name)
            self.dark_data = dark_avg
            self.dark_filename = dark_name
        else:
            self.log("\nSkipping dark current processing (no files or disabled)")
            self.dark_data = None
            self.dark_filename = None

        # Process flat field sequence
        if y_files and nfy1 > 0 and nfy2 > 0:
            flat_avg, flat_name, flat_count = self.process_sequence(
                'flat', y_files, nfy1, nfy2
            )
            self.write_averaged_file(flat_avg, flat_name)
            self.flat_data = flat_avg
            self.flat_filename = flat_name
        else:
            self.log("\nSkipping flat field processing (no files or disabled)")
            self.flat_data = None
            self.flat_filename = None

        self.log("\n" + "="*60)
        self.log("AVERAGING COMPLETE")
        self.log("="*60)

        # Return filenames for next processing step
        return {
            'dark': self.dark_filename,
            'flat': self.flat_filename,
            'dark_data': self.dark_data,
            'flat_data': self.flat_data
        }

    def __del__(self):
        """Close log files on cleanup."""
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'channel_file'):
            self.channel_file.close()


def main():
    """Main entry point for ms1.py"""
    print("="*60)
    print("MSDP DATA PROCESSING - STEP 1")
    print("Averaging of dark current and flat field sequences")
    print("="*60)

    # Create processor and run
    processor = MSDPProcessor('Literal translation/ms.yml')
    results = processor.run()

    print("\n" + "="*60)
    print("Processing complete. Results:")
    print(f"  Dark current file: {results['dark']}")
    print(f"  Flat field file: {results['flat']}")
    print("\nReady for Step 2 (Geometry): run ms2.py")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
