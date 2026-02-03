"""
PSD Configuration Template
==========================

Copy this file and customize for your specific analysis needs.
All parameters are documented with recommended values.
"""

# ============================================================================
# INPUT DATA
# ============================================================================

# Path to your segmented 3D volume
# Expected format: Numpy array (.npy) or TIFF stack
# Data type: Boolean (True = pore space, False = solid phase)
INPUT_VOLUME_PATH = "/content/drive/MyDrive/data/scan_001_segmented.npy"

# Physical voxel dimensions in micrometers (Z, Y, X)
# Common μCT: Isotropic (1.0, 1.0, 1.0) or anisotropic (2.0, 1.0, 1.0)
VOXEL_SPACING_UM = (2.0, 1.0, 1.0)


# ============================================================================
# COMPUTATIONAL PARAMETERS
# ============================================================================

# GPU Acceleration
# Set to True for CuPy acceleration (recommended in Colab with GPU runtime)
USE_GPU = True

# Block Processing (for large volumes that don't fit in memory)
# Set to True for volumes > 512³ voxels or if GPU memory is limited
USE_CHUNKING = True

# Chunk size for block processing (before halo padding)
# Smaller chunks = less memory, more overhead
# Recommended: (128, 128, 128) for 16GB GPU, (64, 64, 64) for 8GB GPU
CHUNK_SIZE = (128, 128, 128)

# Halo width for overlap padding (critical for boundary-correct EDT)
# Must be >= maximum expected pore diameter in voxels
# Recommended: 2 × max_expected_pore_diameter
# Example: If max pore is 40 voxels, use halo_width = 80
HALO_WIDTH = 50


# ============================================================================
# CHECKPOINTING (Colab Timeout Resilience)
# ============================================================================

# Enable checkpointing to Google Drive
# Allows resuming interrupted computations
ENABLE_CHECKPOINTING = True

# Checkpoint directory (must be on Google Drive for persistence)
CHECKPOINT_DIR = "/content/drive/MyDrive/psd_checkpoints"

# Unique run identifier (auto-generated if None)
RUN_ID = "scan_001_psd"

# Resume from existing checkpoint if available
RESUME_FROM_CHECKPOINT = True


# ============================================================================
# PSD PARAMETERS
# ============================================================================

# Custom diameter bin edges in micrometers (None = auto-generate)
# Example: np.logspace(0, 2, 50) for 50 logarithmic bins from 1-100 μm
# If None, automatically generates 50 logarithmic bins covering data range
BIN_EDGES_UM = None

# Border exclusion (Vogel et al. constraint)
# If True, pores touching volume edges are excluded from PSD
# Recommended: True (ensures accurate sizing)
EXCLUDE_BORDER_PORES = True


# ============================================================================
# OUTPUT
# ============================================================================

# Output directory for results
OUTPUT_DIR = "/content/drive/MyDrive/results"

# Output filename (without extension)
OUTPUT_FILENAME = "scan_001_psd"

# Export formats (list of: 'csv', 'json', 'hdf5', 'excel')
# CSV: Human-readable, portable
# JSON: Includes metadata
# HDF5: Compressed, best for large datasets
# Excel: Spreadsheet format (requires openpyxl)
EXPORT_FORMATS = ['csv', 'json']

# Metadata to include in output
METADATA = {
    'scan_id': 'scan_001',
    'sample_name': 'soil_core_A1',
    'sample_type': 'agricultural_soil',
    'depth_cm': '0-5',
    'acquisition_date': '2026-02-03',
    'resolution_um': 1.0,
    'scanner': 'SkyScan 1272',
    'notes': 'Topsoil sample from field site A'
}


# ============================================================================
# VISUALIZATION
# ============================================================================

# Generate PSD plot
GENERATE_PLOT = True

# Show unreliable region (d < 5 voxels) in plot
# If False, unreliable region is greyed out
SHOW_UNRELIABLE_IN_PLOT = False

# Use logarithmic scale for diameter axis
LOG_SCALE_PLOT = True


# ============================================================================
# VALIDATION
# ============================================================================

# Run synthetic validation tests before processing real data
RUN_VALIDATION_FIRST = False

# Minimum porosity check (warning if below threshold)
# Typical soil: 0.3-0.6, highly compacted: < 0.2
MIN_EXPECTED_POROSITY = 0.1


# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Connectivity for component labeling (26 = face+edge+corner, 6 = face only)
# Vogel et al. uses 26-connectivity
CONNECTIVITY = 26

# Morphological reconstruction method
# 'hildebrand': Full reconstruction (accurate, slow)
# 'approximate': Maximum filter approximation (fast, less accurate)
OPENING_METHOD = 'hildebrand'

# Checkpoint compression (reduces Drive storage but slower)
CHECKPOINT_COMPRESSION = True
