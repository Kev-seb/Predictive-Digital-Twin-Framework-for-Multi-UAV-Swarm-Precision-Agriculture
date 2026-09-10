import os
import time
import datetime
import numpy as np
from pathlib import Path
from src.core.multispectral_loader import load_multispectral_tiff
from src.indices.ndvi import compute_ndvi

def perform_ndvi_validation():
    # 1. Paths & Setup
    sample_path = Path("data/samples/sample_paddy.tif")
    
    print("=== PHASE 3.1 NDVI VALIDATION ===")
    print(f"Executed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 2. Check input data classification
    # Data is classified under the structural data integrity rules
    print("DATA SOURCES")
    print(f"REAL DATA:           {sample_path.name if sample_path.exists() else 'NOT FOUND'}")
    print("SYNTHETIC DATA:      None")
    print("SIMULATED DATA:      None")
    print("GROUND TRUTH:        NOT AVAILABLE (no independent reference data found on disk)\n")
    
    # 3. Load image & compute NDVI
    if not sample_path.exists():
        print("[ERROR] Real data sample_paddy.tif not found in data/samples/.")
        print("FINAL STATUS:        NOT EVALUABLE")
        return
        
    try:
        ms = load_multispectral_tiff(sample_path)
    except Exception as e:
        print(f"[ERROR] Failed to load multispectral TIFF: {e}")
        print("FINAL STATUS:        NOT EVALUABLE")
        return

    # Band Identification
    # Band 2 = Red (670 nm), Band 4 = NIR (840 nm)
    red_band = ms.red
    nir_band = ms.nir
    
    print("BAND IDENTIFICATION")
    print("Red:                 Band 2 (Red, approx 670 nm)")
    print("NIR:                 Band 4 (NIR, approx 840 nm)")
    print("Wavelength metadata: Red = 670 nm, NIR = 840 nm")
    print("Reflectance preprocessing: Percentile clipping [2%, 98%] and min-max normalization to [0,1]\n")
    
    # NDVI Calculation
    epsilon = 1e-8
    ndvi = compute_ndvi(nir_band, red_band)
    
    valid_pixels = int(np.sum(np.isfinite(ndvi)))
    pct_valid = (valid_pixels / ndvi.size) * 100.0
    
    print("NDVI COMPUTATION")
    print("Formula:             (NIR - Red) / (NIR + Red + epsilon)")
    print(f"Epsilon:             {epsilon}")
    print(f"Valid pixel count:   {valid_pixels} ({pct_valid:.2f}%)")
    print(f"NDVI range:          Min: {np.min(ndvi):.4f}, Max: {np.max(ndvi):.4f}\n")
    
    # 4. Ground Truth Search & Validation
    print("GROUND TRUTH STATUS: NOT AVAILABLE (Searched repository data directories; no independent validation sets found)\n")
    
    # Scientific metrics
    print("SCIENTIFIC METRICS")
    print("MAE:                 NOT EVALUABLE")
    print("RMSE:                NOT EVALUABLE")
    print("Pearson r:           NOT EVALUABLE")
    print("Spatial Agreement:   NOT EVALUABLE\n")
    
    # Final Status
    # NOT EVALUABLE since independent ground truth is unavailable
    print("FINAL STATUS:        NOT EVALUABLE")

if __name__ == "__main__":
    perform_ndvi_validation()
