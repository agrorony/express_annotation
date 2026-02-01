#!/usr/bin/env python3
"""Test GPU flag behavior and error handling."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "src")
from metrics_engine import compute_z_metrics, compute_z_run_length_stability, compute_2d_metrics

def test_gpu_error_handling():
    """Test that GPU flag properly raises errors when not available."""
    print("="*60)
    print("Testing GPU Error Handling")
    print("="*60)
    
    # Create a small test volume
    volume = np.zeros((5, 20, 20), dtype=np.uint8)
    volume[:, 5:15, 5:15] = 1
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: compute_z_metrics with GPU
    tests_total += 1
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    
    if not TORCH_AVAILABLE:
        print("\nTest 1: compute_z_metrics with use_gpu=True (PyTorch not installed)")
        try:
            metrics = compute_z_metrics(volume, 1, 3, use_gpu=True)
            print("  ✗ FAIL: Should have raised NotImplementedError")
        except NotImplementedError as e:
            if "PyTorch" in str(e):
                print(f"  ✓ PASS: Correctly raised NotImplementedError")
                print(f"    Message: {str(e)}")
                tests_passed += 1
            else:
                print(f"  ✗ FAIL: Wrong error message: {str(e)}")
        except Exception as e:
            print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {str(e)}")
    else:
        print("\nTest 1: compute_z_metrics with use_gpu=True (PyTorch installed)")
        if not torch.cuda.is_available():
            try:
                metrics = compute_z_metrics(volume, 1, 3, use_gpu=True)
                print("  ✗ FAIL: Should have raised RuntimeError (no CUDA)")
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"  ✓ PASS: Correctly raised RuntimeError for missing CUDA")
                    print(f"    Message: {str(e)}")
                    tests_passed += 1
                else:
                    print(f"  ✗ FAIL: Wrong error message: {str(e)}")
            except Exception as e:
                print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {str(e)}")
        else:
            print("  CUDA available - GPU execution should work")
            try:
                metrics = compute_z_metrics(volume, 1, 3, use_gpu=True)
                print(f"  ✓ PASS: GPU execution successful")
                tests_passed += 1
            except Exception as e:
                print(f"  ✗ FAIL: GPU execution failed: {str(e)}")
    
    # Test 2: compute_z_run_length_stability with GPU
    tests_total += 1
    if not TORCH_AVAILABLE:
        print("\nTest 2: compute_z_run_length_stability with use_gpu=True (PyTorch not installed)")
        try:
            run_stats = compute_z_run_length_stability(volume, 1, use_gpu=True)
            print("  ✗ FAIL: Should have raised NotImplementedError")
        except NotImplementedError as e:
            if "PyTorch" in str(e):
                print(f"  ✓ PASS: Correctly raised NotImplementedError")
                tests_passed += 1
            else:
                print(f"  ✗ FAIL: Wrong error message: {str(e)}")
        except Exception as e:
            print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {str(e)}")
    else:
        print("\nTest 2: compute_z_run_length_stability with use_gpu=True (PyTorch installed)")
        if not torch.cuda.is_available():
            try:
                run_stats = compute_z_run_length_stability(volume, 1, use_gpu=True)
                print("  ✗ FAIL: Should have raised RuntimeError (no CUDA)")
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"  ✓ PASS: Correctly raised RuntimeError for missing CUDA")
                    tests_passed += 1
                else:
                    print(f"  ✗ FAIL: Wrong error message: {str(e)}")
            except Exception as e:
                print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {str(e)}")
        else:
            print("  CUDA available - GPU execution should work")
            try:
                run_stats = compute_z_run_length_stability(volume, 1, use_gpu=True)
                print(f"  ✓ PASS: GPU execution successful")
                tests_passed += 1
            except Exception as e:
                print(f"  ✗ FAIL: GPU execution failed: {str(e)}")
    
    # Test 3: compute_2d_metrics with GPU (not implemented)
    tests_total += 1
    print("\nTest 3: compute_2d_metrics with use_gpu=True (not implemented)")
    try:
        slice_mask = volume[0]
        metrics = compute_2d_metrics(slice_mask, 1, use_gpu=True)
        print("  ✗ FAIL: Should have raised NotImplementedError")
    except NotImplementedError as e:
        if "2D metrics" in str(e):
            print(f"  ✓ PASS: Correctly raised NotImplementedError")
            print(f"    Message: {str(e)}")
            tests_passed += 1
        else:
            print(f"  ✗ FAIL: Wrong error message: {str(e)}")
    except Exception as e:
        print(f"  ✗ FAIL: Wrong exception type: {type(e).__name__}: {str(e)}")
    
    # Test 4: CPU fallback behavior
    tests_total += 1
    print("\nTest 4: CPU execution with use_gpu=False")
    try:
        metrics = compute_z_metrics(volume, 1, 3, use_gpu=False)
        print(f"  ✓ PASS: CPU execution successful")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAIL: CPU execution failed: {str(e)}")
    
    print("\n" + "="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print("="*60)
    
    return 0 if tests_passed == tests_total else 1

if __name__ == "__main__":
    sys.exit(test_gpu_error_handling())
