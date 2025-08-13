#!/usr/bin/env python3
"""
Test script to demonstrate both hardcoded and automated variable selection approaches.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wa_congressional_analysis import main
import time

def test_both_approaches():
    """Test both hardcoded and automated variable selection approaches."""
    print("Testing Variable Selection Approaches")
    print("=" * 60)
    
    # Test 1: Hardcoded Selection (Fast, Predictable)
    print("\n1. HARDCODED Variable Selection")
    print("-" * 40)
    print("Purpose: Fast, predictable variable selection for demonstrations")
    print("Method: Uses carefully curated list of diverse demographic variables")
    print("Expected: 20 pre-selected diverse variables")
    
    start_time = time.time()
    try:
        main(
            use_hardcoded_selection=True,
            max_variables=20,
            years=[2020],  # Just one year for faster testing
            corr_threshold=0.95
        )
        hardcoded_time = time.time() - start_time
        print(f"✓ Hardcoded selection completed in {hardcoded_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Hardcoded selection failed: {e}")
        hardcoded_time = None
    
    # Test 2: Automated Selection (Comprehensive, Production-ready)
    print("\n2. AUTOMATED Variable Selection")
    print("-" * 40)
    print("Purpose: Comprehensive variable discovery using full pipeline")
    print("Method: Category-based filtering and automated selection")
    print("Expected: Variable number of variables based on categories")
    
    start_time = time.time()
    try:
        main(
            use_hardcoded_selection=False,
            max_variables=20,
            years=[2020],  # Just one year for faster testing
            corr_threshold=0.95
        )
        automated_time = time.time() - start_time
        print(f"✓ Automated selection completed in {automated_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Automated selection failed: {e}")
        automated_time = None
    
    # Comparison
    print("\n3. COMPARISON")
    print("-" * 40)
    if hardcoded_time and automated_time:
        speedup = automated_time / hardcoded_time
        print(f"Hardcoded selection: {hardcoded_time:.2f} seconds")
        print(f"Automated selection: {automated_time:.2f} seconds")
        print(f"Speedup factor: {speedup:.1f}x faster with hardcoded selection")
        
        if speedup > 2:
            print("✓ Hardcoded selection is significantly faster (good for demos)")
        else:
            print("⚠ Automated selection performance is acceptable")
    else:
        print("⚠ Could not compare performance due to errors")
    
    print("\n4. USE CASES")
    print("-" * 40)
    print("HARDCODED Selection:")
    print("  ✓ Fast demonstrations and testing")
    print("  ✓ Predictable results")
    print("  ✓ Avoids highly correlated variables")
    print("  ✓ Good for educational purposes")
    
    print("\nAUTOMATED Selection:")
    print("  ✓ Comprehensive variable discovery")
    print("  ✓ Production-ready analysis")
    print("  ✓ Can handle larger variable sets")
    print("  ✓ Leverages full pipeline capabilities")
    
    print("\n5. OUTPUT FILES")
    print("-" * 40)
    print("Check the data/processed/test_examples/ directory for:")
    print("  - wa_congressional_analysis_hardcoded_*.csv")
    print("  - wa_congressional_analysis_automated_*.csv")
    print("  - wa_congressional_variables_hardcoded_*.csv")
    print("  - wa_congressional_variables_automated_*.csv")

if __name__ == "__main__":
    test_both_approaches()
