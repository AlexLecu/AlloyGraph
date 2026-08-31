from ..models.feature_engineering import compute_alloy_features


def print_test_header(title: str):
    """Print test section header."""
    print("\n" + "=" * 80)
    print(f"🧪 {title}")
    print("=" * 80)


def test_1_waspaloy_features():
    """Test 1: Feature computation for a known alloy (Waspaloy)."""
    print_test_header("TEST: Known Alloy (Waspaloy)")
    
    waspaloy = {
        "Ni": 58.0, "Cr": 19.4, "Co": 13.5,
        "Mo": 4.3, "Ti": 3.0, "Al": 1.4,
        "Fe": 0.4
    }
    
    features = compute_alloy_features(waspaloy)
    
    print("\nComputed Features:")
    print(f"  Md_gamma: {features['Md_gamma']:.4f}")
    print(f"  Lattice mismatch: {features['lattice_mismatch_pct']:.3f}%")
    print(f"  γ' fraction: {features['gamma_prime_estimated_vol_pct']:.1f}%")
    print(f"  Density: {features['density_calculated_gcm3']:.2f} g/cm³")
    
   # Sanity checks
    assert 0.9 < features['Md_gamma'] < 1.0, "Waspaloy should have Md ~0.93"
    assert abs(features['lattice_mismatch_pct']) < 1.0, "Lattice mismatch should be < 1%"
    assert 15 < features['gamma_prime_estimated_vol_pct'] < 25, "γ' should be ~20%"
    # Density uses the inverse rule of mixtures (1/ρ = Σ wᵢ/ρᵢ), which is the
    # correct form for mass fractions. Gives 8.07 here; literature Waspaloy is
    # 8.19 g/cm³. The earlier 8.2–8.6 window was calibrated to the arithmetic
    # form Σ(wᵢ·ρᵢ), which overestimates badly for refractory-rich alloys
    # (CMSX-4: 9.90 vs literature 8.70).
    assert 7.9 < features['density_calculated_gcm3'] < 8.3, "Density should be ~8.1 g/cm³"
    
    print("\n✅ PASS: Waspaloy features computed correctly")


def test_2_edge_cases():
    """Test 2: Edge cases (minimal composition, high refractories)."""
    print_test_header("TEST: Edge Cases")
    
    # Pure Ni baseline
    print("\n📋 Test Case 1: Pure Ni")
    pure_ni = {"Ni": 100.0}
    features = compute_alloy_features(pure_ni)
    
    assert features['Md_gamma'] < 0.8, "Pure Ni should have low Md"
    assert features['gamma_prime_estimated_vol_pct'] == 0, "Pure Ni has no γ'"
    print(f"  Md_gamma: {features['Md_gamma']:.4f} ✓")
    print(f"  γ' fraction: {features['gamma_prime_estimated_vol_pct']:.1f}% ✓")
    
    # High refractory superalloy
    print("\n📋 Test Case 2: High Refractory Content")
    high_re = {
        "Ni": 50.0, "Cr": 8.0, "Co": 10.0,
        "Re": 6.0, "W": 8.0, "Mo": 3.0,
        "Al": 6.0, "Ti": 0.5, "Ta": 8.5
    }
    features = compute_alloy_features(high_re)
    
    # Md_gamma is computed from the γ-matrix composition after partitioning,
    # which is normalised to 100%. Before that normalisation this returned
    # 1.2743 — outside the physical range of the Morinaga scale (~0.9–1.05),
    # which is why the old assertion read > 1.0 while its message said > 0.97.
    assert features['Md_gamma'] > 0.97, "High Re/W should have Md > 0.97"
    # Inverse-ROM density: 8.56 here, in line with comparable refractory-rich
    # alloys (MAR-M 247 8.54, CMSX-4 8.70). The old > 8.8 bound reflected the
    # arithmetic form, which returned an unphysical 10.63 for this composition.
    assert features['density_calculated_gcm3'] > 8.4, "Heavy refractories increase density"
    print(f"  Md_gamma: {features['Md_gamma']:.4f} ✓")
    print(f"  Density: {features['density_calculated_gcm3']:.2f} g/cm³ ✓")
    
    print("\n✅ ALL TESTS PASSED: Edge Cases")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🚀 FEATURE ENGINEERING TEST SUITE")
    print("=" * 80)
    
    test_1_waspaloy_features()
    test_2_edge_cases()
    
    print("\n" + "=" * 80)
    print("✅ ALL FEATURE ENGINEERING TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
