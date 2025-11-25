"""
Test script to verify all API endpoints are working
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

def test_endpoint(name, method, endpoint, expected_status=200):
    """Test a single endpoint"""
    url = f"{API_BASE}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ PASS")
            try:
                data = response.json()
                print(f"Response preview: {json.dumps(data, indent=2)[:200]}...")
            except:
                print(f"Response: {response.text[:200]}...")
            return True
        else:
            print(f"❌ FAIL - Expected {expected_status}, got {response.status_code}")
            print(f"Error: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAIL - Connection refused. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ FAIL - {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("GNSS API Integration Test")
    print("="*60)
    print(f"Testing API at: {API_BASE}")
    print(f"Time: {datetime.now()}")
    
    results = []
    
    # System Endpoints
    print("\n\n### SYSTEM ENDPOINTS ###")
    results.append(test_endpoint("Root Status", "GET", "/"))
    results.append(test_endpoint("Health Check", "GET", "/health"))
    
    # Prediction Endpoints
    print("\n\n### PREDICTION ENDPOINTS ###")
    results.append(test_endpoint("MEO Predictions (All Horizons)", "GET", "/predict/MEO"))
    results.append(test_endpoint("GEO Predictions (All Horizons)", "GET", "/predict/GEO"))
    results.append(test_endpoint("MEO Prediction (15min)", "GET", "/predict/MEO/15min"))
    results.append(test_endpoint("GEO Prediction (1h)", "GET", "/predict/GEO/1h"))
    
    # Data Endpoints
    print("\n\n### DATA ENDPOINTS ###")
    results.append(test_endpoint("MEO Data Stats", "GET", "/data/stats/MEO"))
    results.append(test_endpoint("GEO Data Stats", "GET", "/data/stats/GEO"))
    results.append(test_endpoint("MEO Data Sample", "GET", "/data/sample/MEO?limit=10"))
    results.append(test_endpoint("GEO Data Sample", "GET", "/data/sample/GEO?limit=10"))
    
    # Model Endpoints
    print("\n\n### MODEL ENDPOINTS ###")
    results.append(test_endpoint("MEO Model Metrics", "GET", "/models/metrics/MEO"))
    results.append(test_endpoint("GEO Model Metrics", "GET", "/models/metrics/GEO"))
    results.append(test_endpoint("Model Comparison", "GET", "/models/comparison"))
    
    # Feature Endpoints
    print("\n\n### FEATURE ENDPOINTS ###")
    results.append(test_endpoint("MEO Feature Importance", "GET", "/features/importance/MEO"))
    results.append(test_endpoint("GEO Feature Importance", "GET", "/features/importance/GEO"))
    results.append(test_endpoint("MEO Feature Stats", "GET", "/features/stats/MEO"))
    results.append(test_endpoint("GEO Feature Stats", "GET", "/features/stats/GEO"))
    
    # Analysis Endpoints
    print("\n\n### ANALYSIS ENDPOINTS ###")
    results.append(test_endpoint("MEO Residuals", "GET", "/residuals/MEO"))
    results.append(test_endpoint("GEO Residuals", "GET", "/residuals/GEO"))
    results.append(test_endpoint("MEO Historical Predictions", "GET", "/predictions/historical/MEO"))
    results.append(test_endpoint("GEO Historical Predictions", "GET", "/predictions/historical/GEO"))
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
