"""
Test script to check connectivity to Metacritic
Run this before scraping to diagnose issues
"""

import requests
import sys
from datetime import datetime

def test_basic_connection():
    """Test basic connection to Metacritic"""
    print("\n" + "=" * 60)
    print("Testing Basic Connection to Metacritic")
    print("=" * 60)
    
    try:
        print("\n📡 Attempting to connect to metacritic.com...")
        response = requests.get(
            "https://www.metacritic.com/",
            timeout=10,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        if response.status_code == 200:
            print("✅ SUCCESS! Can reach Metacritic")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {response.elapsed.total_seconds():.2f}s")
            return True
        else:
            print(f"⚠️  WARNING: Received status code {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Could not connect to Metacritic (timeout)")
        print("   This suggests network issues or Metacritic is blocking requests")
        return False
    except requests.exceptions.ConnectionError as e:
        print("❌ CONNECTION ERROR: Could not connect to Metacritic")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_game_page(game_name="the-last-of-us-part-ii", platform="playstation-4"):
    """Test connection to a specific game page"""
    print("\n" + "=" * 60)
    print(f"Testing Game Page Access")
    print("=" * 60)
    
    url = f"https://www.metacritic.com/game/{platform}/{game_name}"
    print(f"\n🎮 Testing game: {game_name}")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
        )
        
        if response.status_code == 200:
            print("✅ SUCCESS! Game page accessible")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {response.elapsed.total_seconds():.2f}s")
            print(f"   Page Size: {len(response.content) / 1024:.2f} KB")
            
            # Check for reviews
            if 'user-reviews' in response.text:
                print("✅ User reviews section found!")
            else:
                print("⚠️  Warning: User reviews section not found in HTML")
            
            return True
        elif response.status_code == 404:
            print("❌ NOT FOUND (404): Game page does not exist")
            print("   Check the game name and platform are correct")
            return False
        elif response.status_code == 403:
            print("❌ FORBIDDEN (403): Access denied by Metacritic")
            print("   You may be temporarily blocked")
            return False
        else:
            print(f"⚠️  Received status code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Could not load game page (timeout)")
        return False
    except requests.exceptions.ConnectionError as e:
        print("❌ CONNECTION ERROR")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_user_reviews_page(game_name="the-last-of-us-part-ii", platform="playstation-4"):
    """Test connection to user reviews page"""
    print("\n" + "=" * 60)
    print("Testing User Reviews Page Access")
    print("=" * 60)
    
    url = f"https://www.metacritic.com/game/{platform}/{game_name}/user-reviews"
    print(f"\n📝 Testing reviews page...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
        )
        
        if response.status_code == 200:
            print("✅ SUCCESS! Reviews page accessible")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {response.elapsed.total_seconds():.2f}s")
            
            # Quick check for review content
            if 'review' in response.text.lower():
                print("✅ Review content found in page!")
            else:
                print("⚠️  Warning: No review content detected")
            
            return True
        else:
            print(f"⚠️  Received status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    print("\nResults:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All tests passed!")
        print("\nYou should be able to scrape Metacritic successfully.")
        print("\nTo start scraping, run:")
        print("  python run_scraper.py")
        print("\nOr:")
        print("  scrapy crawl metacritic_reviews -a game_name=\"the-last-of-us-part-ii\" -a platform=\"playstation-4\" -a max_reviews=10")
    else:
        print("⚠️  Some tests failed!")
        print("\nPossible issues:")
        print("  - Network connectivity problems")
        print("  - Metacritic is blocking automated requests")
        print("  - Firewall or antivirus interference")
        print("  - The specific game doesn't exist on Metacritic")
        print("\nRecommendations:")
        print("  1. Check your internet connection")
        print("  2. Try a different game (if game page test failed)")
        print("  3. Wait a few minutes and try again")
        print("  4. Check TROUBLESHOOTING.md for more help")
    
    print("=" * 60 + "\n")


def main():
    """Run all connectivity tests"""
    print("\n" + "=" * 60)
    print(" " * 15 + "METACRITIC CONNECTIVITY TEST")
    print("=" * 60)
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will test if you can connect to Metacritic...")
    
    # Run tests
    results = {}
    
    results['Basic Connection'] = test_basic_connection()
    
    if results['Basic Connection']:
        results['Game Page'] = test_game_page()
        if results['Game Page']:
            results['Reviews Page'] = test_user_reviews_page()
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user\n")
        sys.exit(0)
