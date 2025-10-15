"""
Quick start script for Metacritic scraper

This script provides an interactive way to start scraping game reviews.
"""

import os
import subprocess
import sys


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print(" " * 15 + "METACRITIC GAME REVIEW SCRAPER")
    print("=" * 70)
    print("\nCollect game reviews with sentiment labels for analysis")
    print()


def get_user_input():
    """Get scraping parameters from user"""
    print("Enter game details:")
    print("-" * 70)
    
    # Get game name
    game_name = input("\n1. Game name (use hyphens, e.g., 'the-last-of-us-part-ii'): ").strip()
    
    if not game_name:
        print("❌ Game name is required!")
        return None
    
    # Show platform options
    print("\n2. Select platform:")
    platforms = {
        "1": ("PlayStation 5", "playstation-5"),
        "2": ("PlayStation 4", "playstation-4"),
        "3": ("Xbox Series X/S", "xbox-series-x"),
        "4": ("Xbox One", "xbox-one"),
        "5": ("Nintendo Switch", "switch"),
        "6": ("PC", "pc"),
    }
    
    for key, (name, _) in platforms.items():
        print(f"   {key}. {name}")
    
    platform_choice = input("\nEnter platform number (or type custom): ").strip()
    
    if platform_choice in platforms:
        platform = platforms[platform_choice][1]
    else:
        platform = input("Enter custom platform value: ").strip()
    
    if not platform:
        print("❌ Platform is required!")
        return None
    
    # Get max reviews
    max_reviews = input("\n3. Maximum reviews to scrape (press Enter for all): ").strip()
    
    return {
        'game_name': game_name,
        'platform': platform,
        'max_reviews': max_reviews if max_reviews else None
    }


def run_scraper(params):
    """Run the scraper with given parameters"""
    print("\n" + "=" * 70)
    print("Starting scraper...")
    print("=" * 70)
    
    command = [
        "scrapy", "crawl", "metacritic_reviews",
        "-a", f"game_name={params['game_name']}",
        "-a", f"platform={params['platform']}"
    ]
    
    if params['max_reviews']:
        command.extend(["-a", f"max_reviews={params['max_reviews']}"])
    
    print(f"\nCommand: {' '.join(command)}\n")
    print("🕷️  Scraping in progress...")
    print("=" * 70 + "\n")
    
    try:
        result = subprocess.run(command, check=True)
        
        print("\n" + "=" * 70)
        print("✅ Scraping completed successfully!")
        print("=" * 70)
        print("\n📁 Check the 'data' folder for your scraped reviews.\n")
        
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print("❌ Scraping failed!")
        print("=" * 70)
        print(f"\nError: {e}\n")
        return False
    except FileNotFoundError:
        print("\n" + "=" * 70)
        print("❌ Scrapy not found!")
        print("=" * 70)
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt\n")
        return False


def quick_examples():
    """Show quick example commands"""
    print("\n" + "=" * 70)
    print("QUICK EXAMPLES")
    print("=" * 70)
    
    examples = [
        {
            "description": "Scrape The Last of Us Part II (PS4)",
            "command": 'scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4"'
        },
        {
            "description": "Scrape Elden Ring (PS5) - First 100 reviews",
            "command": 'scrapy crawl metacritic_reviews -a game_name="elden-ring" -a platform="playstation-5" -a max_reviews=100'
        },
        {
            "description": "Scrape Hades (Switch)",
            "command": 'scrapy crawl metacritic_reviews -a game_name="hades" -a platform="switch"'
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print(f"   {example['command']}")
    
    print("\n" + "=" * 70)


def main():
    """Main function"""
    print_banner()
    
    while True:
        print("\nWhat would you like to do?")
        print("  1. Interactive scraping (recommended for beginners)")
        print("  2. Show example commands")
        print("  3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            params = get_user_input()
            if params:
                proceed = input("\n▶️  Ready to start scraping? (y/n): ").strip().lower()
                if proceed == 'y':
                    run_scraper(params)
                else:
                    print("Scraping cancelled.")
        
        elif choice == "2":
            quick_examples()
        
        elif choice == "3":
            print("\n👋 Goodbye!\n")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!\n")
        sys.exit(0)
