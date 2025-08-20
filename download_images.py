#!/usr/bin/env python3
"""
Profile Image Downloader for Fight Club X
==========================================
Scans community data and downloads any missing profile images.
Run this anytime to ensure all images are ready for battle.

Usage: python download_images.py
"""

import json
import os
import requests
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configuration
COMMUNITY_DATA_FILE = 'twitter_community_data.json'
IMAGES_DIR = 'images'

def load_community_data():
    """Load community data from JSON file"""
    if not os.path.exists(COMMUNITY_DATA_FILE):
        print(f"❌ {COMMUNITY_DATA_FILE} not found!")
        return None
    
    try:
        with open(COMMUNITY_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if len(data) == 0:
            print("❌ No community members found in data file")
            return None
            
        print(f"📊 Found {len(data)} community members")
        return data
        
    except Exception as e:
        print(f"❌ Error loading {COMMUNITY_DATA_FILE}: {e}")
        return None

def find_missing_images(community_data):
    """Check which profile images are missing"""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    missing = []
    existing = 0
    
    for member in community_data:
        username = member['screen_name']
        image_path = os.path.join(IMAGES_DIR, f"{username}.png")
        
        if os.path.exists(image_path):
            existing += 1
        else:
            missing.append(member)
    
    print(f"🔍 Found {existing} existing images, {len(missing)} missing")
    return missing

def download_image(member):
    """Download profile image for a single member"""
    username = member['screen_name']
    url = member['profile_image_url_https'].replace('_normal', '_400x400')
    image_path = os.path.join(IMAGES_DIR, f"{username}.png")
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Convert and save as PNG
        img = Image.open(BytesIO(response.content)).convert('RGBA')
        img.save(image_path, 'PNG', optimize=True)
        return True
        
    except Exception as e:
        print(f"❌ Failed @{username}: {e}")
        return False

def main():
    print("🖼️ Profile Image Downloader")
    print("=" * 40)
    
    # Load community data
    community_data = load_community_data()
    if not community_data:
        return
    
    # Find missing images
    missing_members = find_missing_images(community_data)
    
    if not missing_members:
        print("✅ All profile images already downloaded!")
        return
    
    # Download missing images
    print(f"📥 Downloading {len(missing_members)} missing images...")
    
    downloaded = 0
    failed = 0
    
    for member in tqdm(missing_members, desc="Downloading"):
        if download_image(member):
            downloaded += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(0.2)
    
    print(f"\n🎯 Download complete!")
    print(f"✅ Downloaded: {downloaded}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total images ready: {len(community_data) - failed}")

if __name__ == "__main__":
    main()
