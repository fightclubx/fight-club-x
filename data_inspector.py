import json

def inspect_twitter_data(json_file_path):
    """
    Load and inspect the Twitter community JSON data
    to understand its structure
    """
    
    print("🔍 Inspecting Twitter Community Data...")
    
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded JSON file!")
        print(f"📊 Total members found: {len(data)}")
        print()
        
        # Show key info for first few members
        print("👥 First 3 members preview:")
        for i, member in enumerate(data[:3]):
            print(f"  {i+1}. @{member['screen_name']} ({member['name']}) - {member['community_role']}")
        print()
        
        # Count roles
        admins = sum(1 for m in data if m['community_role'] == 'Admin')
        members = sum(1 for m in data if m['community_role'] == 'Member')
        print(f"👑 Admins: {admins}")
        print(f"👤 Members: {members}")
        print()
        
        # Show what we'll use for the fight
        print("⚔️ Battle Setup Info:")
        print(f"  - Usernames: screen_name field (@{data[0]['screen_name']})")
        print(f"  - Display names: name field ({data[0]['name']})")
        print(f"  - Profile images: profile_image_url_https")
        print(f"  - Sample image URL: {data[0]['profile_image_url_https']}")
        print()
        
        # Check for any issues
        issues = []
        for i, member in enumerate(data):
            if not member.get('screen_name'):
                issues.append(f"Member {i+1} missing screen_name")
            if not member.get('profile_image_url_https'):
                issues.append(f"Member {i+1} missing profile image")
        
        if issues:
            print("⚠️ Potential issues found:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        else:
            print("✅ All members have required data!")
        
        return data
        
    except FileNotFoundError:
        print("❌ Error: JSON file not found!")
        print("Make sure the file path is correct.")
        return None
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format!")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Run the inspector
if __name__ == "__main__":
    # CHANGE THIS to your actual JSON file name
    json_file = "twitter_community_data.json"
    
    print("🚀 Twitter Fight Club - Data Inspector")
    print("=" * 50)
    
    data = inspect_twitter_data(json_file)
    
    if data:
        print("✅ Data inspection complete!")
        print("\nNext steps:")
        print("1. Note which fields contain usernames and profile images")
        print("2. We'll use this info to adapt the fight simulation")
        print("3. Then we'll download profile images and run battles!")
    else:
        print("❌ Please fix the JSON file issue and try again")
