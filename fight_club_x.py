# fight_club_x.py  (updated)
# --------------------------------------------------------------
# Exports video, thumbnail, per-simulation stats & ranking JSON,
# updates /data/manifest.json, and renders a winner overlay
# with GOLD/SILVER/BRONZE podium finishes.
# NOW WITH DYNAMIC RESIZING AS PARTICIPANTS ARE ELIMINATED!
# LOADS COMMUNITY DATA DURING BATTLE START FOR FRESH DATA!
# --------------------------------------------------------------

import pygame
import json
import os
import random
import math
import numpy as np
import datetime
import moviepy as mpy
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
import time


# ================== DUAL RESOLUTION CONFIG ==================
# Display resolution (what you see while running)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080

# Recording resolution (output video file size) 
RECORD_WIDTH, RECORD_HEIGHT = 1280, 720  # Change this for different output sizes

print(f"🖥️  Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"🎬 Recording: {RECORD_WIDTH}x{RECORD_HEIGHT}")


# -------------------- WEBSITE OUTPUT SETTINGS --------------------
# Adjust WEB_ROOT if your static root differs (e.g., "public" for Next.js)
WEB_ROOT = os.environ.get("WEB_ROOT", "public")
MEDIA_DIR = os.path.join(WEB_ROOT, "media")
DATA_DIR  = os.path.join(WEB_ROOT, "data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
RANK_DIR  = os.path.join(DAILY_DIR, "ranking")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(RANK_DIR, exist_ok=True)

def _to_url(local_path: str) -> str:
    """Map an absolute path under WEB_ROOT to a site URL (/media/... or /data/...)."""
    lp = os.path.abspath(local_path).replace("\\", "/")
    root = os.path.abspath(WEB_ROOT).replace("\\", "/")
    if lp.startswith(root):
        rel = lp[len(root):]
        return rel if rel.startswith("/") else "/" + rel
    return local_path  # fallback

# -------------------- PARTICLE --------------------

class TwitterParticle:
    def __init__(self, username, display_name, image_path, radius, max_hp, max_speed, acc_magnitude, width, height, position):
        self.id = username
        self.username = username
        self.display_name = display_name
        self.image_path = image_path
        self.radius = radius
        self.max_hp = max_hp
        self.max_speed = max_speed
        self.acc_magnitude = acc_magnitude
        self.width = width
        self.height = height

        # Rendering surfaces (NEW: base + cache for scaled sizes)
        self.base_surface = None      # masked, clamped once
        self._scale_cache = {}        # {size_px: pygame.Surface}

        # Physics
        self.pos = position
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.1, 0.4)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
        self.acc_mag = 0.005
        self.hp = max_hp
        self.alive = True
        self.mass = 1.0

        # Back-compat for places that expect .image (e.g., podium)
        self.image = None

    def load_twitter_image(self):
        """Load once, clamp once, mask once. No per-frame PIL work."""
        if self.base_surface is not None:
            return
        try:
            pil_img = Image.open(self.image_path).convert('RGBA')
        except Exception as e:
            print(f"Error loading image for @{self.username}: {e}")
            size = max(64, self.radius * 2)
            pil_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(pil_img)
            color = (
                hash(self.username) % 156 + 100,
                (hash(self.username) >> 8) % 156 + 100,
                (hash(self.username) >> 16) % 156 + 100
            )
            draw.ellipse([0, 0, size-1, size-1], fill=color)

        # Clamp once to a reasonable maximum so scaling stays cheap
        MAX = 192  # tune 128–256 if you like
        if pil_img.size[0] != MAX or pil_img.size[1] != MAX:
            pil_img = pil_img.resize((MAX, MAX), Image.Resampling.LANCZOS)

        # Convert to pygame surface + apply circular mask ONCE
        pg = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)
        self.base_surface = circular_mask(pg).convert_alpha()

        # Maintain .image for existing code paths (e.g., podium scaling)
        self.image = self.base_surface

    def _get_scaled(self, size_px: int) -> pygame.Surface:
        """Fast render-time scaling with a tiny cache."""
        s = self._scale_cache.get(size_px)
        if s is None:
            # base_surface guaranteed after load_twitter_image()
            s = pygame.transform.smoothscale(self.base_surface, (size_px, size_px))
            self._scale_cache[size_px] = s
        return s

    def update_radius(self, new_radius):
        """Only update numeric radius; no image processing here."""
        self.radius = new_radius
        # intentionally no re-mask/re-PIL here

    def move(self):
        if not self.alive:
            return
        if np.linalg.norm(self.vel) > 0:
            acc_dir = self.vel / np.linalg.norm(self.vel)
            acc = acc_dir * self.acc_mag
        else:
            acc = np.array([0.0, 0.0])
        self.vel += acc
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        self.pos += self.vel

        # Bounce
        THR = 0.1
        for i, limit in enumerate([self.width, self.height]):
            min_pos = self.radius + THR
            max_pos = limit - self.radius - THR
            if self.pos[i] < min_pos:
                self.pos[i] = min_pos
                self.vel[i] *= -1
            elif self.pos[i] > max_pos:
                self.pos[i] = max_pos
                self.vel[i] *= -1

    def draw(self, surface):
        if not self.alive:
            return
        if self.base_surface is None:
            self.load_twitter_image()

        size = self.radius * 2
        scaled = self._get_scaled(size)
        img_rect = scaled.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        surface.blit(scaled, img_rect)

        # Health bar
        if self.radius >= 10:
            hp_ratio = max(0, min(self.hp / self.max_hp, 1))
            bar_width = self.radius * 2
            bar_height = max(3, self.radius // 6)
            x = int(self.pos[0]) - self.radius
            y = int(self.pos[1]) - self.radius - bar_height - 2
            pygame.draw.rect(surface, (40, 40, 40), (x, y, bar_width, bar_height), border_radius=1)
            hp_bar_width = int(hp_ratio * bar_width)
            if hp_bar_width > 0:
                r = int(255 * (1 - hp_ratio))
                g = int(255 * hp_ratio)
                color = (r, g, 40)
                pygame.draw.rect(surface, color, (x, y, hp_bar_width, bar_height), border_radius=1)
            pygame.draw.rect(surface, (150, 150, 150), (x, y, bar_width, bar_height), width=1, border_radius=1)

    def damage(self, force):
        self.hp -= force
        if self.hp <= 0:
            self.alive = False
            self.vel = np.array([0, 0], dtype=float)

# -------------------- HELPERS --------------------

def circular_mask(image):
    size = image.get_size()
    mask_surface = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (size[0]//2, size[1]//2), min(size)//2)
    image.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return image

def get_dynamic_radius(num_particles, width, height, min_radius, max_radius, change_radius=True, particles=None):
    """Calculate optimal radius for current number of particles and optionally resize them"""
    best_radius = min_radius
    for r in range(max_radius, min_radius - 1, -1):
        cell_size = 4 * r
        cols = width // cell_size
        rows = height // cell_size
        if cols * rows >= num_particles:
            best_radius = r
            break
    
    # Update particle radii if requested and particles are provided
    if change_radius and particles:
        for particle in particles:
            if particle.alive:  # Only update living particles
                particle.update_radius(best_radius)
    
    return best_radius

def assign_position(radius, width, height, num_particles):
    cell_size = 2 * radius + 2
    cols = int(width // cell_size)
    rows = int(height // cell_size)
    if num_particles > cols * rows:
        raise ValueError(f"Not enough space for {num_particles} particles. Max: {cols * rows}")
    available_cells = [(col, row) for col in range(cols) for row in rows_range(rows)]
    random.shuffle(available_cells)
    positions = []
    for _ in range(num_particles):
        col, row = available_cells.pop()
        x = col * cell_size + radius + 10
        y = row * cell_size + radius + 10
        positions.append(np.array([float(x), float(y)]))
    return positions

def rows_range(rows):
    return range(rows)

def get_cell_coords(pos, cell_size):
    return int(pos[0] // cell_size), int(pos[1] // cell_size)

def load_battle_floor(width, height, floor_image_path='battle_floor.png'):
    """Load and prepare the battle floor texture"""
    try:
        # Try to load your custom floor texture
        floor_texture = pygame.image.load(floor_image_path)
        print(f"✅ Loaded battle floor: {floor_image_path}")
        
        # Scale to fit the arena size
        floor_texture = pygame.transform.scale(floor_texture, (width, height))
        
        return floor_texture
        
    except Exception as e:
        print(f"⚠️  Could not load {floor_image_path}: {e}")
        print("🎨 Creating procedural battle floor...")
        
        # Fallback: Create a procedural battle floor
        floor_surface = pygame.Surface((width, height))
        
        # Base dark color
        floor_surface.fill((25, 30, 40))
        
        # Add grid pattern for battle arena feel
        grid_color = (40, 50, 65)
        grid_size = 50
        
        # Main grid
        for x in range(0, width, grid_size):
            pygame.draw.line(floor_surface, grid_color, (x, 0), (x, height), 1)
        for y in range(0, height, grid_size):
            pygame.draw.line(floor_surface, grid_color, (0, y), (width, y), 1)
            
        # Add some battle arena markings
        center_x, center_y = width // 2, height // 2
        
        # Center circle
        pygame.draw.circle(floor_surface, (50, 60, 80), (center_x, center_y), 100, 3)
        pygame.draw.circle(floor_surface, (35, 45, 60), (center_x, center_y), 200, 2)
        
        # Corner markings
        corner_size = 80
        corner_color = (45, 55, 70)
        corners = [
            (50, 50), (width - 50, 50), 
            (50, height - 50), (width - 50, height - 50)
        ]
        for cx, cy in corners:
            pygame.draw.lines(floor_surface, corner_color, False, 
                            [(cx - corner_size//2, cy), (cx, cy), (cx, cy - corner_size//2)], 3)
        
        return floor_surface

# ---- Stats accounting ----

def apply_damage(victim: TwitterParticle, dmg: float, attacker_username: str, now_seconds: float, stats):
    was_alive = victim.alive
    victim.damage(dmg)

    pv = stats["players"].setdefault(victim.username, {"kills":0,"deaths":0,"damageDealt":0.0,"damageReceived":0.0,"eliminatedAt":None})
    pa = stats["players"].setdefault(attacker_username, {"kills":0,"deaths":0,"damageDealt":0.0,"damageReceived":0.0,"eliminatedAt":None})

    pv["damageReceived"] += float(dmg)
    pa["damageDealt"]    += float(dmg)

    if was_alive and not victim.alive:
        pv["deaths"] += 1
        pa["kills"]  += 1
        if pv["eliminatedAt"] is None:
            pv["eliminatedAt"] = float(now_seconds)

def check_collisions(radius, cell_size, grid_width, grid_height, particles, now_seconds, stats):
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]
    for p in particles:
        if p.alive:
            cx, cy = get_cell_coords(p.pos, cell_size)
            if 0 <= cx < grid_width and 0 <= cy < grid_height:
                grid[cx][cy].append(p)

    for i in range(grid_width):
        for j in range(grid_height):
            cell_particles = grid[i][j]
            for a in cell_particles:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < grid_width and 0 <= nj < grid_height:
                            for b in grid[ni][nj]:
                                if b is a or not b.alive or not a.alive:
                                    continue
                                dist_pos = a.pos - b.pos
                                dist = np.linalg.norm(dist_pos)
                                if dist < radius * 2:
                                    direction = dist_pos / dist if dist != 0 else np.array([1.0, 0.0])
                                    repel_distance = 0.1 * radius
                                    a.pos += direction * repel_distance
                                    b.pos -= direction * repel_distance

                                    force_a = np.linalg.norm(a.vel) * 2  # to b
                                    force_b = np.linalg.norm(b.vel) * 2  # to a
                                    if force_b > 0:
                                        apply_damage(a, min(force_b, a.hp), b.username, now_seconds, stats)
                                    if force_a > 0:
                                        apply_damage(b, min(force_a, b.hp), a.username, now_seconds, stats)

                                    v1 = np.dot(a.vel, direction)
                                    v2 = np.dot(b.vel, direction)
                                    m1, m2 = a.mass, b.mass
                                    new_v1 = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2)
                                    new_v2 = (v2 * (m2 - m1) + 2 * m1 * v1) / (m1 + m2)
                                    a.vel += (new_v1 - v1) * direction
                                    b.vel += (new_v2 - v2) * direction

# -------------------- Assets & fighters --------------------

def download_missing_profile_images(twitter_data):
    print("🔍 Checking for missing profile images...")
    os.makedirs('images', exist_ok=True)
    missing = []
    for member in twitter_data:
        username = member['screen_name']
        if not os.path.exists(f'images/{username}.png'):
            missing.append(member)
    if not missing:
        print("✅ All profile images already downloaded!")
        return len(twitter_data)

    print(f"📥 Need to download {len(missing)} missing profile images...")
    downloaded = 0
    failed = 0
    for member in tqdm(missing, desc="Downloading missing images"):
        try:
            username = member['screen_name']
            url = member['profile_image_url_https'].replace('_normal','_400x400')
            r = requests.get(url, timeout=10); r.raise_for_status()
            Image.open(BytesIO(r.content)).convert('RGBA').save(f'images/{username}.png','PNG')
            downloaded += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"❌ Failed @{username}: {e}")
            failed += 1
    print(f"\n🎯 Download complete! ✅ {downloaded}  ❌ {failed}")
    return len(twitter_data) - failed

def create_placeholder(path, radius, username):
    size = radius * 2
    img = Image.new('RGB', (size, size), (40, 40, 60))
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    draw = ImageDraw.Draw(img)
    color = (hash(username)%156+100, (hash(username)>>8)%156+100, (hash(username)>>16)%156+100)
    draw.ellipse([2,2,size-3,size-3], fill=color)
    if radius >= 10:
        try:
            font_size = max(8, radius//3)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            initials = username[:2].upper()
            bbox = draw.textbbox((0,0), initials, font=font)
            x = (size - (bbox[2]-bbox[0]))//2
            y = (size - (bbox[3]-bbox[1]))//2
            draw.text((x,y), initials, fill=(255,255,255), font=font)
        except:
            pass
    img.save(path)

def load_all_twitter_fighters(json_file='twitter_community_data.json'):
    """
    MOVED TO BE CALLED DURING LOADING SEQUENCE!
    Now loads fresh community data when battle actually starts.
    """
    print("🔥 Loading FRESH Twitter Community Data...")
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"❌ ERROR: {json_file} not found!")
        print("💡 Make sure to scrape and save community data before starting battle.")
        return None, None, None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"📊 Fresh community members loaded: {len(data)}")
    except Exception as e:
        print(f"❌ ERROR loading {json_file}: {e}")
        return None, None, None
    
    if len(data) == 0:
        print("❌ ERROR: No community members in data file!")
        return None, None, None
        
    available = download_missing_profile_images(data)
    print(f"🖼️ Profile images available: {available}")

    n = len(data)
    WIDTH, HEIGHT = 1920, 1080
    MIN_R, MAX_R = 4, 25
    R = get_dynamic_radius(n, WIDTH, HEIGHT, MIN_R, MAX_R, change_radius=False)
    print(f"🎯 Initial particle radius: {R}px")

    try:
        positions = assign_position(R, WIDTH, HEIGHT, n)
    except ValueError:
        WIDTH, HEIGHT = 2560, 1440
        positions = assign_position(R, WIDTH, HEIGHT, n)
        print(f"✅ Using {WIDTH}x{HEIGHT} resolution")

    fighters = []
    for i, m in enumerate(tqdm(data, desc="Creating fighters")):
        u = m['screen_name']; name = m['name']
        img_path = f'images/{u}.png'
        if not os.path.exists(img_path):
            os.makedirs('placeholders', exist_ok=True)
            ph = f'placeholders/{u}.png'
            create_placeholder(ph, R, u)
            img_path = ph
        fighters.append(TwitterParticle(
            username=u, display_name=name, image_path=img_path,
            radius=R, max_hp=100, max_speed=3, acc_magnitude=0.01,
            width=WIDTH, height=HEIGHT, position=positions[i]
        ))
    
    print(f"✅ {len(fighters)} fighters ready for battle!")
    return fighters, WIDTH, HEIGHT

# -------------------- Export helpers --------------------

def _build_interactions_and_ranking(stats_players, winner_username, battle_duration):
    interactions = {}
    for p, s in stats_players.items():
        interactions[p] = {
            "kills": int(s.get("kills", 0)),
            "killsTotal": int(s.get("kills", 0)),
            "damageDealt": float(s.get("damageDealt", 0.0)),
            "damageReceived": float(s.get("damageReceived", 0.0)),
            "deaths": int(s.get("deaths", 0)),
        }
    order = []
    for p, s in stats_players.items():
        elim = s.get("eliminatedAt")
        t = float(elim) if elim is not None else float(battle_duration) + 1.0
        order.append((p, t))
    order.sort(key=lambda x: x[1], reverse=True)

    ranking = {}
    if winner_username:
        ranking[winner_username] = 0
    rank = 1 if winner_username else 0
    for p, _ in order:
        if p == winner_username:
            continue
        ranking[p] = rank
        rank += 1
    return interactions, ranking

def _save_thumbnail_from_frame(frame_rgb, dest_path_jpg):
    Image.fromarray(frame_rgb).save(dest_path_jpg, "JPEG", quality=86, optimize=True, progressive=True)

def _update_manifest(slug, date_str, winner, video_abs, thumb_abs, stats_abs, ranking_abs, top_kills):
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    sims = manifest.get("simulations", [])
    sims = [s for s in sims if s.get("slug") != slug]
    sims.insert(0, {
        "slug": slug,
        "date": date_str,
        "winner": winner,
        "thumb": _to_url(thumb_abs),
        "video": _to_url(video_abs),
        "stats": _to_url(stats_abs),
        "ranking": _to_url(ranking_abs),
        "topKills": int(top_kills)
    })
    manifest["simulations"] = sims
    manifest.setdefault("nextBattle", None)
    manifest.setdefault("contractAddress", None)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"🗂  Updated manifest: {MANIFEST_PATH}")

# -------------------- MAIN --------------------

def main():
    print("🚀 FIGHT CLUB X - REAL-TIME COMMUNITY BATTLE")
    print("="*60)
    print("💡 Community data will be loaded when battle starts")
    print("🔄 This allows for fresh data scraped minutes before battle")
    print("="*60)

    # NO LOADING OF FIGHTERS HERE - moved to loading sequence!
    # Just initialize pygame and show waiting screen
    
    pygame.init()
    
    # Use default resolution for waiting screen, will be set properly after data load
    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fight Club X")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 24)

    # These will be set after loading
    fighters = None
    all_fighters = None
    by_user = {}
    stats = {}
    initial_count = 0

    ts = datetime.datetime.now()
    slug = ts.strftime("%Y-%m-%d_%H-%M-%S")
    date_str = ts.strftime("%Y-%m-%d")

    winner_username = None

    running = True
    battle_started = False
    loading_phase = False
    battle_over = False
    target_fps = 30

    # Loading sequence phases - VIEWER-FRIENDLY!
    loading_phases = [
        "LOADING COMMUNITY DATA...",
        "CALCULATING BATTLE METRICS...",
        "OPTIMIZING ARENA LAYOUT...",
        "CREATING FIGHTER PARTICLES...",
        "INITIALIZING PHYSICS ENGINE...",
        "BATTLE READY - COMMENCING..."
    ]
    current_loading_phase = 0
    loading_progress = 0.0
    phase_duration = 60  # frames per phase (2 seconds at 30fps)
    loading_frame_count = 0

    print("🎮 Controls (STREAMER ONLY):")
    print("   SPACE = Start battle (load fresh data)")
    print("   ESC   = Exit")
    print("   E     = Emergency crown winner (during battle)")
    print("🎥 NOTE: Stream viewers will only see countdown and loading screens")
    print()

    # Pre-battle countdown/waiting screen
    while running and not battle_started:
        clock.tick(target_fps)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not loading_phase:
                    loading_phase = True
                    loading_frame_count = 0
                    current_loading_phase = 0
                    loading_progress = 0.0
                    print("\n🚀 BATTLE START INITIATED - LOADING FRESH DATA!")
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not running:
            break
        
        # Handle REAL loading sequence
        if loading_phase:
            loading_frame_count += 1
            
            # Update progress within current phase
            phase_progress = (loading_frame_count % phase_duration) / phase_duration
            loading_progress = (current_loading_phase + phase_progress) / len(loading_phases)
            
            # Move to next phase and ACTUALLY DO THE WORK
            if loading_frame_count > 0 and loading_frame_count % phase_duration == 0:
                current_loading_phase += 1
                
                # PHASE 1: Check for data file (SILENT - no viewer feedback needed)
                if current_loading_phase == 1:
                    if not os.path.exists('twitter_community_data.json'):
                        print("❌ twitter_community_data.json not found!")
                        print("💡 Please scrape community data first!")
                        loading_phase = False
                        continue
                
                # PHASE 2: Actually load the JSON data (SILENT - just do the work)
                elif current_loading_phase == 2:
                    print("📂 Loading twitter_community_data.json...")
                    fighters, WIDTH, HEIGHT = load_all_twitter_fighters()
                    if fighters is None:
                        print("❌ Failed to load fighters - aborting battle")
                        loading_phase = False
                        continue
                    
                    # Update screen resolution if needed
                    if screen.get_width() != WIDTH or screen.get_height() != HEIGHT:
                        screen = pygame.display.set_mode((WIDTH, HEIGHT))
                    
                    all_fighters = list(fighters)
                    by_user = {f.username: f for f in all_fighters}
                    initial_count = len(fighters)
                    stats = {"players": {u.username: {"kills":0,"deaths":0,"damageDealt":0.0,"damageReceived":0.0,"eliminatedAt":None} for u in fighters}}
                    
                    print(f"✅ {initial_count} fighters loaded and ready!")
                
                # PHASES 3-7: Let the existing loading logic handle these
                
                # PHASE 8: Battle ready!
                elif current_loading_phase >= len(loading_phases):
                    battle_started = True
                    loading_phase = False
                    t0 = time.time()
                    frames = []
                    
                    # Load battle floor texture
                    battle_floor = load_battle_floor(WIDTH, HEIGHT, 'battle_floor.png')
                    
                    # Dynamic sizing parameters
                    MIN_RADIUS = 4
                    MAX_RADIUS = 50
                    RESIZE_CHECK_INTERVAL = 60
                    SMOOTH_TRANSITION_SPEED = 0.3
                    frame_count = 0
                    last_alive_count = len(fighters)
                    current_radius = float(fighters[0].radius)
                    target_radius = float(fighters[0].radius)
                    
                    print(f"🎬 Recording started: {slug}.mp4 @ {target_fps}fps")
                    print(f"⚔️ BATTLE COMMENCING: {initial_count} fighters!")
                    break

            
        # [WAITING SCREEN RENDER ... unchanged]
        now_utc = datetime.datetime.utcnow()
        current_hour = now_utc.hour
        
        # Find next battle time (next 6-hour mark)
        next_battle_hours = [0, 6, 12, 18]
        next_battle_hour = None
        for hour in next_battle_hours:
            if hour > current_hour:
                next_battle_hour = hour
                break
        
        if next_battle_hour is None:  # Past 18:00, next is tomorrow at 00:00
            next_battle_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        else:
            next_battle_time = now_utc.replace(hour=next_battle_hour, minute=0, second=0, microsecond=0)
        
        # Calculate time remaining (can be negative)
        time_remaining = next_battle_time - now_utc
        total_seconds = int(time_remaining.total_seconds())
        
        is_overdue = total_seconds < 0
        if is_overdue:
            total_seconds = abs(total_seconds)
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
            
        # Draw waiting screen (backgrounds, text, countdown, etc.)
        try:
            background = pygame.image.load('background.png')
            background = pygame.transform.scale(background, (WIDTH, HEIGHT))
            screen.blit(background, (0, 0))
        except:
            try:
                for bg_file in ['background.jpg', 'background.jpeg', 'bg.png', 'bg.jpg']:
                    try:
                        background = pygame.image.load(bg_file)
                        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
                        screen.blit(background, (0, 0))
                        break
                    except:
                        continue
                else:
                    screen.fill((15, 20, 35))
                    grid_color = (40, 45, 60)
                    grid_size = 40
                    for x in range(0, WIDTH, grid_size):
                        pygame.draw.line(screen, grid_color, (x, 0), (x, HEIGHT), 1)
                    for y in range(0, HEIGHT, grid_size):
                        pygame.draw.line(screen, grid_color, (0, y), (WIDTH, y), 1)
                    for i in range(0, WIDTH + HEIGHT, grid_size * 3):
                        pygame.draw.line(screen, (25, 30, 45), (i, 0), (i - HEIGHT, HEIGHT), 1)
            except:
                screen.fill((15, 20, 35))
        
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(40)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        try:
            logo = pygame.image.load('logo.png')
            logo_size = 200
            logo = pygame.transform.scale(logo, (logo_size, logo_size))
            screen.blit(logo, (WIDTH//2 - logo_size//2, HEIGHT//2 - 320))
        except:
            logo_font = pygame.font.Font(None, 96)
            logo_text = logo_font.render("FIGHT", True, (255, 255, 255))
            logo_text2 = logo_font.render("CLUB X", True, (216, 32, 32))
            screen.blit(logo_text, (WIDTH//2 - logo_text.get_width()//2, HEIGHT//2 - 340))
            screen.blit(logo_text2, (WIDTH//2 - logo_text2.get_width()//2, HEIGHT//2 - 270))
        
        welcome_font = pygame.font.Font(None, 42)
        welcome_shadow = welcome_font.render("WELCOME TO THE ALGORITHM'S UNDERGROUND", True, (0, 0, 0))
        screen.blit(welcome_shadow, (WIDTH//2 - welcome_shadow.get_width()//2 + 2, HEIGHT//2 - 120 + 2))
        welcome = welcome_font.render("WELCOME TO THE ALGORITHM'S UNDERGROUND", True, (255, 255, 255))
        screen.blit(welcome, (WIDTH//2 - welcome.get_width()//2, HEIGHT//2 - 120))
        
        desc_font = pygame.font.Font(None, 28)
        desc_lines = [
            "This is a simulated environment where our X community",
            "fight it out to prove who's the best. Join our X to be",
            "automatically included in the next battle."
        ]
        line_height = 30
        start_y = HEIGHT//2 - 80
        for i, line in enumerate(desc_lines):
            desc_shadow = desc_font.render(line, True, (0, 0, 0))
            screen.blit(desc_shadow, (WIDTH//2 - desc_shadow.get_width()//2 + 1, start_y + (i * line_height) + 1))
            desc_text = desc_font.render(line, True, (180, 180, 180))
            screen.blit(desc_text, (WIDTH//2 - desc_text.get_width()//2, start_y + (i * line_height)))

        if loading_phase:
            # FUNCTIONAL LOADING SEQUENCE DISPLAY
            battle_rect = pygame.Rect(WIDTH//2 - 300, HEIGHT//2 + 40, 600, 160)
            pygame.draw.rect(screen, (216, 32, 32), battle_rect, 2)  # #D82020 border
            # Semi-transparent dark background for readability
            dark_bg = pygame.Surface((600, 160))
            dark_bg.set_alpha(180)
            dark_bg.fill((10, 15, 25))
            screen.blit(dark_bg, (WIDTH//2 - 300, HEIGHT//2 + 40))
            pygame.draw.rect(screen, (216, 32, 32), battle_rect, 2)  # Border on top
            
            # Loading phase text
            phase_font = pygame.font.Font(None, 32)
            if current_loading_phase < len(loading_phases):
                phase_text = loading_phases[current_loading_phase]
            else:
                phase_text = "BATTLE COMMENCING..."
                
            # Shadow
            phase_shadow = phase_font.render(phase_text, True, (0, 0, 0))
            screen.blit(phase_shadow, (WIDTH//2 - phase_shadow.get_width()//2 + 1, HEIGHT//2 + 65 + 1))
            # Main text
            phase_main = phase_font.render(phase_text, True, (255, 255, 255))
            screen.blit(phase_main, (WIDTH//2 - phase_main.get_width()//2, HEIGHT//2 + 65))
            
            # Progress bar
            bar_width = 500
            bar_height = 20
            bar_x = WIDTH//2 - bar_width//2
            bar_y = HEIGHT//2 + 110
            
            # Background bar
            pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height), border_radius=10)
            
            # Progress fill with gradient effect
            progress_width = int(loading_progress * bar_width)
            if progress_width > 0:
                for i in range(progress_width):
                    ratio = i / bar_width
                    r = int(216 * (1 - ratio * 0.3))  # Fade from red
                    g = int(32 + ratio * 100)         # Add some green
                    b = int(32 + ratio * 50)          # Add some blue
                    pygame.draw.rect(screen, (r, g, b), (bar_x + i, bar_y, 1, bar_height))
            
            # Progress bar border
            pygame.draw.rect(screen, (216, 32, 32), (bar_x, bar_y, bar_width, bar_height), width=2, border_radius=10)
            
            # Progress percentage (no technical details for viewers)
            progress_font = pygame.font.Font(None, 24)
            progress_text = f"{int(loading_progress * 100)}%"
            progress_surface = progress_font.render(progress_text, True, (255, 255, 255))
            screen.blit(progress_surface, (WIDTH//2 - progress_surface.get_width()//2, HEIGHT//2 + 140))
            
        else:
            # NORMAL COUNTDOWN DISPLAY
            battle_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 + 40, 500, 140)
            pygame.draw.rect(screen, (216, 32, 32), battle_rect, 2)  # #D82020 border
            # Semi-transparent dark background for readability
            dark_bg = pygame.Surface((500, 140))
            dark_bg.set_alpha(180)
            dark_bg.fill((10, 15, 25))
            screen.blit(dark_bg, (WIDTH//2 - 250, HEIGHT//2 + 40))
            pygame.draw.rect(screen, (216, 32, 32), battle_rect, 2)  # Border on top
            
            # Next Battle text with shadow
            countdown_font = pygame.font.Font(None, 48)
            
            # Show different text based on timing
            if is_overdue:
                title_text = "BATTLE OVERDUE:"
                title_color = (255, 100, 100)  # Light red for overdue
            else:
                title_text = "NEXT BATTLE:"
                title_color = (255, 255, 255)
            
            # Shadow
            countdown_shadow = countdown_font.render(title_text, True, (0, 0, 0))
            screen.blit(countdown_shadow, (WIDTH//2 - countdown_shadow.get_width()//2 + 1, HEIGHT//2 + 55 + 1))
            # Main text
            countdown_title = countdown_font.render(title_text, True, title_color)
            screen.blit(countdown_title, (WIDTH//2 - countdown_title.get_width()//2, HEIGHT//2 + 55))
            
            # Countdown timer with negative sign if overdue
            timer_font = pygame.font.Font(None, 72)
            timer_text = f"{'-' if is_overdue else ''}{hours:02d}:{minutes:02d}:{seconds:02d}"
            timer_color = (255, 100, 100) if is_overdue else (216, 32, 32)
            
            # Shadow
            timer_shadow = timer_font.render(timer_text, True, (0, 0, 0))
            screen.blit(timer_shadow, (WIDTH//2 - timer_shadow.get_width()//2 + 2, HEIGHT//2 + 100 + 2))
            # Main timer
            countdown_timer = timer_font.render(timer_text, True, timer_color)
            screen.blit(countdown_timer, (WIDTH//2 - countdown_timer.get_width()//2, HEIGHT//2 + 100))
        
        # Schedule info with tech styling and shadows (only show if not loading)
        if not loading_phase:
            schedule_font = pygame.font.Font(None, 28)
            # Shadow
            schedule_shadow = schedule_font.render("BATTLES COMMENCE EVERY 6 HOURS UTC", True, (0, 0, 0))
            screen.blit(schedule_shadow, (WIDTH//2 - schedule_shadow.get_width()//2 + 1, HEIGHT//2 + 220 + 1))
            # Main text
            schedule_text = schedule_font.render("BATTLES COMMENCE EVERY 6 HOURS UTC", True, (200, 200, 200))
            screen.blit(schedule_text, (WIDTH//2 - schedule_text.get_width()//2, HEIGHT//2 + 220))
            
            # Battle times with dots and shadows
            times_font = pygame.font.Font(None, 32)
            # Shadow
            times_shadow = times_font.render("00:00  •  06:00  •  12:00  •  18:00", True, (0, 0, 0))
            screen.blit(times_shadow, (WIDTH//2 - times_shadow.get_width()//2 + 1, HEIGHT//2 + 250 + 1))
            # Main text
            times_text = times_font.render("00:00  •  06:00  •  12:00  •  18:00", True, (180, 180, 180))
            screen.blit(times_text, (WIDTH//2 - times_text.get_width()//2, HEIGHT//2 + 250))
            
            # Add some subtle tech elements
            try:
                pygame.image.load('background.png')
            except:
                bracket_color = (60, 65, 80)
                bracket_size = 30
                # Top left
                pygame.draw.lines(screen, bracket_color, False, [(50, 50+bracket_size), (50, 50), (50+bracket_size, 50)], 2)
                # Top right  
                pygame.draw.lines(screen, bracket_color, False, [(WIDTH-50-bracket_size, 50), (WIDTH-50, 50), (WIDTH-50, 50+bracket_size)], 2)
                # Bottom left
                pygame.draw.lines(screen, bracket_color, False, [(50, HEIGHT-50-bracket_size), (50, HEIGHT-50), (50+bracket_size, HEIGHT-50)], 2)
                # Bottom right
                pygame.draw.lines(screen, bracket_color, False, [(WIDTH-50-bracket_size, HEIGHT-50), (WIDTH-50, HEIGHT-50), (WIDTH-50, HEIGHT-50-bracket_size)], 2)        
        pygame.display.flip()

    if not running or fighters is None:
        pygame.quit()
        return

    # Main battle loop
    while running:
        clock.tick(target_fps)
        frame_count += 1
        alive_count = sum(1 for f in fighters if f.alive)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; battle_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False; battle_over = True
                elif event.key == pygame.K_e:
                    if alive_count > 1:
                        best = max([f for f in fighters if f.alive], key=lambda x: x.hp)
                        now_secs = time.time() - t0
                        for fighter in fighters:
                            if fighter.alive and fighter != best:
                                apply_damage(fighter, fighter.hp, best.username, now_secs, stats)
                        winner_username = best.username
                    alive_count = sum(1 for f in fighters if f.alive)

        screen.blit(battle_floor, (0, 0))

        # SMOOTH DYNAMIC RESIZING LOGIC (unchanged)
        if frame_count % RESIZE_CHECK_INTERVAL == 0 or alive_count != last_alive_count:
            if alive_count > 0:
                optimal_radius = get_dynamic_radius(
                    alive_count, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS, 
                    change_radius=False, particles=None
                )
                target_radius = float(optimal_radius)
                last_alive_count = alive_count
        
        if abs(current_radius - target_radius) > 0.1:
            if current_radius < target_radius:
                current_radius = min(target_radius, current_radius + SMOOTH_TRANSITION_SPEED)
            else:
                current_radius = max(target_radius, current_radius - SMOOTH_TRANSITION_SPEED)
            alive_fighters = [f for f in fighters if f.alive]
            for fighter in alive_fighters:
                fighter.update_radius(int(current_radius))

        RADIUS = int(current_radius)
        CELL_SIZE = RADIUS * 2
        grid_width = WIDTH // CELL_SIZE + 1
        grid_height = HEIGHT // CELL_SIZE + 1

        for f in fighters: f.move()
        now_secs = time.time() - t0
        check_collisions(RADIUS, CELL_SIZE, grid_width, grid_height, fighters, now_secs, stats)
        for f in fighters: f.draw(screen)

        # HUD
        battle_time = time.time() - t0
        hud = [
            f"Fighters Remaining: {alive_count}/{initial_count}",
            f"Battle Time: {int(battle_time//60):02d}:{int(battle_time%60):02d}"
        ]
        for i, line in enumerate(hud):
            txt = small_font.render(line, True, (255, 255, 255))
            screen.blit(txt, (30, 30 + i*30))

        # Progress bar
        pw, ph = 400, 8
        px, py = WIDTH//2 - pw//2, HEIGHT - 40
        pygame.draw.rect(screen, (40,40,40), (px,py,pw,ph), border_radius=4)
        prog = 1 - (alive_count/initial_count)
        fill = int(prog * pw)
        for i in range(fill):
            ratio = i / pw
            r = int(255*ratio); g = int(255*(1 - ratio*0.5)); b = int(100*(1 - ratio))
            pygame.draw.rect(screen, (r,g,b), (px+i,py,1,ph))
        pygame.draw.rect(screen, (150,150,150), (px,py,pw,ph), width=1, border_radius=4)

        # End conditions + podium overlay
        if alive_count == 1 and not battle_over:
            winner = next(f for f in fighters if f.alive)
            winner_username = winner.username
            battle_over = True

            # Determine 2nd/3rd
            elapsed = time.time() - t0
            elim_list = []
            for u, s in stats["players"].items():
                if u == winner_username: continue
                t = s.get("eliminatedAt")
                elim_list.append((u, -1.0 if t is None else float(t)))
            elim_list.sort(key=lambda x: x[1], reverse=True)
            second = elim_list[0][0] if len(elim_list) > 0 else None
            third  = elim_list[1][0] if len(elim_list) > 1 else None

            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(220); overlay.fill((0,0,0)); screen.blit(overlay,(0,0))
            title = font.render("CHAMPION", True, (255,215,0))
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 220))

            win_img_size = 160
            if winner.image is None: winner.load_twitter_image()
            try:
                win_disp = pygame.transform.smoothscale(winner.image, (win_img_size, win_img_size))
            except:
                win_disp = pygame.Surface((win_img_size, win_img_size), pygame.SRCALPHA)
                fc = (hash(winner.username)%156+100, (hash(winner.username)>>8)%156+100, (hash(winner.username)>>16)%156+100)
                pygame.draw.circle(win_disp, fc, (win_img_size//2, win_img_size//2), win_img_size//2)
            screen.blit(win_disp, win_disp.get_rect(center=(WIDTH//2, HEIGHT//2 - 120)))

            name = font.render(f"@{winner.username}", True, (255,255,255))
            screen.blit(name, (WIDTH//2 - name.get_width()//2, HEIGHT//2 - 20))
            sub = small_font.render(f"Last survivor out of {initial_count} community members", True, (200,200,200))
            screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2 + 10))

            podium_y = HEIGHT - 200
            specs = [(-120, 50, (192,192,192), "SILVER", second, 45),
                     (   0, 80, (255,215,0),  "GOLD",   winner_username, 60),
                     ( 120, 40, (205,127,50), "BRONZE", third,  45)]

            for xoff, h, color, medal, user, img_size in specs:
                if not user: continue
                x = WIDTH//2 + xoff
                rect = pygame.Rect(x-50, podium_y - h, 100, h)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (255,255,255), rect, 3)
                mtxt = small_font.render(medal, True, color)
                screen.blit(mtxt, (x - mtxt.get_width()//2, podium_y - h - 70))
                fighter = by_user.get(user)
                if fighter is None:
                    portrait = pygame.Surface((img_size, img_size), pygame.SRCALPHA)
                    fc = (hash(user)%156+100, (hash(user)>>8)%156+100, (hash(user)>>16)%156+100)
                    pygame.draw.circle(portrait, fc, (img_size//2, img_size//2), img_size//2)
                else:
                    if fighter.image is None: fighter.load_twitter_image()
                    try:
                        portrait = pygame.transform.smoothscale(fighter.image, (img_size, img_size))
                    except:
                        portrait = pygame.Surface((img_size, img_size), pygame.SRCALPHA)
                        fc = (hash(user)%156+100, (hash(user)>>8)%156+100, (hash(user)>>16)%156+100)
                        pygame.draw.circle(portrait, fc, (img_size//2, img_size//2), img_size//2)
                pr = portrait.get_rect(center=(x, podium_y - h - 35))
                screen.blit(portrait, pr)
                uname = small_font.render(f"@{user}", True, (255,255,255))
                screen.blit(uname, (x - uname.get_width()//2, podium_y + 15))

        elif alive_count == 0 and not battle_over:
            battle_over = True
            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(200); overlay.fill((0,0,0)); screen.blit(overlay,(0,0))
            txt = font.render("TOTAL ANNIHILATION!", True, (255,100,100))
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2))

        # Record frame
        if RECORD_WIDTH != DISPLAY_WIDTH or RECORD_HEIGHT != DISPLAY_HEIGHT:
            # Scale down for recording
            scaled_surface = pygame.transform.smoothscale(screen, (RECORD_WIDTH, RECORD_HEIGHT))
            frame = pygame.surfarray.array3d(scaled_surface).transpose([1,0,2])
        else:
            # No scaling needed
            frame = pygame.surfarray.array3d(screen).transpose([1,0,2])

        frames.append(frame)
        pygame.display.flip()

        if battle_over:
            for _ in range(target_fps * 3):
                frames.append(frames[-1])
            break

        if alive_count < len(fighters) * 0.8:
            fighters = [f for f in fighters if f.alive]

    pygame.quit()

    # --------- ENCODE + EXPORT ---------
    print("\n🎬 Encoding video…")
    clip = mpy.ImageSequenceClip(frames, fps=target_fps)
    video_abs = os.path.join(MEDIA_DIR, f"{slug}.mp4")
    clip.write_videofile(
    video_abs, codec='libx264', audio=False, fps=target_fps, preset='fast',
    ffmpeg_params=['-pix_fmt','yuv420p','-crf','23','-maxrate','3M','-bufsize','6M']
    )

    # Thumbnail
    thumb_abs = os.path.join(MEDIA_DIR, f"{slug}.jpg")
    mid = max(0, min(len(frames)-1, len(frames)//3))
    _save_thumbnail_from_frame(frames[mid], thumb_abs)

    # Winner fallback (e.g., annihilation)
    if not winner_username:
        winner_username = max(stats["players"].items(), key=lambda kv: (kv[1]["kills"], kv[1]["damageDealt"]))[0]

    duration = time.time() - t0
    interactions, ranking = _build_interactions_and_ranking(stats["players"], winner_username, duration)

    stats_abs = os.path.join(DAILY_DIR, f"{slug}.json")
    with open(stats_abs, "w") as f:
        json.dump({"winner": winner_username, "interactions": interactions}, f, indent=2)

    ranking_abs = os.path.join(RANK_DIR, f"{slug}_ranking.json")
    with open(ranking_abs, "w") as f:
        json.dump(ranking, f, indent=2)

    top_kills = max((v["kills"] for v in stats["players"].values()), default=0)
    _update_manifest(slug, date_str, winner_username, video_abs, thumb_abs, stats_abs, ranking_abs, top_kills)

    print("\n✅ Complete!")
    print(f"📹 Video:   {_to_url(video_abs)}")
    print(f"🖼️ Thumb:   {_to_url(thumb_abs)}")
    print(f"📊 Stats:   {_to_url(stats_abs)}")
    print(f"🏁 Ranking: {_to_url(ranking_abs)}")
    print(f"📚 Manifest:{MANIFEST_PATH}")

if __name__ == "__main__":
    main()
