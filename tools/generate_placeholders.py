import json
import os

LOGO_DIR = 'frontend/public/assets/logos'
SLUG_MAP = os.path.join(LOGO_DIR, 'slug_map.json')

def make_svg(initials, color):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
  <rect width="100%" height="100%" fill="{color}"/>
  <text x="50%" y="55%" font-size="40" text-anchor="middle" fill="#fff" font-family="Arial" dy=".35em">{initials}</text>
</svg>'''

def slug_to_initials(name: str) -> str:
    parts = name.replace('-', ' ').split()
    if len(parts) == 0:
        return 'X'
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()

def main():
    if not os.path.exists(SLUG_MAP):
        print('slug_map.json not found; run tools/generate_logo_slugs.py first')
        return

    with open(SLUG_MAP, encoding='utf-8') as f:
        data = json.load(f)

    teams = data.get('teams', {})
    os.makedirs(LOGO_DIR, exist_ok=True)

    colors = ['#e63946','#457b9d','#2a9d8f','#ffb703','#8d99ae','#6a4c93']
    i = 0
    for team, slug in teams.items():
        initials = slug_to_initials(slug)
        color = colors[i % len(colors)]
        svg = make_svg(initials, color)
        out = os.path.join(LOGO_DIR, f"{slug}.svg")
        if not os.path.exists(out):
            with open(out, 'w', encoding='utf-8') as of:
                of.write(svg)
        i += 1

    print(f'Wrote placeholder SVGs for {len(teams)} teams into {LOGO_DIR}')

if __name__ == '__main__':
    main()
