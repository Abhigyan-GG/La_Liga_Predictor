import csv
import json
import os

INPUT = 'data/LaLiga.csv'
OUT = 'frontend/public/assets/logos/slug_map.json'


def slugify(name: str) -> str:
    return ''.join(c for c in name.lower().replace(' ', '-') if c.isalnum() or c == '-')


def main():
    if not os.path.exists(INPUT):
        print(f"Input file {INPUT} not found. Run this from repo root.")
        return

    teams = {}
    stadiums = {}
    with open(INPUT, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            home = r.get('home_team') or r.get('Home')
            away = r.get('away_team') or r.get('Away')
            venue = r.get('venue') or r.get('stadium') or ''
            if home:
                teams[home] = slugify(home)
            if away:
                teams[away] = slugify(away)
            if venue:
                stadiums[venue] = slugify(venue)

    out = {
        'teams': teams,
        'stadiums': stadiums
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'Wrote {OUT} with {len(teams)} teams and {len(stadiums)} stadiums')


if __name__ == '__main__':
    main()
