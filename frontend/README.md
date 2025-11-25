LaLiga Predictor - Frontend

Minimal React + TypeScript frontend scaffold (Vite-style).

Quick start (dev machine)

1. From repo root install dependencies for frontend (optional; this scaffold assumes you have Node installed):

```bash
cd frontend
npm install
npm run dev
```

2. The frontend expects either:
- A backend endpoint at `/api/predictions` that returns JSON array of predictions; or
- A CSV file at `/data/future_predictions.csv` served from repo root. We ship a CSV in `data/`.

3. Add team logos to `frontend/public/assets/logos/` using slugified filenames (lowercase, spaces -> `-`, remove special chars), e.g. `real-madrid.svg`, `barcelona.svg`. A helper script is provided to generate slug map:

```bash
python tools/generate_logo_slugs.py
# writes frontend/public/assets/logos/slug_map.json
```

4. For a quick demo, keep logos minimal: `default.svg` is used as fallback.

Uploading logos (how-to)
------------------------

- Preferred formats: SVG (recommended) or PNG. SVG scales cleanly and keeps the theme sharp.
- File naming: use the slug value from `slug_map.json` or follow this rule: lowercase, spaces -> `-`, remove punctuation. Example: `Real Madrid` -> `real-madrid.svg`.
- Location: place files in `frontend/public/assets/logos/`.
- Example steps (Windows cmd):

```powershell
# from repo root
copy C:\path\to\logo-files\real-madrid.svg frontend\public\assets\logos\real-madrid.svg
copy C:\path\to\logo-files\barcelona.png frontend\public\assets\logos\barcelona.png
```

- After adding new logos, restart the frontend dev server (`npm run dev`) so Vite picks up the static files.
- If a logo file is missing for a team, the app will use `default.svg` as fallback.

Getting logos quickly
---------------------
- Option 1: Use Wikipedia or club official sites and download the SVG/PNG (check license).
- Option 2: Use a logo CDN (if permissible) and download files to `frontend/public/assets/logos/`.
- Option 3: For demo purposes you can create simple placeholder SVGs (a colored circle plus initials). I can generate placeholders for the top 10 teams if you want.

Files added
- `frontend/` — Vite-style frontend scaffold
- `frontend/src/` — React entry and components
- `frontend/public/assets/logos/default.svg` — fallback logo
- `tools/generate_logo_slugs.py` — generates slug_map.json from `data/LaLiga.csv`

Notes
- This is a lightweight scaffold to help demo the UI. You can expand styling and add bundler config as needed.
