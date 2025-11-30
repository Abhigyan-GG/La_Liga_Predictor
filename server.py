from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import subprocess
import os
import json
from typing import Optional
import requests
from bs4 import BeautifulSoup
import threading
import time
import pickle
from utils.feature_engineer import FeatureEngineer

app = FastAPI(title="LaLiga Predictor Server")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories: frontend and data

# Mount lowercase `frontend` if present (legacy)
if os.path.isdir('frontend'):
    app.mount('/', StaticFiles(directory='frontend', html=True), name='frontend')

# Mount the `Frontend` directory (capitalized) used in this workspace
if os.path.isdir('Frontend'):
    app.mount('/', StaticFiles(directory='Frontend', html=True), name='Frontend')

# Assets: prefer `Frontend/public/assets` if present
if os.path.isdir('Frontend/public/assets'):
    app.mount('/assets', StaticFiles(directory='Frontend/public/assets'), name='assets')
elif os.path.isdir('frontend/public/assets'):
    app.mount('/assets', StaticFiles(directory='frontend/public/assets'), name='assets')

if os.path.isdir('data'):
    app.mount('/data', StaticFiles(directory='data'), name='data')


def read_predictions_csv(path='data/future_predictions.csv'):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # convert to records and replace NaN
        records = df.fillna('').to_dict(orient='records')
        return records
    except Exception:
        return []

def read_cached_standings(season: str):
    path = f'data/standings_{season}.json'
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_standings(season: str, data):
    path = f'data/standings_{season}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_fbref_standings_html(html: str):
    # FBref often hides tables inside HTML comments; extract commented HTML first
    # Try to find commented tables and append to the soup
    soup = BeautifulSoup(html, 'html.parser')
    # search for commented blocks that contain <table
    import re
    comments = re.findall(r'<!--([\s\S]*?)-->', html)
    for c in comments:
        if '<table' in c:
            try:
                soup2 = BeautifulSoup(c, 'html.parser')
                for t in soup2.find_all('table'):
                    # insert into main soup so it can be found
                    soup.append(t)
            except Exception:
                pass
    # find a table that looks like a standings table
    tables = soup.find_all('table')
    for table in tables:
        thead = table.find('thead')
        if not thead:
            continue
        headers = [th.get_text(strip=True).lower() for th in thead.find_all('th')]
        # look for common columns
        if any(h in headers for h in ('pos', 'rank', 'w', 'wins')) and ('pts' in ' '.join(headers) or 'points' in ' '.join(headers)):
            # parse rows
            rows = []
            tbody = table.find('tbody')
            if not tbody:
                continue
            for tr in tbody.find_all('tr'):
                cols = [td.get_text(strip=True) for td in tr.find_all(['th','td'])]
                if not cols:
                    continue
                # heuristic mapping
                team = None
                pts = None
                played = None
                w = d = l = gf = ga = None
                # try to find team name and points
                team_cell = tr.find('td', {'data-stat':'team'}) or tr.find('th', {'scope':'row'})
                if team_cell:
                    team = team_cell.get_text(strip=True)
                def get_stat(tg, stat):
                    el = tg.find('td', {'data-stat': stat})
                    return el.get_text(strip=True) if el else None
                pts = get_stat(tr, 'points') or get_stat(tr, 'pts')
                played = get_stat(tr, 'games') or get_stat(tr, 'matches') or get_stat(tr, 'games_played')
                w = get_stat(tr, 'wins')
                d = get_stat(tr, 'draws')
                l = get_stat(tr, 'losses')
                gf = get_stat(tr, 'goals_for')
                ga = get_stat(tr, 'goals_against')

                row = {
                    'team': team or (cols[1] if len(cols) > 1 else ''),
                    'played': int(played) if played and played.isdigit() else (int(cols[2]) if len(cols) > 2 and cols[2].isdigit() else None),
                    'wins': int(w) if w and w.isdigit() else None,
                    'draws': int(d) if d and d.isdigit() else None,
                    'losses': int(l) if l and l.isdigit() else None,
                    'goals_for': int(gf) if gf and gf.isdigit() else None,
                    'goals_against': int(ga) if ga and ga.isdigit() else None,
                    'points': int(pts) if pts and pts.isdigit() else (int(cols[-1]) if cols and cols[-1].isdigit() else None)
                }
                rows.append(row)
            if rows:
                return rows
    return None


# --- Model loading and inference ---
MODELS = {
    'goal_model': None,
    'result_model': None,
}

def load_models():
    base = 'models'
    goal_path = os.path.join(base, 'goal_model.pkl')
    result_path = os.path.join(base, 'result_model.pkl')
    if os.path.exists(goal_path):
        try:
            with open(goal_path, 'rb') as f:
                MODELS['goal_model'] = pickle.load(f)
        except Exception:
            MODELS['goal_model'] = None
    if os.path.exists(result_path):
        try:
            with open(result_path, 'rb') as f:
                MODELS['result_model'] = pickle.load(f)
        except Exception:
            MODELS['result_model'] = None


load_models()

# --- Simple scheduler for periodic standings scraping ---
_scheduler = {
    'thread': None,
    'stop_event': None,
    'running': False,
    'config': None
}

def _scheduler_worker(url, season, interval_minutes, stop_event):
    while not stop_event.is_set():
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                standings = parse_fbref_standings_html(resp.text)
                if standings:
                    cache_standings(season, standings)
        except Exception:
            pass
        # sleep with early exit
        for _ in range(int(interval_minutes * 60)):
            if stop_event.is_set():
                break
            time.sleep(1)


@app.post('/scheduler/start')
def scheduler_start(payload: dict = Body(...)):
    """Start periodic scraping. JSON: {"url":"<fbref_url>", "season":"2024-2025", "interval_minutes": 60}
    """
    url = payload.get('url')
    season = payload.get('season')
    interval = payload.get('interval_minutes', 60)
    if not url or not season:
        raise HTTPException(status_code=400, detail='url and season are required')
    if _scheduler['running']:
        raise HTTPException(status_code=400, detail='Scheduler already running')
    stop_event = threading.Event()
    t = threading.Thread(target=_scheduler_worker, args=(url, season, interval, stop_event), daemon=True)
    _scheduler['thread'] = t
    _scheduler['stop_event'] = stop_event
    _scheduler['running'] = True
    _scheduler['config'] = {'url': url, 'season': season, 'interval_minutes': interval}
    t.start()
    return JSONResponse({'ok': True, 'started': True, 'config': _scheduler['config']})


@app.post('/scheduler/stop')
def scheduler_stop():
    if not _scheduler['running']:
        return JSONResponse({'ok': True, 'running': False})
    _scheduler['stop_event'].set()
    _scheduler['thread'].join(timeout=5)
    _scheduler['thread'] = None
    _scheduler['stop_event'] = None
    _scheduler['running'] = False
    cfg = _scheduler['config']
    _scheduler['config'] = None
    return JSONResponse({'ok': True, 'stopped': True, 'previous_config': cfg})


@app.get('/scheduler/status')
def scheduler_status():
    return JSONResponse({'running': _scheduler['running'], 'config': _scheduler['config']})





@app.get('/api/predictions')
def api_predictions():
    records = read_predictions_csv()
    return JSONResponse(records)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str] = None


@app.post('/api/predict')
def api_predict(req: PredictRequest):
    # Try to find matching row in future_predictions.csv
    records = read_predictions_csv()
    if not records:
        raise HTTPException(status_code=404, detail='No predictions CSV found')

    for r in records:
        # match by team names (case-insensitive)
        if r.get('home_team','').strip().lower() == req.home_team.strip().lower() and r.get('away_team','').strip().lower() == req.away_team.strip().lower():
            # optional date check
            if req.date and r.get('date') and str(r.get('date')).strip() != str(req.date).strip():
                continue
            return JSONResponse(r)

    # Not found; return helpful message
    # If models are available, attempt to perform inference using historical data
    if MODELS.get('goal_model') is None and MODELS.get('result_model') is None:
        raise HTTPException(status_code=404, detail='Prediction not found in CSV; no models available. Try running /api/scrape or add the fixture to data/future_predictions.csv')

    # Build a temporary dataframe from historical matches and append the requested fixture
    laledata_path = os.path.join('data', 'LaLiga.csv')
    if not os.path.exists(laledata_path):
        raise HTTPException(status_code=500, detail='Historical dataset data/LaLiga.csv not found; cannot run model inference')

    try:
        df_hist = pd.read_csv(laledata_path)
        # ensure date column
        df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
        # create a new row for prediction
        new_row = {
            'season': req.date.split('-')[0] if req.date else df_hist['season'].iloc[-1] if 'season' in df_hist.columns else None,
            'date': req.date,
            'home_team': req.home_team,
            'away_team': req.away_team,
            'home_goals': None,
            'away_goals': None,
            'result': None
        }
        df = df_hist.copy()
        df = df.append(new_row, ignore_index=True)

        # Run feature engineering on the combined dataset and take the last row features
        fe = FeatureEngineer(df)
        X_all, _, _, _ = fe.create_advanced_features()
        if X_all is None or len(X_all) == 0:
            raise HTTPException(status_code=500, detail='Feature engineering failed')
        X_last = X_all[-1].reshape(1, -1)

        # Predict goals
        goal_model = MODELS.get('goal_model')
        result_model = MODELS.get('result_model')
        preds = {}
        if goal_model is not None:
            try:
                gpred = goal_model.predict(X_last)
                # handle multioutput regressors
                if hasattr(gpred, 'shape') and gpred.shape[1] >= 2:
                    ph, pa = float(gpred[0][0]), float(gpred[0][1])
                else:
                    # if single output, assume predicts home goals; fallback
                    ph = float(gpred[0][0]) if hasattr(gpred[0], '__len__') else float(gpred[0])
                    pa = None
                preds['pred_home_goals'] = ph
                preds['pred_away_goals'] = pa
            except Exception as e:
                preds['goal_error'] = str(e)

        if result_model is not None:
            try:
                rpred = result_model.predict(X_last)
                # normalize to -1/0/1 if necessary
                rp = rpred[0]
                preds['pred_result'] = float(rp)
            except Exception as e:
                preds['result_error'] = str(e)

        return JSONResponse(preds)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/scrape')
def api_scrape():
    """Run project scraper (if present) and regenerate slug_map.json"""
    # Try to run Scraper/Scraper.py or Scraper.py inside Scraper
    scraper_paths = [os.path.join('Scraper','Scraper.py'), os.path.join('Scraper','scraper.py'), 'Scraper.py']
    ran = False
    for p in scraper_paths:
        if os.path.exists(p):
            try:
                subprocess.check_call(['python', p])
                ran = True
            except subprocess.CalledProcessError as e:
                return JSONResponse({'ok': False, 'error': str(e)})

    # Run data processor to preprocess if available
    processor_path = os.path.join('utils','data_processor.py')
    if os.path.exists(processor_path):
        try:
            # run a small script to preprocess
            subprocess.check_call(['python', '-c', "from utils.data_processor import DataProcessor; p=DataProcessor('data/LaLiga.csv'); p.load_data(); p.preprocess_data(); p.save_processed_data('data/LaLiga_processed.csv')"])
        except subprocess.CalledProcessError:
            pass

    # regenerate slug_map.json
    gen = os.path.join('tools','generate_logo_slugs.py')
    if os.path.exists(gen):
        try:
            subprocess.check_call(['python', gen])
        except subprocess.CalledProcessError:
            pass

    return JSONResponse({'ok': True, 'ran_scraper': ran})


@app.post('/api/upload_logo')
def upload_logo(file: UploadFile = File(...), slug: str = ''):
    """Upload a logo file for a team. Saves to frontend/public/assets/logos/<slug>.<ext>"""
    if not slug:
        raise HTTPException(status_code=400, detail='slug query param is required')
    dest_dir = os.path.join('frontend','public','assets','logos')
    os.makedirs(dest_dir, exist_ok=True)
    filename = file.filename
    ext = os.path.splitext(filename)[1] or '.svg'
    dest = os.path.join(dest_dir, f"{slug}{ext}")
    with open(dest, 'wb') as f:
        f.write(file.file.read())
    return JSONResponse({'ok': True, 'path': dest})


@app.get('/api/standings')
def api_get_standings(season: Optional[str] = None):
    """Return cached standings for a season if available. Query param: ?season=2024-2025"""
    if not season:
        raise HTTPException(status_code=400, detail='season query parameter is required (e.g. 2024-2025)')
    data = read_cached_standings(season)
    if data is None:
        raise HTTPException(status_code=404, detail='No cached standings for that season. POST /api/standings/scrape with a FBref URL to scrape and cache.')
    return JSONResponse(data)


@app.post('/api/standings/scrape')
def api_scrape_standings(payload: dict = Body(...)):
    """Scrape standings from a provided FBref URL and cache them. JSON body: { "url": "<fbref_url>", "season": "2024-2025" }
    If scraping succeeds the standings are cached to `data/standings_<season>.json` and returned.
    """
    url = payload.get('url')
    season = payload.get('season')
    if not url or not season:
        raise HTTPException(status_code=400, detail='Both "url" and "season" are required in the JSON body')

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f'Failed to fetch URL: status {resp.status_code}')
        standings = parse_fbref_standings_html(resp.text)
        if not standings:
            raise HTTPException(status_code=500, detail='Failed to parse standings from the provided page')
        cache_standings(season, standings)
        return JSONResponse({'ok': True, 'season': season, 'standings': standings})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/leaderboard')
def leaderboard_page():
    fp = os.path.join('frontend', 'Leaderboard.html')
    if os.path.exists(fp):
        return FileResponse(fp, media_type='text/html')
    raise HTTPException(status_code=404, detail='Leaderboard.html not found in frontend')


@app.get('/standings_page')
def standings_page():
    fp = os.path.join('frontend', 'Standings.html')
    if os.path.exists(fp):
        return FileResponse(fp, media_type='text/html')
    raise HTTPException(status_code=404, detail='Standings.html not found in frontend')


@app.get('/match_detail')
def match_detail_page():
    fp = os.path.join('frontend', 'Match_detail.html')
    if os.path.exists(fp):
        return FileResponse(fp, media_type='text/html')
    raise HTTPException(status_code=404, detail='Match_detail.html not found in frontend')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
