"""
La Liga Match Results Scraper from FBref
Focuses on basic match data: teams, scores, winners
"""

import os
import time
import pandas as pd
import random
import requests
import re
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== Constants ==========
SEASONS = list(range(2000, 2025))  # Past 10 seasons (2015-16 to 2024-25)
BASE_URL = "https://fbref.com"
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_fixtures_url(season):
    """Get the URL for La Liga fixtures for a specific season"""
    if season == 2024:
        return "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures"
    else:
        season_str = f"{season}-{season+1}"
        return f"https://fbref.com/en/comps/12/{season_str}/schedule/{season_str}-La-Liga-Scores-and-Fixtures"

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session

def scrape_with_requests(url, session=None, max_retries=3):
    if session is None:
        session = create_session()
        
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: {url}")
            time.sleep(random.uniform(2, 4))  # Reduced delay
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                logger.warning(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(random.uniform(3, 6))
            
    return None

def find_fixtures_table(soup):
    """Find the fixtures table in the page"""
    logger.info("Looking for fixtures table...")
    
    # Try to find by ID first (FBref uses specific IDs for their tables)
    table = soup.find('table', id=lambda x: x and 'sched' in x.lower())
    if table:
        logger.info("Found fixtures table by ID")
        return table
        
    # Try to find by class or other attributes
    for table in soup.find_all('table'):
        if table.has_attr('class') and any('sched' in c.lower() for c in table['class']):
            logger.info("Found fixtures table by class")
            return table
            
    # Fallback: find any table with match data
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if len(rows) > 5:  # Reasonable number of rows
            first_row = rows[0]
            headers = [th.get_text(strip=True).lower() for th in first_row.find_all(['th', 'td'])]
            if 'date' in headers and any(x in headers for x in ['home', 'away']) and 'score' in headers:
                logger.info("Found fixtures table by header content")
                return table
                
    logger.warning("Could not find fixtures table")
    return None

def extract_team_name(cell):
    """Extract team name from a table cell"""
    if not cell:
        return ''
        
    link = cell.find('a')
    if link:
        return link.get_text(strip=True)
        
    return cell.get_text(strip=True)

def parse_date(date_str):
    """Parse date string into consistent format"""
    if not date_str:
        return ''
        
    date_str = date_str.strip()
    
    # Try to extract date from various formats
    try:
        # Handle month names in different languages
        month_replacements = {
            'ene': 'jan', 'feb': 'feb', 'mar': 'mar', 'abr': 'apr', 'may': 'may', 'jun': 'jun',
            'jul': 'jul', 'ago': 'aug', 'sep': 'sep', 'oct': 'oct', 'nov': 'nov', 'dic': 'dec'
        }
        
        for esp, eng in month_replacements.items():
            date_str = date_str.lower().replace(esp, eng)
            
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d %b %Y', '%b %d, %Y', '%d %B %Y', '%B %d, %Y'):
            try:
                parsed_date = pd.to_datetime(date_str, format=fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                continue
                
    except:
        pass
        
    return date_str

def parse_score(score_str):
    """Parse score string into home and away goals"""
    if not score_str:
        return None, None
        
    # Replace various dash characters with standard hyphen
    score_str = score_str.replace('−', '-').replace('–', '-').replace(':', '-').replace('‒', '-').replace('—', '-')
    
    # Extract numbers from score
    score_parts = re.findall(r'\d+', score_str)
    if len(score_parts) >= 2:
        try:
            home_goals = int(score_parts[0])
            away_goals = int(score_parts[1])
            return home_goals, away_goals
        except:
            return None, None
            
    return None, None

def determine_result(home_goals, away_goals):
    """Determine match result based on score"""
    if home_goals is None or away_goals is None:
        return 'Unknown'
        
    if home_goals > away_goals:
        return 'Home Win'
    elif away_goals > home_goals:
        return 'Away Win'
    else:
        return 'Draw'

def scrape_season_fixtures(season, session=None):
    """Scrape all fixtures for a single La Liga season"""
    logger.info(f"Scraping fixtures for season {season}-{season+1}")
    
    if session is None:
        session = create_session()
        
    fixtures_url = get_fixtures_url(season)
    logger.info(f"Fetching fixtures from: {fixtures_url}")
    
    soup = scrape_with_requests(fixtures_url, session)
    if not soup:
        logger.error("Could not fetch fixtures page")
        return []
        
    fixtures_table = find_fixtures_table(soup)
    if not fixtures_table:
        logger.error("Could not find fixtures table")
        return []
        
    matches = []
    rows = fixtures_table.find_all('tr')
    
    if not rows:
        logger.error("No rows found in fixtures table")
        return []
        
    # Identify column indices
    headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
    logger.info(f"Table headers: {headers}")
    
    col_map = {}
    for i, header in enumerate(headers):
        if 'date' in header:
            col_map['date'] = i
        elif 'time' in header:
            col_map['time'] = i
        elif 'home' in header:
            col_map['home'] = i
        elif 'away' in header or 'visitor' in header:
            col_map['away'] = i
        elif 'score' in header or 'result' in header:
            col_map['score'] = i
        elif 'attendance' in header:
            col_map['attendance'] = i
        elif 'venue' in header:
            col_map['venue'] = i
            
    # If we couldn't identify key columns, make educated guesses
    if 'home' not in col_map:
        for i, header in enumerate(headers):
            if header in ['squad', 'team'] and 'home' not in col_map:
                col_map['home'] = i
            elif header in ['squad', 'team'] and 'home' in col_map:
                col_map['away'] = i
                
    # Process each row
    for row_idx, row in enumerate(rows[1:], 1):
        cells = row.find_all(['td', 'th'])
        if len(cells) < max(col_map.values()) + 1:
            continue
            
        try:
            # Get the column indices based on our mapping
            date_idx = col_map.get('date', 0)
            time_idx = col_map.get('time', 1)
            home_idx = col_map.get('home', 2)
            away_idx = col_map.get('away', 3)
            score_idx = col_map.get('score', 4)
            venue_idx = col_map.get('venue', 5)
            attendance_idx = col_map.get('attendance', 6)
            
            match_data = {
                'season': f"{season}-{season+1}",
                'date': parse_date(cells[date_idx].get_text(strip=True)) if date_idx < len(cells) else '',
                'time': cells[time_idx].get_text(strip=True) if time_idx < len(cells) else '',
                'home_team': extract_team_name(cells[home_idx]) if home_idx < len(cells) else '',
                'away_team': extract_team_name(cells[away_idx]) if away_idx < len(cells) else '',
                'score': cells[score_idx].get_text(strip=True) if score_idx < len(cells) else '',
                'venue': cells[venue_idx].get_text(strip=True) if venue_idx < len(cells) else '',
                'attendance': cells[attendance_idx].get_text(strip=True) if attendance_idx < len(cells) else ''
            }
            
            # Parse score and determine result
            home_goals, away_goals = parse_score(match_data['score'])
            match_data['home_goals'] = home_goals
            match_data['away_goals'] = away_goals
            match_data['result'] = determine_result(home_goals, away_goals)
            
            # Skip invalid rows
            if (not match_data['home_team'] or not match_data['away_team'] or 
                len(match_data['home_team']) < 2 or len(match_data['away_team']) < 2):
                continue
                
            if (match_data['home_team'].lower() in ['home', 'team', 'date'] or 
                match_data['away_team'].lower() in ['away', 'team', 'visitor']):
                continue
                
            matches.append(match_data)
            
            if len(matches) <= 3:
                logger.info(f"Match {len(matches)}: {match_data['home_team']} vs {match_data['away_team']} ({match_data['score']})")
                
        except Exception as e:
            if row_idx <= 10:
                logger.warning(f"Error parsing row {row_idx}: {e}")
            continue
            
    logger.info(f"Found {len(matches)} matches for season {season}-{season+1}")
    return matches

def main():
    """Main function to scrape La Liga data for the past 10 seasons"""
    logger.info("Starting La Liga Match Scraper...")
    logger.info(f"Seasons to scrape: {SEASONS}")
    
    all_matches = []
    session = create_session()

    # Scrape each season
    for i, season in enumerate(SEASONS):
        logger.info(f"Processing season {season}-{season+1} ({i+1}/{len(SEASONS)})")
        
        if i > 0:
            delay = random.uniform(3, 8)
            logger.info(f"Waiting {delay:.1f} seconds between seasons...")
            time.sleep(delay)
            
        try:
            season_matches = scrape_season_fixtures(season, session)
            all_matches.extend(season_matches)
            logger.info(f"Completed season {season}-{season+1}, total matches: {len(all_matches)}")
        except Exception as e:
            logger.error(f"Error scraping season {season}: {e}")
            continue
            
    if not all_matches:
        logger.error("No matches were collected")
        return
        
    # Save data to CSV
    logger.info(f"Saving data for {len(all_matches)} matches...")
    df = pd.DataFrame(all_matches)
    
    # Reorder columns for better readability
    columns = ['season', 'date', 'time', 'home_team', 'away_team', 'home_goals', 'away_goals', 
               'score', 'result', 'venue', 'attendance']
    df = df[[col for col in columns if col in df.columns]]
    
    csv_path = os.path.join(DATA_DIR, 'la_liga_results_10_years.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Data saved to: {csv_path}")
    
    # Print summary
    logger.info("\n=== SCRAPING SUMMARY ===")
    logger.info(f"Total matches: {len(df)}")
    
    if len(df) > 0:
        # Results distribution
        result_counts = df['result'].value_counts()
        for result, count in result_counts.items():
            logger.info(f"{result}: {count} matches ({count/len(df)*100:.1f}%)")
            
        # Latest season in data
        latest_season = df['season'].max()
        latest_season_matches = df[df['season'] == latest_season]
        logger.info(f"\nLatest season ({latest_season}): {len(latest_season_matches)} matches")
        
        # Show a few sample matches
        logger.info("\nSample matches:")
        for i, row in df.head(3).iterrows():
            logger.info(f"{row['date']}: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} ({row['result']})")

if __name__ == "__main__":
    main()
