import React, { useEffect, useMemo, useState } from 'react'
import PredictionsTable from './components/PredictionsTable'
import MatchCard from './components/MatchCard'
import TeamSelector from './components/TeamSelector'

type Row = Record<string, any>

function parseCSV(text: string): Row[] {
  // Basic CSV parser that normalizes common header variants to standard keys
  const lines = text.split(/\r?\n/).filter(Boolean)
  if (lines.length === 0) return []
  const rawHeaders = lines[0].split(',').map(h => h.trim())

  // map common header variants to standard names used across the frontend
  const headerMap: Record<string, string> = {
    'home_team': 'home_team', 'home': 'home_team', 'home team': 'home_team', 'home_team ': 'home_team',
    'away_team': 'away_team', 'away': 'away_team', 'away team': 'away_team',
    'home_goals': 'home_goals', 'home goals': 'home_goals', 'home_goals ': 'home_goals',
    'away_goals': 'away_goals', 'away goals': 'away_goals',
    'date': 'date', 'match_date': 'date', 'datetime': 'date',
    'venue': 'venue', 'stadium': 'venue'
  }

  const headers = rawHeaders.map(h => {
    const key = h.toLowerCase().replace(/\s+/g, ' ').trim()
    return headerMap[key] ?? h
  })

  const rows: Row[] = []
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',')
    const obj: Row = {}
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j]] = cols[j] !== undefined ? cols[j].trim() : ''
    }
    rows.push(obj)
  }
  return rows
}

export default function App() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [teams, setTeams] = useState<string[]>([])

  useEffect(() => {
    let mounted = true
    let slugMap: any = null
    async function load() {
      // try to load slug map for logos (optional)
      try {
        const sm = await fetch('/assets/logos/slug_map.json')
        if (sm.ok) {
          slugMap = await sm.json()
          console.log('Loaded slug_map.json for logos')
        }
      } catch (err) {
        // ignore
      }
      // try API first
      try {
        const resp = await fetch('/api/predictions')
        if (resp.ok) {
          const json = await resp.json()
          if (mounted) {
            setData(json)
            setTeams(Array.from(new Set(json.flatMap((r: any) => [r.home_team, r.away_team]))))
            setLoading(false)
            return
          }
        }
      } catch (e) {
        // fallback to CSV
      }

      try {
        const csvResp = await fetch('/data/future_predictions.csv')
        const text = await csvResp.text()
        const rows = parseCSV(text)
        if (mounted) {
          setData(rows)
          const teamSet = new Set<string>()
          rows.forEach(r => { if (r.home_team) teamSet.add(r.home_team); if (r.away_team) teamSet.add(r.away_team) })
          setTeams(Array.from(teamSet))
          setLoading(false)
        }
      } catch (e) {
        console.error('Failed to load prediction data', e)
        setLoading(false)
      }
    }
    load()
    return () => { mounted = false }
  }, [])

  const matchweek = useMemo(() => {
    // pick next 8 matches for a simple home screen
    return data.slice(0, 8)
  }, [data])

  return (
    <div className="app-root">
      <header className="header">
        <h1>LaLiga Predictor</h1>
        <p className="subtitle">Matchweek predictions • quick search • team vs team</p>
      </header>

      <main className="container">
        <section className="matchweek">
          <h2>Matchweek Preview</h2>
          {loading ? <div>Loading...</div> : (
            <div className="match-grid">
              {matchweek.map((m, i) => (
                <MatchCard key={i} row={m} />
              ))}
            </div>
          )}
        </section>

        <section className="team-vs">
          <h2>Team vs Team — Quick Predict</h2>
          <TeamSelector teams={teams} />
        </section>

        <section className="table-section">
          <h2>All Predictions</h2>
          <PredictionsTable data={data} />
        </section>
      </main>

      <footer className="footer">Built with ❤️ — use the CSV or API endpoint `/api/predictions`</footer>
    </div>
  )
}
