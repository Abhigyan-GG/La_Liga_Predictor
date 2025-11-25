import React, { useState } from 'react'

export default function TeamSelector({ teams }: { teams: string[] }) {
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [result, setResult] = useState<any>(null)

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault()
    setResult('Loading...')
    // Try backend predict endpoint first
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: home, away_team: away, date: new Date().toISOString() })
      })
      if (resp.ok) {
        const json = await resp.json()
        setResult(json)
        return
      }
    } catch (err) {
      // fall back to client-side search
    }

    // Fallback: show message — frontend expects CSV to be present and searches it
    setResult({ message: 'No backend predict available. Use CSV upload or API.' })
  }

  return (
    <div className="team-selector">
      <form onSubmit={handlePredict}>
        <div className="selects">
          <select value={home} onChange={(e)=>setHome(e.target.value)}>
            <option value="">Select Home</option>
            {teams.map((t,i)=> <option key={i} value={t}>{t}</option>)}
          </select>

          <span className="vs">vs</span>

          <select value={away} onChange={(e)=>setAway(e.target.value)}>
            <option value="">Select Away</option>
            {teams.map((t,i)=> <option key={i} value={t}>{t}</option>)}
          </select>

          <button type="submit" disabled={!home || !away}>Predict</button>
        </div>
      </form>

      <div className="team-selector-result">
        {result ? (
          typeof result === 'string' ? <div>{result}</div> : (
            <pre style={{whiteSpace: 'pre-wrap'}}>{JSON.stringify(result, null, 2)}</pre>
          )
        ) : <div className="muted">Select two teams and hit Predict</div>}
      </div>
    </div>
  )
}
