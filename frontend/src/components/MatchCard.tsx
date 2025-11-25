import React from 'react'

function slugify(name: string) {
  return name ? name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9\-]/g, '') : ''
}

export default function MatchCard({ row }: { row: Record<string, any> }) {
  const home = row.home_team || 'Home'
  const away = row.away_team || 'Away'
  const stadium = row.venue || row.stadium || ''
  const homeLogo = `/assets/logos/${slugify(home)}.svg`
  const awayLogo = `/assets/logos/${slugify(away)}.svg`
  const defaultLogo = '/assets/logos/default.svg'

  return (
    <div className="match-card">
      <div className="teams">
        <div className="team">
          <img src={homeLogo} onError={(e:any)=>{e.currentTarget.onerror=null; e.currentTarget.src=defaultLogo}} alt={home} className="team-logo" />
          <div className="team-name">{home}</div>
        </div>

        <div className="score">
          <div className="pred">{row.pred_home_goals ?? '-'} - {row.pred_away_goals ?? '-'}</div>
          <div className="result">{row.pred_result ?? ''}</div>
        </div>

        <div className="team">
          <img src={awayLogo} onError={(e:any)=>{e.currentTarget.onerror=null; e.currentTarget.src=defaultLogo}} alt={away} className="team-logo" />
          <div className="team-name">{away}</div>
        </div>
      </div>
      <div className="meta">{stadium}</div>
    </div>
  )
}
