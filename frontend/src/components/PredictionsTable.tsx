import React, { useMemo, useState } from 'react'

export default function PredictionsTable({ data }:{ data: Record<string, any>[] }){
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)
  const pageSize = 12

  const filtered = useMemo(()=>{
    const q = query.trim().toLowerCase()
    if (!q) return data
    return data.filter(r => (
      (r.home_team && r.home_team.toLowerCase().includes(q)) ||
      (r.away_team && r.away_team.toLowerCase().includes(q)) ||
      (r.date && r.date.toLowerCase().includes(q))
    ))
  }, [data, query])

  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const pageRows = filtered.slice(page * pageSize, (page+1)*pageSize)

  return (
    <div className="pred-table">
      <div className="toolbar">
        <input placeholder="Search team or date" value={query} onChange={(e)=>setQuery(e.target.value)} />
      </div>

      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Home</th>
            <th>Score</th>
            <th>Away</th>
            <th>Stadium</th>
          </tr>
        </thead>
        <tbody>
          {pageRows.map((r,i)=> (
            <tr key={i}>
              <td>{r.date}</td>
              <td>{r.home_team}</td>
              <td>{r.pred_home_goals ?? '-'} - {r.pred_away_goals ?? '-'}</td>
              <td>{r.away_team}</td>
              <td>{r.venue ?? r.stadium ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="pagination">
        <button onClick={()=>setPage(Math.max(0,page-1))} disabled={page===0}>Prev</button>
        <span>Page {page+1} / {pageCount}</span>
        <button onClick={()=>setPage(Math.min(pageCount-1,page+1))} disabled={page===pageCount-1}>Next</button>
      </div>
    </div>
  )
}
