import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, ErrorState, Loading, RiskBadge } from "../components/ui";

const display = (value: unknown) => value == null || value === "" ? "—" : String(value);

export default function WorkspaceCustomers() {
  const { runId = "" } = useParams();
  const [result, setResult] = useState<any>(null);
  const [search, setSearch] = useState("");
  const [risk, setRisk] = useState("");
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    api.workspaceCustomers(runId, { page, pageSize: 25, search, risk })
      .then((data) => { if (!cancelled) setResult(data); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [runId, page, search, risk]);

  const updateSearch = (value: string) => {
    setPage(1);
    setSearch(value);
  };
  const updateRisk = (value: string) => {
    setPage(1);
    setRisk(value);
  };

  if (error) return <ErrorState message={error} />;
  if (!result) return <Loading label="Loading all uploaded customer rows..." />;
  return <div className="space-y-5 max-w-[1600px]">
    <Link to="/workspace" className="text-sm text-brand-700 hover:underline">← Back to My Workspace</Link>
    <div>
      <h1 className="text-xl font-semibold">Uploaded customer explorer</h1>
      <p className="text-sm text-slate-400 mt-1">{result.total.toLocaleString()} matching rows · click any customer to inspect model drivers</p>
    </div>
    <Card className="p-4">
      <div className="flex flex-wrap gap-3 items-center">
        <input value={search} onChange={(e) => updateSearch(e.target.value)} placeholder="Search uploaded rows..." className="border border-slate-200 rounded-lg px-3 py-2 text-sm w-72" />
        <select value={risk} onChange={(e) => updateRisk(e.target.value)} className="border border-slate-200 rounded-lg px-3 py-2 text-sm">
          <option value="">All risk levels</option>
          <option>Low</option><option>Medium</option><option>High</option><option>Critical</option>
        </select>
      </div>
    </Card>
    <Card className="p-1">
      <CardHeader title="All uploaded rows" subtitle="Columns and values come directly from the uploaded dataset." />
      <div className="overflow-x-auto">
        <table className="w-full text-sm min-w-[1100px]">
          <thead><tr className="text-left text-xs text-slate-400 border-b">
            <th className="py-3 px-4 sticky left-0 bg-white">Customer ID</th>
            <th>Churn probability</th><th>Risk</th>
            {result.columns.filter((column: string) => column !== result.customer_id_column).slice(0, 12).map((column: string) => <th key={column} className="px-3">{column}</th>)}
          </tr></thead>
          <tbody>{result.rows.map((row: any) => <tr key={row.row} className="border-b border-slate-100 hover:bg-slate-50">
            <td className="py-3 px-4 sticky left-0 bg-white"><Link to={`/workspace/runs/${runId}/customers/${row.row}`} className="font-medium text-brand-700 hover:underline">{row.customer_id}</Link><div className="text-[11px] text-slate-400">row {row.row + 1}</div></td>
            <td>{(row.probability * 100).toFixed(1)}%</td><td><RiskBadge level={row.risk} /></td>
            {result.columns.filter((column: string) => column !== result.customer_id_column).slice(0, 12).map((column: string) => <td key={column} className="px-3 max-w-[180px] truncate" title={display(row.values[column])}>{display(row.values[column])}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <div className="flex items-center justify-between px-5 py-4 text-sm text-slate-500">
        <span>Page {result.page} of {result.pages}</span>
        <div className="flex gap-2">
          <button onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={result.page <= 1} className="px-3 py-1.5 border rounded-lg disabled:opacity-40">Previous</button>
          <button onClick={() => setPage((value) => Math.min(result.pages, value + 1))} disabled={result.page >= result.pages} className="px-3 py-1.5 border rounded-lg disabled:opacity-40">Next</button>
        </div>
      </div>
    </Card>
  </div>;
}
