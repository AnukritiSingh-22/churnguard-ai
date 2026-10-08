import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Search } from "lucide-react";
import { api } from "../api/client";
import { Card, Loading, ErrorState, RiskBadge } from "../components/ui";
import { useDataset, money, pct } from "../context/DatasetContext";

const fmt = (v: any, type: string, cur: string) =>
  v == null ? "n/a" : type === "money" ? money(v, cur) : type === "pct" ? pct(v) : typeof v === "number" ? v.toLocaleString() : String(v);

export default function RiskExplorer() {
  const { dataset, schema } = useDataset();
  const [rows, setRows] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const [riskLevel, setRiskLevel] = useState("");
  const [segment, setSegment] = useState("");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("churn_probability");
  const [sortDir, setSortDir] = useState("desc");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setPage(1); setSegment(""); setSearch(""); setRiskLevel(""); setSortBy("churn_probability"); setSortDir("desc"); }, [dataset]);

  useEffect(() => {
    setLoading(true); setError(null);
    api.ds.customers(dataset, { risk_level: riskLevel, segment, search, sort_by: sortBy, sort_dir: sortDir, page, page_size: pageSize })
      .then((d) => { setRows(d.customers); setTotal(d.total); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [dataset, riskLevel, segment, search, sortBy, sortDir, page]);

  const toggleSort = (col: string) => {
    if (sortBy === col) setSortDir(sortDir === "desc" ? "asc" : "desc");
    else { setSortBy(col); setSortDir("desc"); }
  };
  const arrow = (col: string) => (sortBy === col ? (sortDir === "desc" ? " ↓" : " ↑") : "");
  const cur = schema?.currency ?? "";
  const th = "px-4 py-2.5 font-medium cursor-pointer select-none";

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Risk Explorer</h1>
        <p className="text-sm text-slate-400 mt-0.5">{total.toLocaleString()} customers in {schema?.name} · filter, sort and drill into any record</p>
      </div>
      <Card className="p-4 flex flex-wrap gap-3 items-center">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-2.5 text-slate-400" />
          <input className="pl-8 pr-3 py-1.5 text-sm border border-slate-200 rounded-lg w-56 focus:outline-none focus:ring-2 focus:ring-brand-200"
            placeholder="Search customer ID..." value={search} onChange={(e) => { setPage(1); setSearch(e.target.value); }} />
        </div>
        <select className="text-sm border border-slate-200 rounded-lg px-2.5 py-1.5" value={riskLevel} onChange={(e) => { setPage(1); setRiskLevel(e.target.value); }}>
          <option value="">All risk levels</option>
          {["Critical", "High", "Medium", "Low"].map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
        <select className="text-sm border border-slate-200 rounded-lg px-2.5 py-1.5" value={segment} onChange={(e) => { setPage(1); setSegment(e.target.value); }}>
          <option value="">All · {schema?.segment_label}</option>
          {schema?.segment_values.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      </Card>
      <Card className="overflow-hidden">
        {error && <div className="p-4"><ErrorState message={error} /></div>}
        {loading && <Loading />}
        {!loading && !error && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-400 bg-slate-50">
                  <th className="px-4 py-2.5 font-medium">Customer ID</th>
                  <th className={th} onClick={() => toggleSort("churn_probability")}>Churn Probability{arrow("churn_probability")}</th>
                  <th className="px-4 py-2.5 font-medium">Risk</th>
                  <th className={th} onClick={() => toggleSort("CLV")}>CLV{arrow("CLV")}</th>
                  {schema?.columns.map((c) => <th key={c.key} className={th} onClick={() => toggleSort(c.key)}>{c.label}{arrow(c.key)}</th>)}
                  <th className="px-4 py-2.5 font-medium">Top Driver</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.customer_id} className="border-t border-slate-100 hover:bg-slate-50">
                    <td className="px-4 py-2.5"><Link to={`/customers/${r.customer_id}`} className="font-medium text-brand-700 hover:underline">{r.customer_id}</Link></td>
                    <td className="px-4 py-2.5 font-medium">{pct(r.churn_probability)}</td>
                    <td className="px-4 py-2.5"><RiskBadge level={r.risk_level} /></td>
                    <td className="px-4 py-2.5">{money(r.CLV, cur)}</td>
                    {schema?.columns.map((c) => <td key={c.key} className="px-4 py-2.5 text-slate-600">{fmt(r[c.key], c.type, cur)}</td>)}
                    <td className="px-4 py-2.5 text-slate-500">{r.top_driver ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex items-center justify-between px-4 py-3 border-t border-slate-100 text-xs text-slate-500">
          <span>Page {page} of {Math.max(1, Math.ceil(total / pageSize))}</span>
          <div className="space-x-2">
            <button className="px-2.5 py-1 border border-slate-200 rounded disabled:opacity-40" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</button>
            <button className="px-2.5 py-1 border border-slate-200 rounded disabled:opacity-40" disabled={page >= Math.ceil(total / pageSize)} onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </div>
      </Card>
    </div>
  );
}
