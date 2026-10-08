import React, { useEffect, useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { Card, CardHeader, ErrorState, KpiCard, Loading, RiskBadge, SimulationTag } from "../components/ui";

const RISK_COLORS: Record<string, string> = { Low: "#10b981", Medium: "#f59e0b", High: "#f97316", Critical: "#e11d48" };
const number = (value: number | undefined) => value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
const percent = (value: number | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

export default function Workspace() {
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<any>(null);
  const [uploads, setUploads] = useState<any[]>([]);
  const [target, setTarget] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.uploads().then(async (response) => {
      setUploads(response.uploads);
      const latestRun = response.uploads.find((upload: any) => upload.run_id);
      if (latestRun) setResult(await api.workspaceRun(latestRun.run_id));
    }).catch((e) => setError(e.message));
  }, []);

  const upload = async () => {
    if (!file) return;
    try {
      setBusy(true); setError(""); setResult(null);
      const nextProfile = await api.uploadCsv(file);
      setProfile(nextProfile);
      setTarget(nextProfile.columns.find((column: any) => column.role === "possible_target")?.name || "");
      setCustomerId(nextProfile.columns.find((column: any) => column.role === "possible_id")?.name || "");
      if (nextProfile.auto_run) setResult(nextProfile.auto_run);
      if (nextProfile.auto_train_error) setError(nextProfile.auto_train_error);
      setUploads((await api.uploads()).uploads);
    } catch (e: any) { setError(e.message); } finally { setBusy(false); }
  };

  const train = async () => {
    if (!profile || !target) return;
    try {
      setBusy(true); setError("");
      setResult(await api.trainUpload(profile.upload_id, target, customerId || undefined));
    } catch (e: any) { setError(e.message); } finally { setBusy(false); }
  };

  const dashboard = result?.dashboard;
  return <div className="space-y-5 max-w-7xl">
    <div>
      <h1 className="text-xl font-semibold">My data workspace</h1>
      <p className="text-sm text-slate-500 mt-1">Upload your file and see the same decision-intelligence views as the demo, computed only from your data.</p>
    </div>
    {error && <ErrorState message={error} />}
    <Card className="p-5">
      <CardHeader title="Upload a customer CSV" subtitle="CSV only, maximum 100 MB. No labels are invented." />
      <div className="flex flex-wrap gap-3 items-center px-5 pb-2">
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={upload} disabled={!file || busy} className="px-4 py-2 rounded-lg bg-brand-700 text-white text-sm disabled:opacity-40">{busy ? "Working…" : "Profile file"}</button>
      </div>
    </Card>
    {profile && <Card className="p-5">
      <CardHeader title={profile.filename} subtitle={`${profile.total_rows.toLocaleString()} rows · ${profile.columns.length} columns`} />
      {profile.normalization?.status === "normalized" && <div className="mx-5 mb-4 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
        <strong>Upload normalized automatically.</strong> {profile.normalization.column ? `“${profile.normalization.column}” was converted to 0/1 using ${profile.normalization.method}.` : "Export-only columns were removed."}
        {profile.auto_trained ? " The churn baseline has also been trained automatically." : ""}
      </div>}
      {profile.normalization?.dropped_columns?.length > 0 && <p className="px-5 mb-3 text-xs text-slate-500">Ignored export columns: {profile.normalization.dropped_columns.join(", ")}</p>}
      <div className="grid md:grid-cols-2 gap-4 px-5">
        <label className="text-sm">Target column<select value={target} onChange={(e) => setTarget(e.target.value)} className="block mt-1 border rounded-lg p-2 w-full">
          <option value="">Select a binary churn/exit label</option>{profile.columns.map((c: any) => <option key={c.name} value={c.name}>{c.name} ({c.role})</option>)}</select></label>
        <label className="text-sm">Customer ID (optional)<select value={customerId} onChange={(e) => setCustomerId(e.target.value)} className="block mt-1 border rounded-lg p-2 w-full">
          <option value="">None</option>{profile.columns.map((c: any) => <option key={c.name} value={c.name}>{c.name}</option>)}</select></label>
      </div>
      <p className="text-xs text-slate-400 px-5 mt-3">{profile.note}</p>
      {!profile.auto_trained && <button onClick={train} disabled={!target || busy} className="mx-5 mt-4 px-4 py-2 rounded-lg bg-brand-700 text-white text-sm disabled:opacity-40">{busy ? "Training…" : "Train churn baseline"}</button>}
    </Card>}
    {!dashboard && <Card className="p-6"><p className="text-sm text-slate-500">After training, this page will show risk charts, performance, explanations, drift, review queue, and revenue context for the uploaded file. The bundled benchmark pages remain unchanged.</p></Card>}
    {dashboard && result?.run_id && <WorkspaceDashboard dashboard={dashboard} runId={result.run_id} />}
    <Card className="p-5"><CardHeader title="Your previous uploads" />{uploads.length ? uploads.map((u) => <div key={u.upload_id} className="text-sm border-b py-2">{u.filename} · {u.rows.toLocaleString()} rows</div>) : <p className="text-sm text-slate-400">No uploads yet.</p>}</Card>
  </div>;
}

function WorkspaceDashboard({ dashboard, runId }: { dashboard: any; runId: string }) {
  const m = dashboard.metrics;
  return <div className="space-y-5">
    <div className="flex items-center justify-between"><div><h2 className="text-lg font-semibold">Uploaded-data results</h2><p className="text-xs text-slate-400">{dashboard.honesty_note}</p></div><span className="text-xs px-2 py-1 rounded-full bg-brand-50 text-brand-700">Private run</span></div>
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      <KpiCard label="Rows scored" value={number(m.train_rows + m.test_rows)} sub={`${m.test_rows.toLocaleString()} holdout rows`} />
      <KpiCard label="ROC-AUC" value={number(m.roc_auc)} sub="Stratified holdout" tone="good" />
      <KpiCard label="PR-AUC" value={number(m.pr_auc)} sub="Positive-class ranking" tone="good" />
      <KpiCard label="Brier score" value={number(m.brier_score)} sub="Lower is better" />
      <KpiCard label="Predicted churn" value={percent(dashboard.predicted_rate)} sub={`Observed ${percent(dashboard.positive_rate)}`} tone="warn" />
    </div>
    <div className="grid lg:grid-cols-2 gap-5">
      <Card className="p-1"><CardHeader title="Risk distribution" subtitle="All uploaded rows scored by predicted churn probability" /><div className="h-72"><ResponsiveContainer><PieChart><Pie data={dashboard.risk_distribution} dataKey="count" nameKey="risk_level" innerRadius={60} outerRadius={100} paddingAngle={3}>{dashboard.risk_distribution.map((entry: any) => <Cell key={entry.risk_level} fill={RISK_COLORS[entry.risk_level]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></div></Card>
      <Card className="p-1"><CardHeader title="Calibration check" subtitle="Holdout predicted probability versus observed churn" /><div className="h-72 px-3"><ResponsiveContainer><LineChart data={dashboard.calibration}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="bucket" fontSize={10} /><YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} /><Tooltip /><Line type="monotone" dataKey="predicted" name="Predicted" stroke="#6366f1" strokeWidth={2} /><Line type="monotone" dataKey="actual" name="Actual" stroke="#10b981" strokeWidth={2} /></LineChart></ResponsiveContainer></div></Card>
    </div>
    <div className="grid lg:grid-cols-2 gap-5">
      <Card className="p-1"><CardHeader title="Model drivers" subtitle="Positive coefficients increase predicted churn; association, not causation" /><div className="h-80 px-3"><ResponsiveContainer><BarChart data={dashboard.feature_drivers} layout="vertical" margin={{ left: 50 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={130} fontSize={10} /><Tooltip /><Bar dataKey="impact" fill="#6366f1" /></BarChart></ResponsiveContainer></div></Card>
      <Card className="p-1"><CardHeader title="Data drift" subtitle="Holdout partition compared with training partition" /><div className="h-80 px-3"><ResponsiveContainer><BarChart data={dashboard.drift.slice(0, 12)} layout="vertical" margin={{ left: 50 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={130} fontSize={10} /><Tooltip /><Bar dataKey="shift" fill="#f59e0b" /></BarChart></ResponsiveContainer></div></Card>
    </div>
    <Card className="p-5"><CardHeader title="Holdout confusion matrix" subtitle="Threshold = 50%" /><div className="grid grid-cols-4 gap-3 text-center text-sm"><div className="rounded-lg bg-emerald-50 p-4">True negative<strong className="block text-2xl">{dashboard.confusion_matrix.tn}</strong></div><div className="rounded-lg bg-rose-50 p-4">False positive<strong className="block text-2xl">{dashboard.confusion_matrix.fp}</strong></div><div className="rounded-lg bg-rose-50 p-4">False negative<strong className="block text-2xl">{dashboard.confusion_matrix.fn}</strong></div><div className="rounded-lg bg-emerald-50 p-4">True positive<strong className="block text-2xl">{dashboard.confusion_matrix.tp}</strong></div></div></Card>
    <Card className="p-5"><CardHeader title="Human review queue" subtitle="Highest-risk uploaded records; select a customer to inspect model drivers" right={<a href={`#/workspace/runs/${runId}/customers`} className="text-sm font-medium text-brand-700 hover:underline">Show more</a>} /><div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Customer ID</th><th>Risk</th><th>Probability</th><th>Action</th></tr></thead><tbody>{dashboard.review_queue.map((row: any) => <tr key={row.row} className="border-b border-slate-100"><td className="py-2"><a href={`#/workspace/runs/${runId}/customers/${row.row}`} className="font-medium text-brand-700 hover:underline">{row.customer_id}</a><div className="text-[11px] text-slate-400">row {row.row + 1}</div></td><td><RiskBadge level={row.risk} /></td><td>{percent(row.probability)}</td><td className="text-slate-500">Review customer context before contact</td></tr>)}</tbody></table></div></Card>
    <RevenuePanel data={dashboard.revenue_forecast ?? { available: false, note: "No uploaded sales forecast is available for this run." }} runId={runId} />
  </div>;
}

function RevenuePanel({ data, runId }: { data: any; runId: string }) {
  const chartRows = data.available ? [
    ...(data.history || []).map((row: any) => ({ ...row, forecast: null, lower: null, upper: null })),
    ...(data.forecast || []).map((row: any) => ({ period: row.period, actual: null, forecast: row.prediction, lower: row.lower, upper: row.upper })),
  ] : [];
  return <Card className="p-1"><CardHeader title="Sales forecast and revenue context" subtitle="Historical monthly sales, leakage-safe backtest, and future forecast for uploaded data." />
    <div className="px-5 pb-5">{data.available ? <><div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4"><KpiCard label={`Next ${data.forecast.length} months`} value={number(data.forecast.reduce((sum: number, row: any) => sum + row.prediction, 0))} sub="Point estimate" tone="good" /><KpiCard label="Backtest MAPE" value={`${data.backtest.mape}%`} sub={`RMSE ${number(data.backtest.rmse)}`} /><KpiCard label="Forecast model" value="Seasonal naive" sub={`${data.seasonal_period_months}-month window`} /><KpiCard label="Data source" value={data.revenue_column} sub={`Date: ${data.date_column}`} /></div><div className="h-72"><ResponsiveContainer><ComposedChart data={chartRows}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="period" angle={-20} textAnchor="end" height={48} fontSize={10} /><YAxis /><Tooltip /><Area dataKey="upper" stroke="none" fill="#c7d2fe" fillOpacity={0.5} name="Upper interval" /><Area dataKey="lower" stroke="none" fill="#fff" fillOpacity={1} name="Lower interval" /><Line dataKey="actual" stroke="#172033" strokeWidth={2.5} dot={false} name="Actual" /><Line dataKey="forecast" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4 }} name="Forecast" /></ComposedChart></ResponsiveContainer></div><div className="flex items-center justify-between mt-3"><p className="text-xs text-slate-500">{data.interval_note}</p><a href={`#/workspace/runs/${runId}/revenue`} className="text-sm font-medium text-brand-700 hover:underline">Show more</a></div></> : <div className="rounded-lg bg-slate-50 border border-slate-200 p-5"><div className="flex items-center gap-2"><SimulationTag /><span className="text-sm font-medium">Future sales forecast not available for this upload</span></div><p className="text-sm text-slate-500 mt-2">{data.note}</p><p className="text-xs text-slate-400 mt-2">Upload transaction-level data with a date/time column and revenue, sales, amount, or quantity × unit-price field. Static churn snapshots cannot support a genuine future-sales forecast.</p></div>}</div>
  </Card>;
}
