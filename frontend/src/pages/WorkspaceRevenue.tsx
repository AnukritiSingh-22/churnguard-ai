import React, { useEffect, useState } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, ErrorState, KpiCard, Loading } from "../components/ui";

const money = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: 0 });

export default function WorkspaceRevenue() {
  const { runId = "" } = useParams();
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.workspaceRun(runId).then(setResult).catch((e) => setError(e.message)); }, [runId]);
  if (error) return <ErrorState message={error} />;
  if (!result) return <Loading label="Loading uploaded sales forecast..." />;
  const data = result.dashboard.revenue_forecast ?? {
    available: false,
    note: "This run was created before uploaded sales forecasting was available. Re-upload the file to generate forecast details.",
  };
  if (!data.available) return <div className="max-w-5xl"><Card className="p-6"><h1 className="text-xl font-semibold">Uploaded sales forecast</h1><p className="text-sm text-slate-500 mt-2">{data.note}</p></Card></div>;
  const rows = [
    ...data.history.map((row: any) => ({ ...row, forecast: null, lower: null, upper: null })),
    ...data.forecast.map((row: any) => ({ period: row.period, actual: null, forecast: row.prediction, lower: row.lower, upper: row.upper })),
  ];
  return <div className="space-y-5 max-w-7xl">
    <div><h1 className="text-xl font-semibold">Uploaded sales forecast</h1><p className="text-sm text-slate-400 mt-1">Future monthly sales estimated from <strong>{data.revenue_column}</strong>, grouped by <strong>{data.date_column}</strong>. Run ID: {runId}</p></div>
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <KpiCard label={`Next ${data.forecast.length} months`} value={money(data.forecast.reduce((s: number, r: any) => s + r.prediction, 0))} sub="Point estimate" tone="good" />
      <KpiCard label="Backtest MAPE" value={`${data.backtest.mape}%`} sub={`RMSE ${money(data.backtest.rmse)}`} />
      <KpiCard label="Backtest folds" value={String(data.backtest.folds.length)} sub="Rolling-origin holdout" />
      <KpiCard label="Forecast method" value="Seasonal naive" sub={`${data.seasonal_period_months}-month window`} />
    </div>
    <Card className="border-amber-200 bg-amber-50/40"><div className="p-5"><h2 className="text-sm font-semibold text-amber-900">Read this before using the forecast</h2><p className="text-sm text-amber-900/80 mt-2">{data.backtest.honesty_note} {data.interval_note}</p></div></Card>
    <Card className="p-1"><CardHeader title="Actual, backtest, and future forecast" subtitle="The shaded area is the empirical forecast interval." /><div className="h-[430px] px-3 pb-4"><ResponsiveContainer><ComposedChart data={rows}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="period" angle={-25} textAnchor="end" height={55} fontSize={11} /><YAxis /><Tooltip formatter={(value: any) => [money(Number(value)), "Sales"]} /><Area dataKey="upper" stroke="none" fill="#c7d2fe" fillOpacity={0.5} name="Upper interval" /><Area dataKey="lower" stroke="none" fill="#fff" fillOpacity={1} name="Lower interval" /><Line dataKey="actual" stroke="#172033" strokeWidth={2.5} dot={false} name="Actual" /><Line dataKey="forecast" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4 }} name="Forecast" /></ComposedChart></ResponsiveContainer></div></Card>
    <div className="grid lg:grid-cols-2 gap-5">
      <Card className="p-1"><CardHeader title="Next forecast periods" subtitle="Point estimate and interval" /><div className="px-5 pb-5 overflow-x-auto"><table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Month</th><th>Forecast</th><th>Low</th><th>High</th></tr></thead><tbody>{data.forecast.map((row: any) => <tr key={row.period} className="border-b border-slate-100"><td className="py-2 font-medium">{row.period}</td><td>{money(row.prediction)}</td><td>{money(row.lower)}</td><td>{money(row.upper)}</td></tr>)}</tbody></table></div></Card>
      <Card className="p-1"><CardHeader title="Rolling-origin backtest" subtitle="Every fold uses only earlier months" /><div className="px-5 pb-5 overflow-x-auto"><table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Period</th><th>Actual</th><th>Prediction</th><th>Error</th></tr></thead><tbody>{data.backtest.folds.map((row: any) => <tr key={row.period} className="border-b border-slate-100"><td className="py-2">{row.period}</td><td>{money(row.actual)}</td><td>{money(row.prediction)}</td><td>{money(row.prediction - row.actual)}</td></tr>)}</tbody></table></div></Card>
    </div>
    <Card className="p-1"><CardHeader title="Monthly sales history" subtitle="Observed monthly revenue used as forecast context" /><div className="h-72 px-3 pb-4"><ResponsiveContainer><BarChart data={data.history}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="period" fontSize={11} /><YAxis /><Tooltip formatter={(value: any) => [money(Number(value)), "Actual"]} /><Bar dataKey="actual" fill="#64748b" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></Card>
  </div>;
}
