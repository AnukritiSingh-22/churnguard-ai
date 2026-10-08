import React, { useEffect, useState } from "react";
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import { Card, CardHeader, ErrorState, KpiCard, Loading } from "../components/ui";

const money = (value: number | null | undefined) =>
  value == null ? "—" : `£${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export default function RevenueForecast() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.revenueForecast(3).then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading label="Building leakage-safe revenue backtest..." />;

  const backtest = data.backtest;
  const forecastTotal = data.forecast.reduce((sum: number, row: any) => sum + row.prediction, 0);
  const intervalLow = data.forecast.reduce((sum: number, row: any) => sum + row.lower, 0);
  const intervalHigh = data.forecast.reduce((sum: number, row: any) => sum + row.upper, 0);
  const bias = backtest.folds.length
    ? backtest.folds.reduce((sum: number, row: any) => sum + ((row.prediction - row.actual) / Math.max(row.actual, 1)), 0) / backtest.folds.length
    : 0;
  const actual = data.history.map((row: any) => ({ period: row.period, actual: row.actual }));
  const predictions = new Map(data.backtest.folds.map((row: any) => [row.period, row.prediction]));
  const historical = actual.map((row: any) => ({ ...row, backtest: predictions.get(row.period) ?? null }));
  const future = data.forecast.map((row: any) => ({
    period: row.period, actual: null, backtest: null, forecast: row.prediction,
    lower: row.lower, upper: row.upper,
  }));
  const chartRows = [...historical, ...future];

  return <div className="space-y-5 max-w-7xl">
    <div>
      <h1 className="text-xl font-semibold text-slate-900">Sales Forecast</h1>
      <p className="text-sm text-slate-400 mt-1">
        Next-month revenue for Online Retail. Model: <strong>{data.model.replace(/_/g, " ")}</strong>,
        evaluated with rolling-origin holdout and conformal-style uncertainty intervals.
      </p>
    </div>

    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <KpiCard label={`Next ${data.forecast.length} months`} value={money(forecastTotal)} sub={`80% range ${money(intervalLow)} – ${money(intervalHigh)}`} />
      <KpiCard label="Backtest MAPE" value={`${backtest.mape}%`} sub={`RMSE ${money(backtest.rmse)}`} tone="good" />
      <KpiCard label="Rolling-origin folds" value={String(backtest.folds.length)} sub="No future month used in earlier folds" tone="warn" />
      <KpiCard label="Average bias" value={`${(bias * 100).toFixed(2)}%`} sub={bias > 0 ? "positive = over-forecast" : "negative = under-forecast"} tone={Math.abs(bias) < .1 ? "good" : "warn"} />
    </div>

    <Card className="border-amber-200 bg-amber-50/40">
      <div className="p-5">
        <h2 className="text-sm font-semibold text-amber-900">Read this before trusting the number</h2>
        <ul className="mt-3 space-y-2 text-sm text-amber-900/80 list-disc pl-5">
          <li>{data.backtest.honesty_note} The dataset has a short history, so seasonal behavior cannot be learned reliably.</li>
          <li>Revenue is computed from identified retail transactions only. This is a forecasting baseline, not a causal estimate of retention impact.</li>
          <li>{data.interval_note} The range is empirical uncertainty, not a guarantee of future sales.</li>
        </ul>
      </div>
    </Card>

    <Card className="p-1">
      <CardHeader title="Predicted vs actual monthly sales" subtitle="Actual history, rolling-origin backtest predictions, then the future forecast with an uncertainty band" />
      <div className="h-[420px] px-3 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartRows} margin={{ top: 15, right: 24, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="period" angle={-25} textAnchor="end" height={55} fontSize={11} />
            <YAxis tickFormatter={(value) => `£${(value / 1000).toFixed(0)}k`} fontSize={11} />
            <Tooltip formatter={(value: any, name: string) => [money(Number(value)), name]} />
            <Legend />
            <Area type="monotone" dataKey="upper" stroke="none" fill="#c7d2fe" fillOpacity={0.45} name="Upper interval" />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" fillOpacity={1} name="Lower interval" />
            <Line type="monotone" dataKey="actual" stroke="#172033" strokeWidth={2.5} dot={false} name="Actual" connectNulls={false} />
            <Line type="monotone" dataKey="backtest" stroke="#10a982" strokeWidth={2} dot={false} name="Backtest prediction" connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4 }} name="Forecast" connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>

    <div className="grid lg:grid-cols-2 gap-5">
      <Card className="p-1">
        <CardHeader title="Next forecast periods" subtitle="Point estimate and empirical uncertainty range" />
        <div className="px-5 pb-5 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Month</th><th>Forecast</th><th>Low</th><th>High</th></tr></thead>
            <tbody>{data.forecast.map((row: any) => <tr key={row.period} className="border-b border-slate-100"><td className="py-2 font-medium">{row.period}</td><td>{money(row.prediction)}</td><td>{money(row.lower)}</td><td>{money(row.upper)}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
      <Card className="p-1">
        <CardHeader title="Backtest: every fold, nothing hidden" subtitle="Rolling-origin holdout; lower error is better" />
        <div className="px-5 pb-5 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Period</th><th>Actual</th><th>Prediction</th><th>Error</th></tr></thead>
            <tbody>{backtest.folds.map((row: any) => <tr key={row.period} className="border-b border-slate-100"><td className="py-2">{row.period}</td><td>{money(row.actual)}</td><td>{money(row.prediction)}</td><td>{money(row.prediction - row.actual)}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
    </div>

    <Card className="p-1">
      <CardHeader title="Monthly revenue history" subtitle="Complete observed months; context for the forecast" />
      <div className="h-72 px-3 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data.history} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="period" fontSize={11} />
            <YAxis tickFormatter={(value) => `£${(value / 1000).toFixed(0)}k`} fontSize={11} />
            <Tooltip formatter={(value: any) => money(Number(value))} />
            <Bar dataKey="actual" fill="#64748b" radius={[4, 4, 0, 0]} name="Revenue" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>

    <p className="text-xs text-slate-400">Honesty note: this is a seasonal-naive baseline on the bundled Online Retail transactions, not a deep temporal model or a causal revenue-impact estimate.</p>
  </div>;
}
