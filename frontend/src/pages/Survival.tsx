import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function Survival() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const supported = schema?.capabilities.survival;

  useEffect(() => {
    setData(null); setError(null);
    if (supported) api.ds.survivalPopulation(dataset).then(setData).catch((e) => setError(e.message));
  }, [dataset, supported]);

  if (!schema) return <Loading />;
  if (!supported) return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-900">Survival Analysis</h1>
      <Card className="p-6 max-w-2xl">
        <div className="font-semibold text-slate-800">Not available for {schema.name}</div>
        <p className="text-sm text-slate-500 mt-2">Survival modelling needs a duration and an event indicator per customer (for example tenure plus churn). This dataset does not provide a usable time-to-event field, and we do not fabricate one.</p>
        <p className="text-sm text-slate-500 mt-2">To enable it: supply customer start dates and churn dates (or an observation end date) so duration and censoring can be computed.</p>
        <Link to="/" className="inline-block mt-4 text-xs font-medium text-brand-600 hover:underline">← Back to Overview</Link>
      </Card>
    </div>);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;

  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-semibold text-slate-900">Survival Analysis</h1><p className="text-sm text-slate-400 mt-0.5">{data.method}</p></div>
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        {Object.entries(data.checkpoint_survival).map(([m, p]: any) => (
          <Card key={m} className="p-4"><div className="text-xs text-slate-400">Survival at {m} mo</div><div className="text-xl font-semibold mt-1">{p !== null ? `${(p * 100).toFixed(1)}%` : "N/A"}</div></Card>))}
      </div>
      <Card className="p-4">
        <div className="text-xs text-slate-400">Cox PH concordance index (held-out customers)</div>
        <div className="text-2xl font-semibold mt-1">{data.c_index_test}</div>
        <div className="text-xs text-slate-400 mt-1">tenure and TotalCharges are excluded from the covariates (they define the time axis); an earlier ~0.93 was inflated by this circularity.</div>
        {data.caveat && <div className="text-xs text-amber-700 mt-2">{data.caveat}</div>}
      </Card>
      <Card className="p-1">
        <CardHeader title="Population Kaplan-Meier survival curve" subtitle="Probability the average customer remains active, by tenure month" />
        <div className="h-80 px-2 pb-4"><ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.population_curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="tenure_months" fontSize={11} /><YAxis domain={[0, 1]} fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} />
            <Line type="stepAfter" dataKey="survival_probability" stroke="#4f46e5" strokeWidth={2} dot={false} />
          </LineChart></ResponsiveContainer></div>
      </Card>
    </div>
  );
}
