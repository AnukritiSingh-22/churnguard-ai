import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState, StatusBadge } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function Drift() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setData(null); setError(null); api.ds.drift(dataset).then(setData).catch((e) => setError(e.message)); }, [dataset]);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  const pd = data.prediction_drift;
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-semibold text-slate-900">Drift & Monitoring: {schema?.name}</h1><p className="text-sm text-slate-400 mt-0.5">PSI and KS statistics, computed for real</p></div>
      <Card className="p-4 bg-amber-50 border-amber-200 text-xs text-amber-800">{data.note}</Card>
      <Card className="p-5">
        <CardHeader title="Prediction drift" subtitle="Population Stability Index between reference and comparison cohorts" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-2">
          <div><div className="text-xs text-slate-400">PSI</div><div className="text-xl font-semibold">{pd.psi}</div></div>
          <div><div className="text-xs text-slate-400">KS statistic</div><div className="text-xl font-semibold">{pd.ks_statistic}</div></div>
          <div><div className="text-xs text-slate-400">Status</div><StatusBadge status={pd.status} /></div>
          <div><div className="text-xs text-slate-400">Mean score (ref → comp)</div><div className="text-sm font-medium">{pd.reference_mean_score} → {pd.comparison_mean_score}</div></div>
        </div>
      </Card>
      <Card>
        <CardHeader title="Feature drift" subtitle="Per-feature PSI/KS on numeric columns" />
        <div className="overflow-x-auto"><table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">{["Feature", "PSI", "KS", "KS p-value", "Ref mean", "Comp mean", "Status"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{data.feature_drift.map((f: any) => (
            <tr key={f.feature} className="border-t border-slate-100"><td className="px-4 py-2.5 font-medium">{f.feature}</td><td className="px-4 py-2.5">{f.psi}</td><td className="px-4 py-2.5">{f.ks_statistic}</td>
              <td className="px-4 py-2.5">{f.ks_p_value}</td><td className="px-4 py-2.5">{f.reference_mean}</td><td className="px-4 py-2.5">{f.comparison_mean}</td><td className="px-4 py-2.5"><StatusBadge status={f.status} /></td></tr>))}</tbody>
        </table></div>
      </Card>
    </div>
  );
}
