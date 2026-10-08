import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function Experiments() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setData(null); setError(null); api.ds.experiments(dataset).then(setData).catch((e) => setError(e.message)); }, [dataset]);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-semibold text-slate-900">Experiments: {schema?.name}</h1><p className="text-sm text-slate-400 mt-0.5">One row per real training run</p></div>
      <Card><CardHeader title="Experiment log" />
        <div className="overflow-x-auto"><table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">{["Experiment", "Model", "Split", "Val PR-AUC", "CV PR-AUC", "Val ROC-AUC", "Brier", "Status"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{data.experiments.map((e: any) => (
            <tr key={e.experiment_id} className={`border-t border-slate-100 ${e.is_production ? "bg-brand-50/40" : ""}`}>
              <td className="px-4 py-2.5 font-medium">{e.experiment_id}</td><td className="px-4 py-2.5 capitalize">{e.model.replace(/_/g, " ")}</td>
              <td className="px-4 py-2.5 text-slate-500 text-xs">{e.split}</td><td className="px-4 py-2.5">{e.pr_auc}</td><td className="px-4 py-2.5">{e.cv_pr_auc ?? "n/a"}</td>
              <td className="px-4 py-2.5">{e.roc_auc}</td><td className="px-4 py-2.5">{e.brier}</td>
              <td className="px-4 py-2.5"><span className="text-[11px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">{e.status}</span></td></tr>))}</tbody>
        </table></div></Card>
    </div>
  );
}
