import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Check } from "lucide-react";
import { api } from "../api/client";
import { Card, Loading, ErrorState } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function Datasets() {
  const { dataset, setDataset } = useDataset();
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.getDatasets().then(setData).catch((e) => setError(e.message)); }, []);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  const active = data.datasets.filter((d: any) => d.status === "active");
  const open = (k: string, to = "/") => { setDataset(k); navigate(to); };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Datasets</h1>
        <p className="text-sm text-slate-400 mt-0.5">{active.length} datasets loaded, trained and live. Open any one to switch every page (Overview, Risk Explorer, Models, Drift, Data Quality) to it. You can also use the dropdown in the header.</p>
      </div>

      <Card className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">
            {["Dataset", "Domain", "Rows", "Churn rate", "Production model", "Test ROC-AUC", "Test PR-AUC", ""].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{active.map((d: any) => (
            <tr key={d.key} className={`border-t border-slate-100 ${d.key === dataset ? "bg-brand-50/40" : ""}`}>
              <td className="px-4 py-2.5 font-medium">{d.name}{d.key === dataset && <Check size={13} className="inline ml-1.5 text-brand-600" />}</td>
              <td className="px-4 py-2.5 text-slate-500">{d.domain}</td><td className="px-4 py-2.5">{d.rows?.toLocaleString()}</td>
              <td className="px-4 py-2.5">{d.result ? `${(d.result.positive_rate * 100).toFixed(1)}%` : "n/a"}</td>
              <td className="px-4 py-2.5 capitalize">{d.result?.production_model.replace(/_/g, " ")}</td>
              <td className="px-4 py-2.5">{d.result?.test_roc_auc}</td><td className="px-4 py-2.5">{d.result?.test_pr_auc}</td>
              <td className="px-4 py-2.5 text-right"><button onClick={() => open(d.key)} className="text-xs font-medium text-brand-600 hover:underline">Open</button></td>
            </tr>))}</tbody>
        </table>
        <p className="text-[11px] text-slate-400 px-4 py-2">AUCs are not comparable across datasets: different tasks, label definitions and signal strength.</p>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.datasets.map((d: any) => (
          <Card key={d.key} className="p-5 flex flex-col">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-800">{d.name}</h3>
              <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full border ${d.status === "active" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-slate-50 text-slate-500 border-slate-200"}`}>{d.status.replace(/_/g, " ")}</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">{d.domain}</div>
            {d.rows && (
              <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                <div><span className="text-slate-400">Rows:</span> <b>{d.rows.toLocaleString()}</b></div>
                <div><span className="text-slate-400">Features:</span> <b>{d.features}</b></div>
                <div><span className="text-slate-400">Temporal:</span> <b>{d.temporal ? "Yes" : "No"}</b></div>
                <div><span className="text-slate-400">Labels:</span> <b>{d.labels ? "Provided" : "Derived"}</b></div>
              </div>)}
            <p className="text-xs text-slate-500 mt-3 flex-1">{d.notes}</p>
            {d.status === "active" && (
              <div className="flex flex-wrap gap-2 mt-4">
                <button onClick={() => open(d.key)} className="inline-flex items-center gap-1 text-xs font-medium text-white bg-brand-600 hover:bg-brand-700 rounded-lg px-3 py-1.5">Open dashboard <ArrowRight size={12} /></button>
                <button onClick={() => open(d.key, "/models")} className="text-xs font-medium text-slate-600 border border-slate-200 rounded-lg px-3 py-1.5 hover:bg-slate-50">Models</button>
                <button onClick={() => open(d.key, "/data-quality")} className="text-xs font-medium text-slate-600 border border-slate-200 rounded-lg px-3 py-1.5 hover:bg-slate-50">Data quality</button>
              </div>)}
          </Card>))}
      </div>
    </div>
  );
}
