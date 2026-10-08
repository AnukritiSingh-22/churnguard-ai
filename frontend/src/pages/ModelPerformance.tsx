import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function ModelPerformance() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [cal, setCal] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null); setCal(null); setError(null);
    Promise.all([api.ds.models(dataset), api.ds.calibration(dataset)])
      .then(([m, c]) => { setData(m); setCal(c); })
      .catch((e) => setError(e.message));
  }, [dataset]);

  if (error) return <ErrorState message={error} />;
  if (!data || !cal) return <Loading />;
  const cc = cal.calibration_comparison;
  const ch = data.challenger;
  const ta = data.threshold_analysis;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Model Performance: {schema?.name}</h1>
        <p className="text-sm text-slate-400 mt-0.5">
          Production model: <span className="font-medium text-brand-700">{data.production_model.replace(/_/g, " ")}</span> — {data.selection_reason}
        </p>
      </div>

      <Card className="overflow-x-auto">
        <CardHeader title="Model comparison" subtitle="Held-out test split (one split) and 5×3 repeated CV on the train split. Models within noise of each other are statistically tied." />
        <table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">
            {["Model", "Test ROC-AUC", "Test PR-AUC", "Test F1", "Test Brier", "CV PR-AUC (mean ± sd)", "CV ROC-AUC", "Train (s)"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>
            {data.models.map((m: any) => (
              <tr key={m.model} className={`border-t border-slate-100 ${m.is_production ? "bg-brand-50/40" : ""}`}>
                <td className="px-4 py-2.5 font-medium capitalize">{m.model.replace(/_/g, " ")}{m.is_production && <span className="ml-2 text-[10px] font-bold text-brand-600 uppercase">Production</span>}</td>
                <td className="px-4 py-2.5">{m.test.roc_auc}</td><td className="px-4 py-2.5">{m.test.pr_auc}</td>
                <td className="px-4 py-2.5">{m.test.f1}</td><td className="px-4 py-2.5">{m.test.brier_score}</td>
                <td className="px-4 py-2.5">{m.cv ? `${m.cv.pr_auc_mean} ± ${m.cv.pr_auc_std}` : "n/a"}</td>
                <td className="px-4 py-2.5">{m.cv ? m.cv.roc_auc_mean : "n/a"}</td>
                <td className="px-4 py-2.5">{m.train_time_seconds}</td>
              </tr>))}
          </tbody>
        </table>
      </Card>

      {ch && (
        <Card className="p-1 overflow-x-auto">
          <CardHeader title="Advanced-model gate: stacking and voting ensembles" subtitle={ch.rule} />
          <div className={`mx-5 mb-3 text-xs rounded-lg border px-3 py-2 ${ch.promoted?.length ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-amber-50 border-amber-200 text-amber-800"}`}>{ch.decision}</div>
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">
              {["Candidate", "CV PR-AUC", "Gain vs baseline", "SE of gain", "> 1 SE", "≥ 0.02", "Test PR-AUC"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>
              {Object.entries(ch.cv).map(([name, v]: any) => (
                <tr key={name} className="border-t border-slate-100">
                  <td className="px-4 py-2.5 font-medium capitalize">{name === "baseline" ? `Baseline (${ch.baseline_model.replace(/_/g, " ")})` : name.replace(/_/g, " ")}</td>
                  <td className="px-4 py-2.5">{v.pr_auc_mean}</td>
                  <td className="px-4 py-2.5">{v.pr_auc_gain_vs_baseline ?? "—"}</td><td className="px-4 py-2.5">{v.gain_se ?? "—"}</td>
                  <td className="px-4 py-2.5">{v.statistically_better === undefined ? "—" : v.statistically_better ? "Yes" : "No"}</td>
                  <td className="px-4 py-2.5">{v.practically_meaningful === undefined ? "—" : v.practically_meaningful ? "Yes" : "No"}</td>
                  <td className="px-4 py-2.5">{ch.test[name]?.pr_auc}</td>
                </tr>))}
            </tbody>
          </table>
          <p className="text-[11px] text-slate-400 px-5 py-3">Sequence/graph models are intentionally not included: none of these datasets has per-customer event sequences or a relationship graph.</p>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="p-1">
          <CardHeader title="Calibration reliability diagram" subtitle="Production model after sigmoid calibration, held-out test data" />
          <div className="h-64 px-2 pb-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cal.reliability_curve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="predicted_avg" type="number" domain={[0, 1]} fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis domain={[0, 1]} fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} /><Legend />
                <Line type="monotone" dataKey="observed_freq" name="Observed frequency" stroke="#4f46e5" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="predicted_avg" name="Perfect calibration" stroke="#cbd5e1" strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {cc && (
            <table className="text-xs mx-5 mb-4 w-[calc(100%-2.5rem)]">
              <thead><tr className="text-left text-slate-400"><th className="py-1">Method</th><th>PR-AUC</th><th>Brier</th><th>ECE</th></tr></thead>
              <tbody>{["raw", "sigmoid", "isotonic"].filter((k) => cc[k]).map((k) => (
                <tr key={k} className="border-t border-slate-100"><td className="py-1 capitalize">{k}{k === cc.production_method ? " (production)" : ""}</td>
                  <td>{cc[k].pr_auc}</td><td>{cc[k].brier_score}</td><td>{cc[k].ece}</td></tr>))}</tbody>
            </table>
          )}
        </Card>
        <Card className="p-1">
          <CardHeader title="ROC curve" subtitle="Production model, held-out test set" />
          <div className="h-64 px-2 pb-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cal.roc_curve.fpr.map((f: number, i: number) => ({ fpr: f, tpr: cal.roc_curve.tpr[i] }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} fontSize={11} /><YAxis domain={[0, 1]} fontSize={11} /><Tooltip />
                <Line type="monotone" dataKey="tpr" stroke="#e11d48" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {ta && (
        <Card className="p-5">
          <CardHeader title="Cost-based threshold analysis" subtitle={ta.note} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            {(["raw", "calibrated"] as const).map((k) => ta[k] && (
              <div key={k} className="border border-slate-200 rounded-lg p-4 text-sm">
                <div className="text-xs font-semibold uppercase text-slate-400">{k} probabilities</div>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div><span className="text-slate-400 text-xs block">Best threshold</span><b>{ta[k].best_threshold.threshold}</b></div>
                  <div><span className="text-slate-400 text-xs block">Recall / Precision</span><b>{ta[k].best_threshold.recall} / {ta[k].best_threshold.precision}</b></div>
                  <div><span className="text-slate-400 text-xs block">Customers flagged</span><b>{ta[k].best_threshold.flagged}</b></div>
                  <div><span className="text-slate-400 text-xs block">Top-decile lift</span><b>{ta[k].top_decile_lift}×</b></div>
                </div>
              </div>))}
          </div>
        </Card>
      )}

      <Card className="p-5">
        <CardHeader title="Confusion matrix" subtitle="Production model, calibrated, threshold 0.5, held-out test set (the cost-optimal threshold above differs)" />
        <div className="grid grid-cols-2 gap-2 max-w-xs mt-3">
          {([["True Negative", cal.confusion_matrix[0][0], "bg-emerald-50 border-emerald-200"], ["False Positive", cal.confusion_matrix[0][1], "bg-amber-50 border-amber-200"],
            ["False Negative", cal.confusion_matrix[1][0], "bg-rose-50 border-rose-200"], ["True Positive", cal.confusion_matrix[1][1], "bg-brand-50 border-brand-200"]] as any[]).map(([l, v, cls]) => (
            <div key={l} className={`${cls} border rounded-lg p-4 text-center`}><div className="text-xs text-slate-500">{l}</div><div className="text-xl font-semibold">{v}</div></div>))}
        </div>
      </Card>
    </div>
  );
}
