import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState, StatusBadge } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

export default function DataQuality() {
  const { dataset, schema } = useDataset();
  const [dq, setDq] = useState<any>(null);
  const [audit, setAudit] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setDq(null); setAudit(null); setError(null);
    Promise.all([api.ds.dataQuality(dataset), api.ds.leakage(dataset)]).then(([d, a]) => { setDq(d); setAudit(a); }).catch((e) => setError(e.message));
  }, [dataset]);
  if (error) return <ErrorState message={error} />;
  if (!dq || !audit) return <Loading />;
  const ab = audit.ablation;
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-semibold text-slate-900">Data Quality & Leakage Audit: {schema?.name}</h1>
        <p className="text-sm text-slate-400 mt-0.5">Source: {dq.source}</p></div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[["Rows", dq.rows.toLocaleString()], ["Columns", dq.columns], ["Duplicate rows", dq.duplicate_rows],
          ["Class imbalance (No:Yes)", `${dq.imbalance_ratio_no_to_yes}:1`], ["Fields with missing values", Object.keys(dq.missing_values).length]].map(([l, v]: any) => (
          <Card key={l} className="p-4"><div className="text-xs text-slate-400">{l}</div><div className="text-xl font-semibold">{v}</div></Card>))}
      </div>
      <Card className="p-5">
        <CardHeader title="Outlier counts (IQR method)" subtitle="Numeric columns" />
        <div className="flex flex-wrap gap-x-8 gap-y-3 mt-2">{Object.entries(dq.outlier_counts_iqr).map(([k, v]: any) => (
          <div key={k}><div className="text-xs text-slate-400">{k}</div><div className="text-lg font-semibold">{v}</div></div>))}</div>
      </Card>
      {audit.note && <Card className="p-4 bg-amber-50 border-amber-200 text-xs text-amber-800">{audit.note}</Card>}
      {ab && (
        <Card className="p-5">
          <CardHeader title="Feature-group ablation" subtitle="CV ROC-AUC with suspicious feature groups removed: does the signal survive?" />
          <table className="w-full text-sm mt-2"><thead><tr className="text-left text-xs text-slate-400"><th className="py-1">Scenario</th><th>ROC-AUC</th><th>± sd</th><th>Features</th></tr></thead>
            <tbody>{Object.entries(ab).filter(([k]) => k !== "interpretation").map(([k, v]: any) => (
              <tr key={k} className="border-t border-slate-100"><td className="py-2 font-medium">{k.replace(/_/g, " ")}</td><td>{v.roc_auc_mean}</td><td>{v.roc_auc_std}</td><td>{v.n_features}</td></tr>))}</tbody></table>
          {ab.interpretation && <p className="text-xs text-slate-500 mt-3">{ab.interpretation}</p>}
        </Card>
      )}
      <Card>
        <CardHeader title="Leakage audit results" subtitle={`${audit.excluded_structural.length} excluded structurally · ${audit.flagged_for_review.length} flagged for review · ${audit.cleared.length} cleared`} />
        <div className="overflow-x-auto max-h-[420px] overflow-y-auto"><table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-50"><tr className="text-left text-xs text-slate-400">{["Feature", "Check", "Association", "Status", "Reason"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{audit.details.map((d: any) => (
            <tr key={d.feature} className="border-t border-slate-100"><td className="px-4 py-2.5 font-medium">{d.feature}</td><td className="px-4 py-2.5 text-slate-500 capitalize">{d.check}</td>
              <td className="px-4 py-2.5">{d.association_score ?? "—"}</td><td className="px-4 py-2.5"><StatusBadge status={d.status} /></td><td className="px-4 py-2.5 text-slate-500 max-w-md">{d.reason}</td></tr>))}</tbody>
        </table></div>
      </Card>
    </div>
  );
}
