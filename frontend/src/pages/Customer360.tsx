import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";
import { ArrowLeft } from "lucide-react";
import { api } from "../api/client";
import { Card, CardHeader, Loading, ErrorState, RiskBadge, SimulationTag } from "../components/ui";
import { useDataset, money, pct } from "../context/DatasetContext";

export default function Customer360() {
  const { id } = useParams<{ id: string }>();
  const { dataset, schema } = useDataset();
  const [customer, setCustomer] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [survival, setSurvival] = useState<any>(null);
  const [counterfactual, setCounterfactual] = useState<any>(null);
  const [recs, setRecs] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setCustomer(null); setError(null);
    const caps = schema?.capabilities;
    Promise.all([
      api.ds.customer(dataset, id),
      api.ds.explanation(dataset, id).catch(() => null),
      caps?.survival ? api.ds.survival(dataset, id).catch(() => null) : Promise.resolve(null),
      caps?.counterfactual ? api.ds.counterfactual(dataset, id).catch(() => null) : Promise.resolve(null),
      api.ds.recommendations(dataset, id).catch(() => null),
    ]).then(([c, exp, surv, cf, rec]) => { setCustomer(c); setExplanation(exp); setSurvival(surv); setCounterfactual(cf); setRecs(rec); })
      .catch((e) => setError(e.message));
  }, [id, dataset, schema]);

  if (error) return (<div className="space-y-4"><Link to="/risk-explorer" className="text-xs text-brand-700 hover:underline">← Back to Risk Explorer</Link>
    <ErrorState message={`${error} (the selected dataset is "${schema?.name ?? dataset}". Pick the right one in the header.)`} /></div>);
  if (!customer || !schema) return <Loading />;
  const cur = customer.currency ?? "";

  const drivers = explanation
    ? [...explanation.risk_increasing_factors, ...explanation.risk_decreasing_factors].sort((a: any, b: any) => b.contribution - a.contribution)
    : [];

  return (
    <div className="space-y-6">
      <Link to="/risk-explorer" className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-brand-700"><ArrowLeft size={14} /> Back to Risk Explorer</Link>
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Customer {customer.customer_id}</h1>
        <div className="flex items-center gap-2 mt-1">
          <RiskBadge level={customer.risk_level} />
          <span className="text-xs text-slate-400">{schema.name} · Model: {customer.production_model}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4"><div className="text-xs text-slate-400">Churn Probability</div><div className="text-2xl font-semibold mt-1">{pct(customer.churn_probability)}</div></Card>
        <Card className="p-4"><div className="text-xs text-slate-400">Customer Value (approx. CLV)</div><div className="text-2xl font-semibold mt-1">{money(customer.CLV, cur)}</div></Card>
        {schema.capabilities.survival ? (
          <Card className="p-4"><div className="text-xs text-slate-400">6-Month Churn Risk (survival)</div><div className="text-2xl font-semibold mt-1">{pct(customer.churn_risk_6m)}</div></Card>
        ) : (
          <Card className="p-4"><div className="text-xs text-slate-400">Observed outcome</div><div className="text-2xl font-semibold mt-1">{customer.churn_flag ? "Churned" : "Retained"}</div></Card>
        )}
        <Card className="p-4"><div className="text-xs text-slate-400">Top model driver</div><div className="text-sm font-semibold mt-2 break-words">{customer.top_driver ?? "n/a"}</div></Card>
      </div>

      <Card className="p-1">
        <CardHeader title="Customer attributes" subtitle="Raw fields used by the model for this dataset" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3 px-5 pb-4">
          {Object.entries(customer.attributes).map(([k, v]: any) => (
            <div key={k}><div className="text-[11px] text-slate-400">{k}</div><div className="text-sm font-medium text-slate-700">{typeof v === "number" ? v.toLocaleString() : String(v)}</div></div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {schema.capabilities.survival && (
          <Card className="p-1">
            <CardHeader title="Time-to-Churn (Survival Curve)" subtitle="Cox model on tenure/churn (tenure excluded from covariates)" />
            <div className="h-64 px-2 pb-4">
              {survival ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={survival.curve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="month" fontSize={11} /><YAxis domain={[0, 1]} fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} labelFormatter={(l) => `Month ${l}`} />
                    <Line type="monotone" dataKey="survival_probability" stroke="#4f46e5" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div className="text-xs text-slate-400 px-4">Survival curve not available.</div>}
            </div>
            {survival?.conditional_churn_risk && (
              <div className="px-5 pb-3 text-xs text-slate-500">
                Conditional risk given current tenure: 3 mo <b className="text-slate-700">{pct(survival.conditional_churn_risk["3m"])}</b> ·
                6 mo <b className="text-slate-700">{pct(survival.conditional_churn_risk["6m"])}</b> ·
                12 mo <b className="text-slate-700">{pct(survival.conditional_churn_risk["12m"])}</b>
              </div>
            )}
          </Card>
        )}
        <Card className={`p-1 ${schema.capabilities.survival ? "" : "lg:col-span-2"}`}>
          <CardHeader title="Risk Drivers" subtitle="SHAP contributions: what the model associated with this prediction, not causal proof" />
          <div className="h-64 px-2 pb-4">
            {explanation ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={drivers} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" fontSize={11} /><YAxis type="category" dataKey="feature" width={150} fontSize={10} /><Tooltip />
                  <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>{drivers.map((e: any, i: number) => <Cell key={i} fill={e.contribution >= 0 ? "#e11d48" : "#10b981"} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div className="text-xs text-slate-400 px-4">No explanation available for this customer.</div>}
          </div>
        </Card>
      </div>

      {schema.capabilities.counterfactual && (
        <Card className="p-5">
          <CardHeader title="What Would Change the Prediction?" subtitle="Model re-scoring on edited inputs (correlational, not causal)" />
          {counterfactual ? (<>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-2">
              {counterfactual.scenarios.map((s: any) => (
                <div key={s.name} className="border border-slate-200 rounded-lg p-3">
                  <div className="text-xs font-medium text-slate-600">{s.description}</div>
                  <div className="text-lg font-semibold mt-2">{pct(s.predicted_churn_probability)}</div>
                  {s.delta !== 0 && <div className={`text-xs font-medium ${s.delta < 0 ? "text-emerald-600" : "text-rose-600"}`}>{s.delta > 0 ? "+" : ""}{(s.delta * 100).toFixed(1)} pts</div>}
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-400 mt-3">{counterfactual.disclaimer}</p>
          </>) : <div className="text-xs text-slate-400">Not available for this customer.</div>}
        </Card>
      )}

      <Card className="p-5">
        <div className="flex items-center gap-2 mb-1"><CardHeader title="Retention Optimizer: Recommended Actions" subtitle="" /><SimulationTag /></div>
        {recs && (<>
          <p className="text-xs text-slate-400 -mt-2 mb-3 px-5">{recs.note}</p>
          <div className="overflow-x-auto"><table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-slate-400 bg-slate-50">
              {["Action", "Assumed Uplift", "Cost", "Value Preserved", "Net Value", "Confidence"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>{recs.actions.map((a: any) => (
              <tr key={a.action} className="border-t border-slate-100">
                <td className="px-4 py-2 font-medium">{a.label}</td><td className="px-4 py-2">{a.assumed_retention_uplift_pct}%</td>
                <td className="px-4 py-2">{money(a.intervention_cost, recs.currency)}</td><td className="px-4 py-2">{money(a.expected_clv_preserved, recs.currency)}</td>
                <td className={`px-4 py-2 font-medium ${a.expected_net_value >= 0 ? "text-emerald-600" : "text-rose-600"}`}>{money(a.expected_net_value, recs.currency)}</td>
                <td className="px-4 py-2 text-slate-500">{a.confidence}</td>
              </tr>))}
            </tbody></table></div>
        </>)}
      </Card>
    </div>
  );
}
