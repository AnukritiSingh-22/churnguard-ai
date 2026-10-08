import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from "recharts";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, KpiCard, Loading, ErrorState, RiskBadge } from "../components/ui";
import { useDataset, money, pct } from "../context/DatasetContext";

const RISK_COLORS: Record<string, string> = { Low: "#10b981", Medium: "#f59e0b", High: "#f97316", Critical: "#e11d48" };

export default function Overview() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null); setError(null);
    api.ds.summary(dataset).then(setData).catch((e) => setError(e.message));
  }, [dataset]);

  if (error) return <ErrorState message={error} />;
  if (!data || !schema) return <Loading />;
  const cur = data.currency;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">{schema.name}: Customer Intelligence Overview</h1>
        <p className="text-sm text-slate-400 mt-0.5">Every figure is computed from the trained production model for this dataset. {data.label_note}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <KpiCard label="Total Customers" value={data.total_customers.toLocaleString()} sub={`Observed churn ${pct(data.observed_churn_rate)}`} />
        <KpiCard label="High-Risk Customers" value={data.high_risk_customers.toLocaleString()}
          sub={`${((data.high_risk_customers / data.total_customers) * 100).toFixed(1)}% of base`} tone="warn" />
        <KpiCard label="Mean Predicted Churn" value={pct(data.predicted_churn_rate)} sub="Mean calibrated probability" tone="bad" />
        <KpiCard label="Revenue at Risk (proxy)" value={money(data.revenue_at_risk, cur)} sub="CLV of High + Critical" tone="bad" />
        <KpiCard label="Expected Loss (proxy)" value={money(data.expected_loss_proxy, cur)} sub="Σ probability × CLV" tone="good" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2 p-1">
          <CardHeader title="Customer Risk Distribution" subtitle="Counts by risk tier" />
          <div className="h-64 px-2 pb-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.risk_distribution} dataKey="n" nameKey="risk_level" innerRadius={55} outerRadius={90} paddingAngle={2}>
                  {data.risk_distribution.map((e: any, i: number) => <Cell key={i} fill={RISK_COLORS[e.risk_level] || "#94a3b8"} />)}
                </Pie>
                <Tooltip /><Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="p-1">
          <CardHeader title={`Risk by ${data.segment_label}`} subtitle="Average predicted churn probability" />
          <div className="h-64 px-2 pb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.by_segment.slice(0, 8)} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} fontSize={11} />
                <YAxis type="category" dataKey="segment" width={100} fontSize={11} />
                <Tooltip formatter={(v: any) => `${(v * 100).toFixed(1)}%`} />
                <Bar dataKey="avg_risk" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="p-1">
        <CardHeader title={`Revenue at Risk by ${data.segment_label}`} subtitle="Sum of CLV for High/Critical customers (a proxy, not realised loss)" />
        <div className="h-56 px-4 pb-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.by_segment}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
              <XAxis dataKey="segment" fontSize={12} />
              <YAxis fontSize={11} tickFormatter={(v) => money(v, cur)} />
              <Tooltip formatter={(v: any) => `${cur}${Number(v).toLocaleString()}`} />
              <Bar dataKey="revenue_at_risk" fill="#4f46e5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card>
        <CardHeader title="Top Retention Opportunities" subtitle="Highest-CLV customers currently at High/Critical risk"
          right={<Link to="/risk-explorer" className="text-xs font-medium text-brand-600 hover:underline">View all →</Link>} />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-400 border-t border-slate-100">
                <th className="px-5 py-2 font-medium">Customer</th><th className="px-5 py-2 font-medium">Risk</th>
                <th className="px-5 py-2 font-medium">Probability</th><th className="px-5 py-2 font-medium">CLV</th>
                <th className="px-5 py-2 font-medium">Top model driver</th>
              </tr>
            </thead>
            <tbody>
              {data.top_retention_opportunities.map((c: any) => (
                <tr key={c.customer_id} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-5 py-2.5"><Link to={`/customers/${c.customer_id}`} className="font-medium text-brand-700 hover:underline">{c.customer_id}</Link></td>
                  <td className="px-5 py-2.5"><RiskBadge level={c.risk_level} /></td>
                  <td className="px-5 py-2.5">{pct(c.churn_probability)}</td>
                  <td className="px-5 py-2.5">{money(c.CLV, cur)}</td>
                  <td className="px-5 py-2.5 text-slate-500">{c.top_driver ?? "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
