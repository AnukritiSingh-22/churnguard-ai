import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, ErrorState, KpiCard, Loading, RiskBadge } from "../components/ui";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function WorkspaceCustomer() {
  const { runId, row } = useParams<{ runId: string; row: string }>();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (runId && row) api.workspaceCustomer(runId, Number(row)).then(setData).catch((e) => setError(e.message));
  }, [runId, row]);

  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading label="Loading uploaded customer explanation..." />;

  const driverData = [...data.drivers].sort((a, b) => a.contribution - b.contribution);
  return <div className="space-y-5 max-w-6xl">
    <Link to="/workspace" className="text-sm text-brand-700 hover:underline">← Back to My Workspace</Link>
    <div>
      <div className="flex items-center gap-3"><h1 className="text-xl font-semibold">Customer {data.customer_id}</h1><RiskBadge level={data.risk_level} /></div>
      <p className="text-sm text-slate-400 mt-1">Uploaded-data run · row {data.row + 1}</p>
    </div>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <KpiCard label="Churn probability" value={percent(data.churn_probability)} sub="Calibrated baseline output" tone={data.churn_probability >= .5 ? "bad" : "good"} />
      <KpiCard label="Observed outcome" value={data.observed_outcome} sub="Provided label in upload" />
      <KpiCard label="Risk level" value={data.risk_level} sub="Probability threshold band" tone="warn" />
      <KpiCard label="Feature signals" value={String(data.drivers.length)} sub="Non-zero model contributions" />
    </div>
    <Card className="p-1">
      <CardHeader title="Customer attributes" subtitle="Raw fields supplied in the uploaded CSV" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4 px-5 pb-5">
        {Object.entries(data.attributes).map(([key, value]) => <div key={key}><div className="text-[11px] text-slate-400">{key}</div><div className="text-sm font-medium text-slate-700 truncate">{String(value ?? "—")}</div></div>)}
      </div>
    </Card>
    <Card className="p-1">
      <CardHeader title="Risk drivers" subtitle={data.note} />
      <div className="h-96 px-3 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={driverData} layout="vertical" margin={{ left: 70, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={150} fontSize={11} />
            <Tooltip formatter={(value: any) => Number(value).toFixed(5)} />
            <Bar dataKey="contribution" name="Model contribution">
              {driverData.map((entry: any) => <Cell key={entry.feature} fill={entry.contribution >= 0 ? "#e11d48" : "#10b981"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
    <Card className="p-5">
      <CardHeader title="Contribution details" subtitle="Positive values push the baseline toward churn; negative values push it away." />
      <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400 border-b"><th className="py-2">Feature</th><th>Encoded value</th><th>Contribution</th><th>Interpretation</th></tr></thead>
        <tbody>{driverData.map((driver: any) => <tr key={driver.feature} className="border-b border-slate-100"><td className="py-2 font-medium">{driver.feature}</td><td>{driver.value.toFixed(3)}</td><td className={driver.contribution >= 0 ? "text-rose-600" : "text-emerald-600"}>{driver.contribution.toFixed(5)}</td><td>{driver.contribution >= 0 ? "Raises predicted churn" : "Lowers predicted churn"}</td></tr>)}</tbody>
      </table></div>
    </Card>
  </div>;
}
