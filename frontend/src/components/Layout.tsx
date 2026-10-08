import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard, Search, Activity, GitBranch, Database, FlaskConical, Radio, ShieldAlert, Settings, CircleDot, TrendingUp, Upload,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/risk-explorer", label: "Risk Explorer", icon: Search },
  { to: "/models", label: "Model Performance", icon: Activity },
  { to: "/survival", label: "Survival Analysis", icon: GitBranch },
  { to: "/drift", label: "Drift & Monitoring", icon: Radio },
  { to: "/data-quality", label: "Data Quality & Leakage", icon: ShieldAlert },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/experiments", label: "Experiments", icon: FlaskConical },
  { to: "/revenue-forecast", label: "Revenue Forecast", icon: TrendingUp },
  { to: "/workspace", label: "My Workspace", icon: Upload },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const { dataset, setDataset, datasets, schema, error } = useDataset();
  const navigate = useNavigate();
  const email = localStorage.getItem("churnguard.email") || "Workspace user";
  const logout = () => {
    localStorage.removeItem("churnguard.token");
    localStorage.removeItem("churnguard.email");
    window.location.reload();
  };
  return (
    <div className="flex h-screen bg-slate-50 text-slate-900">
      <aside className="w-64 shrink-0 border-r border-slate-200 bg-white flex flex-col">
        <div className="h-16 flex items-center gap-2 px-5 border-b border-slate-200">
          <div className="h-8 w-8 rounded-lg bg-brand-700 flex items-center justify-center text-white font-bold text-sm">CG</div>
          <div>
            <div className="font-semibold text-sm leading-tight">ChurnGuard AI</div>
            <div className="text-[11px] text-slate-400 leading-tight">Customer Intelligence</div>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"}`}>
              <item.icon size={16} strokeWidth={2} />
              {item.label}
              {item.to === "/survival" && schema && !schema.capabilities.survival && (
                <span className="ml-auto text-[9px] font-semibold text-slate-400 border border-slate-200 rounded px-1">N/A</span>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-slate-200 p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <CircleDot size={12} className="text-emerald-500" /> Live model &middot; real predictions
          </div>
          <div className="flex items-center gap-2 px-1 py-1.5">
            <div className="h-7 w-7 rounded-full bg-brand-100 text-brand-700 flex items-center justify-center text-xs font-semibold">A</div>
            <div className="text-xs min-w-0"><div className="font-medium text-slate-700 truncate max-w-[130px]">{email}</div><button onClick={logout} className="text-slate-400 hover:text-brand-700">Sign out</button></div>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 shrink-0 border-b border-slate-200 bg-white flex items-center justify-between px-6 gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <label htmlFor="dataset-select" className="text-xs font-medium text-slate-400 shrink-0">Dataset</label>
            <select id="dataset-select" value={dataset}
              onChange={(e) => { setDataset(e.target.value); navigate("/"); }}
              className="text-sm font-medium text-slate-800 border border-slate-200 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-brand-200 max-w-md">
              {datasets.length === 0 && <option value={dataset}>{dataset}</option>}
              {datasets.map((d) => <option key={d.key} value={d.key}>{d.name} — {d.domain}</option>)}
            </select>
            <span className="text-xs text-slate-400 truncate hidden md:inline">
              {error ? "API unreachable" : schema ? `${schema.rows.toLocaleString()} customers · ${schema.label_note}` : "Loading…"}
            </span>
          </div>
          <Settings size={16} className="text-slate-500 shrink-0" />
        </header>
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
