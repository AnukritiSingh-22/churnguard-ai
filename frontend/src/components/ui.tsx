import React from "react";
import { Loader2, AlertTriangle } from "lucide-react";

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white border border-slate-200 rounded-xl shadow-sm ${className}`}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between px-5 pt-4 pb-2">
      <div>
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function KpiCard({
  label, value, sub, tone = "neutral",
}: {
  label: string; value: string; sub?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const toneMap: Record<string, string> = {
    neutral: "text-slate-500",
    good: "text-emerald-600",
    warn: "text-amber-600",
    bad: "text-rose-600",
  };
  return (
    <Card className="p-4">
      <div className="text-xs font-medium text-slate-400">{label}</div>
      <div className="text-2xl font-semibold text-slate-900 mt-1">{value}</div>
      {sub && <div className={`text-xs mt-1 font-medium ${toneMap[tone]}`}>{sub}</div>}
    </Card>
  );
}

export function RiskBadge({ level }: { level: string }) {
  const map: Record<string, string> = {
    Low: "bg-emerald-50 text-emerald-700 border-emerald-200",
    Medium: "bg-amber-50 text-amber-700 border-amber-200",
    High: "bg-orange-50 text-orange-700 border-orange-200",
    Critical: "bg-rose-50 text-rose-700 border-rose-200",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold border ${map[level] || "bg-slate-50 text-slate-600 border-slate-200"}`}>
      {level}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    stable: "bg-emerald-50 text-emerald-700 border-emerald-200",
    healthy: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warning: "bg-amber-50 text-amber-700 border-amber-200",
    significant_drift: "bg-rose-50 text-rose-700 border-rose-200",
    cleared: "bg-emerald-50 text-emerald-700 border-emerald-200",
    flagged_for_review: "bg-amber-50 text-amber-700 border-amber-200",
    excluded: "bg-rose-50 text-rose-700 border-rose-200",
  };
  const label = status.replace(/_/g, " ");
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold border capitalize ${map[status] || "bg-slate-50 text-slate-600 border-slate-200"}`}>
      {label}
    </span>
  );
}

export function Loading({ label = "Loading real data from the API..." }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400 py-16 justify-center">
      <Loader2 size={16} className="animate-spin" />
      {label}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-lg px-4 py-3">
      <AlertTriangle size={16} />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center text-sm text-slate-400 py-12">
      {message}
    </div>
  );
}

export function SimulationTag() {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold border bg-violet-50 text-violet-700 border-violet-200 uppercase tracking-wide">
      Simulation mode
    </span>
  );
}
