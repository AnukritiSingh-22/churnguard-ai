import React, { createContext, useContext, useEffect, useState } from "react";
import { api } from "../api/client";

export type DatasetInfo = { key: string; name: string; domain: string; rows?: number; status: string; notes?: string };
export type Schema = {
  key: string; name: string; domain: string; currency: string; rows: number;
  segment_col: string; segment_label: string; segment_values: string[];
  columns: { key: string; label: string; type: string }[];
  capabilities: { survival: boolean; counterfactual: boolean };
  label_note: string;
};
type Ctx = {
  dataset: string; setDataset: (k: string) => void; datasets: DatasetInfo[];
  schema: Schema | null; error: string | null;
};

const DatasetContext = createContext<Ctx | null>(null);
const STORAGE_KEY = "churnguard.dataset";

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [dataset, setDatasetState] = useState<string>(() => {
    try { return localStorage.getItem(STORAGE_KEY) || "telco"; } catch { return "telco"; }
  });
  const [schema, setSchema] = useState<Schema | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDatasets()
      .then((d) => {
        const active = (d.datasets as DatasetInfo[]).filter((x) => x.status === "active");
        setDatasets(active);
        if (active.length && !active.some((x) => x.key === dataset)) setDatasetState(active[0].key);
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setSchema(null);
    api.ds.schema(dataset).then(setSchema).catch((e) => setError(e.message));
  }, [dataset]);

  const setDataset = (k: string) => {
    setDatasetState(k);
    try { localStorage.setItem(STORAGE_KEY, k); } catch { /* storage unavailable: fine */ }
  };

  return <DatasetContext.Provider value={{ dataset, setDataset, datasets, schema, error }}>{children}</DatasetContext.Provider>;
}

export function useDataset(): Ctx {
  const c = useContext(DatasetContext);
  if (!c) throw new Error("useDataset must be used inside DatasetProvider");
  return c;
}

export const money = (n: number | null | undefined, cur = "") =>
  n == null ? "n/a"
    : `${cur}${Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(2) + "M" : Math.abs(n) >= 1e4 ? (n / 1e3).toFixed(1) + "K" : Math.round(n).toLocaleString()}`;
export const pct = (v: number | null | undefined, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
