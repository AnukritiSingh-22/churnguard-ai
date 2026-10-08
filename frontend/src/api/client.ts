const BASE = "/api";

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("churnguard.token");
  return token ? { Authorization: "Bearer " + token } : {};
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function post<T>(path: string, data: any): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

const qs = (params: Record<string, string | number | undefined>) => {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") query.set(key, String(value));
  });
  return query.toString();
};
const enc = encodeURIComponent;

const ds = {
  schema: (key: string) => get<any>(`/datasets/${key}/schema`),
  summary: (key: string) => get<any>(`/datasets/${key}/summary`),
  customers: (key: string, params: Record<string, string | number | undefined> = {}) => get<any>(`/datasets/${key}/customers?${qs(params)}`),
  customer: (key: string, id: string) => get<any>(`/datasets/${key}/customers/${enc(id)}`),
  explanation: (key: string, id: string) => get<any>(`/datasets/${key}/customers/${enc(id)}/explanation`),
  recommendations: (key: string, id: string) => get<any>(`/datasets/${key}/customers/${enc(id)}/recommendations`),
  survival: (key: string, id: string) => get<any>(`/datasets/${key}/customers/${enc(id)}/survival`),
  counterfactual: (key: string, id: string) => get<any>(`/datasets/${key}/customers/${enc(id)}/counterfactual`),
  survivalPopulation: (key: string) => get<any>(`/datasets/${key}/survival/population`),
  models: (key: string) => get<any>(`/datasets/${key}/models`),
  calibration: (key: string) => get<any>(`/datasets/${key}/calibration`),
  drift: (key: string) => get<any>(`/datasets/${key}/drift`),
  leakage: (key: string) => get<any>(`/datasets/${key}/leakage-audit`),
  dataQuality: (key: string) => get<any>(`/datasets/${key}/data-quality`),
  experiments: (key: string) => get<any>(`/datasets/${key}/experiments`),
};

export const api = {
  ds,
  health: () => get<any>("/health"),
  dashboardSummary: () => get<any>("/dashboard/summary"),
  listCustomers: (params: Record<string, string | number | undefined>) => get<any>(`/customers?${qs(params)}`),
  getCustomer: (id: string) => get<any>(`/customers/${id}`),
  getCustomerRisk: (id: string) => get<any>(`/customers/${id}/risk`),
  getCustomerExplanation: (id: string) => get<any>(`/customers/${id}/explanation`),
  getCustomerSurvival: (id: string) => get<any>(`/customers/${id}/survival`),
  getCustomerCounterfactual: (id: string) => get<any>(`/customers/${id}/counterfactual`),
  getCustomerRecommendations: (id: string) => get<any>(`/customers/${id}/recommendations`),
  getSurvivalPopulation: () => get<any>("/survival/population"),
  listModels: () => get<any>("/models"),
  getModelMetrics: (name: string) => get<any>(`/models/${name}/metrics`),
  getProductionCalibration: () => get<any>("/models/production/calibration"),
  getDataQuality: () => get<any>("/data-quality"),
  getLeakageAudit: () => get<any>("/leakage-audit"),
  getDrift: () => get<any>("/drift"),
  getDatasets: () => get<any>("/datasets"),
  getDatasetMetrics: (key: string) => get<any>(`/datasets/${key}/metrics`),
  getDatasetCustomers: (key: string, params: Record<string, string | number | undefined> = {}) => get<any>(`/datasets/${key}/customers?${qs(params)}`),
  getDatasetLeakageAudit: (key: string) => get<any>(`/datasets/${key}/leakage-audit`),
  getDatasetDrift: (key: string) => get<any>(`/datasets/${key}/drift`),
  getExperiments: () => get<any>("/experiments"),
  predict: (payload: any) => post<any>("/predict", payload),
  optimizeRetention: (payload: any) => post<any>("/retention/optimize", payload),
  revenueForecast: (horizon = 3) => get<any>(`/revenue/forecast?horizon=${horizon}`),
  monitoringStatus: () => get<any>("/monitoring/status"),
  reviewQueue: (limit = 50) => get<any>(`/review-queue?limit=${limit}`),
  powerBiManifest: () => get<any>("/powerbi/manifest"),
  register: (email: string, password: string) => post<any>("/auth/register", { email, password }),
  login: (email: string, password: string) => post<any>("/auth/login", { email, password }),
  uploads: () => get<any>("/workspace/uploads"),
  uploadCsv: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(`${BASE}/workspace/uploads`, { method: "POST", headers: authHeaders(), body });
    if (!res.ok) throw new Error(`POST /workspace/uploads failed: ${res.status} ${await res.text()}`);
    return res.json();
  },
  uploadProfile: (id: string) => get<any>(`/workspace/uploads/${encodeURIComponent(id)}`),
  trainUpload: (id: string, target: string, customerId?: string) =>
    post<any>(`/workspace/uploads/${encodeURIComponent(id)}/train`, { target, customer_id: customerId }),
  workspaceRun: (id: string) => get<any>(`/workspace/runs/${encodeURIComponent(id)}`),
  workspaceCustomer: (runId: string, row: number) => get<any>(`/workspace/runs/${encodeURIComponent(runId)}/customers/${row}`),
};
