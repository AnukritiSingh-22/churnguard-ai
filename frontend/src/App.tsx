import React from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import RiskExplorer from "./pages/RiskExplorer";
import Customer360 from "./pages/Customer360";
import ModelPerformance from "./pages/ModelPerformance";
import Survival from "./pages/Survival";
import Drift from "./pages/Drift";
import DataQuality from "./pages/DataQuality";
import Datasets from "./pages/Datasets";
import Experiments from "./pages/Experiments";
import RevenueForecast from "./pages/RevenueForecast";
import Workspace from "./pages/Workspace";
import WorkspaceCustomer from "./pages/WorkspaceCustomer";
import Login from "./pages/Login";
import { DatasetProvider } from "./context/DatasetContext";

export default function App() {
  const [loggedIn, setLoggedIn] = React.useState(() => Boolean(localStorage.getItem("churnguard.token")));
  if (!loggedIn) return <Login onLogin={() => setLoggedIn(true)} />;
  return (
    <DatasetProvider>
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/risk-explorer" element={<RiskExplorer />} />
          <Route path="/customers/:id" element={<Customer360 />} />
          <Route path="/models" element={<ModelPerformance />} />
          <Route path="/survival" element={<Survival />} />
          <Route path="/drift" element={<Drift />} />
          <Route path="/data-quality" element={<DataQuality />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/revenue-forecast" element={<RevenueForecast />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="/workspace/runs/:runId/customers/:row" element={<WorkspaceCustomer />} />
        </Routes>
      </Layout>
    </HashRouter>
    </DatasetProvider>
  );
}
