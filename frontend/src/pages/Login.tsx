import React, { useState } from "react";
import { api } from "../api/client";

export default function Login({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [register, setRegister] = useState(false); const [error, setError] = useState("");
  const submit = async (e: React.FormEvent) => { e.preventDefault(); try { const r = register ? await api.register(email, password) : await api.login(email, password); localStorage.setItem("churnguard.token", r.access_token); localStorage.setItem("churnguard.email", r.email); onLogin(); } catch (x: any) { setError(x.message); } };
  return <div className="min-h-screen bg-slate-50 flex items-center justify-center"><form onSubmit={submit} className="bg-white border rounded-xl p-6 w-96 shadow-sm space-y-4">
    <div><h1 className="text-xl font-semibold">ChurnGuard workspace</h1><p className="text-sm text-slate-500 mt-1">Local account for private uploads</p></div>
    {error && <p className="text-sm text-rose-600">{error}</p>}<input required type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full border rounded-lg p-2" />
    <input required minLength={8} type="password" placeholder="Password (8+ characters)" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full border rounded-lg p-2" />
    <button className="w-full bg-brand-700 text-white rounded-lg py-2">{register ? "Create account" : "Sign in"}</button>
    <button type="button" onClick={() => setRegister(!register)} className="w-full text-sm text-brand-700">{register ? "Already have an account? Sign in" : "Create a local account"}</button>
  </form></div>;
}
