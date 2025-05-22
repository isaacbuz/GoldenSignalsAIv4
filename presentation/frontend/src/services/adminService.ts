import { authFetch } from "./authFetch";

export async function fetchAdminStats() {
  const res = await authFetch("/admin/stats");
  return await res.json();
}

export async function fetchRecentSignals() {
  const res = await authFetch("/admin/recent");
  return await res.json();
}

export async function fetchSectors() {
  const res = await authFetch("/admin/sectors");
  return await res.json();
}

export async function fetchAgentStats() {
  const res = await authFetch("/admin/agents");
  return await res.json();
}
