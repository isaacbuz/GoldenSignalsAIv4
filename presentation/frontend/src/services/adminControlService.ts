import { authFetch } from "./authFetch";

export async function toggleAgent(id: string) {
  await authFetch(`/admin/agent/${id}/toggle`, { method: "POST" });
}

export async function restartAgent(id: string) {
  await authFetch(`/admin/agent/${id}/restart`, { method: "POST" });
}

export async function purgeSignals() {
  await authFetch(`/admin/purge-signals`, { method: "POST" });
}
