export async function authFetch(
  input: RequestInfo,
  init: RequestInit = {}
): Promise<Response> {
  const token = localStorage.getItem("token"); // or use cookie/session

  const headers: HeadersInit = {
    ...init.headers,
    "Content-Type": "application/json",
  };

  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(input, { ...init, headers });

  if (response.status === 401) {
    window.location.href = "/login";
  }

  return response;
}
