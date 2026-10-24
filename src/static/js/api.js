// The API of this origin, with a bearer token.
//
// The access token lives in memory only; the single-use refresh token in
// sessionStorage, or in localStorage with "Stay signed in".

const REMEMBER_KEY = "nocturna.remember";
const REFRESH_KEY = "nocturna.refresh";

export class AuthError extends Error {}

const session = { access: null, expiresAt: 0 };
let refreshing = null;

function tokenStore() {
  try {
    return localStorage.getItem(REMEMBER_KEY) === "1" ? localStorage : sessionStorage;
  } catch {
    return null;
  }
}

export function readRefreshToken() {
  try {
    return tokenStore()?.getItem(REFRESH_KEY) ?? null;
  } catch {
    return null;
  }
}

function saveTokens(data) {
  session.access = data.access_token;
  session.expiresAt = Date.now() + (data.expires_in ?? 3600) * 1000;
  try {
    tokenStore()?.setItem(REFRESH_KEY, data.refresh_token);
  } catch {
    // Storage blocked: the session lasts as long as the page
  }
}

function clearTokens() {
  session.access = null;
  session.expiresAt = 0;
  for (const store of [localStorage, sessionStorage]) {
    try {
      store.removeItem(REFRESH_KEY);
    } catch {
      // nothing stored
    }
  }
  try {
    localStorage.removeItem(REMEMBER_KEY);
  } catch {
    // nothing stored
  }
}

async function rotateTokens() {
  // A refresh token works once; presenting a used one ends every session. Tabs
  // that share localStorage take turns, and each reads the token inside the lock.
  const rotate = async () => {
    const token = readRefreshToken();
    if (!token) throw new AuthError("signed out");
    const response = await fetch("/api/v1/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: token }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.success) {
      clearTokens();
      throw new AuthError("session ended");
    }
    saveTokens(body.data);
  };
  return navigator.locks ? navigator.locks.request("nocturna-refresh", rotate) : rotate();
}

export function freshTokens() {
  // One rotation at a time in this tab, however many requests need it
  if (!refreshing) refreshing = rotateTokens().finally(() => { refreshing = null; });
  return refreshing;
}

export async function api(method, path, body) {
  if (!session.access || Date.now() > session.expiresAt - 60000) await freshTokens();
  const send = () => fetch(path, {
    method,
    headers: {
      Authorization: `Bearer ${session.access}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const used = session.access;
  let response = await send();
  if (response.status === 401) {
    if (session.access === used) await freshTokens();
    response = await send();
    if (response.status === 401) throw new AuthError("unauthorized");
  }
  const json = await response.json().catch(() => ({}));
  return {
    ok: response.ok && json.success !== false,
    status: response.status,
    data: json.data,
    error: json.error || json.message || `HTTP ${response.status}`,
    message: json.message,
  };
}

export async function signIn(username, password, remember) {
  const response = await fetch("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || !body.success) throw new Error(body.error || `HTTP ${response.status}`);
  clearTokens();
  try {
    if (remember) localStorage.setItem(REMEMBER_KEY, "1");
  } catch {
    // Storage blocked: signed in for this page only
  }
  saveTokens(body.data);
}

export async function signOut() {
  const refreshToken = readRefreshToken();
  try {
    if (session.access) {
      await fetch("/api/v1/auth/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${session.access}`, "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    }
  } finally {
    clearTokens();
  }
}
