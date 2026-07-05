// Thin wrapper around better-auth's vanilla client (framework-agnostic — no
// React needed, works fine from online.ts's plain-DOM code). Talks to the
// server's /api/auth/* routes (see packages/server/src/auth.ts) and defaults
// to `credentials: "include"` automatically, so the session cookie travels
// on every call without anything extra here.

import { createAuthClient } from 'better-auth/client';
import { SERVER_URL } from './server-url';

export const authClient = createAuthClient({ baseURL: SERVER_URL });

export interface AuthUser {
    id: string;
    name: string;
    email: string;
}

export async function getCurrentUser(): Promise<AuthUser | null> {
    const { data } = await authClient.getSession();
    return data?.user ?? null;
}
