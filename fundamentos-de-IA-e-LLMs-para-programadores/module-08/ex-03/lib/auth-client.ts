import { authClient } from "better-auth/client";

export const { signIn, signOut, useSession } = authClient();
