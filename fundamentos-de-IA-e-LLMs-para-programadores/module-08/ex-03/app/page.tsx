"use client";

import { useSession, signOut } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function Home() {
  const { data, isPending } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (!isPending && !data?.session) {
      router.push("/login");
    }
  }, [data, isPending, router]);

  if (isPending) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center p-24">
        <p className="text-lg">Carregando...</p>
      </main>
    );
  }

  if (!data?.session) {
    return null;
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">Hello World</h1>
        <p className="text-xl text-gray-600 mb-8">
          Logado como <span className="font-semibold">{data.user?.email || data.user?.name}</span>
        </p>
        <button
          onClick={async () => {
            await signOut();
            router.push("/login");
          }}
          className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
        >
          Sair
        </button>
      </div>
    </main>
  );
}
