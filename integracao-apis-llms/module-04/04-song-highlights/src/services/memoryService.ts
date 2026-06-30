import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { config } from "../config.ts";

export interface MemoryService {
  checkPointer: PostgresSaver;
  store: PostgresStore;
}

export async function getMemoryService(): Promise<MemoryService> {
  const dbUri = config.memory.dbUri;
  const store = PostgresStore.fromConnString(dbUri);
  const checkPointer = PostgresSaver.fromConnString(dbUri);

  await store.setup();
  await checkPointer.setup();

  console.log("Memory service initialized with Postgres store and saver.");

  return {
    checkPointer,
    store,
  };
}
