import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../src/server.ts";
import { config } from "../src/config.ts";
import { OpenRouterService } from "../src/openRouterService.ts";

console.assert(
  process.env.OPENROUTER_API_KEY,
  "OPENROUTER_API_KEY is not set in environment variables",
);

test.todo("routes to chepest model by default", async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: {
        ...config.provider.sort,
        by: "price",
      },
    },
  };

  const routerService = new OpenRouterService(customConfig);
  const app = createServer(routerService);

  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: { question: "What is the capital of France?" },
  });

  assert.equal(response.statusCode, 200);
  const body = response.json();
  console.log("Response body:", body);
  assert.ok(
    body.model.includes("inclusionai/ling-2.6-1t"),
    "Expected to use the cheaper model",
  );
});
// test.todo("POST /chat should return a response from the LLM", async () => {});
