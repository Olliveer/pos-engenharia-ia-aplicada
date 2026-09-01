import { describe, it, after, before } from "node:test";
import assert from "node:assert";
import { createClient } from "../helpers.ts";

describe("Customers Resource suite", () => {
  let client: Awaited<ReturnType<typeof createClient>>;
  before(async () => {
    client = await createClient();
  });

  after(async () => {
    await client.close();
  });

  it("should list  customers://api-info resource", async () => {
    const { resources } = await client.listResources();
    const info = resources.find((r) => r.uri === "customers://api-info");

    assert.deepStrictEqual(
      info?.description,
      "Provides information about the customers API",
      "Expected customers://api-info resource to have correct description",
    );
  });
});
