import { describe, it, after, before } from "node:test";
import assert from "node:assert";
import { createClient } from "../helpers.ts";
import { Client } from "@modelcontextprotocol/sdk/client";

describe("Customer Prompts", async () => {
  let client: Client;

  before(async () => {
    client = await createClient();
  });

  after(async () => {
    await client.close();
  });

  it("should return the find_customer_prompt", async () => {
    const result = await client.getPrompt({
      name: "find-customer-prompt",
      arguments: { name: "John" },
    });
    const text = result.messages[0].content;
    assert.ok(
      "text" in text && text.text.includes("get_customer"),
      "Prompt should reference the get_customer tool",
    );
    assert.ok(
      "text" in text && text.text.includes("John"),
      "Prompt should include the query",
    );
  });
});
