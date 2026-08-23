import { describe, it, after, before } from "node:test";
import assert from "node:assert";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { createClient } from "./helpers.ts";

async function encryptMessage(client: Client, message: string, key: string) {
  const result = (await client.callTool({
    name: "encrypt-message",
    arguments: {
      message,
      encryptionKey: key,
    },
  })) as unknown as { structuredContent: { encryptedMessage: string } };

  return result;
}

async function decryptMessage(
  client: Client,
  encryptedMessage: string,
  encryptionKey: string,
) {
  const result = (await client.callTool({
    name: "decrypt-message",
    arguments: {
      encryptedMessage,
      encryptionKey,
    },
  })) as unknown as { structuredContent: { decryptedMessage: string } };

  return result;
}

describe("MCP Server", () => {
  let client: Client;
  const encryptionKey = "my-secret-key";

  before(async () => {
    client = await createClient();
  });

  after(async () => {
    await client.close();
  });

  it("Should encrypt a message correctly", async () => {
    const message = "Hello, World!";
    const result = await encryptMessage(client, message, encryptionKey);
    assert.ok(
      result.structuredContent.encryptedMessage.length > 60,
      "Encrypted message is not long enough",
    );
  });

  it("Should decrypt a message correctly", async () => {
    const message = "Hello, sssss!";
    const {
      structuredContent: { encryptedMessage },
    } = await encryptMessage(client, message, encryptionKey);

    const result = await decryptMessage(
      client,
      encryptedMessage,
      encryptionKey,
    );

    assert.deepStrictEqual(
      result.structuredContent.decryptedMessage,
      message,
      "Decrypted message does not match original",
    );
  });

  it("should list de encpryption resources", async () => {
    const { resources } = await client.listResources();
    const info = resources.find((item) => item.uri === "encryption://info");

    assert.ok(info, "encryption://info resource should be listed");
  });

  it("should return the encrypt_message_prompt", async () => {
    const result = await client.getPrompt({
      name: "encrypt_message_prompt",
      arguments: {
        message: "Secret text",
        encryptionKey,
      },
    });

    const item = result.messages.at(0)?.content as unknown as { text: string };
    const expected = `Please encrypt the following message using the encrypt_message tool.
Message: Secret text
Encryption key: my-super-passphrase`;
    assert.deepStrictEqual(
      item.text,
      expected,
      "Prompt should be in the correct format",
    );
  });
});
