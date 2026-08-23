import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { decrypt, encrypt } from "./service.ts";

export const server = new McpServer({
  name: "@Olliveer/cipher-suite-mcp",
  version: "0.0.1",
});

server.registerTool(
  "encrypt-message",
  {
    description: "Encrypts a message using a given key",
    inputSchema: z
      .object({
        message: z.string().describe("The message to encrypt"),
        key: z.string().optional().describe("The encryption key"),
        encryptionKey: z
          .string()
          .optional()
          .describe("The encryption key (alias)"),
      })
      .refine((data) => Boolean(data.key ?? data.encryptionKey), {
        message: "A key or encryptionKey is required",
        path: ["key"],
      }),
    outputSchema: z.object({
      encryptedMessage: z.string().describe("The encrypted message"),
    }),
  },
  async ({ message, key, encryptionKey }) => {
    try {
      const resolvedKey = encryptionKey ?? key;
      if (!resolvedKey) {
        throw new Error("A key or encryptionKey is required");
      }

      const encryptedMessage = encrypt(message, resolvedKey);
      return {
        content: [{ type: "text", text: encryptedMessage }],
        structuredContent: { encryptedMessage },
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Error encrypting message: ${errorMessage}`,
          },
        ],
        structuredContent: { error: errorMessage },
      };
    }
  },
);

server.registerTool(
  "decrypt-message",
  {
    description: "Decrypts a message using a given key",
    inputSchema: z
      .object({
        encryptedMessage: z.string().describe("The encrypted message"),
        key: z.string().optional().describe("The encryption key"),
        encryptionKey: z
          .string()
          .optional()
          .describe("The encryption key (alias)"),
      })
      .refine((data) => Boolean(data.key ?? data.encryptionKey), {
        message: "A key or encryptionKey is required",
        path: ["key"],
      }),
    outputSchema: z.object({
      decryptedMessage: z.string().describe("The decrypted message"),
    }),
  },
  async ({ encryptedMessage, key, encryptionKey }) => {
    try {
      const resolvedKey = encryptionKey ?? key;
      if (!resolvedKey) {
        throw new Error("A key or encryptionKey is required");
      }

      const decryptedMessage = decrypt(encryptedMessage, resolvedKey);
      return {
        content: [{ type: "text", text: decryptedMessage }],
        structuredContent: { decryptedMessage },
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Error decrypting message: ${errorMessage}`,
          },
        ],
        structuredContent: { error: errorMessage },
      };
    }
  },
);

server.registerResource(
  "encryption://info",
  "encryption://info",
  {
    description: "describes the encryption capabilities of this MCP",
  },
  () => ({
    contents: [
      {
        uri: "encryption://info",
        mimeType: "text/plain",
        text: `
Algorithm : AES-256-CBC
Key derivation: scrypt (passphrase + fixed server salt → 32-byte key)
Output format: <16-byte IV in hex>:<ciphertext in hex>  (separated by ":")
Notes:
  - Users pass any passphrase — the server derives a strong 32-byte key automatically using scrypt.
  - A random IV is generated for every encryption — the same message encrypted twice will produce different output.
  - Use the exact same passphrase to decrypt.
  - Keep the full "iv:ciphertext" string to decrypt later.
                `.trim(),
      },
    ],
  }),
);

server.registerPrompt(
  "encrypt_message_prompt",
  {
    description:
      "Prompt to encrypt a plain-text message using the encrypt_message tool",
    argsSchema: {
      message: z.string().describe("The message to encrypt"),
      encryptionKey: z
        .string()
        .describe(
          "Any passphrase to use for encryption — the server derives a strong key from it automatically",
        ),
    },
  },
  ({ message, encryptionKey }) => ({
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: `Please encrypt the following message using the encrypt_message tool.\nMessage: ${message}\nEncryption key: ${encryptionKey}`,
        },
      },
    ],
  }),
);
