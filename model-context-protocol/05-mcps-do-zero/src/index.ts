import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { decrypt, encrypt } from "./service.ts";
import { server } from "./mcp.ts";

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP server is running. Waiting for requests...");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
