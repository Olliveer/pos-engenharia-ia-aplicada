import { MultiServerMCPClient } from "@langchain/mcp-adapters";

export const getMcpTools = async () => {
  const mcpClient = new MultiServerMCPClient({
    filesystem: {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", process.cwd()],
    },
  });

  return mcpClient.getTools();
};
