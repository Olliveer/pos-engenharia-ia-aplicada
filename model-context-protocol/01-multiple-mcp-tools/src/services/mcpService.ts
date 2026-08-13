import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { getMongoDBTool } from "../tools/mongodbTool.ts";
import { csvToJSONTool } from "../tools/csvToJSONTool.ts";
import { getFsTool } from "../tools/fsTool.ts";

export const getMCPTools = async () => {
  const client = new MultiServerMCPClient({
    mcpServers: {
      ...getMongoDBTool(),
      ...getFsTool(),
    },
    onMessage: (log, source) => {
      console.log("MCP Message:", log.data, "from:", source.server);
    },
  });

  const mcpTools = await client.getTools();

  return [...mcpTools, csvToJSONTool()];
};
