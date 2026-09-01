import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../application/customer-service.ts";
import { registerListCustomersTool } from "./tools/list-customers.ts";
import { registerApiInfoResource } from "./resources/api-info.ts";
import { registerCreateCustomerTool } from "./tools/create-customer.ts";
import { registerGetCustomerTool } from "./tools/get-customer.ts";
import { registerFindCustomerPrompt } from "./prompts/find-customer.ts";
import { registerUpdateCustomerTool } from "./tools/update-customer.ts";
import { registerDeleteCustomerTool } from "./tools/delete-customer.ts";

const BASE_URL = "http://localhost:9999/v1";

const service = new CustomerService(BASE_URL);

export const server = new McpServer({
  name: "@erickwendel/ew-customers-mcp",
  version: "0.0.1",
});

registerListCustomersTool(server, service);
registerApiInfoResource(server, BASE_URL);
registerCreateCustomerTool(server, service);
registerGetCustomerTool(server, service);
registerFindCustomerPrompt(server);
registerUpdateCustomerTool(server, service);
registerDeleteCustomerTool(server, service);
