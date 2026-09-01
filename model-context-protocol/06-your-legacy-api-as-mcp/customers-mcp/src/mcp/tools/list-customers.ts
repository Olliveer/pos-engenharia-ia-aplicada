import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customer-service.ts";
import { z } from "zod";
import { CustomerSchema } from "../../domain/customer.ts";

export function registerListCustomersTool(
  server: McpServer,
  service: CustomerService,
) {
  server.registerTool(
    "list-customers",
    {
      description: "List all customers",
      inputSchema: {},
      outputSchema: {
        customers: z.array(CustomerSchema).describe("List of customers"),
      },
    },
    async () => {
      try {
        const customers = await service.listCustomers();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ customers }, null, 2),
            },
          ],
          structuredContent: { customers },
        };
      } catch (error) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `Error listing customers: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    },
  );
}
