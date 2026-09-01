import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customer-service.ts";
import {
  type CustomerQuery,
  CustomerQuerySchema,
  CustomerSchema,
} from "../../domain/customer.ts";

export function registerGetCustomerTool(
  server: McpServer,
  service: CustomerService,
) {
  server.registerTool(
    "get-customer",
    {
      description: "Retrieve a customer by name and phone or by ID",
      inputSchema: CustomerQuerySchema,
      outputSchema: {
        customer: CustomerSchema.nullable().describe(
          "The retrieved customer or null if not found",
        ),
      },
    },
    async (query: CustomerQuery) => {
      try {
        const customer = await service.findCustomer(query);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(customer, null, 2),
            },
          ],
          structuredContent: { customer },
        };
      } catch (error) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `Error creating customer: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    },
  );
}
