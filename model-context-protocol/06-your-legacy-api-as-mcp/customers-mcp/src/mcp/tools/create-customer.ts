import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customer-service.ts";
import { z } from "zod";

export function registerCreateCustomerTool(
  server: McpServer,
  service: CustomerService,
) {
  server.registerTool(
    "create-customer",
    {
      description: "Create a new customer",
      inputSchema: {
        name: z.string().describe("Customer's name"),
        phone: z.string().describe("Customer's phone number"),
      },
      outputSchema: {
        message: z.string().describe("Confirmation message"),
        id: z.string().describe("Newly created customer's ID"),
      },
    },
    async ({ name, phone }) => {
      try {
        const customer = await service.createCustomer({ name, phone });
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(customer, null, 2),
            },
          ],
          structuredContent: customer,
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
