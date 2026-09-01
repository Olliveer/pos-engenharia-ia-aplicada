import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customer-service.ts";
import {
  CustomerMutationSchema,
  CustomerUpdateSchema,
} from "../../domain/customer.ts";

export function registerUpdateCustomerTool(
  server: McpServer,
  service: CustomerService,
) {
  server.registerTool(
    "update-customer",
    {
      description: "Update a customer",
      inputSchema: CustomerUpdateSchema.shape,
      outputSchema: CustomerMutationSchema.shape,
    },
    async (customer) => {
      try {
        const result = await service.updateCustomer(customer);

        return {
          content: [
            {
              type: "text",
              text: result.message ?? "",
            },
          ],
          structuredContent: result,
        };
      } catch (error) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `Error updating customer: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    },
  );
}
