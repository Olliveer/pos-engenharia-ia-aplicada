import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customer-service.ts";
import {
  CustomerMutationSchema,
  CustomerUpdateSchema,
} from "../../domain/customer.ts";
import { z } from "zod";

export function registerDeleteCustomerTool(
  server: McpServer,
  service: CustomerService,
) {
  server.registerTool(
    "delete-customer",
    {
      description: "Delete a customer",
      inputSchema: {
        _id: z.string().describe("customer id for deletion"),
      },
      outputSchema: CustomerMutationSchema.shape,
    },
    async ({ _id }) => {
      try {
        const result = await service.deleteCustomer(_id);

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
              text: `Error delete customer: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    },
  );
}
