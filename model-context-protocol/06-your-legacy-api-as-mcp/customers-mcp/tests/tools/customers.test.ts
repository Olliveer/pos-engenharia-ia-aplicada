import { describe, it, after, before } from "node:test";
import assert from "node:assert";
import { createClient } from "../helpers.ts";
import type {
  Customer,
  CustomerMutation,
  CustomerUpdate,
} from "../../src/domain/customer.ts";

type CustomersResult = {
  structuredContent: { customers: Customer[] };
};
type CustomerResult = {
  structuredContent: { customer: Customer };
};
type CustomerMutationResult = {
  structuredContent: CustomerMutation;
};

describe("Customers MCP suite", () => {
  let client: Awaited<ReturnType<typeof createClient>>;
  before(async () => {
    client = await createClient();
  });

  after(async () => {
    await client.close();
  });

  it("should list all customers", async () => {
    const result = (await client.callTool({
      name: "list-customers",
      arguments: {},
    })) as unknown as CustomersResult;
    assert(
      Array.isArray(result.structuredContent.customers),
      "Expected customers to be an array of Customer objects",
    );
  });

  it("should create a new customer", async () => {
    const customerData = { name: "John Doe", phone: "123-456-7890" };
    const result = (await client.callTool({
      name: "create-customer",
      arguments: customerData,
    })) as unknown as CustomerMutationResult;

    assert.ok(
      result.structuredContent.id,
      "Expected a valid customer ID to be returned",
    );

    assert.deepStrictEqual(
      result.structuredContent.message,
      `user ${customerData.name} created!`,
      "Expected confirmation message to match the created customer name",
    );
  });

  it("should retrieve a new customer", async () => {
    const customerData = { name: "Euu", phone: "123-456-7890" };

    (await client.callTool({
      name: "create-customer",
      arguments: customerData,
    })) as unknown as CustomerMutationResult;

    const result = (await client.callTool({
      name: "get-customer",
      arguments: customerData,
    })) as unknown as CustomerResult;

    assert.ok(
      result.structuredContent.customer._id,
      "Expected a valid customer ID to be returned",
    );

    assert.deepStrictEqual(
      result.structuredContent.customer.name,
      customerData.name,
      "Expected customer name to match the created customer name",
    );
  });

  it("should update a customer", async () => {
    const customerData = {
      name: "Eu mesmo",
      phone: "123-456-7890",
    };

    const {
      structuredContent: { id },
    } = (await client.callTool({
      name: "create-customer",
      arguments: customerData,
    })) as unknown as CustomerMutationResult;

    const result = (await client.callTool({
      name: "update-customer",
      arguments: {
        _id: id,
        name: "eu mesmo updated",
        phone: "123-456-7890",
      } as CustomerUpdate,
    })) as unknown as CustomerMutationResult;

    assert.ok(result.structuredContent.message, "Should contain message");

    assert.deepStrictEqual(
      result.structuredContent.id,
      id,
      "Expected customer id to match the updated customer id",
    );
  });

  it("should delete a customer", async () => {
    const customerData = {
      name: "Eu del",
      phone: "123-456-7890",
    };

    const {
      structuredContent: { id },
    } = (await client.callTool({
      name: "create-customer",
      arguments: customerData,
    })) as unknown as CustomerMutationResult;

    const result = (await client.callTool({
      name: "delete-customer",
      arguments: {
        _id: id,
      } as CustomerUpdate,
    })) as unknown as CustomerMutationResult;

    assert.ok(result.structuredContent.message, "Should contain message");

    assert.deepStrictEqual(
      result.structuredContent.id,
      id,
      "Expected customer id to match the updated customer id",
    );
  });
});
