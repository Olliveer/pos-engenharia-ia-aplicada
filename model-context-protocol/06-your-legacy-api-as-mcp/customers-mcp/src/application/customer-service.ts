import type {
  Customer,
  CustomerMutation,
  CustomerQuery,
  CustomerUpdate,
} from "../domain/customer.ts";
import { CustomerHttpClient } from "../infrastructure/customer-http-client.ts";

export class CustomerService {
  private readonly client: CustomerHttpClient;

  constructor(baseUrl: string) {
    // bad practice to create a new instance of CustomerHttpClient for each service instance
    this.client = new CustomerHttpClient(baseUrl);
  }

  async listCustomers(): Promise<Customer[]> {
    return this.client.listCustomers();
  }

  async createCustomer(
    customer: Omit<Customer, "_id">,
  ): Promise<CustomerMutation> {
    return this.client.createCustomer(customer);
  }

  async updateCustomer(customer: CustomerUpdate): Promise<CustomerMutation> {
    return this.client.updateCustomer(customer);
  }

  async deleteCustomer(id: string): Promise<CustomerMutation> {
    return this.client.deleteCustomer(id);
  }

  async findCustomer(query: CustomerQuery): Promise<Customer | null> {
    if (query._id) {
      return this.client.getCustomerById(query._id);
    }
    const customers = await this.client.listCustomers();
    return (
      customers.find(
        (customer) =>
          (query.name ? customer.name === query.name : true) &&
          (query.phone ? customer.phone === query.phone : true),
      ) || null
    );
  }
}
