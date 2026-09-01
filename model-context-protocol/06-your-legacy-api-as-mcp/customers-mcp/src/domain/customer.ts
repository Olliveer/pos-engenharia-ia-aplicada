import { z } from "zod";

export const CustomerSchema = z.object({
  _id: z.string().optional(),
  name: z.string(),
  phone: z.string(),
});

export type Customer = z.infer<typeof CustomerSchema>;

export const CustomerQuerySchema = z.object({
  _id: z.string().optional().describe("Customer's ID"),
  name: z.string().optional().describe("Customer's name"),
  phone: z.string().optional().describe("Customer's phone number"),
});

export type CustomerQuery = z.infer<typeof CustomerQuerySchema>;

export const CustomerUpdateSchema = CustomerQuerySchema.extend({
  _id: z.string().describe("Customer's ID"),
});

export type CustomerUpdate = z.infer<typeof CustomerUpdateSchema>;

export const CustomerMutationSchema = z.object({
  id: z.string().optional().describe("Customer's ID"),
  message: z.string().optional().describe("Confirmation mesage"),
  isError: z.boolean().optional().describe("Indicates is an error occurred"),
});

export type CustomerMutation = z.infer<typeof CustomerMutationSchema>;
