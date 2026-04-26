import Fastify from "fastify";
import { OpenRouterService } from "./openRouterService.ts";

export const createServer = (routerService: OpenRouterService) => {
  const app = Fastify();

  app.post(
    "/chat",
    {
      schema: {
        body: {
          type: "object",
          required: ["question"],
          properties: {
            question: { type: "string", minLength: 1 },
          },
        },
      },
    },
    async (request, reply) => {
      try {
        const { question } = request.body as { question: string };
        const response = await routerService.generate(question);
        reply.send(response);
      } catch (error) {
        console.error("Error handling /chat request:", error);
        reply.status(500).send({ error: "Internal Server Error" });
      }
    },
  );

  return app;
};
