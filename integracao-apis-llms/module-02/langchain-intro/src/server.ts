import Fastify from "fastify";
import { buildGraph } from "./graph/graph.ts";
import { HumanMessage } from "langchain";

const graph = buildGraph();

export const createServer = () => {
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
        const result = await graph.invoke({
          messages: [new HumanMessage(question)],
        });
        reply.send(result.messages);
      } catch (error) {
        console.error("Error handling /chat request:", error);
        reply.status(500).send({ error: "Internal Server Error" });
      }
    },
  );

  return app;
};
