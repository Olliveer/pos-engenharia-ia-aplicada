import {
  END,
  MessagesValue,
  MessagesZodMeta,
  START,
  StateGraph,
  StateSchema,
} from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { BaseMessage } from "langchain";
import { z } from "zod";
import { chatResponseNode } from "./nodes/chatResponseNode.ts";
import { lowerCaseNode } from "./nodes/lowerCaseNode.ts";
import { upperCaseNode } from "./nodes/upperCaseNode.ts";
import { fallbackNode } from "./nodes/fallbackNode.ts";
import { identifyIntentNode } from "./nodes/IdentifyIntentNode.ts";

const graphState = z.object({
  messages: withLangGraph(z.array(z.any()), MessagesZodMeta),
  output: z.string(),
  command: z.enum(["uppercase", "lowercase", "unknown"]).optional(),
});

export type GraphState = z.infer<typeof graphState>;

export function buildGraph() {
  const workflow = new StateGraph({
    state: graphState,
  })
    .addNode("identifyIntent", identifyIntentNode)
    .addNode("chatResponse", chatResponseNode)
    .addNode("upperCase", upperCaseNode)
    .addNode("lowerCase", lowerCaseNode)
    .addNode("fallback", fallbackNode)
    // .addNode("identifyIntent", (state: GraphState) => {
    //   return {
    //     ...state,
    //     output: "test",
    //   };
    // })
    .addEdge(START, "identifyIntent")
    .addConditionalEdges(
      "identifyIntent",
      (state: GraphState) => {
        switch (state.command) {
          case "uppercase":
            return "uppercase";
          case "lowercase":
            return "lowercase";
          default:
            return "fallback";
        }
      },
      {
        uppercase: "upperCase",
        lowercase: "lowerCase",
        fallback: "fallback",
      },
    )
    .addEdge("upperCase", "chatResponse")
    .addEdge("lowerCase", "chatResponse")
    .addEdge("fallback", "chatResponse")
    .addEdge("chatResponse", END);

  return workflow.compile({});
}
