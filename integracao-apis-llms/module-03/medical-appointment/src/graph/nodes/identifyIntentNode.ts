import {
  getSystemPrompt,
  getUserPromptTemplate,
  IntentSchema,
} from "../../prompts/v1/identifyIntent.ts";
import { professionals } from "../../services/appointmentService.ts";
import { OpenRouterService } from "../../services/openRouterService.ts";
import type { GraphState } from "../graph.ts";

export function createIdentifyIntentNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<GraphState> => {
    console.log(`🔍 Identifying intent...`);
    const input = state.messages.at(-1)!.text;

    try {
      const systemPrompt = getSystemPrompt(professionals);
      const userPrompt = getUserPromptTemplate(input);

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        IntentSchema,
      );
      return {
        ...state,
      };
    } catch (error) {
      console.error("❌ Error in identifyIntent node:", error);
      return {
        ...state,
        intent: "unknown",
        error:
          error instanceof Error
            ? error.message
            : "Intent identification failed",
      };
    }
  };
}
