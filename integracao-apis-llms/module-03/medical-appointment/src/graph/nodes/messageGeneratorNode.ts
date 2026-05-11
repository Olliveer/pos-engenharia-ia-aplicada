import {
  getSystemPrompt,
  getUserPromptTemplate,
  MessageSchema,
} from "../../prompts/v1/messageGenerator.ts";
import { AppointmentService } from "../../services/appointmentService.ts";
import { OpenRouterService } from "../../services/openRouterService.ts";
import type { GraphState } from "../graph.ts";
import { AIMessage } from "langchain";

export function createMessageGeneratorNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    console.log(`💬 Generating response message...`);

    try {
      const hasSucessed = state.actionSuccess ? "success" : "error";
      const scenario = `${state.intent ?? "unknown"}_${hasSucessed}`;
      const details = {
        professionalName: state.professionalName,
        datetime: state.datetime,
        reason: state.reason,
        error: state.actionError,
      };

      const systemPrompt = getSystemPrompt();
      const userPrompt = getUserPromptTemplate({
        scenario,
        details,
      });

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        MessageSchema,
      );

      console.log(`✅ Message generated successfully`);

      if (result.error) {
        console.error("❌ Failed to generate message:", result.error);
        return {
          messages: [
            ...state.messages,
            new AIMessage(
              "An error occurred while generating the response message.",
            ),
          ],
        };
      }

      return {
        messages: [...state.messages, new AIMessage(result.data?.message)],
      };
    } catch (error) {
      console.error("❌ Error in messageGenerator node:", error);
      return {
        ...state,
        messages: [
          ...state.messages,
          new AIMessage("An error occurred while processing your request."),
        ],
      };
    }
  };
}
