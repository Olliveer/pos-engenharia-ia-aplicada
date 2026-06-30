import type { Runtime } from "@langchain/langgraph";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  ChatResponseSchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/chatResponse.ts";
import { AIMessage, HumanMessage } from "langchain";
import { PreferencesService } from "../../services/preferencesService.ts";
import { config } from "../../config.ts";

export function createChatNode(
  llmClient: OpenRouterService,
  preferenceService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime?: Runtime,
  ): Promise<Partial<GraphState>> => {
    const userId = String(runtime?.context?.userId || state.userId || "unknow");

    const userContext =
      state.userContext ?? (await preferenceService.getBasicInfo(userId));
    const systemPrompt = getSystemPrompt(userContext);

    const converssationHistory = state.messages
      .map(
        (msg) =>
          `${HumanMessage.isInstance(msg) ? "User" : "AI"}: ${msg.content}`,
      )
      .join("\n");

    const userMessage = state.messages.at(-1)?.text as string;

    const userPrompt = getUserPromptTemplate(userMessage, converssationHistory);

    const result = await llmClient.generateStructured(
      systemPrompt,
      userPrompt,
      ChatResponseSchema,
    );

    if (!result.success || !result.data) {
      console.error("Erro ao gerar resposta estruturada:", result);
      return {
        messages: [
          new AIMessage(
            "Desculpe, ocorreu um erro ao processar sua solicitação.",
          ),
        ],
      };
    }

    const response = result.data;

    const totalMessages = state.messages.length;

    const needsSummarization = totalMessages >= config.maxMessagesToSummary;
    return {
      messages: [new AIMessage(response.message)],
      extractedPreferences: response.shouldSavePreferences
        ? response.preferences
        : undefined,
      needsSummarization,
    };
  };
}
