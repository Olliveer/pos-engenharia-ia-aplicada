import { ChatOpenAI } from "@langchain/openai";
import { config, ModelConfig } from "../config";
import { z } from "zod/v3";
import { createAgent, providerStrategy } from "langchain";

export class OpenRouterService {
  private config: ModelConfig;
  private llmClient: ChatOpenAI;
  constructor(configOverride?: ModelConfig) {
    this.config = configOverride ?? config;
    this.llmClient = new ChatOpenAI({
      modelName: this.config.apiKey,
      temperature: this.config.temperature,
      model: this.config.models.at(0),
      configuration: {
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
          "http-referer": this.config.httpReferer,
          "x-title": this.config.xTitle,
        },
      },

      modelKwargs: {
        models: this.config.models,
        provider: this.config.provider,
      },
    });
  }

  async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema: z.ZodSchema<T>,
  ) {
    const agent = createAgent({
      model: this.llmClient,
      tools: [],
      responseFormat: providerStrategy(schema),
    });
  }
}
