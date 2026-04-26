console.assert(
  process.env.OPENROUTER_API_KEY,
  "OPENROUTER_API_KEY is not set in environment variables",
);

export type ModelConfig = {
  apiKey: string;
  httpReferer: string;
  xTitle: string;
  port: number;
  models: string[];
  temperature: number;
  maxTokens: number;
  systemPrompt: string;

  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };
};

export const config: ModelConfig = {
  apiKey: process.env.OPENROUTER_API_KEY!,
  httpReferer: "http://localhost:5173",
  xTitle: "Smart Model Router Gateway",
  port: 3000,
  models: ["inclusionai/ling-2.6-1t:free", "baidu/qianfan-ocr-fast:free"],
  temperature: 0.2,
  maxTokens: 100,
  systemPrompt: "Your are a helpfull assistent.",
  provider: {
    sort: {
      by: "price",
      partition: "none",
    },
  },
};
