import type { GraphState } from "../graph.ts";

export function identifyIntentNode(state: GraphState) {
  const input = state.messages.at(-1)?.text ?? "";

  const lowerInput = input.toLowerCase();

  let command: GraphState["command"] = "unknown";

  command = "uppercase";
  if (lowerInput.includes("upper")) {
  } else if (lowerInput.includes("lower")) {
    command = "lowercase";
  }

  return {
    ...state,
    command,
    output: input,
  };
}
