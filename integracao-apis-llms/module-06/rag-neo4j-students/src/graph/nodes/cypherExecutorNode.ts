import config from "../../config.ts";
import { Neo4jService } from "../../services/neo4jService.ts";
import type { GraphState } from "../graph.ts";

async function executeQueury(query: string, neo4jService: Neo4jService) {
  try {
    const isValid = await neo4jService.validateQuery(query);
    if (!isValid) {
      return {
        results: null,
        error: "Invalid Cypher query",
      };
    }

    const results = await neo4jService.query(query);
    if (!results.length) {
      return {
        results: [],
        error: "No results found for the query",
      };
    }

    console.log("✅ Cypher query executed successfully:", query);

    return {
      results,
      error: null,
    };
  } catch (error) {
    return {
      results: null,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}

function hasMoreSteps(state: GraphState): boolean {
  if (
    !state.isMultiStep ||
    !state.subQuestions?.length ||
    state.currentStep === undefined
  ) {
    return false;
  }
  return state.currentStep < state.subQuestions.length;
}

function handleCypherExecuteNode(state: GraphState, results: any[]) {
  const updatedSubResults = [...(state.subResults ?? []), results];

  const nextStep = state.currentStep ?? 0 + 1;
  const multiStepState = {
    dbResults: results,
    subResults: updatedSubResults,
    currentStep: nextStep,
    needsCorrection: false,
  };

  const totalSteps = state.subQuestions?.length ?? 0;
  if (hasMoreSteps({ ...state, ...multiStepState })) {
    console.log(`✅ Step ${nextStep} of ${totalSteps} executed successfully.`);
    return multiStepState;
  }
  console.log(`✅ All ${totalSteps} steps executed successfully.`);

  return multiStepState;
}

export function createCypherExecutorNode(neo4jService: Neo4jService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const { results, error } = await executeQueury(
        state.query!,
        neo4jService,
      );

      if (error && results === null) {
        if (state.correctionAttempts ?? 0 < config.maxCorrectionAttempts) {
          console.warn(
            "⚠️ Cypher query execution failed, will attempt correction:",
            error,
          );
          return {
            validationError: error,
            originalQuery: state.originalQuery ?? state.query,
            needsCorrection: true,
          };
        }
        return {
          ...state,
          error: `Failed to execute query: ${error}`,
        };
      }

      if (
        state.isMultiStep &&
        state.subQuestions?.length &&
        state.currentStep !== undefined
      ) {
        const multiStepState = handleCypherExecuteNode(state, results!);

        return {
          ...multiStepState,
        };
      }

      if (!results?.length) {
        return {
          dbResults: [],
          error: "No results found for the query",
        };
      }

      return {
        ...state,
        dbResults: results,
        needsCorrection: false,
      };
    } catch (error) {
      console.error(
        "Error executing Cypher query:",
        error instanceof Error ? error.message : error,
      );

      return {
        ...state,
        error: "Invalid Cypher query - correction failed",
      };
    }
  };
}
