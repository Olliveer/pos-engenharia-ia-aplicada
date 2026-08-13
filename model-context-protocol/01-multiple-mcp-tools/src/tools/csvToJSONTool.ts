import { tool } from "@langchain/core/tools";
import { z } from "zod";
import csvToJSON from "csvtojson";

export function csvToJSONTool() {
  return tool(
    async ({ csvText }) => {
      const result = await csvToJSON().fromString(csvText);
      console.log("[CSV to JSON] conversion result:", result.length);
      return JSON.stringify(result);
    },
    {
      name: "csv_to_json",
      description: "Converts CSV text to JSON format.",
      schema: z.object({
        csvText: z.string().describe("CSV text to convert to JSON"),
      }),
    },
  );
}
