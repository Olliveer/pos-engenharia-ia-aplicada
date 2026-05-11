import { z } from "zod";

export const IntentSchema = z
  .object({
    intent: z
      .enum(["schedule", "cancel", "unknown"])
      .describe("The user intent"),
    professionalId: z
      .number()
      .describe(
        "ID of the medical professional - REQUIRED for schedule/cancel",
      ),
    professionalName: z
      .string()
      .optional()
      .describe("Name of the medical professional"),
    datetime: z
      .string()
      .describe(
        "Appointment date and time in ISO format - REQUIRED for schedule/cancel",
      ),
    patientName: z
      .string()
      .describe(
        "Patient name extracted from question - REQUIRED for schedule/cancel",
      ),
    reason: z
      .string()
      .optional()
      .describe("Reason for appointment (for scheduling)"),
  })
  .refine(
    (data) => {
      if (data.intent === "unknown") {
        return true; // For unknown intents, fields can be empty
      }
      // For schedule/cancel, all required fields must be present
      return data.professionalId && data.datetime && data.patientName;
    },
    {
      message:
        "professionalId, datetime, and patientName are required for schedule/cancel intents",
      path: ["professionalId"],
    },
  );

export type IntentData = z.infer<typeof IntentSchema>;

export const getSystemPrompt = (professionals: any[]) => {
  return JSON.stringify({
    role: "Intent Classifier for Medical Appointments",
    task: "Identify user intent and extract all appointment-related details",
    professionals: professionals.map((p) => ({
      id: p.id,
      name: p.name,
      specialty: p.specialty,
    })),
    current_date: new Date().toISOString(),
    rules: {
      schedule: {
        description: "User wants to book/schedule a new appointment",
        keywords: [
          "schedule",
          "book",
          "appointment",
          "I want to",
          "make an appointment",
        ],
        required_fields: ["professionalId", "datetime", "patientName"],
        optional_fields: ["reason"],
      },
      cancel: {
        description: "User wants to cancel an existing appointment",
        keywords: ["cancel", "remove", "delete", "cancel my appointment"],
        required_fields: ["professionalId", "datetime", "patientName"],
      },
      unknown: {
        description:
          "Anything not related to scheduling or cancelling appointments",
        examples: ["weather questions", "general info", "unrelated queries"],
        note: 'For unknown intents, you can set professionalId=0, datetime="", patientName=""',
      },
    },
    extraction_instructions: {
      professionalId:
        "CRITICAL: Match the professional name mentioned in the question to the ID from the professionals list. Use fuzzy matching. For unknown intent, set to 0.",
      professionalName:
        "Extract the professional name as mentioned by the user",
      datetime:
        "CRITICAL: Parse relative dates (today, tomorrow) and times. Convert to ISO format. Use current_date as reference. For unknown intent, set to empty string.",
      patientName:
        "CRITICAL: Extract the patient name from the question or context. This is essential. For unknown intent, set to empty string.",
      reason:
        "Extract the reason/purpose for the appointment (only for scheduling)",
    },
    critical_notes: [
      "ALWAYS include professionalId even for unknown intents (use 0 as placeholder)",
      "ALWAYS include datetime even for unknown intents (use empty string as placeholder)",
      "ALWAYS include patientName even for unknown intents (use empty string as placeholder)",
      "These three fields are MANDATORY in your response",
    ],
    examples: [
      {
        input:
          "I want to schedule with Dr. Alicio da Silva for tomorrow at 4pm for a check-up",
        output: {
          intent: "schedule",
          professionalId: 1,
          professionalName: "Dr. Alicio da Silva",
          datetime: "2026-02-12T16:00:00.000Z",
          patientName: "Unknown",
          reason: "check-up",
        },
      },
      {
        input: "Cancel my appointment with Dr. Ana Pereira today at 11am",
        output: {
          intent: "cancel",
          professionalId: 2,
          professionalName: "Dr. Ana Pereira",
          datetime: "2026-02-11T11:00:00.000Z",
          patientName: "Unknown",
        },
      },
      {
        input: "What is the weather today?",
        output: {
          intent: "unknown",
          professionalId: 0,
          datetime: "",
          patientName: "",
          professionalName: "",
        },
      },
    ],
  });
};

export const getUserPromptTemplate = (question: string) => {
  return JSON.stringify({
    question,
    instructions: [
      "Carefully analyze the question to determine the user intent",
      "Extract all relevant appointment details",
      "Convert dates and times to ISO format",
      "Match professional names to their IDs",
      "Return only the fields that are present in the question",
    ],
  });
};
