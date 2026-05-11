import { AppointmentService } from "../../services/appointmentService.ts";
import type { GraphState } from "../graph.ts";
import { z } from "zod/v3";

const schedulerSchema = z.object({
  professionalId: z.number({ required_error: "Professional ID is required" }),
  datetime: z.string({ required_error: "Datetime is required" }),
  patientName: z.string({ required_error: "Patient name is required" }),
});

export function createCancellerNode(appointmentService: AppointmentService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    console.log(`❌ Cancelling appointment...`);
    console.log("State data:", {
      professionalId: state.professionalId,
      datetime: state.datetime,
      patientName: state.patientName,
    });

    const validation = schedulerSchema.safeParse(state);

    if (validation.error) {
      const errorMessages = validation.error.errors
        .map((item) => `${item.path.join(".")}: ${item.message}`)
        .join("; ");

      console.log(`❌ Cancellation validation failed: ${errorMessages}`);

      return {
        actionSuccess: false,
        actionError: `Validation failed: ${errorMessages}`,
      };
    }

    appointmentService.cancelAppointment(
      validation.data.professionalId,
      validation.data.patientName,
      new Date(validation.data.datetime),
    );

    try {
      return {
        actionSuccess: true,
      };
    } catch (error) {
      console.log(
        `❌ Cancellation failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
      return {
        actionSuccess: false,
        actionError:
          error instanceof Error ? error.message : "Cancellation failed",
      };
    }
  };
}
