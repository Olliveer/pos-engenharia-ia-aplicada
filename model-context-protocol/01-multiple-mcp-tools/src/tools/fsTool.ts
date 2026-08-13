export const getFsTool = () => {
  return {
    filesystem: {
      transport: "stdio" as const,
      command: "npx",
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        `${process.cwd()}/reports`,
      ],
    },
  };
};
