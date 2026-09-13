import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Integration configuration: the real UI components against the **live** API.
 *
 * Separate from the unit config because these tests need the stack running
 * (`make ragflow-up && make api`) and they perform real network calls. They skip
 * themselves when the API is not reachable rather than failing.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/integration/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/integration/setup.ts"],
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
    },
  },
});
