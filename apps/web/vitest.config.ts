import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/unit/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/unit/setup.ts"],
    // jsdom only provides `localStorage` for a real origin. Without a URL the
    // document has an opaque origin, `window.localStorage` is undefined, and the
    // feedback store silently degrades to its no-storage path - which is correct
    // behaviour but means the persistence it is supposed to have goes untested.
    environmentOptions: {
      jsdom: { url: "http://127.0.0.1:3000/" },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
    },
  },
});
