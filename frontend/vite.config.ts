import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// In development the API runs on :8000; Vite proxies /api so the browser
// talks to a single origin, exactly like production behind nginx.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.VITE_API_PROXY ?? "http://localhost:8000", changeOrigin: true },
    },
  },
});
