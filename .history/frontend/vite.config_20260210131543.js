import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // 不 rewrite — 后端路由本身就是 /check、/check/files 等
        // 前端请求 /api/check → 后端 /api/check
        // 但后端路由注册在 router 上无前缀，所以需要 rewrite 去掉 /api
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
