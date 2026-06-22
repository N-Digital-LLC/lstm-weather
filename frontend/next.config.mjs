import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const IS_STATIC = process.env.NEXT_PUBLIC_STATIC === "1";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Pin the workspace root so Next/Turbopack doesn't pick up an unrelated
  // parent lockfile (e.g. C:\Users\PC\package-lock.json).
  turbopack: {
    root: __dirname,
  },
  // Static results package: emit a fully static, backend-free site into ./out.
  ...(IS_STATIC
    ? {
        output: "export",
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
