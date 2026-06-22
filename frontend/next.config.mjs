import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const IS_STATIC = process.env.NEXT_PUBLIC_STATIC === "1";
// When hosting under a subpath (e.g. GitHub Project Pages at /lstm-weather),
// set NEXT_PUBLIC_BASE_PATH="/lstm-weather". Empty = served from domain root.
const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH || "").replace(/\/$/, "");

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
  ...(BASE_PATH
    ? {
        basePath: BASE_PATH,
        assetPrefix: BASE_PATH,
      }
    : {}),
};

export default nextConfig;
