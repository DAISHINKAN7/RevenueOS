import type { Config } from "tailwindcss";

/**
 * A deliberately narrow palette. Accent is reserved for the selected action and
 * interactive affordances; semantic colour is reserved for economic value and
 * policy outcomes. Probability is never coloured — it is neutral information,
 * and colouring it would imply a high-converting action is a good one.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#08090c",
          900: "#0c0e13",
          850: "#11141a",
          800: "#161a22",
          700: "#1e232d",
          600: "#2a303c",
          500: "#3a4150",
        },
        accent: { DEFAULT: "#3ba9d4", muted: "#1e6b8a", soft: "#0f2d3a" },
        pos: { DEFAULT: "#3fb984", soft: "#0f2a20" },
        neg: { DEFAULT: "#e0555f", soft: "#2c1316" },
        warn: { DEFAULT: "#d99b3c", soft: "#2a1f0e" },
        muted: { DEFAULT: "#8a93a5", dim: "#5d6675" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        "metric": ["1.75rem", { lineHeight: "1.1", letterSpacing: "-0.02em" }],
        "metric-lg": ["2.5rem", { lineHeight: "1.05", letterSpacing: "-0.03em" }],
      },
      borderRadius: { xl: "0.625rem" },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(4px)" },
                     to: { opacity: "1", transform: "translateY(0)" } },
        "pulse-soft": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.45" } },
      },
      animation: {
        "fade-up": "fade-up 240ms ease-out both",
        "pulse-soft": "pulse-soft 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;