import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        sidebar: "#111111",
        primary: "#111827",
      },
      animation: {
        "bounce-dot":  "bounce-dot 1.4s infinite ease-in-out both",
        "fade-in":     "fade-in 0.25s ease forwards",
        "slide-up":    "slide-up 0.3s ease forwards",
        blink:         "blink 1s step-end infinite",
        shimmer:       "shimmer 3s linear infinite",
        float:         "float 4s ease-in-out infinite",
        "fade-up":     "fade-up 0.7s ease forwards",
        "fade-up-2":   "fade-up 0.7s ease 0.15s forwards",
        "fade-up-3":   "fade-up 0.7s ease 0.3s forwards",
        "fade-up-4":   "fade-up 0.7s ease 0.45s forwards",
        "pulse-slow":  "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "spin-slow":   "spin 8s linear infinite",
        "thinking":    "thinking 1.8s ease-in-out infinite",
      },
      keyframes: {
        "bounce-dot": {
          "0%, 80%, 100%": { transform: "scale(0)", opacity: "0.5" },
          "40%": { transform: "scale(1)", opacity: "1" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(12px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% center" },
          "100%": { backgroundPosition: "200% center" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-10px)" },
        },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(24px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        thinking: {
          "0%, 100%": { opacity: "0.4" },
          "50%":      { opacity: "1" },
        },
      },
      backgroundSize: {
        "300%": "300%",
      },
    },
  },
  plugins: [],
};

export default config;
