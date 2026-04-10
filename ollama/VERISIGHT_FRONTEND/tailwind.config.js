/** @type {import('tailwindcss').Config} */
export default {
      content: [
            "./index.html",
            "./src/**/*.{js,ts,jsx,tsx}",
      ],
      theme: {
            extend: {
                  colors: {
                        success: "#22c55e",
                        warning: "#f59e0b",
                        alert: "#f97316",
                        danger: "#ef4444",
                        slate: {
                              50: "#f8fafc",
                              100: "#f1f5f9",
                              200: "#e2e8f0",
                              300: "#cbd5e1",
                              400: "#94a3b8",
                              500: "#64748b",
                              600: "#475569",
                              700: "#334155",
                              800: "#1e293b",
                              900: "#0f172a",
                        },
                  },
                  fontFamily: {
                        sans: ["Inter", "system-ui", "sans-serif"],
                  },
                  animation: {
                        fade: "fadeIn 0.3s ease-in-out",
                        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
                  },
                  keyframes: {
                        fadeIn: {
                              "0%": { opacity: "0" },
                              "100%": { opacity: "1" },
                        },
                  },
            },
      },
      plugins: [],
};
