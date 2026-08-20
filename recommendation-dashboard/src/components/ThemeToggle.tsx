import { useEffect, useState } from "react";
import { Moon, Sun } from "@phosphor-icons/react";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <button
      type="button"
      onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
      aria-label="Toggle theme"
      className="flex size-8 items-center justify-center rounded-full border border-[var(--line)] text-[var(--ink-soft)] transition-colors hover:bg-[var(--surface-2)]"
    >
      {theme === "light" ? <Moon size={15} /> : <Sun size={15} />}
    </button>
  );
}
