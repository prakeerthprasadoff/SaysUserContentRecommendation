import type { Category } from "./types";

/** Single source of truth for the category → color mapping used across every component. */
export function accentFor(category: Category): string {
  return category === "speech" ? "var(--speech)" : "var(--non-speech)";
}
