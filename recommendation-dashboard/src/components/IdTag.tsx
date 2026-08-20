import { useState } from "react";
import { Check, Copy } from "@phosphor-icons/react";

export function IdTag({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <button
      type="button"
      onClick={copy}
      title={id}
      className="group flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[10px] text-[var(--ink-muted)] transition-colors hover:bg-[var(--surface-2)]"
    >
      {id.slice(0, 8)}&hellip;{id.slice(-4)}
      {copied ? (
        <Check size={11} weight="bold" className="text-[var(--pos)]" />
      ) : (
        <Copy size={11} className="opacity-0 transition-opacity group-hover:opacity-100" />
      )}
    </button>
  );
}
