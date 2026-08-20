import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "@phosphor-icons/react";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioPlayer({ src, accent }: { src: string; accent: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setDuration(audio.duration);
    const onEnded = () => setIsPlaying(false);

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("ended", onEnded);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("ended", onEnded);
    };
  }, []);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      // Pause every other player on the page so only one clip plays at a time.
      document.querySelectorAll("audio").forEach((el) => {
        if (el !== audio) el.pause();
      });
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const seek = (event: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    audio.currentTime = ratio * duration;
  };

  const progress = duration ? (currentTime / duration) * 100 : 0;

  return (
    <div className="flex items-center gap-2.5">
      <audio ref={audioRef} src={src} preload="metadata" onPause={() => setIsPlaying(false)} />
      <button
        type="button"
        onClick={togglePlay}
        aria-label={isPlaying ? "Pause" : "Play"}
        className="flex size-8 shrink-0 items-center justify-center rounded-full text-white transition-transform active:scale-90"
        style={{ backgroundColor: accent }}
      >
        {isPlaying ? <Pause size={14} weight="fill" /> : <Play size={14} weight="fill" className="ml-0.5" />}
      </button>
      <div
        onClick={seek}
        className="group h-6 flex-1 cursor-pointer"
        role="slider"
        aria-valuenow={Math.round(progress)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="relative mt-[11px] h-[3px] rounded-full bg-[var(--surface-2)]">
          <div
            className="absolute inset-y-0 left-0 rounded-full transition-[width]"
            style={{ width: `${progress}%`, backgroundColor: accent }}
          />
        </div>
      </div>
      <span className="w-[3.5rem] shrink-0 text-right font-mono text-[11px] tabular-nums text-[var(--ink-muted)]">
        {formatTime(isPlaying || currentTime ? currentTime : duration)}
      </span>
    </div>
  );
}
