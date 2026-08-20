export interface YamnetClass {
  class_name: string;
  score: number;
}

export interface ClipSummary {
  id: string;
  filename: string;
  is_speech: boolean;
  yamnet: YamnetClass[];
  transcript: string | null;
}

export type Category = "speech" | "non_speech";

export interface CategoryProfile {
  count: number;
  top_classes: YamnetClass[];
}

export interface Recommendation {
  recordingId: string;
  filename: string;
  is_speech: boolean;
  recommended_because: Category;
  clap_sim: number;
  yamnet_overlap: number | null;
  transcript_sim: number | null;
  z_clap: number;
  z_yamnet: number | null;
  z_transcript: number | null;
  score: number;
  yamnet: YamnetClass[];
  transcript: string | null;
}

export interface SignalStats {
  mean: number;
  std: number;
  n: number;
}

export type PoolStats = Record<"clap_sim" | "yamnet_overlap" | "transcript_sim", SignalStats | null>;

export interface RatioData {
  non_speech_count: number;
  speech_count: number;
  label: string;
  mock_history: ClipSummary[];
  category_profiles: Partial<Record<Category, CategoryProfile>>;
  recommendations: Recommendation[];
  pool_stats: Partial<Record<Category, PoolStats>>;
}

export interface Pipeline {
  top_k: number;
  candidate_pool: number;
  ratios: RatioData[];
}
