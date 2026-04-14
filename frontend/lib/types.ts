export interface GPTConfig {
  vocab_size: number;
  context_length: number;
  emb_dim: number;
  n_heads: number;
  n_layers: number;
  drop_rate: number;
  qkv_bias: boolean;
}

export interface ParamCount {
  token_embedding: number;
  position_embedding: number;
  per_block: {
    attention_qkv: number;
    attention_out: number;
    feedforward: number;
    layer_norm_1: number;
    layer_norm_2: number;
    subtotal: number;
  };
  num_blocks: number;
  all_blocks: number;
  final_layer_norm: number;
  output_head: number;
  total: number;
  total_M: number;
}

export interface MemoryEstimate {
  parameters_mb: number;
  gradients_mb: number;
  optimizer_states_mb: number;
  activations_mb: number;
  total_training_mb: number;
  inference_only_mb: number;
}

export interface BPEStep {
  step: number;
  merged_pair: [number, number];
  merged_pair_str: [string, string];
  new_token_id: number;
  new_token: string;
  current_vocab_size: number;
  total_tokens: number;
  compression_ratio: number;
}

export interface TokenResult {
  token_ids: number[];
  tokens: string[];
  num_tokens: number;
}

export interface AttentionResult {
  tokens: string[];
  weights: number[][] | number[][][];
  type: string;
  num_heads?: number;
  is_masked?: boolean;
}

export interface TrainingMetric {
  type: "eval" | "epoch_end" | "completed" | "cancelled" | "error";
  epoch?: number;
  global_step?: number;
  train_loss?: number;
  val_loss?: number;
  tokens_seen?: number;
  learning_rate?: number;
  sample_text?: string;
  message?: string;
  model_path?: string;
}
