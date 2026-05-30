/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
  cloudProvider?: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
}

export type CloudProviderId = 'hf-jobs' | 'gcp-vertex';
export type TrainingGoal = 'smoke-test' | 'production' | 'agent-decide';
export type OutputPolicy = 'cloud-private' | 'hf-hub' | 'cloud-and-hf-hub';

export type DatasetSourceFormat = 'csv' | 'json' | 'jsonl' | 'pdf' | 'docx' | 'xlsx';

export interface DatasetUploadResponse {
  session_id: string;
  repo_id: string;
  repo_type: 'dataset';
  private: true;
  upload_id: string;
  config_name: string;
  filename: string;
  path_in_repo: string;
  raw_path_in_repo: string;
  normalized_path_in_repo: string;
  normalized_format: 'jsonl';
  normalized_row_count: number;
  source_format: DatasetSourceFormat;
  supports_training: boolean;
  size_bytes: number;
  format: DatasetSourceFormat;
  hub_url: string;
  load_dataset_snippet: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
  needsAttention: boolean;
  model?: string | null;
  cloudProvider?: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
  /** True when the backend no longer recognizes this session id (e.g.
   *  after a backend restart). The UI shows a recovery banner and
   *  disables input until the user chooses to restore-with-summary or
   *  start fresh. */
  expired?: boolean;
  autoApprovalEnabled?: boolean;
  autoApprovalCostCapUsd?: number | null;
  autoApprovalEstimatedSpendUsd?: number;
  autoApprovalRemainingUsd?: number | null;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
  namespace?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
}
