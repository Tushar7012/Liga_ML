import type { CloudProviderId, OutputPolicy, TrainingGoal } from '../types/agent.js';

export const DEFAULT_TRAINING_GOAL: TrainingGoal = 'agent-decide';
export const DEFAULT_OUTPUT_POLICY: OutputPolicy = 'cloud-and-hf-hub';

export const TRAINING_GOAL_OPTIONS: Array<{
  value: TrainingGoal;
  label: string;
}> = [
  { value: 'smoke-test', label: 'Quick smoke test' },
  { value: 'production', label: 'Production-ready fine-tuning' },
  { value: 'agent-decide', label: 'Let the agent decide' },
];

export const OUTPUT_POLICY_OPTIONS: Array<{
  value: OutputPolicy;
  label: string;
}> = [
  { value: 'cloud-private', label: 'Google Cloud Storage only' },
  { value: 'hf-hub', label: 'Hugging Face Hub only' },
  { value: 'cloud-and-hf-hub', label: 'Both Google Cloud Storage and Hugging Face Hub' },
];

export function trainingGoalLabel(value: TrainingGoal | undefined): string {
  return TRAINING_GOAL_OPTIONS.find((option) => option.value === value)?.label
    ?? trainingGoalLabel(DEFAULT_TRAINING_GOAL);
}

export function outputPolicyLabel(value: OutputPolicy | undefined): string {
  return OUTPUT_POLICY_OPTIONS.find((option) => option.value === value)?.label
    ?? outputPolicyLabel(DEFAULT_OUTPUT_POLICY);
}

export function storageDestinationLabel(value: OutputPolicy | undefined): string {
  switch (value) {
    case 'cloud-private':
      return 'Google Cloud Storage only';
    case 'hf-hub':
      return 'Hugging Face Hub only';
    case 'cloud-and-hf-hub':
    default:
      return 'Google Cloud Storage and Hugging Face Hub';
  }
}

export function buildGcloudChatRequestMetadata({
  cloudProvider,
  trainingGoal,
  outputPolicy,
}: {
  cloudProvider: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
}): {
  cloud_provider: CloudProviderId;
  training_goal?: TrainingGoal;
  output_policy?: OutputPolicy;
} {
  if (cloudProvider !== 'gcp-vertex') {
    return { cloud_provider: cloudProvider };
  }
  return {
    cloud_provider: cloudProvider,
    training_goal: trainingGoal ?? DEFAULT_TRAINING_GOAL,
    output_policy: outputPolicy ?? DEFAULT_OUTPUT_POLICY,
  };
}
