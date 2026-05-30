import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DEFAULT_OUTPUT_POLICY,
  DEFAULT_TRAINING_GOAL,
  buildGcloudChatRequestMetadata,
  outputPolicyLabel,
  storageDestinationLabel,
  trainingGoalLabel,
} from '../src/lib/gcloud-preflight.js';
import {
  clearExplicitToolApprovalsForTesting,
  consumeExplicitApprovalDecision,
  registerExplicitToolApprovals,
} from '../src/lib/explicit-tool-approvals.js';
import type { OutputPolicy, TrainingGoal } from '../src/types/agent.js';

test('defines GCloud preflight option labels and defaults', () => {
  assert.equal(DEFAULT_TRAINING_GOAL, 'agent-decide');
  assert.equal(DEFAULT_OUTPUT_POLICY, 'cloud-and-hf-hub');

  const goals: Record<TrainingGoal, string> = {
    'smoke-test': 'Quick smoke test',
    production: 'Production-ready fine-tuning',
    'agent-decide': 'Let the agent decide',
  };
  const policies: Record<OutputPolicy, string> = {
    'cloud-private': 'Google Cloud Storage only',
    'hf-hub': 'Hugging Face Hub only',
    'cloud-and-hf-hub': 'Both Google Cloud Storage and Hugging Face Hub',
  };

  for (const [value, label] of Object.entries(goals)) {
    assert.equal(trainingGoalLabel(value as TrainingGoal), label);
  }
  for (const [value, label] of Object.entries(policies)) {
    assert.equal(outputPolicyLabel(value as OutputPolicy), label);
  }
});

test('builds snake_case chat request metadata without duplicating fields', () => {
  assert.deepEqual(
    buildGcloudChatRequestMetadata({
      cloudProvider: 'gcp-vertex',
      trainingGoal: 'smoke-test',
      outputPolicy: 'cloud-private',
    }),
    {
      cloud_provider: 'gcp-vertex',
      training_goal: 'smoke-test',
      output_policy: 'cloud-private',
    },
  );
});

test('omits GCloud-only fields for non-GCloud providers', () => {
  assert.deepEqual(
    buildGcloudChatRequestMetadata({
      cloudProvider: 'hf-jobs',
      trainingGoal: 'production',
      outputPolicy: 'hf-hub',
    }),
    {
      cloud_provider: 'hf-jobs',
    },
  );
});

test('formats storage destination for Vertex panel summaries', () => {
  assert.equal(storageDestinationLabel('cloud-private'), 'Google Cloud Storage only');
  assert.equal(storageDestinationLabel('hf-hub'), 'Hugging Face Hub only');
  assert.equal(storageDestinationLabel('cloud-and-hf-hub'), 'Google Cloud Storage and Hugging Face Hub');
});

test('requires an explicit user approval before consuming a tool approval decision', () => {
  clearExplicitToolApprovalsForTesting();

  assert.equal(consumeExplicitApprovalDecision('session-1', 'tool-1'), null);

  registerExplicitToolApprovals('session-1', [
    { tool_call_id: 'tool-1', approved: true },
  ]);

  assert.deepEqual(consumeExplicitApprovalDecision('session-1', 'tool-1'), {
    approved: true,
    feedback: null,
    edited_script: null,
    namespace: null,
  });
  assert.equal(consumeExplicitApprovalDecision('session-1', 'tool-1'), null);
});
