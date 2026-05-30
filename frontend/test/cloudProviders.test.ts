import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CLOUD_PROVIDER_OPTIONS, isCloudProviderId } from '../src/lib/cloud-providers.js';
import type { CloudProviderId } from '../src/types/agent.js';

test('cloud provider options include AWS SageMaker AI', () => {
  const ids = CLOUD_PROVIDER_OPTIONS.map((provider) => provider.id);
  const labels = CLOUD_PROVIDER_OPTIONS.map((provider) => provider.name);

  assert.deepEqual(ids, ['hf-jobs', 'gcp-vertex', 'aws-sagemaker']);
  assert.ok(labels.includes('AWS SageMaker AI'));
});

test('cloud provider type accepts AWS SageMaker provider id', () => {
  const provider: CloudProviderId = 'aws-sagemaker';

  assert.equal(provider, 'aws-sagemaker');
});

test('cloud provider guard accepts only known providers', () => {
  assert.equal(isCloudProviderId('hf-jobs'), true);
  assert.equal(isCloudProviderId('gcp-vertex'), true);
  assert.equal(isCloudProviderId('aws-sagemaker'), true);
  assert.equal(isCloudProviderId('azure-ml'), false);
});
