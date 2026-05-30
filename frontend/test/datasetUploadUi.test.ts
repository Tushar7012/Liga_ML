import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const chatInputSource = readFileSync(new URL('../src/components/Chat/ChatInput.tsx', import.meta.url), 'utf8');

test('chat input accepts Markdown uploads', () => {
  assert.match(chatInputSource, /DATASET_UPLOAD_ACCEPT\s*=\s*['"][^'"]*\.md/);
  assert.match(chatInputSource, /DATASET_UPLOAD_EXTENSIONS[\s\S]*['"]md['"]/);
});

test('uploaded data section renders training metadata', () => {
  assert.match(chatInputSource, /Uploaded Data/);
  assert.match(chatInputSource, /Ready for training/);
  assert.match(chatInputSource, /normalized_row_count/);
  assert.match(chatInputSource, /config_name/);
});
