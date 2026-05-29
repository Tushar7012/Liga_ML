import type { PanelData } from '@/store/agentStore';

export interface VertexToolState {
  state?: string;
  jobName?: string;
  jobUrl?: string;
  outputDir?: string;
}

const VERTEX_SUMMARY_FIELDS = [
  ['Dataset', 'dataset_name'],
  ['Dataset config', 'dataset_config'],
  ['Dataset split', 'dataset_split'],
  ['Model', 'model_name'],
  ['HF target', 'hub_model_id'],
  ['Machine type', 'machine_type'],
  ['Accelerator type', 'accelerator_type'],
  ['Accelerator count', 'accelerator_count'],
  ['Output dir', 'output_dir'],
  ['Staging bucket', 'staging_bucket'],
  ['Trackio project', 'trackio_project'],
  ['Trackio Space', 'trackio_space_id'],
] as const;

function valueToString(value: unknown): string | null {
  if (value === undefined || value === null || value === '') return null;
  if (Array.isArray(value)) return value.map(String).join(' ');
  return String(value);
}

export function buildVertexSftSummary(args: Record<string, unknown>): string {
  const rows = VERTEX_SUMMARY_FIELDS
    .map(([label, key]) => {
      const value = valueToString(args[key]);
      return value ? `| ${label} | \`${value}\` |` : null;
    })
    .filter((row): row is string => Boolean(row));

  return [
    '## Vertex AI SFT Training',
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ].join('\n');
}

export function buildVertexStateMarkdown(state: VertexToolState): string {
  const rows = [
    ['State', state.state],
    ['Vertex job', state.jobName],
    ['GCS output directory', state.outputDir],
    ['Vertex console', state.jobUrl],
  ]
    .map(([label, value]) => {
      const text = valueToString(value);
      if (!text) return null;
      const rendered = label === 'Vertex console' ? `[${text}](${text})` : `\`${text}\``;
      return `| ${label} | ${rendered} |`;
    })
    .filter((row): row is string => Boolean(row));

  if (rows.length === 0) return '';

  return [
    '## Vertex AI Job State',
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ].join('\n');
}

export function createVertexRunPanel(args: Record<string, unknown>): {
  data: PanelData;
  view: 'script' | 'output';
  editable: boolean;
} | null {
  if (args.operation !== 'run') return null;

  if (typeof args.script === 'string' && args.script) {
    return {
      data: {
        title: 'Vertex AI Script',
        script: { content: args.script, language: 'python' },
        parameters: args,
      },
      view: 'script',
      editable: false,
    };
  }

  if (args.template === 'sft') {
    return {
      data: {
        title: 'Vertex AI SFT Training',
        output: { content: buildVertexSftSummary(args), language: 'markdown' },
        parameters: args,
      },
      view: 'output',
      editable: false,
    };
  }

  if (Array.isArray(args.command) && args.command.length > 0) {
    return {
      data: {
        title: 'Vertex AI Command',
        script: { content: args.command.map(String).join(' '), language: 'bash' },
        parameters: args,
      },
      view: 'script',
      editable: false,
    };
  }

  return null;
}
