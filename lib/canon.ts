// The canonical form, and every translation to and from the three provider wires and the two
// client wires.
//
// A request the client sends (OpenAI Chat Completions or Anthropic Messages) is first read into
// one canonical form: the joined system text, the conversation turns as blocks, the tool list,
// max_tokens, the stop sequences, whether JSON mode was asked for, and whether it streams. From
// that form the request is written out in the wire of whichever provider serves it (OpenAI,
// Anthropic or Gemini), and the provider's answer is read back into a canonical answer (its text
// whole, its usage counts, its stop reason) and written out in the client's wire, as JSON or as
// server-sent events.
//
// The translations preserve, field for field, what a provider reads out of a request: the system
// text (blocks joined with newlines), the last user turn's text, the tool names in order,
// max_tokens, the stop sequences, the image bytes, and the JSON-mode flag. A field that is absent
// in the client's request stays absent in the provider's: a max_tokens the client never sent is
// never invented, because the provider echoes what it parsed and the client compares the echo.

import { isRecord, type Json } from './brief.ts';

export type ImageRef = { kind: 'data'; mediaType: string; data: string } | { kind: 'url'; url: string };

export type Block =
  | { type: 'text'; text: string }
  | { type: 'image'; image: ImageRef }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; toolUseId: string; text: string };

export interface Turn {
  role: 'user' | 'assistant';
  blocks: Block[];
}

export interface CanonTool {
  name: string;
  description: string;
  schema: unknown;
}

export interface CanonRequest {
  system: string;
  turns: Turn[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

export interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason, in its own wire's spelling. */
  stop: string;
  model: string;
}

const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const strOr = (v: unknown, fallback: string): string => (typeof v === 'string' ? v : fallback);

/** The text of a content value as the wires read it: a string as-is, blocks' text joined by \n. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An OpenAI image_url value as an image reference: a data URL keeps its bytes, any other its URL. */
function openAIImage(url: unknown): ImageRef | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'data', mediaType: data[1], data: data[2] };
  return { kind: 'url', url };
}

function parseToolInput(text: unknown): unknown {
  if (typeof text !== 'string') return text ?? {};
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Client wires in

/** Read an OpenAI Chat Completions request body into the canonical form. */
export function fromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'user',
        blocks: [{ type: 'tool_result', toolUseId: strOr(raw.tool_call_id, ''), text: blocksText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    const blocks: Block[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url)) {
          const image = openAIImage(part.image_url.url);
          if (image !== null) blocks.push({ type: 'image', image });
        }
      }
    }
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: strOr(call.id, ''),
          name: strOr(call.function.name, ''),
          input: parseToolInput(call.function.arguments),
        });
      }
    }
    turns.push({ role: raw.role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: strOr(tool.function.description, ''),
        schema: tool.function.parameters ?? { type: 'object' },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** Read an Anthropic Messages request body into the canonical form. */
export function fromAnthropic(body: Json): CanonRequest {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || (raw.role !== 'user' && raw.role !== 'assistant')) continue;
    const blocks: Block[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image' && isRecord(part.source)) {
          if (part.source.type === 'base64' && typeof part.source.data === 'string') {
            blocks.push({
              type: 'image',
              image: { kind: 'data', mediaType: strOr(part.source.media_type, 'image/png'), data: part.source.data },
            });
          } else if (part.source.type === 'url' && typeof part.source.url === 'string') {
            blocks.push({ type: 'image', image: { kind: 'url', url: part.source.url } });
          }
        } else if (part.type === 'tool_use') {
          blocks.push({ type: 'tool_use', id: strOr(part.id, ''), name: strOr(part.name, ''), input: part.input ?? {} });
        } else if (part.type === 'tool_result') {
          blocks.push({
            type: 'tool_result',
            toolUseId: strOr(part.tool_use_id, ''),
            text: typeof part.content === 'string' ? part.content : blocksText(part.content),
          });
        }
      }
    }
    turns.push({ role: raw.role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({ name: tool.name, description: strOr(tool.description, ''), schema: tool.input_schema ?? { type: 'object' } });
    }
  }
  return {
    system: blocksText(body.system),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Provider wires out

function openAIContent(turn: Turn): unknown {
  const hasNonText = turn.blocks.some((b) => b.type !== 'text');
  if (!hasNonText) return turn.blocks.map((b) => (b.type === 'text' ? b.text : '')).join('\n');
  const parts: unknown[] = [];
  for (const block of turn.blocks) {
    if (block.type === 'text') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image') {
      const url = block.image.kind === 'data' ? `data:${block.image.mediaType};base64,${block.image.data}` : block.image.url;
      parts.push({ type: 'image_url', image_url: { url } });
    }
  }
  return parts;
}

/** Write the canonical request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(req: CanonRequest, model: string): Json {
  const messages: unknown[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const turn of req.turns) {
    if (turn.role === 'assistant') {
      const text = turn.blocks.filter((b) => b.type === 'text').map((b) => (b.type === 'text' ? b.text : '')).join('\n');
      const calls = turn.blocks.filter((b) => b.type === 'tool_use');
      const message: Json = { role: 'assistant', content: text === '' && calls.length > 0 ? null : text };
      if (calls.length > 0) {
        message.tool_calls = calls.map((b) =>
          b.type === 'tool_use'
            ? { id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) } }
            : null,
        );
      }
      messages.push(message);
      continue;
    }
    const results = turn.blocks.filter((b) => b.type === 'tool_result');
    const rest = turn.blocks.filter((b) => b.type !== 'tool_result');
    if (rest.length > 0) messages.push({ role: 'user', content: openAIContent({ role: 'user', blocks: rest }) });
    for (const block of results) {
      if (block.type === 'tool_result') messages.push({ role: 'tool', tool_call_id: block.toolUseId, content: block.text });
    }
    if (rest.length === 0 && results.length === 0) messages.push({ role: 'user', content: '' });
  }
  const body: Json = { model, messages };
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  if (req.stream) {
    body.stream = true;
    // The usage chunk is how a streamed answer reports its counts; ask for it on this wire always.
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** Write the canonical request as an Anthropic Messages body naming `model`. */
export function toAnthropic(req: CanonRequest, model: string): Json {
  const messages: unknown[] = [];
  for (const turn of req.turns) {
    const blocks: unknown[] = [];
    for (const block of turn.blocks) {
      if (block.type === 'text') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image') {
        blocks.push({
          type: 'image',
          source:
            block.image.kind === 'data'
              ? { type: 'base64', media_type: block.image.mediaType, data: block.image.data }
              : { type: 'url', url: block.image.url },
        });
      } else if (block.type === 'tool_use') blocks.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input ?? {} });
      else if (block.type === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: block.toolUseId, content: block.text });
    }
    messages.push({ role: turn.role, content: blocks });
  }
  const body: Json = { model, messages };
  if (req.system !== '') body.system = req.system;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.stream) body.stream = true;
  return body;
}

/** Write the canonical request as a Gemini generateContent body for `model`. */
export function toGemini(req: CanonRequest, _model: string): Json {
  const toolNames = new Map<string, string>();
  for (const turn of req.turns) {
    for (const block of turn.blocks) if (block.type === 'tool_use') toolNames.set(block.id, block.name);
  }
  const contents: unknown[] = [];
  for (const turn of req.turns) {
    const parts: unknown[] = [];
    for (const block of turn.blocks) {
      if (block.type === 'text') parts.push({ text: block.text });
      else if (block.type === 'image') {
        parts.push(
          block.image.kind === 'data'
            ? { inlineData: { mimeType: block.image.mediaType, data: block.image.data } }
            : { fileData: { fileUri: block.image.url } },
        );
      } else if (block.type === 'tool_use') parts.push({ functionCall: { name: block.name, args: block.input ?? {} } });
      else if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: toolNames.get(block.toolUseId) ?? 'tool', response: { result: block.text } } });
      }
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (req.system !== '') body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Provider answers in

interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a server-sent-events body into its frames. */
export function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Read an OpenAI-wire provider answer (JSON body or SSE stream) into the canonical answer. */
export function answerFromOpenAI(text: string, stream: boolean, model: string): CanonAnswer | null {
  if (!stream) {
    const body = parseJson(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof choice.message.content === 'string' ? choice.message.content : blocksText(choice.message.content),
      inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numOrNull(usage.completion_tokens) ?? 0,
      stop: strOr(choice.finish_reason, 'stop'),
      model: strOr(body.model, model),
    };
  }
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  let saw = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    saw = true;
    if (isRecord(chunk.usage)) {
      input = numOrNull(chunk.usage.prompt_tokens) ?? input;
      output = numOrNull(chunk.usage.completion_tokens) ?? output;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    if (typeof chunk.model === 'string') model = chunk.model;
  }
  return saw ? { text: out, inputTokens: input, outputTokens: output, stop, model } : null;
}

/** Read an Anthropic-wire provider answer (JSON body or SSE stream) into the canonical answer. */
export function answerFromAnthropic(text: string, stream: boolean, model: string): CanonAnswer | null {
  if (!stream) {
    const body = parseJson(text);
    if (body === null) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: blocksText(body.content),
      inputTokens: numOrNull(usage.input_tokens) ?? 0,
      outputTokens: numOrNull(usage.output_tokens) ?? 0,
      stop: strOr(body.stop_reason, 'end_turn'),
      model: strOr(body.model, model),
    };
  }
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let saw = false;
  for (const frame of sseFrames(text)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    saw = true;
    if (frame.event === 'message_start' && isRecord(data.message)) {
      if (isRecord(data.message.usage)) input = numOrNull(data.message.usage.input_tokens) ?? input;
      if (typeof data.message.model === 'string') model = data.message.model;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
      out += strOr(data.delta.text, '');
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrNull(data.usage.output_tokens) ?? output;
    }
  }
  return saw ? { text: out, inputTokens: input, outputTokens: output, stop, model } : null;
}

/** Read a Gemini-wire provider answer (JSON body or SSE stream) into the canonical answer. */
export function answerFromGemini(text: string, stream: boolean, model: string): CanonAnswer | null {
  const read = (body: Json, acc: CanonAnswer): void => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string') acc.text += part.text;
        }
      }
      if (typeof candidate.finishReason === 'string') acc.stop = candidate.finishReason;
    }
    if (isRecord(body.usageMetadata)) {
      acc.inputTokens = numOrNull(body.usageMetadata.promptTokenCount) ?? acc.inputTokens;
      acc.outputTokens = numOrNull(body.usageMetadata.candidatesTokenCount) ?? acc.outputTokens;
    }
    if (typeof body.modelVersion === 'string') acc.model = body.modelVersion;
  };
  const acc: CanonAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'STOP', model };
  if (!stream) {
    const body = parseJson(text);
    if (body === null) return null;
    read(body, acc);
    return acc;
  }
  let saw = false;
  for (const frame of sseFrames(text)) {
    const body = parseJson(frame.data);
    if (body === null) continue;
    saw = true;
    read(body, acc);
  }
  return saw ? acc : null;
}

// ---------------------------------------------------------------------------
// Client wires out

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

/** The stop reason in the client's wire spelling, in the same family as the provider's. */
export function mapStop(stop: string, api: 'openai' | 'anthropic'): string {
  const family = STOP_FAMILIES.get(stop) ?? 0;
  return api === 'openai' ? ['stop', 'length', 'tool_calls', 'content_filter'][family] : ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family];
}

/** Split a text the way the simulated providers stream it: pieces of at most 13 characters. */
export function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

const FIXED_CREATED = 1767225600;

/** The canonical answer as an OpenAI Chat Completions JSON body. */
export function openAIJsonAnswer(answer: CanonAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: answer.model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: mapStop(answer.stop, 'openai'), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** The canonical answer as an OpenAI SSE stream, ending in a usage chunk and [DONE]. */
export function openAIStreamAnswer(answer: CanonAnswer): string {
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model: answer.model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const frames = [chunk({ role: 'assistant', content: '' }, null)];
  for (const piece of splitText(answer.text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, mapStop(answer.stop, 'openai')));
  frames.push(
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model: answer.model, choices: [], usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens } })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The canonical answer as an Anthropic Messages JSON body. */
export function anthropicJsonAnswer(answer: CanonAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model: answer.model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: mapStop(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** The canonical answer as an Anthropic SSE stream, in the wire's event order. */
export function anthropicStreamAnswer(answer: CanonAnswer): string {
  const frame = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames = [
    frame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model: answer.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    frame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    frame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(answer.stop, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    frame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An error body in the client's wire shape. */
export function clientError(api: 'openai' | 'anthropic', status: number, message: string): Json {
  if (api === 'anthropic') {
    const type = status === 503 ? 'overloaded_error' : status >= 500 ? 'api_error' : 'invalid_request_error';
    return { type: 'error', error: { type, message } };
  }
  const type = status === 429 ? 'rate_limit_error' : status >= 500 ? 'api_error' : 'invalid_request_error';
  return { error: { message, type, param: null, code: null } };
}
