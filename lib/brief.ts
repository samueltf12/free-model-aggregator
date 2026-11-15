// The trial brief: parsing, and the catalog built from it.
//
// A trial hands this server one file (the path in MAKEFUN_PROMPT_FILE) whose prose carries three
// JSON documents, each found by its `schema` member:
//
//   - the provider directory (free-model-aggregator-01/providers/v1): the ten simulated free-tier
//     providers, each with its id, name, wire (openai, anthropic or gemini), auth scheme, the
//     environment variable a real deployment reads its key from, the key itself, and the base path
//     the server joins to MAKEFUN_ENV_URL to dial it;
//   - the quality table (free-model-aggregator-01/ratings/v1): every free model with its rating in
//     milli-points, its context length, and whether it takes tools, images and JSON mode, plus the
//     best rating each workload request could earn;
//   - the workload manifest (free-model-aggregator-01/workload/v1): the requests the env's client
//     will send, in order, each with its api, stream flag, weight and requirements.
//
// The catalog this server registers (and prints for --list) is built from the directory and the
// quality table alone: every provider with its base path, key variable and wire, and its free
// models with their context lengths. It is a pure function of the brief, so the document a live
// pass registers is the document its replay registers, byte for byte.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type ClientApi = 'openai' | 'anthropic';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. `base` is its path (or an absolute base URL). */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table: a model, the provider that serves it, and what it can do. */
export interface ModelRow {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one workload request needs from the model that serves it. */
export interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

/** One request of the workload manifest, in the order the client sends them. */
export interface WorkloadEntry {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The whole brief, parsed. Any document the file does not carry stays null. */
export interface Brief {
  providers: ProviderEntry[];
  models: ModelRow[];
  best: Map<string, number>;
  manifest: WorkloadEntry[];
  error: string | null;
}

/** One provider of the catalog document this server registers and prints. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: { id: string; free: boolean; context_length: number | null }[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The index of the `}` that closes the object opening at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep looking.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = str(v.path) !== '' ? str(v.path) : str(v.base_url) !== '' ? str(v.base_url) : str(v.base);
  return { id, name: str(v.name) !== '' ? str(v.name) : id, wire, auth, base, keyEnv: str(v.key_env), key: str(v.key) };
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const provider = str(v.provider);
  const model = str(v.model);
  if (provider === '' || model === '') return null;
  return {
    provider,
    model,
    ratingMilli: numOrNull(v.rating_milli) ?? 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function entryOf(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: req.tools === true,
      vision: req.vision === true,
      json: req.json === true,
      minContext: numOrNull(req.min_context),
    },
  };
}

/** Read and parse the brief at `path`. A missing file yields an empty brief with `error` set. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], models: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directory.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratings = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const models =
    ratings === undefined
      ? []
      : (ratings.ratings as unknown[]).flatMap((v) => {
          const m = modelOf(v);
          return m === null ? [] : [m];
        });
  const best = new Map<string, number>();
  if (ratings !== undefined && Array.isArray(ratings.best)) {
    for (const v of ratings.best as unknown[]) {
      if (isRecord(v) && str(v.id) !== '') best.set(str(v.id), numOrNull(v.rating_milli) ?? 0);
    }
  }
  const workload = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workload === undefined
      ? []
      : (workload.manifest as unknown[]).flatMap((v) => {
          const e = entryOf(v);
          return e === null ? [] : [e];
        });
  return { providers, models, best, manifest, error: null };
}

/**
 * The catalog document: one entry per directory provider, in the directory's order, holding the
 * provider's own base path (never joined to the env's URL: the env's origin differs between a
 * live pass and its replay, and the registered catalog is part of the export a replay must
 * reproduce), its key variable, its wire, and its free models from the quality table, best first.
 */
export function buildCatalog(brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const models = brief.models
      .filter((m) => m.provider === provider.id && m.free)
      .sort((a, b) => b.ratingMilli - a.ratingMilli || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))
      .map((m) => ({ id: m.model, free: true, context_length: m.contextLength }));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { providers };
}

/** The free model ids across every provider, each once, in directory order: the /v1/models list. */
export function freeModelIds(brief: Brief): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const provider of brief.providers) {
    const rows = brief.models
      .filter((m) => m.provider === provider.id && m.free)
      .sort((a, b) => b.ratingMilli - a.ratingMilli || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    for (const row of rows) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      out.push(row.model);
    }
  }
  return out;
}

/** A provider's base path joined to the env's URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme carries its key in; `none` (or no key) sends nothing. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}
