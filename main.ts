// The best free AI model aggregator: one server, two client APIs, ten free-tier providers.
//
// The server reads the trial brief (MAKEFUN_PROMPT_FILE): the provider directory, the quality
// table and the workload manifest. It builds one catalog from the brief, listens on a port it
// picks, and registers the port and the catalog with the env (MAKEFUN_ENV_URL). The env's
// client then sends the workload's requests, one at a time, to POST /v1/chat/completions and
// POST /v1/messages; each is routed to the best free model that can serve it (lib/route.ts),
// translated into that provider's wire and back (lib/canon.ts), and answered in the client's
// wire, as JSON or as server-sent events. GET /v1/models lists the free catalog in the shape of
// the auth header it arrives with. When GET $MAKEFUN_ENV_URL/work says the workload is done,
// the server exits 0.
//
// Run with --list, it prints the catalog document and exits 0 without starting the server: the
// same document the register body carries, byte for byte. Outside a trial there is no brief to
// read, and the catalog is empty.
//
// Everything the server decides is a function of the brief and of the providers' answers it has
// seen: no wall clock, no randomness, so a replayed trial walks the same routes and leaves the
// same state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModelIds,
  isRecord,
  readBrief,
  type Brief,
  type Catalog,
  type ClientApi,
  type Json,
  type WorkloadEntry,
} from './lib/brief.ts';
import {
  anthropicJsonAnswer,
  anthropicStreamAnswer,
  clientError,
  fromAnthropic,
  fromOpenAI,
  openAIJsonAnswer,
  openAIStreamAnswer,
  type CanonAnswer,
} from './lib/canon.ts';
import { Router } from './lib/route.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief: Brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const catalog: Catalog = buildCatalog(brief);
const modelIds: string[] = freeModelIds(brief);
const router = new Router(brief, ENV_URL);

/** How many client requests have arrived: the manifest is served in arrival order. */
let arrivalIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** GET /v1/models: the free catalog, in the shape of the wire the auth header names. */
function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: modelIds.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: modelIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
  });
}

/** The manifest row for the request now arriving; past the manifest, a row read off the body. */
function entryFor(api: ClientApi, stream: boolean): WorkloadEntry {
  const entry = brief.manifest[arrivalIndex];
  arrivalIndex += 1;
  if (entry !== undefined) return entry;
  return { id: `extra-${arrivalIndex}`, api, stream, weight: 1, requires: { tools: false, vision: false, json: false, minContext: null } };
}

function answer(res: ServerResponse, api: ClientApi, stream: boolean, served: CanonAnswer): void {
  if (api === 'openai') {
    if (stream) sendStream(res, openAIStreamAnswer(served));
    else sendJson(res, 200, openAIJsonAnswer(served));
  } else {
    if (stream) sendStream(res, anthropicStreamAnswer(served));
    else sendJson(res, 200, anthropicJsonAnswer(served));
  }
}

/** POST /v1/chat/completions and POST /v1/messages: route, translate, answer. */
async function serve(req: IncomingMessage, res: ServerResponse, api: ClientApi): Promise<void> {
  const body = await readBody(req);
  if (body === null) {
    sendJson(res, 400, clientError(api, 400, 'the request body is not one JSON object'));
    return;
  }
  const canon = api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const entry = entryFor(api, canon.stream);
  const served = await router.serve(arrivalIndex - 1, entry, canon);
  if (served === null) {
    sendJson(res, 503, clientError(api, 503, 'no free model could serve this request'));
    return;
  }
  answer(res, api, canon.stream, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return serve(req, res, 'openai');
  if (req.method === 'POST' && url.pathname === '/v1/messages') return serve(req, res, 'anthropic');
  const api: ClientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  sendJson(res, 404, clientError(api, 404, `no route ${req.method ?? ''} ${url.pathname}`));
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const api: ClientApi = (req.url ?? '').startsWith('/v1/messages') ? 'anthropic' : 'openai';
        sendJson(res, 500, clientError(api, 500, 'the server failed to answer'));
      }
    });
  });
  let port = await listen(server);
  // 8080 is the env's own port on one layout; if the OS handed it out, take another.
  for (let attempt = 0; port === 8080 && attempt < 5; attempt++) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    port = await listen(server);
  }
  if (port === 8080) throw new Error('the OS keeps handing out 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${modelIds.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
