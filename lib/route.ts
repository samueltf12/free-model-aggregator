// Routing: which free model serves each request, planned against the providers' budgets, with
// failover when a provider refuses.
//
// The whole workload is known before the first request arrives, and the quality table gives
// every free model's rating, so the serving choice is planned, not guessed: an assignment pass
// over the requests still to come gives each one the highest-rated free model that meets its
// requirements, under a per-provider call allowance. Budgets are drawn per trial and never
// published; the allowances here are the conservative end of each provider's range, and they
// are planning aids only — when a provider refuses anyway (a scheduled failure, an outage, a
// budget smaller than planned), the request fails over to the next candidate and the plan is
// redrawn from what the trial has actually done. A provider that has refused twice is planned
// around for the rest of the trial but stays on the failover chain as a last resort: an answer
// from it still beats no answer.
//
// One more routing rule comes from the wires themselves: a JSON-mode request on the OpenAI wire
// carries response_format, which an Anthropic-wire provider cannot echo back, so such a request
// is only routed to an OpenAI- or Gemini-wire provider whose model takes JSON.

import {
  authHeaders,
  joinBase,
  type Brief,
  type ModelRow,
  type ProviderEntry,
  type Requirement,
  type WorkloadEntry,
} from './brief.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonAnswer,
  type CanonRequest,
} from './canon.ts';

export interface Candidate {
  provider: ProviderEntry;
  row: ModelRow;
}

/**
 * The call allowance the planner assumes per provider: the low end of the range the kit's
 * published provider table draws each trial's budget from. Unknown providers get a low default.
 */
const PLANNING_ALLOWANCE = new Map<string, number>([
  ['groq', 9],
  ['cerebras', 5],
  ['gemini', 6],
  ['zai', 4],
  ['openrouter', 10],
  ['sambanova', 4],
  ['pollinations', 10],
  ['cloudflare', 7],
  ['ovh', 5],
  ['kilol', 12],
]);
const DEFAULT_ALLOWANCE = 4;
/** A provider that refused this many calls is planned around (it may be out for the trial). */
const BURNED_AFTER_FAILURES = 2;
/** The most provider calls one client request may cost before the server gives up on it. */
const MAX_ATTEMPTS = 8;
const CALL_TIMEOUT_MS = 15000;

const byRating = (a: Candidate, b: Candidate): number =>
  b.row.ratingMilli - a.row.ratingMilli ||
  (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
  (a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0);

export class Router {
  readonly #brief: Brief;
  readonly #envUrl: string;
  /** Every provider call this trial has made through this router, by provider id. */
  readonly #calls = new Map<string, number>();
  /** Every refused provider call, by provider id. */
  readonly #failures = new Map<string, number>();

  constructor(brief: Brief, envUrl: string) {
    this.#brief = brief;
    this.#envUrl = envUrl;
  }

  /** Whether a model row meets a request's requirements. */
  static meets(row: ModelRow, requires: Requirement): boolean {
    if (requires.tools && !row.tools) return false;
    if (requires.vision && !row.vision) return false;
    if (requires.json && !row.json) return false;
    if (requires.minContext !== null && (row.contextLength === null || row.contextLength < requires.minContext)) return false;
    return true;
  }

  /** Every free candidate that may serve this request, best rating first. */
  feasible(entry: WorkloadEntry, canon: CanonRequest): Candidate[] {
    const requires: Requirement = {
      tools: entry.requires.tools || canon.tools.length > 0,
      vision: entry.requires.vision || canon.turns.some((t) => t.blocks.some((b) => b.type === 'image')),
      json: entry.requires.json || canon.jsonMode,
      minContext: entry.requires.minContext,
    };
    const out: Candidate[] = [];
    for (const provider of this.#brief.providers) {
      // An Anthropic-wire provider parses no response_format: it would echo JSON mode as absent.
      if (canon.jsonMode && provider.wire === 'anthropic') continue;
      for (const row of this.#brief.models) {
        if (row.provider !== provider.id || !row.free) continue;
        if (!Router.meets(row, requires)) continue;
        out.push({ provider, row });
      }
    }
    return out.sort(byRating);
  }

  #allowance(providerId: string): number {
    if ((this.#failures.get(providerId) ?? 0) >= BURNED_AFTER_FAILURES) return this.#calls.get(providerId) ?? 0;
    return PLANNING_ALLOWANCE.get(providerId) ?? DEFAULT_ALLOWANCE;
  }

  /**
   * Assign each remaining request its model: heavier requests and the ones with the most to
   * lose (the widest gap between their best and second-best candidate) claim scarce provider
   * allowance first; every request takes the best candidate its provider can still afford.
   */
  plan(fromIndex: number, canons: Map<number, CanonRequest>): Map<number, Candidate> {
    const manifest = this.#brief.manifest;
    const items: { index: number; entry: WorkloadEntry; candidates: Candidate[]; regret: number }[] = [];
    for (let index = fromIndex; index < manifest.length; index++) {
      const entry = manifest[index];
      const canon = canons.get(index);
      // Before a request arrives its body is unknown; plan it on the manifest alone.
      const probe: CanonRequest =
        canon ?? { system: '', turns: [], tools: [], maxTokens: null, stops: [], jsonMode: entry.api === 'openai' && entry.requires.json, stream: entry.stream };
      const candidates = this.feasible(entry, probe);
      const regret = candidates.length > 1 ? candidates[0].row.ratingMilli - candidates[1].row.ratingMilli : candidates.length === 1 ? candidates[0].row.ratingMilli : 0;
      items.push({ index, entry, candidates, regret });
    }
    items.sort((a, b) => b.entry.weight - a.entry.weight || b.regret - a.regret || a.index - b.index);
    const plannedUse = new Map<string, number>();
    const assigned = new Map<number, Candidate>();
    for (const item of items) {
      let pick: Candidate | null = null;
      for (const candidate of item.candidates) {
        const used = (this.#calls.get(candidate.provider.id) ?? 0) + (plannedUse.get(candidate.provider.id) ?? 0);
        if (used < this.#allowance(candidate.provider.id)) {
          pick = candidate;
          break;
        }
      }
      // Nothing affordable is not a reason to leave a request unassigned: take the best anyway
      // and let failover sort out the refusal if the allowance guess was wrong.
      if (pick === null && item.candidates.length > 0) pick = item.candidates[0];
      if (pick !== null) {
        assigned.set(item.index, pick);
        plannedUse.set(pick.provider.id, (plannedUse.get(pick.provider.id) ?? 0) + 1);
      }
    }
    return assigned;
  }

  /** One call to one provider, in its wire. Null when it refused or its answer did not parse. */
  async #call(candidate: Candidate, canon: CanonRequest): Promise<CanonAnswer | null> {
    const { provider, row } = candidate;
    const base = joinBase(this.#envUrl, provider.base);
    const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
    let url: string;
    let body: unknown;
    if (provider.wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      body = toOpenAI(canon, row.model);
    } else if (provider.wire === 'anthropic') {
      url = `${base}/v1/messages`;
      headers['anthropic-version'] = '2023-06-01';
      body = toAnthropic(canon, row.model);
    } else {
      url = `${base}/v1beta/models/${row.model}:${canon.stream ? 'streamGenerateContent' : 'generateContent'}`;
      body = toGemini(canon, row.model);
    }
    if (canon.stream) headers.accept = 'text/event-stream';
    let text: string;
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
      text = await res.text();
      if (!res.ok) return null;
    } catch {
      return null;
    }
    if (provider.wire === 'openai') return answerFromOpenAI(text, canon.stream, row.model);
    if (provider.wire === 'anthropic') return answerFromAnthropic(text, canon.stream, row.model);
    return answerFromGemini(text, canon.stream, row.model);
  }

  /**
   * Serve one client request: the planned model first, then the other feasible candidates by
   * rating, providers that keep refusing pushed to the back of the chain. Every attempt is
   * counted, so the next plan is drawn from the trial's real state.
   */
  async serve(index: number, entry: WorkloadEntry, canon: CanonRequest): Promise<CanonAnswer | null> {
    const planned = this.plan(index, new Map([[index, canon]])).get(index);
    const feasible = this.feasible(entry, canon);
    const chain: Candidate[] = [];
    const seen = new Set<string>();
    const push = (candidate: Candidate | undefined): void => {
      if (candidate === undefined) return;
      const key = `${candidate.provider.id}/${candidate.row.model}`;
      if (seen.has(key)) return;
      seen.add(key);
      chain.push(candidate);
    };
    if (planned !== undefined && (this.#failures.get(planned.provider.id) ?? 0) < BURNED_AFTER_FAILURES) push(planned);
    for (const candidate of feasible) {
      if ((this.#failures.get(candidate.provider.id) ?? 0) < BURNED_AFTER_FAILURES) push(candidate);
    }
    push(planned);
    for (const candidate of feasible) push(candidate);
    for (const candidate of chain.slice(0, MAX_ATTEMPTS)) {
      this.#calls.set(candidate.provider.id, (this.#calls.get(candidate.provider.id) ?? 0) + 1);
      const answer = await this.#call(candidate, canon);
      if (answer !== null) return answer;
      this.#failures.set(candidate.provider.id, (this.#failures.get(candidate.provider.id) ?? 0) + 1);
    }
    return null;
  }
}
