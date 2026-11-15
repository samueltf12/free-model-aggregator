# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/samueltf12/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/samueltf12/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs over ten simulated free-tier providers, routing each request to the best free model that can serve it, failing over when a provider refuses, and registering the catalog it read from the brief.

## Run it

```sh
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief file> node main.ts
node main.ts --list   # print the catalog document and exit
```

## How it works

The brief names the ten providers (their wires, auth schemes, base paths and keys), the quality
table (every free model's rating, context length, and whether it takes tools, images and JSON
mode) and the workload manifest (each request's API, stream flag, weight and requirements).

- `lib/brief.ts` parses the brief's three JSON documents and builds the catalog from them: one
  entry per provider with its free models, the same document `--list` prints and the register
  body carries.
- `lib/route.ts` plans the workload: heavier requests and the ones with the most to lose claim
  scarce provider allowance first, each taking the highest-rated free model that meets its
  requirements. When a provider refuses, the request fails over to the next candidate and the
  plan is redrawn from the calls the trial has actually made.
- `lib/canon.ts` holds one canonical form for a request and an answer, and every translation
  between the two client wires (OpenAI, Anthropic) and the three provider wires (OpenAI,
  Anthropic, Gemini), in JSON and in server-sent events. It preserves what a provider reads out
  of a request — the system text, the last user turn's text, the tool names in order,
  max_tokens, the stop sequences, the image bytes and the JSON-mode flag — so the answer comes
  back whole, with the provider's usage counts and its stop reason mapped into the client's
  wire.

Every decision is a function of the brief and of the providers' answers seen so far: no wall
clock, no randomness, so a replayed trial walks the same routes.

---

Entered on [make.fun](https://make.fun/?ref=hcb88zsr&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
