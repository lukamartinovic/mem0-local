# mem0-local - Product Vision (12-24 months)

Vision date: October 2026. Grounded in the market facts verified September/October 2026.

## 1. The problem worth solving

AI coding agents are stateless. Every serious tool - Claude Code, Cursor, Codex, IBM Bob - now recognises this and ships some answer to it. Every answer is the same shape: memory that is per-tool, per-project, invisible, and unauditable.

Claude Code's first-party memory (auto-on since v2.1.59) is stored at `~/.claude/projects/<slugified-cwd>/memory/` - **keyed to the working directory**, so a developer with ten projects owns ten silos, per tool. Users report it is invisible and unverifiable, with no relevance-ranked retrieval and no health check. The proof that this hurts is quantitative: a third-party memory replacement layered on top of Claude Code has accumulated **91,000+ GitHub stars**. Cursor ships an "Auto-Dream Memory" background process that consolidates and prunes on its own schedule - another opaque store the user cannot inspect.

Meanwhile, the interoperability layer is absent on purpose. MCP standardises tool and context access, **not memory semantics**; a proposal to add a Memory Interchange Format to MCP was **closed**, with maintainers stating memory formats are not an MCP concern. Neither A2A nor MCP defines persistent shared-memory semantics - no identity/addressing, no wire schema, no trust/provenance, no consistency model, no permissions. AMCP exists as a public draft (v0.4) with reference implementations and conformance levels L0/L1/L2, but as of September 2026 **there is no ratified interoperability standard for agent memory**. 30+ memory servers exist with incompatible formats - and notably, mem0, Zep/Graphiti, Cognee, Letta and basic-memory have **independently converged on the same core fields**: `id, content, type, timestamp, source, metadata`.

Bridging happens in practice by connecting every agent to one shared store, whose format becomes the de facto protocol. That is exactly the position mem0-local already occupies.

## 2. Market context

Four facts from the verified set define the opening:

1. **Demand is proven and unsatisfied.** 91,000+ stars on a memory replacement for Claude Code, against a first-party feature that is on by default. Users want memory they can see, verify, and query with relevance ranking - and the incumbent does not offer it.
2. **Memory is siloed by construction.** cwd-keyed first-party memory multiplies silos per project and per tool. A developer running three agents across ten projects has thirty disconnected memory stores.
3. **The standards vacuum is structural, not temporary.** The MIF proposal was rejected by MCP maintainers on principle. A W3C Community Group exists, but Community Groups are not standards-track. Waiting for a ratified standard is not a strategy; being a well-shaped de facto store is.
4. **The field is converging.** Five independent projects landed on the same core record schema. AMCP v0.4 formalises something close to it (canonical record, remember/recall, sessions, export/import portability, soft deletion, pinning, version relationships, capability discovery). A store that already exposes `id, memory(content), metadata, created_at(timestamp), user_id(source)` is one mapping layer away from that shape.

## 3. The product thesis

**One local, inspectable, agent-neutral memory store that every tool talks to - small enough to audit, boring enough to trust.**

The thesis rests on three contrarian commitments, all already true of the product today:

- **No intelligence in the store.** The calling agent does fact inference. mem0-local never gains an extraction LLM. This keeps the server small, deterministic, fully offline, and debuggable - and it means memory quality scales with the agent's model, not with a pinned server-side model that ages.
- **Inspectability is the feature.** Against invisible, unverifiable first-party memory, a store where every record can be browsed, filtered from the web UI, exported to JSON/CSV, and health-checked is the differentiated product, not a nicety.
- **Portability over lock-in.** Export and import are first-class today. As the ecosystem converges on a canonical record (`id, content, type, timestamp, source, metadata`) and AMCP matures, mem0-local should be trivially movable to whatever wins - because a memory store you cannot leave is the product users are already fleeing.

## 4. Target users

- **The multi-agent developer** (primary): runs 2-4 coding agents daily, across multiple projects, wants one memory they all share. Today silos multiply per cwd and per tool; mem0-local collapses that to one store, namespaced only where the user chooses.
- **The privacy-constrained professional**: data must not leave the machine. Fully local, no telemetry, no accounts, zero runtime network - structurally, not by toggle.
- **The small self-hosted team**: one instance on a LAN/VPN as shared team memory, inside the loopback/LAN trust model the product honestly documents.

## 5. Strategic pillars

### Pillar 1 - Radical inspectability

**User value:** every memory is visible, searchable with scores, exportable, and the store's health is a single honest endpoint. Against "invisible, unverifiable, no relevance ranking, no health check" first-party memory, this is the reason to switch.
**Defensibility:** it is architectural. An opaque background process (Auto-Dream) cannot bolt on verifiability without becoming a different product; a transparent store is verifiable by default. The existing test-enforced reliability contract (honest health, no curl-exit success, fail-fast startup) is the seed competitors would have to rebuild.

### Pillar 2 - The dumb-store invariant

**User value:** the server never hides an LLM, never phones home, never consolidates without being asked. Memory content is exactly what the user's agents wrote - auditable, deterministic, debuggable.
**Defensibility:** extraction-LLM-based servers (mem0's default path, most of the 30+) carry a model dependency, a quality ceiling set by that model, and a network story to explain. Ours scales quality with the caller's frontier model for free. This pillar is a hard constraint, stated here so it survives future feature pressure.

### Pillar 3 - Agent-neutral shared store with explicit namespaces

**User value:** any MCP-capable agent connects with one config stanza; agents share memory by default and separate by `user_id` where wanted. One store replaces the per-cwd, per-tool silo multiplication.
**Defensibility:** first-party memory is locked to its own tool by strategy, not by accident - Claude Code's memory serves Claude Code. A neutral store's neutrality is durable precisely because the incumbents' lock-in is intentional.

### Pillar 4 - Exit-shaped portability

**User value:** JSON/CSV export and duplicate-safe import today; alignment with the converging core schema (`id, content, type, timestamp, source, metadata`) and AMCP's record model as it matures. The user can always leave, which is precisely why they stay.
**Defensibility:** with the MIF proposal closed, the practical standard will be set by whichever store's format others implement against. Being cleanly shaped, documented, and import/export-complete is how a small project becomes that de facto format - the same mechanism by which teams today bridge agents through one shared store.

## 6. Roadmap

### Near (0-3 months): close the audited gaps

These come straight from the current spec's Known limitations - shipped trust features, not new scope:

- **Memory lifecycle:** expose TTL/expiry through the tools (mem0 already supports `expiration_date` internally); per-fact lifetime as an optional `add` parameter; extend prune beyond age-only with the same dry-run-first, never-delete-unevaluated policy.
- **Usage signal:** `last_used_at` tracking on search hits, so users can see which memories earn their keep and prune dead ones with evidence.
- **Observability depth:** memory count and disk usage in `/health`; time-range filtering in the web UI and in export (export today is entity-scoped only).
- **Scores-in-UI:** relevance-ranked browsing with visible scores in the web UI, matching what the MCP tools already return.

### Mid (3-9 months): make sharing deliberate

- **Namespace ergonomics:** first-class listing/renaming of `user_id` namespaces; a documented convention for project-scoped and shared scopes layered on metadata (not a schema change).
- **Import bridges:** duplicate-safe importers for the formats users actually flee from - Claude Code's `~/.claude/projects/*/memory/` trees chief among them - so migration into the shared store is one command on the host, run by the operator.
- **Optional access guard:** a single static token on HTTP routes as an opt-in hardening step for LAN deployment, documented with the same honesty as today's "no auth" statement. Not accounts, not per-record permissions - the trust model stays loopback/LAN with one added door lock.
- **Schema hygiene:** formalise the stored record against the converged core fields (`id, content, type, timestamp, source, metadata`) so exports are natively shaped like what the ecosystem is converging on.

### Long (9-24 months): become the de facto bridge format

- **AMCP alignment:** implement remember/recall semantics, soft deletion, pinning, version relationships, and capability discovery incrementally; target a published conformance claim (L0, then L1 per AMCP v0.4 levels) when the draft stabilises. Export/import portability becomes protocol-level portability.
- **Sessions:** AMCP-shaped session records layered on namespaces, giving agents conversation-scoped memory without a new backend.
- **Reference-quality documentation of the store format**, so other tools implement against mem0-local's schema - the mechanism by which the 30-server fragmentation resolves in practice.
- **Interop verification suite:** extend the existing tiered tests with cross-format import/export round-trip tests against AMCP reference implementations.

## 7. What we will NOT build, and why

- **An extraction/consolidation LLM inside the server.** This is the dumb-store invariant (Pillar 2). It buys model independence, determinism, offline operation, and debuggability. Cursor's Auto-Dream is the alternative; the 91k stars say where users land.
- **Cloud sync, hosted tiers, accounts, telemetry.** Hard constraint; also the differentiation. The moment sync exists, the trust storey collapses.
- **A second backend or storage engine.** One container, one vector store, one embedder. Small and debuggable beats pluggable.
- **Multi-tenant permissions / RBAC.** The honest trust model is loopback/LAN plus at most one token. Enterprise identity is a different product with a different everything.
- **An MCP proposal or standards lobbying.** MIF was closed on the grounds that memory formats are not an MCP concern - the standard that matters is the one stores implement, so effort goes into being the store worth implementing against.
- **Analytics, ranking-by-"importance", or memory analytics dashboards beyond usage signal.** The store returns memories; judgement lives in the agent.

## 8. Success metrics

- **Adoption of the bridge position:** number of agents per connected store (target: median >= 2 within 12 months - the product exists to collapse silos, and one-agent usage means the silo problem is unsolved).
- **Trust kept:** zero telemetry events emitted (structurally guaranteed; verified by the test suite's no-network tiers); zero authenticated-bypass incidents in LAN deployments.
- **Portability exercised:** export/import round trips in tests and in user reports; successful migrations from first-party memory formats via the import bridges.
- **Reliability contract held:** the test-enforced guarantees (no silent data loss, dimension safety, fatal self-test, honest health) remain green on every release; health accuracy is a regression gate.
- **Standards position:** AMCP conformance level claimed and externally verifiable when the draft stabilises; at least one external tool able to import mem0-local exports unmodified.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| First-party tools improve memory transparency, eroding the inspectability edge. | Inspectability is architectural for us and strategic lock-in for them; keep shipping the audited-gap list fast, and keep export complete so switching costs stay near zero either direction. |
| A ratified standard emerges with an incompatible record shape. | Track AMCP and the converged core fields (`id, content, type, timestamp, source, metadata`); keep the stored record one mapping layer from that shape so alignment is a release, not a rewrite. |
| The standards vacuum never resolves; 30+ incompatible formats persist. | Then the de-facto-store mechanism applies: be the best-documented, easiest-to-bridge local store. Fragmentation is the condition our neutrality monetises. |
| Feature pressure toward an in-server LLM ("just add summarisation"). | The invariant is in writing in SPEC and here; any such proposal must argue against Pillar 2 explicitly, not slip in as a flag. |
| LAN deployments misjudge the no-auth model. | Keep the security section brutally honest, ship the optional token guard in the mid horizon, and say "loopback/LAN trust" in the first paragraph of the docs. |
| mem0/Qdrant/fastembed dependency drift. | Already managed by pins and tested migrations (the Ollama-model aliasing and Qdrant v1.13.2 pin are the proven pattern); keep the dimension-mismatch and self-test gates fatal. |

## 10. Open strategic questions

1. **Sessions vs. namespaces:** should conversation-scoped memory be an AMCP session construct layered on `user_id`, or a first-class field? The draft matures; committing early risks mapping churn, waiting risks being shaped by others.
2. **How far toward AMCP before it stabilises?** Implementing remember/recall and capability discovery now buys de-facto influence; it also risks tracking a moving draft. A reasonable trigger: when AMCP's reference implementations stop breaking between minor versions.
3. **Token auth or none?** The mid-horizon plan is one static token, opt-in. The question is whether small teams on VPNs actually deploy it without it - user evidence should decide before it ships.
4. **Does project scoping belong in the schema?** Today: metadata convention. If the multi-agent developer persona shows consistent cross-project recall failures, a `project` field is cheap to add before the record format hardens; afterwards it is not.