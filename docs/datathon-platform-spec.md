# LatamQA Datathon Platform — Specification

**Status:** Draft v0.5. Changes from v0.4: (1) the **top-30 scoring cap is removed** — a
team's score now sums `score(q)` over **all** its accepted questions (§3.2, §7, D22), and
the committee's counted-question audit widens accordingly (§6.3); (2) the **LLM panel is
served on the Grid'5000 cluster with vLLM** (OpenAI-compatible, via LiteLLM), replacing HF
Inference Endpoints for panel inference (§6.1, §9, §10, D23); (3) a **Wikidata QID may be
supplied directly or resolved from a linked Wikipedia article URL** (§3, §4, §8, D24);
(4) **evaluated and rejected** a Gradio-SDK/ZeroGPU deployment, retaining the Docker Space
(§10, D25). v0.4 (simplified scope) established: software-only scope; in-country reviewers removed (validity
checklist survives as a committee-wide check, §6.3); hosting as a single **Hugging Face
Space** replacing the single-VM Postgres/Argilla architecture (§10, Argilla dropped,
built-in review UI primary). Decision history is retained in §11 with superseded items
marked.
**Date:** 2026-07-13
**Source:** [Note Conceptuelle V2 — Datathon Régional](<V2- Note Conceptuelle _ Datathon Régional sur la Valorisation des Données Ouvertes et des Logiciels Open Source.md>)
**Event date:** Saturday, October 3, 2026 — one global window, **13:30–23:30 UTC**
(07:30–17:30 Mexico City; 10:30–20:30 Santiago / Buenos Aires / Montevideo / Brasília)

---

## 1. Purpose

A web platform through which datathon teams author culturally-grounded multiple-choice
questions (MCQs) in their local language and English, and through which Inria Chile
evaluates the quality of those questions — automatically (schema validation, duplicate
detection, LLM-panel difficulty scoring) and manually (committee review against the
exclusion criteria) — producing per-country and regional leaderboards, and exporting
accepted questions into the LatamQA v2 dataset.

The platform is a single **datathon service** (FastAPI) deployed as a **Hugging Face
Space**, owning every surface: participant authoring, committee review, leaderboards,
public pages, and administration. It reuses the existing `latamqa` evaluation machinery
(LiteLLM-based MCQ evaluation with batching and retries, YAML model configurations) via a
programmatic API, with two deliberate divergences from `eval_mcq.py` specified in §6.2
(seed derivation and error handling).

This document specifies the software. Everything the platform *imposes on participants*
(scoring formula, tie-breaks, quotas, exclusion criteria, rate limits) must be restated in
the published participant rules — producing and publishing those rules is an organizer
task, out of scope here.

### In scope

- Team and participant registration/authentication
- Question authoring, editing, and submission with quota enforcement
- Automatic validation (schema, Wikidata links — direct QID or Wikipedia-article
  resolution, language checks)
- Semantic duplicate detection (multilingual embeddings, threshold 0.92 recalibrated per D2)
- LLM-panel evaluation and per-question difficulty scoring
- Committee review workflow (exclusion criteria, duplicate confirmation) — built-in UI
- Scoring, tie-breaking, per-country and regional leaderboards
- Export of accepted questions to the LatamQA dataset format (Hugging Face Hub)
- Event administration: admin console + authenticated admin API (event window, panel
  freeze, team import, run triggering, restores), monitoring, per-country organizer
  counts page

### Out of scope

- Communication tooling (Discord/Matrix is provided separately)
- Registration/selection of teams *before* the event (the platform only imports the
  confirmed team list)
- The landing page and communication artifacts
- Training or fine-tuning of models
- All non-software workstreams: participant-rules publication, legal/privacy/DPO items,
  contribution-agreement wording, committee recruitment and staffing, prize logistics

---

## 2. Actors and roles

| Role | Who | Capabilities |
| --- | --- | --- |
| **Participant** | Team member (student/researcher) | Author, edit, submit questions for their team; withdraw (with confirmation, §3.1); see own team's status, per-question aggregate scores, and leaderboards |
| **Team** | 3–6 participants, one country | Shares a single question workspace and quota |
| **Committee reviewer** | Inria Chile + invited experts | Review flagged questions, confirm duplicates, apply exclusion criteria incl. the validity checklist (§6.3), annotate decisions — in the platform's review UI |
| **National organizer** | Partner institution staff | Read-only per-country counts page (team submission volumes; no question content before close) |
| **Admin** | Inria Chile technical team | Via the admin console/API only (§10): configure event, import teams, freeze panel at open, trigger runs, restore withdrawn questions, export dataset. Every action is audit-logged |

Teams are created by admins from the confirmed registration list (deadline
September 21, 2026 per the concept note).

**Authentication (D1, revised):** all roles sign in with **Hugging Face OAuth** ("Sign in
with HF" — native to Spaces, no SMTP or password infrastructure). Admins import the team
roster mapping HF usernames to teams; reviewer and admin roles are configured allowlists
of HF usernames. Individual identity per person gives a clean audit trail and supports
per-member citation (§8).

---

## 3. The question object

A question is the central entity. Required fields (per the concept note Annexe):

| Field | Constraint |
| --- | --- |
| `question` | Text in the team's local language (Spanish variant or Brazilian Portuguese) |
| `answer` | The single correct option, local language |
| `distractor1..3` | Three plausible incorrect options, local language, all four options pairwise distinct — and each distractor must be *false*, not merely different (§6.3 criterion 9) |
| `question_en`, `answer_en`, `distractor1..3_en` | Mandatory English translation of every text field |
| `country` | Auto-filled from the team; not editable |
| `region` | Subnational region (state, province…), optional free text with autocomplete |
| `cultural_dimension` | **Multi-select with one designated primary**, from: history, politics, art, music, gastronomy, sport, indigenous peoples, Afro-descendant cultures, gender, language, religion, geography, traditions, literature, science, other. (Gender and Afro-descendant cultures are explicit values because the note names gender and minority bias as headline diagnostic objectives.) |
| `context_note` | Cultural context explanation in Spanish or Portuguese, mandatory |
| `wikidata_qids` | One or more Wikidata entity IDs (`Q…`), each **entered directly or resolved from a linked Wikipedia article URL** (§4) — format/shape checked at submit; resolution and existence verified asynchronously (§4). Stores the canonical resolved `Q…` values |
| `wikidata_source_urls` | Retained/derived: the Wikipedia (or Wikidata) article URLs any `wikidata_qids` entry was resolved from, kept for provenance/export (§4, §8); empty for directly-entered QIDs |
| `cultural_provenance` | Mandatory justification **only if** `wikidata_qids` is empty ("procedencia cultural") |

Length limits (calibrated on LatamQA v1 — an M1 acceptance criterion, not "tune later"):
question ≤ 500 chars (v1 max ≈ 225), **options ≤ 450 chars** (v1 max = 437; a 200-char
limit would have rejected ~3 % of the v1 corpus), context note ≤ 1000 chars. Question and
option text must not contain instruction-like content addressed to a model ("ignore the
options above…"); a heuristic filter flags suspicious submissions for committee review (§6.3).

### 3.1 Lifecycle

```text
draft ──submit──▶ submitted ──panel evaluation──▶ evaluated ──not excluded──▶ accepted
                      │                               │
                      ├────────▶ flagged ◀────────────┤   (flag sources: duplicate,
                      │             │                      heuristic, no-answer rate,
                      │             │                      translation adequacy, reviewer)
                      │             ├─ committee accepts ──▶ returns to prior state
                      │             └─ committee excludes ─▶ excluded      [terminal]
                      │
        withdraw (with confirmation) — allowed from submitted, flagged, or evaluated
                      │
                      ▼
                 withdrawn [terminal] ──clone to draft──▶ new draft
                                        (new question, new sequence number;
                                         10-min grace inherits precedence, below)
```

- **draft** — editable, not counted against quota, never evaluated.
- **submitted** — frozen content; assigned a **submission sequence number** (a per-event
  monotonic counter issued at the commit of the submission transaction — the *sole*
  arbiter of duplicate precedence, tie-breaks, and the close boundary; wall-clock
  timestamps are informational display only). Counts against quota per §3.2.
- **withdrawn** — a distinct terminal state (scores 0, record kept for audit). Requires
  an explicit confirmation step. Withdrawals are **disabled in the final 15 minutes**
  of the event window. Admins can restore a withdrawn question preserving its original
  sequence number (recovery path for accidental or malicious withdrawal; §9 alerts on
  mass-withdrawal patterns). "Clone to draft" creates a *new* question from a withdrawn
  one's content — it does not inherit the sequence number, except: a resubmission within
  a **10-minute grace period** of the withdrawal inherits the original's sequence number
  *for duplicate-precedence purposes only* (so fixing a typo does not forfeit precedence).
- **flagged** — automatically (duplicate > 0.92, heuristic filter, no-answer-rate,
  translation-adequacy) or by a reviewer; awaits committee decision. Flagged questions
  still get panel-evaluated so a decision in their favor doesn't delay scoring. Flag
  notifications to teams are **batched on the 15-minute leaderboard cadence**, never
  instant (an instant notification would itself be a probe channel).
- **excluded** — committee removed it (reason from §6.3); scores 0, quota not refunded.
- **evaluated / accepted** — has a panel difficulty score; **counts toward the team
  score** unless excluded (no top-N cap — every accepted question counts, §7).
- **pending** is *not* a lifecycle state: it is a sub-state of submitted/evaluated
  marking questions with errored evaluation cells awaiting re-run (§6.2).

### 3.2 Quotas (decision D8)

- Max **60 submitted** questions per team — a **lifetime count**, with one exception:
  a withdrawal refunds quota **only if the question has not yet entered any evaluation
  cycle** (in practice a ≤ 15-minute window after submission). Withdrawals of evaluated
  or flagged questions never refund. This kills the submit → read score → withdraw →
  mutate → resubmit oracle loop that an unconditional refund would create, while
  preserving the fix-a-typo path (submitted content is frozen, so withdrawal+resubmit
  is the only correction route).
- **All accepted questions count toward the team's final score** (D22) — there is no
  top-N cap; the team score is the sum of `score(q)` over every accepted question (§7).
  The 60-question lifetime quota (above) is the only bound on how many can count.
- Submission rate limit: at most 10 submissions / 10 minutes / team. Participant
  guidance must state the consequence: placing the full 60-question quota takes at
  least 50 minutes of sustained submission — do not plan a last-minute bulk upload.

---

## 4. Automatic validation (synchronous, at submit time)

Runs in < 2 s using **only local checks** — no external service sits on the blocking
path — and blocks submission with actionable error messages:

1. All required fields present; 4 pairwise-distinct options in each language.
2. Wikidata consistency, as a conjunction of conditionals: (`wikidata_qids` non-empty →
   every entry is a `Q\d+` QID, a Wikipedia article URL, **or** a canonical Wikidata
   entity URL) **and** (`wikidata_qids` empty → `cultural_provenance` non-empty). Only
   the *shape* is checked synchronously (a well-formed QID; a URL on a recognized
   `*.wikipedia.org` host; or a `www.wikidata.org/wiki/Q…` or `/entity/Q…` URL, which is
   stripped to its embedded QID at submit); **QID resolution and existence** are
   verified asynchronously (seconds later, like duplicate detection):
   - A supplied **Wikipedia article URL is resolved to its linked Wikidata item**
     (`wikibase_item`) via a server-side batched MediaWiki lookup on the article's own
     language edition (`action=query&redirects=1&prop=pageprops`, reading the
     `wikibase_item` property together with the `disambiguation` marker — do *not*
     restrict with `ppprop=wikibase_item`, or the disambiguation marker is suppressed),
     and the resolved `Q…` is stored as the canonical value (the source URL is retained
     for provenance/export, §8). Soft-flagged for committee review: an article with no
     linked Wikidata item; a **disambiguation page** (detected by the `disambiguation`
     pageprop — these *do* carry a `wikibase_item`, so filtering it out would silently
     resolve them to a disambiguation-item QID); a **redirect** (followed via
     `redirects=1` and re-checked); or an unresolvable title.
   - A directly supplied **QID** is checked for existence via batched `wbgetentities`
     calls against `www.wikidata.org`.

   Both lookups use a descriptive User-Agent; a nonexistent QID or unresolvable link
   produces a soft flag, and **QID-and-link-only in-place edits on submitted questions
   are permitted** (audited; safe because Wikidata references are neither embedded for
   duplicate detection nor sent to the panel) — this soft flag is *not* the §3.1
   `flagged` state and never affects quota or the refund window. A lookup timeout
   (~1 s) fails open with deferred re-verification. Rationale: Wikidata and Wikipedia
   are no-SLA third-party services; a maxlag or outage episode must not block
   submissions in six countries at once (§9 degraded modes).
3. Language sanity check: lightweight language-ID on `question` and `context_note`
   only (never on answers/options — proper nouns, dates, and indigenous terms like
   the note's own example "milcao" would spuriously warn), and only above a minimum
   text length. Result is an inline warning; it never blocks. A *very-low-confidence*
   result (e.g., regional fields detected as English) additionally soft-flags for
   committee review — the note makes local-language formulation mandatory, and the
   §6.3 translation-adequacy check alone cannot catch English pasted into both
   language fields (identical texts score high cross-lingual similarity).
4. Length limits and instruction-injection heuristics (hard block on limits, flag on
   heuristics).
5. Quota and rate-limit check.

---

## 5. Duplicate detection (asynchronous, seconds after submit)

- Embed `question + " " + answer` (both languages) with `intfloat/multilingual-e5-large`
  (D2, §11). Known residual risk, accepted and recorded: distractor-only
  variants are invisible to this signal; the committee sees full question text when
  confirming, and the D8 quota rule removes the free mutation loop that would exploit it.
- Compare against **(a)** all other submitted questions in the event (any team, any
  country, both languages, max similarity across the four language pairings) and
  **(b)** the existing LatamQA v1 datasets — a datathon question duplicating v1 adds
  nothing to v2.
- At the canonical scale (§6.2) the corpus is ≤ ~8,640 event vectors plus the v1
  datasets: **brute-force cosine similarity in memory (NumPy) — no vector database.**
- Cosine similarity **> 0.92** → auto-flag as duplicate candidate, linked to its match.
- Committee confirms or dismisses. On confirmation: the **earliest sequence number**
  survives; later ones are excluded with reason `duplicate`. Matches against LatamQA v1
  are excluded outright (no precedence question).
- **Orphan rule:** if a surviving original is later withdrawn or excluded for an
  unrelated reason, every duplicate exclusion that named it as the survivor is
  automatically re-opened for committee re-decision (the earliest remaining version
  survives) — otherwise a withdrawal could leave *zero* surviving copies while a rival
  team stays excluded, violating the note's "la première version soumise est conservée".
  A withdrawal of a question that won a confirmed-duplicate contest never refunds quota.
- Both teams are notified of confirmed duplicate **decisions** (content still withheld
  during the event); flag notifications ride the 15-minute cadence (§3.1).

---

## 6. Quality evaluation

### 6.1 LLM panel (decisions D3, D9)

- Panel = **4 small-to-medium open-weight models** (O2/D9), configured with the
  existing `latamqa/models/*.yaml` format; **served on the Grid'5000 cluster with
  vLLM** (which exposes an OpenAI-compatible API) and called through LiteLLM — the YAML
  `LLM URI` field carries each vLLM server's URL (D23). Rationale for small/medium: they
  are *more foolable* by cultural questions than 70B+ models (which often just know the
  answer), giving a wider score distribution that actually separates teams — an
  all-large panel would cluster scores near zero and discriminate poorly — and they are
  cheap and simple to serve (small open-weight checkpoints fit comfortably on a single
  Grid'5000 GPU node under vLLM).
- **Fully undisclosed composition (O2/D9).** Neither the 4 models nor a candidate
  shortlist is published; participants are told only "four small-to-medium open-weight
  models," drawn from a broad pool. Rationale: small open-weight models run offline on
  free hardware, so a *named* panel lets any team optimize questions to fool those
  exact 4 — which not only games scores but degrades the v2 dataset (it would capture
  4 models' quirks, not genuine cultural difficulty). Hiding forces robustly-hard
  questions. The composition is published *after* the event.
- **Panel frozen at event open.** The composition is recorded in the event config and
  cannot be edited during the event. A failing vLLM server is handled by deferral (§6.2
  — errored cells re-queue), never by panel edits. **With only 4 models, one lost server
  is 25 % of the signal**, so provision redundancy on Grid'5000 (reserve spare GPU nodes
  so a model can be re-served without waiting on the scheduler; vLLM can also run
  multiple replicas of a model behind a client-side round-robin in the YAML config) and
  treat a permanent model loss as significant: the final authoritative run then drops
  that model from **all** questions and recomputes every score from archived runs so
  every question shares one denominator (cheap by construction — recomputation from
  archived rows is arithmetic, no inference; the ~36 min/model figure of §6.2 applies
  only when a model's cells must actually be re-executed, e.g., server-restore
  catch-up). The Grid'5000 GPU reservation covering the event window (plus the M2/M4
  rehearsals, with a buffer) must be secured in advance and its servers reachable from
  the Space — an event-critical dependency (§9, §10).
- **Single panel, rolling evaluation:** newly submitted questions are evaluated
  continuously during the event, feeding the provisional leaderboard (§7). A final
  authoritative run at event close re-evaluates every pending question **and every
  cell that ever errored** before any score is used for an announcement or freeze.
- Anti-gaming mitigations:
  - Fully undisclosed panel composition (above).
  - Teams see aggregate per-question scores only, never per-model breakdowns.
  - The leaderboard refreshes on a fixed cadence (default every 15 min, configurable);
    flag notifications are batched on the same cadence.
  - The D8 quota rule caps total scored attempts at 60 per team, so the in-event
    feedback loop cannot be used as an unbounded oracle.
- Evaluation parameters: temperature 0, the existing `DEFAULT_PROPMT_TEMPLATE`-style
  prompt, answer extraction via the existing `extract_answer()` regex logic,
  batching/retries as in `eval_mcq.py`. Evaluation parameters are frozen together with
  the panel at event open: any parameter change (`max_tokens`, prompt, temperature)
  forces re-execution of **all** cells so every scored cell shares one configuration —
  never re-run a subset under different parameters.

### 6.2 Per-question difficulty score (decision D6)

Each question *q* is evaluated by each panel model *m*, in each language
*l ∈ {regional, english}*, under a **balanced 4-permutation design**: four option
orderings per question, derived from the question UUID, placing the correct answer
**exactly once in each of positions A, B, C, D**. (This replaces "R = 3 random seeds":
with 3 random draws, ~6.25 % of questions would have the correct answer in the *same*
position in every run — measured by Monte Carlo through the repo's `shuffle_options` —
and documented LLM position bias would then flip close races. The balanced design is
Zheng et al. 2023's remedy; cost is +33 % compute over R = 3.)

**Seed/permutation derivation — normative:** all four orderings are derived jointly
from **one** hash, `base = sha256(question_uuid)`: `base` fixes a pseudo-random
ordering of the three distractors and a base offset `o = base mod 4`; run
`r ∈ {0,1,2,3}` places the correct answer at position `(o + r) mod 4`, with the fixed
distractor ordering filling the remaining slots (Zheng et al.'s cyclic-permutation
remedy — balance holds by construction). Do **not** derive each run's permutation
independently (e.g., `sha256(uuid || r)` selecting among all 24 permutations): four
independent draws are balanced only ~9.4 % of the time, silently reintroducing the
position-bias defect. The existing `seed + hash(article_id) % 10000` convention in
`eval_mcq.py` **must not be reused**: CPython salts `str` hashes per process
(PYTHONHASHSEED), so it is not deterministic across processes — verified empirically
on this repo's venv (the same ID produced three different correct letters in three
consecutive interpreter runs). Do not rely on pinning PYTHONHASHSEED. The concrete
permutation used is archived on every `model_answer` row, so any score is replayable
from the archive alone.

```text
received(q) = the (m, l, perm) runs for which a model response was actually received
correct(q)  = # of received runs where the model answered correctly
score(q)    = 1 − correct(q) / |received(q)|          ∈ [0, 1]
```

- **Errored runs are never scored** (divergence from a naive fixed denominator — and a
  deliberate alignment with `eval_mcq.py`, which excludes errors from its accuracy
  denominator): an errored `(m, l, perm)` cell marks the question **pending** and is
  re-queued until it completes; the final authoritative run re-executes every cell that
  ever errored. Otherwise a panel-server outage would count as "the model was fooled"
  and inflate scores by up to ~3 team points, unequally by submission time.
- A **received** response with no extractable letter counts as incorrect (the model
  failed to answer) — same convention as `eval_mcq.py`. The per-question **no-answer
  rate** is recorded; questions exceeding a threshold (default > 30 % of runs) are
  auto-flagged for committee review under a dedicated reason, and the rate is surfaced
  in the v2 export. (An engineered refusal-bait exploit was tested empirically during
  review and did not reproduce — this is measurement hygiene, not an active exploit.)
- Both languages count equally: a culturally hard question should resist the panel in
  translation too (decision D4, §11). Counterweights to translation self-sabotage are
  in §6.3 (re-evaluation after edits, adequacy flag).
- Excluded and withdrawn questions: `score = 0`.

**Canonical worst-case scale** (single source for all capacity and timing math;
O4-resolved provisioning target): 6 countries × 12 teams × 60 questions =
**4,320 questions**; × 4 panel models (O2) × 2 languages × 4 permutations =
**32 runs/question = 138,240 completions**. Per model: 34,560 completions; at the
existing batch size (16 concurrent, `eval_mcq.py`) and ~1 s/completion,
**~36 min/model** — with one vLLM server per model on its own Grid'5000 GPU node a full
run is ~36 min wall-clock, so rolling evaluation (§6.1) keeps up comfortably (each
15-minute cycle covers only questions submitted since the last cycle), and the closing
run covers the final cycle plus errored-cell re-runs (budget ≈ 30–45 min between close
and the country announcements). Provision redundant Grid'5000 nodes given each model is
25 % of the signal; re-anchor the ~1 s constant on the chosen Grid'5000 GPU hardware
once the 4 models are picked (O2) — vLLM's continuous batching on a datacenter GPU will
typically beat it, so treat ~36 min/model as a conservative ceiling.

### 6.3 Committee review

Review queue for the committee (built-in review UI, §10), fed by: duplicate flags,
heuristic flags, no-answer-rate flags, translation-adequacy flags, random-sample audits
(e.g., 10 % of all submissions), and post-hoc checks of the questions that count toward a
team's score. With the top-30 cap removed (D22) **every accepted question counts**, so the
counted set is a team's entire accepted corpus rather than its best 30. During the event,
the queue is **priority-ordered by the provisional leaderboard**: flags on leading teams'
questions first, then those teams' full accepted sets — so whatever contributes to a
contender's score is already validated by close, and the same-day country announcement
rests on a finished review rather than a close-time scramble. The platform provides the
ordering, per-question checklists, and review progress views; how the committee is staffed
is out of scope (removing the cap enlarges the counted-question audit — up to the full
~4,320-question corpus at canonical scale instead of ~2,160 — which the organizers must
staff for; the priority ordering ensures prize-deciding questions clear first even if the
tail runs past close). The full counted-question audit across all 72 teams continues
through Oct 5–20 and protects the regional prize and the dataset. At the country freeze, a
still-unadjudicated flag does **not** suppress the question (presumption of validity);
post-freeze exclusions affect only the regional ranking and the export (decision D7, §7).

Exclusion reasons (from the concept note, plus operational ones):

1. Depends on ephemeral or unverifiable data (rumors, opinions, undocumented events)
2. Obscure trivia without meaningful cultural content
3. **Fails the validity checklist** (D17, revised — applied by any committee reviewer;
   there is no per-country reviewer requirement): (a) answerable by an informed local,
   (b) exactly one defensible correct answer, (c) culturally significant rather than
   obscure trivia, (d) sourced (a Wikidata QID — supplied directly or resolved from a
   Wikipedia article link, §4 — or accepted `cultural_provenance`). A question failing
   any item is excluded under this criterion. This check is mandatory for **every counted
   (accepted) question** and subsumes reasons 2 and 9.
4. Factual errors, harmful stereotypes, or discriminatory content
5. Copyright violation / incompatible with the repo's MIT license
6. Confirmed duplicate (§5)
7. Adversarial formatting / prompt injection
8. Wrong, missing, or **quality-degrading** English translation — including
   meaning-preserving but awkward/degraded translations, since English runs are scored
   (O3/D19). Machine translation is permitted but the team owns its quality; the
   committee may instead fix trivial issues and accept — logged as an edit (which
   forces re-evaluation, below).
9. **Answer not uniquely correct** — a distractor is also true, or several options are
   defensible. This is the highest-yield gaming strategy the score rewards (a
   true-distractor question scores ~0.9–1.0 while measuring nothing), so
   **distractor-falsity verification** (each distractor affirmatively false, sourced
   via `wikidata_qids`/`context_note`) is a mandatory checklist item for **every
   counted-question check** and for the same-day contender review.
10. No-answer rate above threshold with refusal-inducing content (from the §6.2 flag)

Reasons 6–10 extend the note's published five criteria; they must appear in the
participant rules (organizer task). The counted-question checklist also verifies the
regional-language obligation (local-language formulation is mandatory per the note; see
§4 item 3 for why the adequacy flag alone cannot enforce it).

Two hard rules:

- **Any committee edit to a scoring question's text (reason 8 fix-and-accept) forces
  re-evaluation of the affected runs before the relevant freeze** — otherwise a
  score earned on the pre-edit text survives the repair.
- An automated **translation-adequacy flag** (cross-lingual embedding similarity
  between the regional and English versions — the §5 e5 infrastructure) routes
  low-similarity pairs to review, as the counterweight to D4's exposure: half of every
  question's runs use the team's own English text, so a degraded translation inflates
  scores for reasons unrelated to cultural knowledge.

Every decision records reviewer, timestamp, reason, and free-text note (audit trail).
Decisions are revocable until the **regional** freeze; country results are final at
announcement (D7).

---

## 7. Scoring and leaderboards

- **Team score** (decision D6, revised by D22) = sum of `score(q)` over **all** the
  team's accepted questions (no top-N cap; the 60-question lifetime quota of §3.2 is the
  only bound). This operationalizes the note's "nombre de questions qui parviennent à
  tromper le panel" as a normalized sum of fooling events (the note's Annexe grades
  questions by "score individuel"). D6/D22, the D7 finality rule (including the
  presumption of validity for flags unadjudicated at the country freeze), and the full
  D10 tie-break chain must all be mirrored verbatim in the published participant rules
  (organizer task) so no team can appeal to the note's looser phrasing after the fact.
- **Tie-breaking** (decision D10): pad every team's per-question score list with zeros
  to length 60 and compare the descending vectors lexicographically (well-defined for
  every team size; subsumes "deeper bench" intuitions). If still tied: lower cumulative
  submission-sequence sum over the counted (all accepted) set (zero-score counted
  questions included), computed at the freeze. Final backstop, unique by construction:
  the lowest single sequence number among counted questions — each sequence number
  belongs to exactly one team, so this cannot tie (cumulative sums *can*: disjoint sets
  can share a sum).
- **Country leaderboard**: teams of that country. **The country result freezes at the
  close of the same-day contender review (§6.3) and the announced country winner is
  final** (decision D7): later exclusions affect the regional ranking and the dataset
  export, never an announced country prize.
- **Regional leaderboard**: the country winners ranked by team score; winner announced
  at the Assises (Oct 28). The regional ranking is frozen and signed off by the
  committee after all reviews close (Oct 5–20 window). **Cross-country comparability
  caveat (decision D14, recorded):** `score(q)` embeds the panel's prior coverage of
  each culture — on the v1 leaderboard all 11 models score higher on es-la than pt-br
  (mean gap +2.5 accuracy points ≈ 0.025 per-question `score(q)`; over the D22 up-to-60
  counted set that is worth ~1.5–4 team-score points, up from the ~0.75–2 that the
  removed top-30 cap implied — the gap scales with the number of counted questions). The
  organizers
  accept raw comparison as mission-aligned ("free" difficulty flows to the cultures the
  panel knows least — exactly what the datathon exists to surface); the committee
  sanity-checks the regional margin against the known coverage gap during sign-off.
- **Visibility during the event** (decision D5): country and regional leaderboards are
  visible to **all teams**, refreshed on the evaluation cadence (default every 15 min)
  and clearly marked *provisional*. Each team additionally sees its own per-question
  provisional scores (aggregate only, §6.1), supporting the submit-and-refine loop —
  with the cap removed every accepted question adds to the team score, so the incentive
  is to fill the 60-question quota with questions that each fool the panel, not to farm
  a disposable pool and keep only the best 30.
- **Between Oct 3 and Oct 28** (decision D15): at event close the public regional
  *ordering* is hidden; the public page shows only "country winners — regional result
  pending committee review" until the Assises reveal (one visibility flag; preserves
  the staged announcement the note plans around).

---

## 8. Dataset export

- One-click admin export of all **accepted** questions to a Hugging Face dataset
  matching the LatamQA MCQ schema (`question`, `answer`, `distractor1–3`, `*_en`
  columns) so `eval_mcq.py` can consume it unchanged, plus new columns:
  `country`, `region`, `cultural_dimension` (primary + full set), `context_note`,
  `wikidata_qids` (resolved `Q…` values), `wikidata_source_urls` (any Wikipedia links
  they were resolved from, §4), `cultural_provenance`, `team_id`, `event_id`,
  `panel_score`, `no_answer_rate`. The Space pushes directly to the Hub with its
  service token.
- Publication layout: one config per existing LatamQA region, mapping countries *into*
  regions **by rule, not enumeration** (Brazil → `pt-br`; every Spanish-speaking
  participating country → `es-la`, e.g. AR/CL/MX/UY; `es-es` receives no datathon
  rows) — the final country roster is open (O4) and must not break the export.
  The `article_id` column is populated with the question's stable `question_uuid`
  (satisfying "consume it unchanged"); permutations replay from the archived seeds, not
  from re-derivation (§6.2).
- License: MIT, consistent with the LatamQA repo. Each participant individually accepts
  a contribution agreement at first login (click-through; acceptance and consent flags
  stored per participant, including consent to individual citation of the regional
  winning team's members in the LatamQA v2 publication). The agreement *text* is
  organizer-owned and out of scope.
- Raw evaluation artifacts (every model response, with its concrete permutation) are
  archived for reproducibility of the published results.

---

## 9. Non-functional requirements

- **Scale:** the canonical worst case of §6.2 (single source of truth; 72 teams, O4).
  User concurrency ≤ ~500 (72 teams × 6 members + staff); write load trivial
  (< 1 submission/s peak).
- **Availability:** the event is one day and cannot be rescheduled. The Space runs on
  paid, always-on hardware (no sleep) with **persistent storage**; SQLite (WAL mode) on
  the persistent volume is the system of record. Off-Space backup: the service commits
  a DB snapshot to a **private HF dataset repo every 5 minutes** and at every freeze,
  so worst-case data loss (RPO) is one snapshot interval — a recorded simplification
  versus v0.3's WAL-archived Postgres (RPO ≈ 0); on restore, the submission sequence
  counter resumes from the snapshot maximum plus a safety gap so precedence ordering is
  never reused. A documented restore runbook (redeploy Space → mount storage → restore
  latest snapshot) is **rehearsed at M4**. The panel runs on the Grid'5000 cluster
  (§6.1, §10): the GPU reservation must span the event window plus the M2/M4 rehearsals
  with a time buffer, and the vLLM servers must be reachable from the Space (a
  public-facing access path, or an SSH tunnel/VPN through a Grid'5000 access node).
  Grid'5000 is a research testbed, not an SLA-backed production service, so treat the
  reservation and reachability as event-critical dependencies rehearsed at M4; a lapsed
  reservation or unreachable node degrades to mode (a). Degraded modes for four failure
  classes: (a) panel-node loss (a Grid'5000 vLLM server or its reserved node) —
  submissions accepted, evaluation deferred and the model's errored cells re-queued
  (§6.1); (b) Wikidata/Wikipedia unavailability — QID existence checks and
  Wikipedia-link resolution defer, submissions flow (§4); (c) Space restart —
  persistent storage survives, background workers resume from the SQLite job queue (all
  jobs idempotent); (d) Space or storage loss — redeploy and restore the latest Hub
  snapshot per runbook.
- **Latency:** submission validation < 2 s (local checks only); provisional
  per-question scores within one leaderboard cycle (≤ 15 min).
- **i18n:** participant UI in Spanish and Portuguese; the review and admin surfaces are
  in-house and may be English/Spanish only.
- **Accessibility:** participant-facing UIs conform to **WCAG 2.1 AA**. Concretely in
  M1's definition of done: complete author→submit flow keyboard-only; programmatic
  labels on all form fields; `lang` attributes distinguishing es/pt fields from the
  mandatory English fields (WCAG 3.1.2 — without them screen readers voice English
  answers with a Spanish/Portuguese synthesizer); validation errors as ARIA status
  messages; leaderboard information not encoded by color alone. M4 includes an
  automated axe scan plus one assistive-technology walkthrough.
- **Time:** one global event window, **13:30–23:30 UTC** on Oct 3, 2026 (decision D13;
  same absolute instant everywhere — keeps first-submitted-wins precedence and the
  shared leaderboard fair; Mexico starts three hours earlier in local time). All
  timestamps stored UTC; the submission **sequence number** (§3.1), not wall-clock, is
  authoritative for ordering. A submission is in-window iff its sequence number is
  assigned before the close transaction.
- **Open source:** the platform itself is released under MIT in the spirit of the
  event. Note: the panel composition (D9) is withheld until after the event; the
  models themselves are open-weight, and the 4 models are named post-event.
- **Privacy (software surface only):** minimal PII (name, email, affiliation, country;
  HF username from OAuth). PII is segregated from competition records in the data model
  (§10) so erasure requests can remove participant PII while competition records
  (questions, scores, decisions) are retained. The governing legal framework, notices,
  and agreement text are organizer/DPO-owned and out of scope.
- **Auditability:** append-only event log for submissions, state changes, review
  decisions, restores, and configuration changes. **Every state-changing operator
  action passes through the datathon service** (admin console or authenticated admin
  API) so it lands in this log — no raw SQL or hand-edited config on event day.
- **Monitoring & alerting:** operators are alerted during the event when a team's
  withdrawal count or rate exceeds a defined threshold (§3.1 mass-withdrawal
  protection), when the evaluation backlog exceeds one cadence cycle, and when an
  external dependency (Wikidata/Wikipedia, a Grid'5000 panel node) degrades.

---

## 10. Architecture

**Decision (D21, supersedes D11): everything runs in one Hugging Face Docker Space.**
(A Gradio-SDK/ZeroGPU deployment was evaluated and rejected — the Docker Space's full
process control is needed for the workflow surfaces, WCAG-AA UI, admin API, and always-on
background workers; see D25.) Argilla is removed entirely — v0.3 had already demoted it to
committee-review-only
(maintenance mode, no Portuguese UI) with a pre-built in-house fallback page; on Space
hosting its mandatory sidecars (Elasticsearch, Redis) are dead weight, so the former
fallback review page is promoted to the primary committee UI. This deletes the one
third-party integration (SDK push, webhooks, reconciliation job) from the design.

```text
                     ┌─────────────────────────────────────────────┐
┌───────────────┐    │  Hugging Face Space (Docker, always-on)     │
│ Participant   │───▶│  Datathon service (FastAPI)                 │
│ UI (es/pt)    │    │  · MCQ submission form + i18n               │
└───────────────┘    │  · validation, quotas, rate limits,         │
┌───────────────┐    │    sequence numbers                         │
│ Public pages  │───▶│  · committee review UI (checklists,         │
│ + organizer   │    │    exclusion workflow)                      │
│ counts        │    │  · admin console + admin API                │
└───────────────┘    │  · background workers (in-process,          │
┌───────────────┐    │    SQLite-queued, idempotent):              │
│ Committee /   │───▶│    duplicate + QID checks, panel evaluator  │
│ Admin UI      │    │    (imports latamqa/LiteLLM), leaderboard   │
└───────────────┘    │    cadence, backup sync                     │
       ▲             │  · SQLite (WAL) on persistent storage       │
       │             └───────┬────────────────────┬────────────────┘
   HF OAuth                  │                    │
 (all roles)                 ▼                    ▼
              ┌───────────────────────┐  ┌───────────────────────────┐
              │ Grid'5000 cluster     │  │ HF Hub                    │
              │ · 4 panel models on   │  │ · private repo: DB        │
              │   vLLM (OpenAI-       │  │   snapshots + eval        │
              │   compatible), via    │  │   archives (5-min sync)   │
              │   LiteLLM             │  │ · public repo: LatamQA v2 │
              │ · (opt.) e5 embedding │  │   dataset export          │
              │   GPU offload         │  └───────────────────────────┘
              └───────────────────────┘
        (e5 embeddings run in-process on the Space CPU by default; §10)
```

### Key architecture points

- **One service owns everything:** authoring, review, leaderboards, admin — no
  cross-system mirroring or reconciliation. `review_decision` rows are written
  natively by the review UI.
- **Storage:** SQLite in WAL mode on the Space's persistent volume. At this write load
  (< 1 submission/s) SQLite is comfortable; embeddings live in an `embedding` table and
  are compared brute-force in memory (§5) — no Postgres, no pgvector.
- **Background work:** Spaces provide no cron and may restart; all periodic work
  (rolling evaluation, duplicate/QID workers, 15-min cadence, Hub backup sync) runs on
  an in-process scheduler with its queue in SQLite, so a restart resumes cleanly
  (jobs idempotent; §9 degraded mode c).
- **Auth:** HF OAuth for all roles (§2); Space secrets hold the service's Hub token and
  inference API keys.
- **Panel serving:** the 4 panel models run on **Grid'5000 GPU nodes under vLLM** (one
  vLLM server per model, plus reserved spare nodes for redundancy), reached through
  LiteLLM as OpenAI-compatible endpoints — the existing YAML config's `LLM URI` field
  carries each vLLM server URL, so the `latamqa` config format is reused unchanged. The
  Grid'5000 GPU reservation must cover the event window and the M2/M4 rehearsals (with a
  buffer), and the servers must be reachable from the Space (public-facing access path,
  or an SSH tunnel/VPN through a Grid'5000 access node); this is an event-critical
  dependency (§9). Embedding inference runs in-process on the Space CPU by default (the
  write load is tiny), with optional offload to a GPU (e.g., a Grid'5000 node).
- **Event-window semantics:** window expiry is authoritative for close; admins may set
  or extend the window only *before* expiry, with a confirmation step; re-opening a
  closed event requires a logged two-person admin action.
- **Reuse:** refactor `eval_mcq` (`evaluate_mcq`, `extract_answer`, prompt building)
  into a programmatic API that accepts a list of question dicts instead of an HF
  dataset name. `shuffle_options` is replaced for the datathon by the §6.2 normative
  permutation derivation (the `hash()`-based convention is process-salted and must not
  be reused).
- New code lives in this repo (e.g., a `datathon/` package) or a sibling repo
  `latamqa-datathon` depending on `latamqa`; the Space builds from that repo.

### Data model

`event` (window, config, frozen panel) · `country` · `team` (country, name, quota) ·
`participant` (team, HF identity, agreement/consent flags; PII segregated) · `question`
(team, all §3 fields ×2 languages incl. `wikidata_qids`/`wikidata_source_urls`, status,
**sequence_number**, uuid) · `embedding`
(question, lang, vector) · `duplicate_flag` (question, matched_question | matched_v1_id,
similarity, decision, survivor link — drives the §5 orphan rule) · `evaluation_run`
(panel, started/finished, config snapshot) · `model_answer` (run, question, model, lang,
**permutation**, raw_response, extracted, correct | errored→requeued) · `review_decision`
(reviewer, action, reason, note, at) · `score_snapshot` (team, run, total_score,
country_freeze | regional_freeze) · `job` (background work queue) · `audit_log`
(append-only, all actors incl. admin console/API)

---

## 11. Decisions

**New or revised at v0.5 (this update):**

- **D22 — Top-30 scoring cap removed:** a team's score is the sum of `score(q)` over
  **all** its accepted questions, not just its best 30 (§3.2, §7). The 60-question
  lifetime quota (D8) remains the only bound on how many questions can count; the D10
  tie-break (zero-padded to 60) is unchanged. Consequence: the committee's
  counted-question audit widens from a team's top 30 to its entire accepted set — up to
  the full ~4,320-question corpus at canonical scale instead of ~2,160 (§6.3), which the
  organizers must staff; the leaderboard-priority ordering clears prize-deciding
  questions first.
- **D23 — Panel hosting on Grid'5000 with vLLM (supersedes the HF-Inference-Endpoints
  portion of D21):** the 4 panel models are served on Grid'5000 GPU nodes under vLLM
  (OpenAI-compatible) and reached through LiteLLM; the YAML `LLM URI` carries each vLLM
  server URL, so the `latamqa` config format is reused unchanged (§6.1, §10). The GPU
  reservation must span the event window and the M2/M4 rehearsals, and the servers must
  be reachable from the Space — an event-critical dependency (§9). Redundancy is
  reserved spare nodes rather than managed autoscaling; the ~1 s/completion timing
  constant is re-anchored on the chosen Grid'5000 GPU (§6.2). Everything else in D21
  (single HF Space, SQLite/WAL as system of record, 5-min Hub snapshots) is unchanged —
  only panel inference moves off HF.
- **D24 — Wikidata reference via Wikipedia link:** a `wikidata_qids` entry may be a raw
  `Q…` QID, a Wikipedia article URL, or a canonical Wikidata entity URL. A Wikipedia URL
  is resolved to the article's linked Wikidata item (`wikibase_item`) via a batched
  MediaWiki `pageprops` lookup (reading the `disambiguation` marker and following
  redirects so those cases soft-flag rather than resolve silently); a Wikidata entity URL
  is stripped to its QID. The resolved `Q…` is stored as the canonical value in
  `wikidata_qids`, with the source URL kept in `wikidata_source_urls` for
  provenance/export (§3, §4, §8). Resolution shares the async, fail-open path of QID
  existence verification; an unresolvable link soft-flags for committee review.
- **D25 — Deployment shell: Docker Space retained; Gradio-SDK/ZeroGPU evaluated and
  rejected.** A Gradio SDK Space on HF **ZeroGPU** was considered and rejected on the
  merits. ZeroGPU is **Gradio-SDK-only**, PRO-gated, caps every GPU call at **120 s**,
  meters a daily quota, and is built for bursty **sleep-on-idle** inference — the wrong
  host for a single-day, unrescheduleable **system of record** running always-on
  background workers (rolling evaluation, 15-min cadence, 5-min backups; §9, §10).
  Decisively, the heavy GPU workload — the 4-model panel — already runs **off-Space on
  Grid'5000 (D23)**, so the Space has no GPU need that ZeroGPU would serve (e5 embeddings
  are CPU-cheap, §5/§10), and ZeroGPU could not host the panel batch (138,240 completions,
  §6.2) under the 120 s cap and quota in any case. The platform's hard parts — the
  multi-surface committee-review/admin **workflow**, the **WCAG 2.1 AA** participant UI
  (§9), the authenticated admin API, and robust always-on background workers — need the
  **full process control** of the Docker FastAPI Space (D21), which is therefore retained.
  HF OAuth (D1) is a **Space-level** feature (`hf_oauth: true` in the Space config), not
  Gradio-specific, so the auth model does not depend on the SDK. If escaping Grid'5000's
  reservation/reachability friction (§9, §10) later becomes the priority, the alternative
  to weigh is a **paid HF GPU Space or HF Inference Endpoints** (the pre-D23 D21 form),
  **not** ZeroGPU.

**Superseded or revised at v0.4 (this simplification):**

- **D1 — Participant authentication (revised):** **Hugging Face OAuth** for all roles
  (participants, reviewers, admins), replacing per-participant email magic links —
  native to Spaces, no SMTP dependency, still one identity per person for the audit
  trail and per-member citation (§8). Participants need a (free) HF account.
- **D11 → D21 — Argilla removed (supersedes committee-review-only scoping):** the
  in-house review page — pre-built as a hot fallback in v0.3 — is now the primary and
  only committee UI; Argilla, Elasticsearch, and Redis leave the stack (§10).
- **D17/D18 (O1/O5) — in-country reviewers removed:** the validity checklist survives
  as §6.3 criterion 3, applied by any committee reviewer; there is no per-country
  reviewer requirement, and the progressive pre-review becomes a leaderboard-priority
  ordering of the shared review queue. Committee staffing is out of scope.
- **D21 — Hosting (new):** one HF Docker Space, always-on hardware, persistent storage,
  SQLite/WAL as system of record, 5-minute DB snapshots to a private Hub repo
  (RPO = snapshot interval, a recorded simplification vs. v0.3's RPO ≈ 0 Postgres),
  panel models served via LiteLLM (hosting superseded by D23: Grid'5000 + vLLM
  replacing HF Inference Endpoints) (§9, §10).
- **Dropped from this document as non-software:** D12 (schedule/staffing posture),
  D16 (participant-rules publication package), D20/O6 (data-controller and legal
  items), O5 staffing math. The software-facing residue remains: rules that must be
  published verbatim are marked where they are defined (§3.2, §6.3, §7).

**Standing decisions (unchanged from v0.3):**

- **D2 — Embedding model:** `multilingual-e5-large` for duplicate detection; the 0.92
  threshold is re-calibrated on LatamQA v1 (thresholds are not transferable between
  models); any calibrated departure from the note's published 0.92 is a recorded
  deviation for the participant rules.
- **D3 — Feedback policy:** provisional leaderboard on a fixed cadence (default
  15 min); teams see their own aggregate per-question scores; per-model results and
  panel composition hidden until after the event (§6.1, §7).
- **D4 — Language weighting:** both languages count equally (§6.2), with the §6.3
  translation-adequacy counterweights.
- **D5 — Leaderboard visibility:** provisional leaderboards visible to all teams
  during the event (§7); post-event visibility governed by D15.
- **D6 — Score aggregation (revised by D22):** sum of continuous `score(q)` over **all**
  accepted questions — the top-30 cap is removed (§7).
- **D7 — Country announcement finality:** country winners final when announced Oct 3
  after the same-day contender review; post-event exclusions affect only the regional
  ranking and the export (§6.3, §7).
- **D8 — Quota rule:** 60 lifetime submissions; withdrawal refunds quota only before
  the question enters any evaluation cycle (§3.2).
- **D9 — Panel composition & disclosure (O2):** 4 small-to-medium open-weight models,
  fully undisclosed until after the event (§6.1). *Still to pick before M2:* the
  specific 4 models (diverse families; validate foolability against LatamQA v1) and
  Grid'5000 GPU-node sizing per model (D23); re-anchor the §6.2 timing constant on the
  chosen Grid'5000 hardware.
- **D10 — Tie-break:** zero-padded descending score-vector lexicographic comparison;
  then cumulative sequence sum; backstop lowest single sequence number (§7).
- **D13 — Event window:** same UTC instant for all countries, 13:30–23:30 UTC (§9).
- **D14 — Regional comparability:** raw cross-country comparison accepted as
  mission-aligned; caveat recorded; committee sanity-check at sign-off (§7).
- **D15 — Post-event visibility:** regional ordering hidden from event close until the
  Assises reveal; only country winners shown as final (§7).
- **D19 (O3) — Translation policy:** machine translation permitted, team owns quality;
  committee excludes meaning-changing or quality-degrading translations (§6.3
  reason 8).
- **O4 — Provisioning target:** sized for 6 countries × 12 teams = 72 teams
  (4,320 questions). If the final roster exceeds this, re-derive the §6.2 scale and
  Grid'5000 node sizing (D23).

---

## 12. Delivery milestones (working backward from Oct 3, 2026)

Scope-cut order if capacity falls short (cut scope, never the test schedule): static
regenerated leaderboard page; drop region autocomplete; admin console degrades to the
authenticated admin API only.

| Milestone | Target | Content |
| --- | --- | --- |
| M0 — Space bootstrap | early Aug 2026 | Docker Space with HF OAuth, persistent storage, SQLite; round-trip a question end-to-end: form → validation → one vLLM evaluation on a Grid'5000 GPU node (proving reachability from the Space) via the `latamqa` programmatic API → score displayed. Prove the backup/restore cycle (snapshot to private Hub repo, restore to a fresh Space) |
| M1 — Core authoring | mid-Aug 2026 | Submission form (es/pt, WCAG 2.1 AA definition-of-done per §9), question lifecycle + sequence numbers + validation + D8 quota, async QID verification, admin console skeleton, contribution-agreement click-through. Acceptance: length limits calibrated on v1; synthetic team roster provisioned |
| M2 — Evaluation | early Sep 2026 | Duplicate detection + orphan rule, panel evaluator (balanced permutations, errored-cell re-queue, archived seeds), rolling evaluation + provisional scores; **first load test** at canonical scale against the Grid'5000/vLLM panel (reservation secured for the window) |
| M3 — Review & leaderboards | ~Sep 11 2026 | Committee review UI (criteria 1–10 checklists), exclusion workflow, forced re-eval on edits, translation-adequacy flag, leaderboards + D15 visibility flag, dataset export |
| M4 — Full rehearsal | ~Sep 18 2026 | Full-day rehearsal with synthetic teams incl. same-day contender review drill and Space restore drill; Grid'5000 reservation + reachability drill; load test at 2× canonical scale against the Grid'5000/vLLM panel; axe scan + assistive-tech walkthrough; ops runbook finalized |
| Smoke test | Sep 28–29 2026 | Light end-to-end pass with the **real** Sep-21 team roster; feature freeze from Sep 25 (fix-only) |
| Event | Oct 3, 2026 | Window 13:30–23:30 UTC; panel frozen at open; on-call rotation; same-day contender review; country announcements (final, D7) |
| Post-event | Oct 5–20, 2026 | Full committee review, regional ranking freeze + sign-off (incl. D14 sanity check), dataset export, panel-composition disclosure, for the Assises announcement (Oct 28) |
