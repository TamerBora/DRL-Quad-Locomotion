# Claude Code Prompt — Single-Agent Guided SAC for Go2-W

You are working in this repository. Your task is to design and implement a **single-agent Guided Soft Actor-Critic** algorithm for **Unitree Go2-W (wheeled quadruped) locomotion on rough terrain**, built on top of my existing FlashSAC implementation.

## Read these first, in this order

1. **`guided_sac_yol_haritasi.md`** — my roadmap and critical analysis of Guided SAC. This is the *governing document*. It defines the working philosophy, the design decisions that must be made (D1–D9), the known failure modes, and the correctness tests. Do not contradict it without telling me why.
2. **`tok.md`** — my most recent work: SAC results on this platform and how FlashSAC is implemented. This tells you what the code base actually looks like and what already works.
3. **`Multi_Agent_Guided_...md`** — Çerçi & Temeltaş (2024), the source paper. The guided actor/control actor logic, the loss functions, and the training loop structure come from here. **Read the `Transcription Notes` section at the end** — it lists errata in both the transcription and the original paper. Use the corrected forms.
4. **`ozet_kodlar_v2/`** — Çerçi's multi-agent implementation. Treat this as a **reference for the guided logic only**, not as code to port. I am building a *single-agent* system. Extract the mechanism, discard the multi-agent scaffolding (centralized critic over joint actions, per-agent adversarial networks, agent indexing).

## Scope

**In scope:** single-agent Guided SAC, Go2-W, rough terrain, IsaacLab + FlashSAC.

**Explicitly out of scope for now** — do not implement these, do not design around them:
- Multi-agent decomposition (per-leg agents)
- Adversarial / state-perturbation networks (Çerçi's `B_δ`)
- The Robust Adaptation Module
- Hyperparameter tuning of any kind

The last point matters. The roadmap is explicit that architecture comes first and tuning comes later. If you find yourself wanting to change a learning rate to fix something, stop and ask whether the architecture is wrong instead.

## Key adaptation you must handle

Çerçi's setup and mine are not the same problem, and the mapping is the core intellectual work here:

| | Çerçi (2024) | This project |
|---|---|---|
| Why the control actor is handicapped | Adversarial perturbation corrupts the state | Genuine partial observability — no height scan, no base linear velocity on the real robot |
| Guiding actor sees | Clean (unperturbed) state | Privileged state: height scan, base linear velocity, friction, contact forces, DR parameters |
| Control actor sees | Perturbed state | Proprioception only, with a history window |

So `s_t^p` in the paper maps to my proprioceptive observation `o_t`, and `s_t` maps to my privileged observation `s_t`. This is **not** a cosmetic renaming: in the paper the two actors see corrupted-vs-clean versions of the *same* vector, whereas here they see *different* vectors of different dimensionality. Anywhere the paper's math assumes identical shapes, flag it and resolve it explicitly.

## Deliverable 1 — Design note (do this first, do not write algorithm code yet)

Produce `docs/guided_sac_design.md` containing:

1. **Answers to D1–D9** (table below), each with a written justification of one short paragraph. For each decision also state the *integration cost* in FlashSAC (what has to change in the replay buffer, network factory, rollout loop, update step) and whether that cost changes your recommendation.
2. **The exact loss functions you will implement**, written out, with every symbol defined, and each one traced back to its equation number in the paper — noting where you deviate and why. At minimum: critic loss, control actor loss, guiding actor loss, latent encoder loss if you use one, entropy temperature update(s).
3. **Observation contract**: the precise contents and dimensionality of `o_t` and `s_t`, and which network consumes which.
4. **A data-flow diagram** (ASCII or mermaid) showing what flows where during training, and separately what runs at deployment.
5. **Open questions and risks** — anything you are unsure about, especially where the paper is ambiguous or where FlashSAC's structure fights the design.

**Stop after the design note and wait for my review.** Do not proceed to implementation until I approve it.

### The D1–D9 design decisions

These are reproduced from the roadmap (section "Faz 1.1"), adapted to the single-agent scope. Every one of them is an architectural choice that cannot be fixed later by tuning. Take a position on each — do not default, and do not answer "either works".

| # | Decision | Options | Why it matters |
|---|---|---|---|
| **D1** | Guiding actor network | Fully separate network / shared trunk with the control actor, separate heads / shared weights with different input adapters | A separate network is the paper's choice and is cleanest, but doubles actor parameters and forward cost. A shared trunk is cheaper but risks privileged information bleeding into shared weights — which would fail the information leakage test. |
| **D2** | Control actor's memory | Frame stack (window K) / GRU / TCN encoder / none | Solving a POMDP requires memory; a memoryless policy cannot estimate base velocity from proprioception. Your choice is constrained by whether FlashSAC's replay buffer can sample sequences — check this before deciding. |
| **D3** | How privileged information reaches the actor | Through the critic only / latent `z_hat` with encoder supervision / action-space imitation / a combination | The central question. The roadmap argues action-space imitation suffers from the imitation gap: a teacher that sees the height scan never learns foot probing or cautious stepping, so imitating it suppresses exactly the behaviours that keep a blind policy alive. Latent supervision avoids this by supervising *representation* instead of *action*. Roadmap's default recommendation: latent + soft imitation. |
| **D4** | Value network | Separate `V(s)` as in Çerçi / twin-Q with targets only, as in modern SAC | The 2024 paper keeps a `V` network; SAC dropped it after the 2018 revision. Keeping it needs a written justification, not inheritance by habit. |
| **D5** | Behavior policy mixing | Fixed alternation as in the paper / probabilistic mixture with weight `p(π_g)` / mixture annealed to zero | Eq. (11)'s strict alternation is arbitrary. Also decide whether the guiding actor acts in the environment at all or only supervises — if it only supervises, mechanism M2 disappears and you should say so explicitly. |
| **D6** | Imitation metric | L2 (`‖a − a_g‖²`) / KL divergence | KL is closed-form for Gaussian policies and matches variance as well as mean. L2 pulls only the mean, forcing the student into false confidence where the teacher is uncertain. |
| **D7** | Which actions the critic sees | Control actor's / guiding actor's / both with separate targets | Çerçi Eq. (13) mixes control action for the agent under evaluation with guiding actions for the others. In single-agent there are no "others", so this construction collapses and must be redefined. This is a mandatory resolution, not a preference. |
| **D8** | Entropy temperature | Shared `α` for both actors / separate `α_g` and `α_f` | The guiding actor solves an easier problem and will want a different exploration level; forcing one target may distort either the teacher or the student. |
| **D9** | DR parameters in the critic | Yes, `Q(s, ξ_DR, a)` / no | Direct remedy for the multimodal-Q pathology described in the roadmap: without them, the critic averages value functions across different randomized MDPs, which is a plausible cause of the crouch collapse I saw in rough-terrain SAC. Architectural, not a tuning knob. |

Related question, not numbered but required: **fixed `λ` or a schedule?** Before the teacher converges, `a_g` is noise, so a constant imitation weight injects noise early in training. See roadmap Faz 2.1.

## Deliverable 2 — Implementation (only after approval)

Follow the design note. Build the guided layer as an **extension** to FlashSAC, not a fork: existing vanilla SAC runs must remain reproducible and unbroken. Keep the guided path behind a config flag.

Implement the correctness tests from the roadmap (Faz 1.3) as actual tests, not manual checks:

- **Information leakage test** — replace `s_priv` with random noise; deployed control actor performance must not change. This is the most important test in the project; a failure here means privileged information is leaking into the deployed policy and the whole approach is invalid.
- **Guiding actor superiority test** — `π_g` must outperform `π_f` by a meaningful margin, otherwise the teacher is not learning and the imitation term is injecting noise.
- **Deploy isolation test** — strip the guiding actor and every privileged input, run the policy, confirm identical behaviour to training-time evaluation.
- **Buffer schema test** — assert `a_g` and `s_priv` are aligned with the correct transition. An off-by-one here fails silently and corrupts everything.

## Ground rules

- **Ask before assuming.** If FlashSAC's structure is ambiguous or the paper's math does not transfer cleanly to the single-agent case, ask me. Do not silently pick an interpretation.
- **Deployability is a design constraint, not a later problem.** The deployed network must be exactly a vanilla SAC actor consuming `o_t`. Nothing privileged may survive into deployment.
- **Cite your sources in code comments.** When a loss or update rule comes from the paper, reference the equation number.
- **No speculative complexity.** If a component is not justified in the design note, it does not go in the code.
- **Report honestly.** If something in my roadmap turns out to be wrong or impractical once you see the code, say so directly.

Start by reading the four sources, then ask me any clarifying questions you need before writing the design note.
