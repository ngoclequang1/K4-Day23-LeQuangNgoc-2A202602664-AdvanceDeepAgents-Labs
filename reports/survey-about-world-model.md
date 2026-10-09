# World Models: A Rigorous Survey of Foundations, Methods, Evidence, and Open Problems

## TL;DR
- A world model is best understood as a learned predictive state-and-dynamics system: it compresses observations, predicts action-conditioned futures, and supports control, planning, or simulation. The term spans decision-centric model-based reinforcement learning (MBRL) and broader interactive simulators. [1][2][3]
- Latent-space models such as PlaNet, Dreamer, DreamerV3, and TD-MPC2 make imagination or planning tractable, but their strongest evidence is still concentrated in simulated control benchmarks rather than open-ended physical understanding. [4][5][6][7]
- Video-generation world models expand scale and modality, learning interactive environments from gameplay or multimodal video; however, visual plausibility alone does not establish causal, physically faithful, or long-horizon simulation. [8][9][10][11]
- The central evaluation gap is functional: conventional video metrics correlate poorly with physics adherence, while newer benchmarks add instruction following, physical violations, consistency, dynamics, and multi-scene tests. [12][13][14]
- The most important open problems are compounding rollout error, persistent memory, uncertainty and OOD generalization, real-time inference, sim-to-real transfer, and closed-loop safety evaluation. [15][16][17][18][19][20]

## Background

### What “world model” means
The phrase does not identify one architecture. In the narrowest and most operational sense, a world model is an internal predictive model used by an agent: a visual encoder maps observations to a compact state, an action-conditioned transition or memory model predicts future states, and a controller or policy uses those predictions. The original paper supports training a controller in an imagined, model-generated environment. [1] The associated project description makes the decomposition explicit as vision, memory, and control, with the learned model serving as a generated training environment. [21]

The MBRL literature supplies the decision-theoretic context: learn environment dynamics, then integrate planning or policy/value learning to optimize an MDP. [2] PlaNet makes the connection concrete by learning dynamics from images and planning in latent space, while Dreamer treats imagined latent trajectories as the substrate for policy and value learning. [4][5] Thus, representation learning is not merely a preprocessing step: its latent state must preserve the information needed for multi-step reward prediction, uncertainty handling, and action selection. [4][2]

A broader usage treats a world model as an interactive simulator or predictive model for counterfactual reasoning, scene generation, robotics, driving, or agent evaluation, even when it is not directly optimizing a reward. Recent surveys explicitly distinguish this simulator-oriented meaning from the narrower MBRL meaning. [3] This distinction prevents a common category error: a model that generates convincing video may be a useful visual predictor without yet being a reliable action-grounded world model.

## Core design pattern: representation, dynamics, and exploitation

### Predictive state is the bottleneck
High-dimensional observations make pixel-space prediction expensive and often unnecessary for control. PlaNet uses deterministic and stochastic latent transition components plus multi-step latent overshooting; the motivation is that planning needs multi-step outcomes and possible futures, not only single-frame reconstruction. [4] Dreamer similarly uses compact latent dynamics so many imagined trajectories can be propagated in parallel, and exploits them through value gradients rather than requiring online search at every step. [5]

This design creates a three-way trade-off. More compression improves speed but can discard task-relevant details; richer decoders or token spaces preserve visual information but increase compute; and stochastic dynamics can represent uncertainty but make long-horizon optimization harder. The practical criterion is therefore predictive usefulness under intervention, not reconstruction quality alone. The foundational system already exposed this issue: model-generated environments can train controllers, but imperfections can be exploited or otherwise distort learning. [1][21]

### Different ways to use the same learned dynamics
World models can support at least three exploitation modes. First, model-predictive control repeatedly plans short latent trajectories and executes the first action; TD-MPC2 is a modern decoder-free example of local trajectory optimization in a learned latent space. [7] Second, imagination-based actor-critic methods learn a policy and value function on model-generated trajectories; DreamerV3 combines a learned world model with an imagined-trajectory actor and critic. [6] Third, differentiable simulators can support first-order policy optimization, as in PWM, which separates broad world-model pretraining from relatively lightweight per-task policy learning. [22]

These modes are complementary rather than a simple leaderboard hierarchy. MPC can react online and exploit local search but pays inference cost; policy learning amortizes decisions but inherits model bias through imagined data; and differentiable optimization can reuse gradients but depends on useful, stable derivatives and may scale less well in high-dimensional action spaces. PWM’s reported comparisons illustrate that the choice of exploitation mechanism changes both the data regime and the relevant evaluation: prediction error alone is not enough to determine policy quality. [22]

## What the control evidence actually shows

The strongest comparative evidence is in controlled, mostly simulated environments. DreamerV3 reports one configuration across more than 150 diverse tasks, with results that include Minecraft diamond collection from pixels without human data or a curriculum. [6] TD-MPC2 provides a contrasting multi-task result over 104 continuous-control tasks and reports one 317M-parameter agent across 80 tasks. [7]

These results establish that learned latent dynamics can support broad task coverage, heterogeneous action spaces, and substantial data reuse. They do not establish general physical understanding. The benchmark evidence is therefore best interpreted as progress in sample-efficient decision-making under specified environments, not proof that a model has a generally valid simulator of the world. [7]

The evidence also favors evaluating the closed loop. A model can have attractive one-step predictions yet drift under repeated intervention, while a slightly imperfect model may still yield a strong controller if its errors are benign in the task-relevant region. Conversely, a visually plausible rollout can systematically mislead planning. The MBRL survey’s emphasis on alignment between predicted and encoded states captures this concern for latent planning. [2]

## From latent control to generative interactive environments

### Video tokens, latent actions, and counterfactual rollouts
The generative branch treats video prediction itself as the central interface. Genie learns tokenized video dynamics and latent actions from unlabeled Internet video rather than ground-truth action labels, and evaluates both visual fidelity and controllability. [8][11] This expands the data regime and makes interactive generation possible, but it should not be conflated with physical robotics competence.

Genie 2 extends the foundation-video direction with an action-conditioned interactive model that generates frames autoregressively from an initial image, latent history, and keyboard or mouse actions. The project description reports counterfactual trajectories from the same starting frame and consistency for up to a minute, while noting that most examples last 10–20 seconds. [9] That qualification matters: short interactive demonstrations are evidence of controllability and visual continuation, not yet evidence of reliable long-horizon causal simulation.

### Embodiment and multimodal conditioning
The same interface is being explored for driving, navigation, and robotics. GAIA-1 combines video, text, and actions for controllable driving scenarios but identifies non-real-time autoregressive generation and limited camera coverage as constraints. [15] UniSim combines heterogeneous action modalities and reports policies trained in simulation that transfer to real-world settings, while explicitly framing “universal” as a unified interface rather than simulation of everything. [16]

Together, these studies support a trend toward action-conditioned, multimodal simulators, while showing that transfer depends on coverage, temporal dynamics, latency, and the fidelity of the action interface—not parameter count alone.

## Evaluation: from visual quality to functional validity

Generic video metrics are inadequate as a complete test of a world model. The Sora technical report is a useful cautionary case: it reports high-fidelity videos and durations up to one minute while also discussing failures in physical interactions and long-duration coherence. [10] In other words, impressive appearance and duration can coexist with incorrect mechanics.

WorldModelBench makes this gap measurable. It uses 67K human labels and adds instruction-following, commonsense/future-state judgments, and explicit physical-adherence categories rather than relying only on generic video quality. [12][14] Its automated judge is useful as a scalable proxy, but the benchmark’s discussion of judge optimization is also a warning that a metric can become an optimization target rather than a transparent measure of capability. [12]

WorldScore broadens the evaluation target beyond video-only physics. It evaluates 3D and 4D or video world generation with controllability, quality, consistency, dynamics, camera control, and multi-scene or long-sequence behavior, and separates static from dynamic scores. [13] Taken together, these benchmarks suggest a layered evaluation protocol: perceptual fidelity, spatial and temporal consistency, intervention and instruction response, physical plausibility, uncertainty calibration, and ultimately closed-loop task utility. No single score should stand in for all of them.

## Trends and open problems

### Long-horizon memory and compounding error
Autoregressive rollout error compounds: a small state or appearance error changes later conditioning and can derail the trajectory. Diffusion world-model results in Atari show a related quality–speed trade-off: fast sampling can be unstable under few denoising steps, while more stable diffusion increases computational cost; the authors also caution that imperfect simulators can produce unexpected agent behavior. [17]

Recent work attacks the problem with explicit memory rather than simply extending the context window. Long-context state-space models combine efficient sequence processing with local attention and report better long-range memory on synthetic tasks, but remain limited in resolution, interactive frame rate, and extrapolation beyond training context. [19] Retrieval/global-state conditioning in VRAG reports better spatiotemporal coherence than naïve context extension or a history buffer, while still facing compute limits for longer sequences and larger models. [20] A robust solution must preserve task-relevant state, not merely more pixels.

### Uncertainty, OOD generalization, and safety
World models are often used precisely where the real environment is expensive or dangerous, so uncertainty is central. Yet the surveyed evidence is stronger for predictive or benchmark performance than for calibrated uncertainty and safety guarantees. Failures under long rollouts show why an agent should distinguish confident predictions from plausible-looking hallucinations. [17]

Safety should therefore be evaluated at the policy–model boundary. A physically invalid rollout can cause unsafe action selection even if its individual frames score well; conversely, a conservative model may be useful if it exposes uncertainty and prevents risky extrapolation. The current sources document this risk and the evaluation gap more clearly than they demonstrate solved mitigation. [12][10][17]

### Scaling versus grounding
Scaling parameters, data, and context is a consistent trend: DreamerV3 reports gains from 12M to 400M parameters, TD-MPC2 reports benefits from designed multi-task scaling, Genie studies scaling of its dynamics model, and GAIA-1 targets multi-modal scale. [6][7][8][15] But scaling is not equivalent to grounding. Training distribution, action coverage, camera geometry, temporal resolution, and the availability of intervention data determine whether a larger model learns controllable mechanisms or only more convincing correlations.

### A practical research agenda
A rigorous next generation of world models should combine: (1) explicit action and intervention coverage; (2) uncertainty-aware latent or token dynamics; (3) memory mechanisms tested beyond the training horizon; (4) evaluation on counterfactual, physical, and closed-loop tasks; (5) real-time inference and efficient planning; and (6) deployment tests that measure failure severity, not only average reward or visual quality. These priorities follow from the consistent boundary in current evidence: world models are increasingly capable predictive interfaces, but their reliability as causal simulators remains an open empirical question.

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (n.d.)
[2] Model-based Reinforcement Learning: A Survey. arxiv. https://arxiv.org/abs/2006.16712 (n.d.)
[3] Understanding World or Predicting Future? A Comprehensive Survey of World Models. arxiv. https://arxiv.org/pdf/2411.14499v3.pdf (n.d.)
[4] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (n.d.)
[5] Dream to Control: Learning Behaviors by Latent Imagination. arxiv. https://arxiv.org/abs/1912.01603 (n.d.)
[6] Mastering Diverse Domains through World Models. arxiv. https://arxiv.org/html/2301.04104 (n.d.)
[7] TD-MPC2: Scalable, Robust World Models for Continuous Control. arxiv. https://arxiv.org/html/2310.16828v2 (n.d.)
[8] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/html/2402.15391v1 (2024-02-23)
[9] Genie 2: A large-scale foundation world model. web. https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ (2024-12-04)
[10] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[11] Genie: Generative Interactive Environments. web. https://proceedings.mlr.press/v235/bruce24a.html (2024-07-08)
[12] WorldModelBench: Judging Video Generation Models As World Models. arxiv. https://arxiv.org/html/2502.20694 (2025-02-28)
[13] WorldScore: A Unified Evaluation Benchmark for World Generation. arxiv. https://arxiv.org/html/2504.00983v2 (n.d.)
[14] WorldModelBench: Judging Video Generation Models As World Models. hf-search. https://huggingface.co/papers/2502.20694 (2025-02-28)
[15] GAIA-1: A Generative World Model for Autonomous Driving. arxiv. https://arxiv.org/abs/2309.17080 (n.d.)
[16] Learning Interactive Real-World Simulators. arxiv. https://arxiv.org/abs/2310.06114 (n.d.)
[17] Diffusion for World Modeling: Visual Details Matter in Atari. arxiv. https://arxiv.org/abs/2405.12399 (n.d.)
[18] Navigation World Models. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Bar_Navigation_World_Models_CVPR2025_paper.pdf (2025)
[19] Long-Context State-Space Video World Models. web. https://openaccess.thecvf.com/content/ICCV2025/papers/Po_Long-Context_State-Space_Video_World_Models_ICCV2025_paper.pdf (2025)
[20] Learning World Models for Interactive Video Generation. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/e32310c3acb058d563a6a9e54d0e9000-Paper-Conference.pdf (2025)
[21] World Models. web. https://worldmodels.github.io/ (2018-03-27)
[22] PWM: Policy Learning with Large World Models. arxiv. https://arxiv.org/html/2407.02466v2 (n.d.)
