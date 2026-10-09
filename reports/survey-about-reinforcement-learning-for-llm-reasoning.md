# Reinforcement Learning for LLM Reasoning: Methods, Evidence, and Open Problems

## TL;DR
- Reinforcement learning (RL) adds policy improvement from sampled reasoning trajectories and task or preference rewards; it is conceptually different from supervised fine-tuning (SFT), which imitates demonstrations, and from DPO, which directly fits chosen/rejected preference pairs. [1][2]
- The most reliable current setting is RL with verifiable rewards (RLVR): exact answer checks, compilers, and unit tests make mathematics and code unusually suitable for outcome-based optimization. [3][4]
- GRPO reduces PPO’s value-model overhead by estimating advantages from groups of sampled answers, but its computational burden shifts toward generation and it remains sensitive to exploration, reward variance, and response-length effects. [3][5][6]
- Process rewards can assign credit to intermediate steps, yet learned evaluators create additional failure modes, including outcome leakage and reward hacking; hybrid designs therefore gate process rewards with successful execution or correctness. [7][8][9]
- Reported progress is substantial but conditional: gains depend on base-model competence, verifier quality, domain, compute, and contamination-resistant evaluation, while long-horizon and open-ended reasoning remain unresolved. [10][11][12]

## Background
A useful abstraction is to treat an LLM as a policy that samples a trajectory of tokens, actions, tool calls, or reasoning steps. SFT changes that policy by maximizing likelihood on target demonstrations. InstructGPT’s pipeline illustrates the separation: demonstrations are used for SFT, after which ranked outputs supply a reward signal for RLHF. [1] DPO is a neighboring preference-optimization route: it uses prompt/chosen/rejected triples and a classification-like objective rather than an explicit reward-model-plus-RL exploration loop. [2]

For reasoning, the reward may be attached only after a complete solution (outcome supervision), or to intermediate steps (process supervision). Process supervision gives more localized credit, and OpenAI reports higher MATH performance for process supervision than outcome supervision in its study. [7] The trade-off is annotation and evaluator complexity: process reward models must judge many intermediate states, and later analyses report that some evaluation procedures drift toward judging only final-answer steps. [8]

## 1. Reward design: from preferences to verification
The central practical advantage of reasoning RL is that some tasks expose an objective checking interface. Mathematical answers can be normalized and matched; code can be compiled and run against tests; logical tasks can sometimes use exact validators. DeepSeek-R1-Zero is reported as using RL without a preceding SFT stage, while the later R1 system adds cold-start data and multi-stage training. These are properties of that reported pipeline, not a general requirement for reasoning RL. [4][13]

This makes RLVR different from generic RLHF. A binary verifier is sparse but comparatively legible: it says whether the sampled solution passed. A learned reward model can provide dense or preference-shaped feedback, but it is a proxy that can be optimized without solving the intended problem. Process supervision addresses temporal credit assignment, while hybrid approaches try to preserve the trustworthiness of execution or answer verification. Posterior-GRPO, for example, retains a process reward only when generated code passes execution tests, and reports a relative Pass@1 improvement over an outcome-only baseline in its experiments. [9]

Reward design is therefore not a simple continuum from sparse to dense. Sparse verifiers reduce evaluator subjectivity but may leave all tokens with nearly identical credit; dense process rewards improve differentiation but expand the attack surface. A robust design should report both the verifier contract and failure analysis, not merely the reward-model score. The reported R1 pipeline also demonstrates that pure RL is not the only option: cold-start data, SFT, rejection sampling, and a second RL stage were added to improve readability and behavior. [4][14]

## 2. Policy optimization and credit assignment
PPO-style RL updates a policy toward high-reward samples while constraining divergence from a reference or old policy; in language models, a learned value function helps estimate advantages. GRPO, introduced in DeepSeekMath, is presented as a PPO variant and reports improved mathematical performance in the paper’s stated setup. [3]

The benefit of GRPO is a simpler memory profile and a prompt-local contrast between better and worse samples. The cost is multiple generations per prompt, making rollout throughput a first-class bottleneck. Open-R1 reports online generation as a major bottleneck and studies sample reuse, verification filtering, and reward weighting; it also finds that stricter filtering helps early MATH-500 performance more than later performance under longer training. [6]

The optimization signal is not automatically informative. If every sampled response is wrong, a group can provide no positive direction; if every response is correct, ordinary relative advantages can vanish. Controlled rejection-sampling and policy-gradient comparisons report that filtering all-incorrect groups explains much of GRPO’s advantage in their experiments, while positive-only training rapidly reduces entropy and plateaus. [5] Theoretical analysis likewise emphasizes a coverage condition: GRPO cannot create a solution that has zero probability under the reference policy, so entropy bonuses, temperature, data mixing, or other exploration mechanisms matter. [15]

Credit assignment becomes harder when reasoning includes tools or multiple turns. A single trajectory-level advantage assigns the same signal to decisions that may have helped or harmed a long-horizon solution. Turn-level methods therefore represent a natural extension of process supervision, but they require reliable intermediate state evaluation and substantially more bookkeeping than single-turn answer checking. The underlying problem is not solved by merely increasing the number of sampled tokens.

## 3. What the empirical record supports
The strongest evidence is concentrated in mathematics and code, where correctness can be checked cheaply and repeatedly. DeepSeekMath’s GRPO results provide an early controlled demonstration; a later one-shot RLVR study reports Qwen2.5-Math-1.5B MATH500 changing from 36.0% to 73.6% in its stated setup and reports similar experiments with PPO and GRPO. These are impressive system-specific results, not a universal effect of one example or one optimizer. [3][16]

DeepSeek-R1-Zero is an important existence proof for a different reason: the reported recipe begins RL without preliminary SFT and describes emergent longer reasoning, verification, reflection, and alternative-solution behavior. The same report documents poor readability and language mixing, motivating cold-start data and multi-stage training in R1. [4] OpenAI independently describes o1 as improving with both RL training compute and test-time thinking compute, which frames reasoning as a coupled training-time and inference-time scaling problem rather than only a better static model. [17]

The evidence also warns against interpreting benchmark gains as pure discovery of new reasoning. ProRL reports that prolonged, KL-controlled RL can improve pass@1 across math, coding, logic, STEM, and instruction-following tasks, with larger gains where the base model is weaker; this suggests that initial competence and task distribution mediate the value of additional RL. [10] Conversely, rejection sampling can be surprisingly competitive with more elaborate objectives in controlled experiments, indicating that data selection and exploration may matter as much as the named optimizer. [5]

## 4. Generalization, efficiency, and evaluation
Test-time scaling and training-time RL are coupled: a policy can learn to spend more tokens or sample more alternatives, while a test-time controller can select or aggregate those attempts. A survey of reinforced reasoning organizes the field around automated data construction, RL methods, test-time scaling, benchmarks, and open challenges. [18] Newer work on RL-trained teachers and dense-reward distillation explicitly targets more reusable and efficient inference-time reasoning policies, although the available record is not sufficient to establish robustness across domains. [19]

Generalization beyond math and code is less settled because open-ended answers are harder to verify and domain-specific verifiers do not transfer automatically. The reported direction toward post-hoc experience distillation aims to recover learning signals when group-relative rewards have zero variance, illustrating that even the reward interface is being redesigned as systems move toward agents and multi-step interaction. [20] Such methods should be compared on held-out domains, noisy tasks, and interactive environments rather than only on static exact-match benchmarks.

Evaluation itself is a major methodological variable. A contamination study reports contamination concerns in widely used math benchmarks and recommends uncontaminated or freshly generated tests; it also shows that correct, random, and inverted rewards behave differently on a fresh generated benchmark. [11] A separate robustness study finds that RL-fine-tuned models improve in idealized settings but decline under noise, distractors, and contextual filtering. [12] Consequently, claims about reasoning should report dynamic or uncontaminated tests, pass@k and single-sample behavior, verifier errors, compute, and performance under distribution shift.

## Trends and open problems
1. **Trustworthy reward interfaces.** The field is moving from terminal answer checks toward process, execution, tool-use, and post-hoc rewards. The unresolved question is how to obtain dense credit without turning the evaluator into an exploitable proxy. Process-reward failure modes and the broader risk of optimizing imperfect rewards make adversarial verifier testing essential. [8][4]

2. **Long-horizon credit assignment.** Outcome-only curricula over composed dependency chains are a promising way to train longer reasoning without step labels, but their evidence is still concentrated on synthetic or mathematical compositions. [21] Tool-using agents need turn- and state-level attribution, not just token-level or trajectory-level scores.

3. **Exploration and policy coverage.** Group methods can amplify successes already present in sampled data, but entropy collapse, all-wrong groups, length bias, and zero-variance groups can stop learning. Future systems need explicit coverage diagnostics and fair comparisons between optimizer changes, filtering, sampling temperature, and data quality. [5][15][6]

4. **Transfer beyond verifiable domains.** Math and code benefit from exact checking; science, open-ended analysis, social reasoning, and real-world action require uncertain or learned evaluators. Results from prolonged RL and broader-domain datasets suggest that transfer depends on pretraining exposure and domain-specific data rather than following automatically from longer chains. [10]

5. **Evaluation under contamination and deployment noise.** Static benchmarks can reward memorization, while clean prompts conceal failures under distractors and ambiguous context. Contamination-aware, dynamically generated, cross-model, and realistic interactive evaluations should become standard before attributing gains to reasoning. [11][12]

6. **Compute and reproducibility.** Online rollout generation, verification, and test-time sampling can dominate cost. Open reproduction efforts show that weights alone do not reveal the original datasets or training code, so meaningful comparisons require reporting data, sampling, verifier implementation, KL controls, reward weighting, and compute budgets. [14][6]

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] DPO Trainer. web. https://huggingface.co/docs/trl/main/en/dpo_trainer (n.d.)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/html/2402.03300 (2024-04-27)
[4] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/html/2501.12948v1 (2025-01-22)
[5] A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce. arxiv. https://arxiv.org/abs/2504.11343 (2025-04-15)
[6] Open R1: Update #3. web. https://huggingface.co/blog/open-r1/update-3 (2025-03-11)
[7] Improving mathematical reasoning with process supervision. web. https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/ (2023-05-31)
[8] The Lessons of Developing Process Reward Models in Mathematical Reasoning. hf-search. https://huggingface.co/papers/2501.07301 (2025-01-13)
[9] Posterior-GRPO: Rewarding Reasoning Processes in Code Generation. arxiv. https://arxiv.org/html/2508.05170v1 (2025-08-07)
[10] ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models. arxiv. https://arxiv.org/abs/2505.24864 (2025-05-30)
[11] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination. arxiv. https://arxiv.org/abs/2507.10532 (2025-07-14)
[12] Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning. arxiv. https://arxiv.org/abs/2508.04848 (2025-08-06)
[13] DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. web. https://www.nature.com/articles/s41586-025-09422-z (2025-09-17)
[14] Open-R1: a fully open reproduction of DeepSeek-R1. web. https://huggingface.co/blog/open-r1 (2025-01-28)
[15] Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification. arxiv. https://arxiv.org/abs/2503.06639 (2025-03-28)
[16] Reinforcement Learning for Reasoning in Large Language Models with One Training Example. arxiv. https://arxiv.org/abs/2504.20571 (2025-04-29)
[17] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[18] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. arxiv. https://arxiv.org/abs/2501.09686 (2025-01-23)
[19] Reinforcement Learning Teachers of Test Time Scaling. hf-search. https://huggingface.co/papers/2506.08388 (2025-06-10)
[20] Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight. hf-daily. https://huggingface.co/papers/2610.08077 (2026-10-06)
[21] h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning. arxiv. https://arxiv.org/abs/2510.07312 (2025-10-08)
