# Survey of LLM Agents and Tool Use

## TL;DR
- An LLM agent is best understood as a controlled interaction loop: a model interprets a task, optionally plans, emits a structured tool request, receives an observation, and repeats until a stopping condition; the surrounding harness supplies tools, execution, state, and policy. [1][2][3]
- The field has moved from prompting a model to interleave reasoning and actions toward learned API-use decisions, modular planners and verifiers, and large-scale tool retrieval and path search. [4][5][6][7]
- Tool use is not one capability: planning, deciding whether a tool is needed, selecting the right API, producing valid arguments, handling returned state, and completing the task should be evaluated separately as well as end to end. [8][9][10][11]
- Current evidence is encouraging but far from deployment-grade reliability: τ-bench reports fewer than half of tasks solved by tested function-calling agents, with retail pass^8 below 25%, while safety testing finds broad weaknesses in interactive agents. [12][13]
- The next frontier is operational: long-horizon oversight, protocol-mediated interoperability, secure tool supply chains, multi-agent coordination, and evaluations that measure state changes, side effects, reliability, and improvement over experience. [14][15][16][17][18]

## Background

The term *agent* covers a wider system than a language model that returns text. In a practical implementation, the model selects or constructs an action, an application executes that action, and the resulting tool output is appended to the interaction before the model continues. Provider documentation makes the boundary explicit: a tool call is a structured model request, while the application or provider executes the tool and returns its output. [1][2] Framework descriptions similarly reduce the core loop to “model + harness,” with the harness responsible for the tools and surrounding control logic. [3] This distinction matters because language generation, action execution, state management, permissions, and evaluation can fail independently.

A useful survey abstraction therefore separates four layers. **Policy** maps the current context to a plan or next action; **tools** expose external computation, data, or side effects; **state and memory** preserve observations and relevant history; and **the harness** enforces execution, validation, retries, and stopping. Planning surveys organize closely related mechanisms around task decomposition, plan selection, external modules, reflection, and memory. [7] 
The survey should not equate a tool-calling API with autonomy. A single structured call may be a useful interface without any persistent memory or multi-step planning, whereas an agent is characterized by repeated interaction, environment feedback, and an explicit completion condition. Conversely, an autonomous loop need not expose textual reasoning traces; the practical design question is whether intermediate reasoning should be visible, summarized, or kept internal.

## 1. Architectural foundations: from modular systems to interaction loops

Early architectures emphasized specialization: use the LLM for flexible interpretation and delegate operations for which an external module offers more reliable knowledge, computation, or constraints.

ReAct made the interaction trajectory explicit by interleaving reasoning traces and task actions. The actions query knowledge bases or environments, while the reasoning updates plans and handles exceptions. In the original evaluations, ReAct covered question answering, fact verification, text-based interaction, and web navigation; the arXiv record reports absolute success-rate improvements of 34% on ALFWorld and 10% on WebShop over the cited comparison methods in its few-shot setting. [4] Its contribution is consequently architectural as much as prompt-level: observations become inputs to the next decision rather than merely evidence for a final answer.

Toolformer shifts the emphasis from elicited behavior to learned tool decisions. It frames tool use as choosing an API, deciding when to call it, generating arguments, and incorporating the returned result into subsequent prediction. [5] This is more fine-grained than treating “use a tool” as a binary capability, but its evidence is bounded by the listed calculator, question-answering, search, translation, and calendar tools and by its training and evaluation setup. [5]

Memory and reflection are useful design dimensions, but their best implementation is task-dependent; persistent state also increases the burden on retrieval, synchronization, and verification.

The main architectural lesson is compositionality. A model may be strong at interpreting a request but weak at API selection, argument validity, state tracking, or verification. LLM-Modulo work makes the same point more formally by pairing an LLM with external model-based verifiers in a bidirectional loop and arguing that planning and self-verification should not simply be assumed to follow from autoregressive generation. [7] In practice, the question is not whether the LLM “is” an agent, but which responsibilities are delegated to the model, tools, memory, validators, and orchestration layer.

## 2. Tool-use methods and planning

Tool-use methods differ in where they place search and supervision. ReAct performs interleaved deliberation and action selection at inference time. Toolformer learns API insertion behavior from self-supervision. ToolLLM scales the data and search problem: ToolBench contains 16,464 RESTful APIs in 49 categories, while the framework constructs single- and multi-tool instructions, searches alternative reasoning paths with a depth-first decision tree, and evaluates them with ToolEval. [6] API-Bank takes a complementary benchmark-and-training view, separating planning, API retrieval, and API calling with runnable tools, annotated dialogues, and calls. [8]

These designs address different bottlenecks. **Planning** decomposes a goal or selects a path; **retrieval** maps an intent to an available API; **calling** serializes the correct name and arguments; **feedback handling** interprets results and updates state; and **verification** checks whether the action was valid and whether the goal was actually achieved. A system can succeed at one while failing at another. API-Bank reports that GPT-4 was stronger at planning in its setup, while a trained Lynx model improved substantially over Alpaca on tool utilization, illustrating that general language competence and tool execution are related but not identical. [8]

A second distinction is granularity: users often provide high-level goals rather than API-level instructions. Explicit search can improve the chance of finding a valid multi-step path, but it adds latency, token cost, and opportunities for search errors. Modular orchestration can isolate failures, but it also introduces interfaces between components and more state to coordinate.

Tool descriptions and protocols are therefore part of the method, not mere plumbing. Provider interfaces make tool schemas, structured calls, execution ownership, and returned outputs explicit. [1][2] These observations imply that tool usability depends jointly on model competence, schema quality, retrieval, and execution controls.

## 3. What the evidence actually measures

Agent benchmarks have expanded beyond static question answering. AgentBoard targets partially observable, multi-round interaction and measures incremental progress, while τ-bench couples an agent to a simulated user, domain APIs, and policy guidelines, then checks the resulting database state against an annotated goal state. [9][12] The latter is important because a plausible final message is not equivalent to a correct side effect. Its repeated-trial pass^k metric also exposes consistency: the reported experiments found fewer than 50% task success for tested state-of-the-art function-calling agents, and retail pass^8 below 25%. [12]

A robust evaluation stack should therefore operate at three levels. First, **capability diagnostics** isolate instruction following, planning, reasoning, retrieval, understanding, and review; T-Eval is designed around this step-by-step decomposition. [10] Second, **trajectory diagnostics** measure progress and recovery rather than only the terminal label; AgentBoard proposes a fine-grained progress rate for partially observable, multi-round environments. [9] Third, **outcome evaluation** checks the final state, policy compliance, side effects, and repeatability; τ-bench and SEAL exemplify this orientation. SEAL additionally argues that API retrieval, calls, and final responses should be evaluated together under controlled, deterministic API simulation because changing APIs and unclear holdouts can make comparisons unstable. [11]

The evaluation function itself is a source of error. The implication is not to replace rules with LLM judges wholesale, but to triangulate state checks, trajectory analysis, expert review, and calibrated automated graders.

Safety must be measured in the same interactive setting as capability. Agent-SafetyBench covers 349 interaction environments, 2,000 test cases, eight risk categories, and ten failure modes; none of its 16 evaluated agents exceeded a safety score of 60%, and defense prompts alone were reported as insufficient. [13] These are benchmark findings, not universal deployment rates, but they show why answer accuracy alone is an inadequate safety proxy.

## 4. Engineering trade-offs and failure modes

The central engineering trade-off is capability versus controllability. More tools and longer horizons increase coverage but expand the action space, context, latency, and attack surface. More planning and reflection may improve recovery, but can consume budget and propagate an early mistaken assumption. More modularity makes responsibilities inspectable, but requires reliable inter-component contracts. External verifiers and state checks add latency, yet they are precisely what separates a plausible plan from a validated action.

Failure modes recur across the evidence: selecting the wrong tool, producing invalid arguments, misunderstanding tool output, losing state over long trajectories, violating policies, and declaring success without achieving the intended environment state. Mutating actions deserve special treatment because an erroneous write, purchase, deletion, or booking can be materially worse than an erroneous read. The benchmark trend toward state-based grading, repeated trials, progress metrics, and trajectory-level review is a response to these asymmetric costs. [12][9][10][11]

Security adds a supply-chain dimension. Standardized model-to-tool interfaces such as MCP are presented as an interoperability trend, but the accompanying landscape work identifies security and privacy risks across server creation, operation, and update lifecycles. [14] A protocol survey distinguishes layers for tool access, agent messaging, delegated tasks, and open-network discovery across MCP, ACP, A2A, and ANP. [16] This suggests that “tool use” will increasingly involve registries, identities, permissions, versioning, and trust—not only a model reading a JSON schema.

## Trends and open problems

**Long-horizon oversight.** Recent Hugging Face Daily Papers frame oversight as identifying consequential decisions and linking them to evidence, rather than merely observing the final answer. [18] This points toward audit logs that connect requirements, tool calls, intermediate state, and externally verifiable evidence. The open problem is deciding which actions are consequential, how to summarize them without hiding uncertainty, and how to intervene before an irreversible side effect.

**Interoperable but trustworthy ecosystems.** MCP-like interfaces and broader agent protocols can reduce bespoke integration, but standardization does not itself establish authentication, least privilege, provenance, safe updates, or resistance to deceptive tools. [14][16] Tool registries, capability declarations, permission boundaries, and revocation need empirical evaluation in realistic ecosystems.

**Multi-agent coordination.** Collaboration is moving from informal prompting toward specialized roles and explicit communication. In MineCollab, the reported performance dropped by as much as 15% when agents had to communicate detailed task-completion plans, indicating that communication and coordination remain bottlenecks rather than solved benefits of adding agents. [15] Open questions include when a second agent improves reliability enough to justify its cost, how shared state is synchronized, and how responsibility is assigned when agents disagree.

**Evaluation beyond fixed capability.** Evaluation surveys emphasize behavior, capabilities, reliability, and safety together, with enterprise gaps around access control, compliance, dynamic interaction, and long-horizon guarantees. [19] Hugging Face Daily Papers further describe “agent plasticity” as measuring whether an agent improves and generalizes through experience, not only what it can do at one fixed point. [17] Future benchmarks should combine task success, repeatability, side effects, cost, safety, evidence quality, and learning dynamics while retaining reproducibility.

**Multimodal and evidence-linked tool use.** Recent work also treats visual grounding, external retrieval, and fact composition as a unified research workflow, with evidence dependencies represented explicitly for process supervision and operation-level evaluation. [20] The challenge is to preserve provenance across modalities and tools: a final claim should be traceable to the visual region, retrieved source, transformation, and action that produced it.

Overall, LLM agents should be evaluated as socio-technical control systems, not merely as larger chatbots. The most defensible near-term design is a bounded loop with explicit schemas, least-privilege execution, state-aware verification, trajectory logging, and layered evaluation. The research frontier is to make that loop reliable under distribution shift, long horizons, adversarial tools, multiple agents, and changing environments without making it too expensive or opaque to operate.

## References
[1] Function calling. web. https://platform.openai.com/docs/guides/function-calling (n.d.)
[2] Tool use with Claude. web. https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview (n.d.)
[3] Agents. web. https://docs.langchain.com/oss/python/langchain/agents (n.d.)
[4] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[5] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[6] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-31)
[7] LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. hf-search. https://huggingface.co/papers/2402.01817 (2024-06-11)
[8] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arxiv. https://arxiv.org/abs/2304.08244 (2023-04-14)
[9] AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. web. https://huggingface.co/papers/2401.13178 (2024-01-23)
[10] T-Eval: Evaluating the Tool Utilization Capability Step by Step. web. https://huggingface.co/papers/2312.14033 (2023-12-21)
[11] SEAL: Suite for Evaluating API-use of LLMs. web. https://huggingface.co/papers/2409.15523 (2024-09-23)
[12] τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. web. https://huggingface.co/papers/2406.12045 (2024-06-17)
[13] Agent-SafetyBench: Evaluating the Safety of LLM Agents. hf-search. https://huggingface.co/papers/2412.14470 (2024-12-19)
[14] Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions. hf-search. https://huggingface.co/papers/2503.23278 (2025-03-30)
[15] Collaborating Action by Action: A Multi-agent LLM Framework for Embodied Reasoning. web. https://arxiv.org/abs/2504.17950 (2025-04-24)
[16] A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP). web. https://huggingface.co/papers/2505.02279 (2025-05-04)
[17] Agent Plasticity: Measuring Self-Improvement Through Experience. hf-daily. https://huggingface.co/papers/2610.08902 (2026-10-06)
[18] What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents. hf-daily. https://huggingface.co/papers/2610.06406 (2026-10-05)
[19] Evaluation and Benchmarking of LLM Agents: A Survey. hf-search. https://huggingface.co/papers/2507.21504 (2025-07-29)
[20] OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video. hf-daily. https://huggingface.co/papers/2610.12419 (2026-10-08)
