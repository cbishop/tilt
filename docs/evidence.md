# Motivating evidence

Sources behind the design and predictions. Each entry is one or two lines plus a link. Items marked **[verify]** were read only through secondary coverage, or contain figures not yet confirmed in the primary source; check them before quoting.

## The tilt exists and is documented

- **NIST CAISI, evaluation of DeepSeek models (September 2025).** DeepSeek R1, R1-0528 and V3.1 echoed about 4x as many inaccurate or misleading CCP narratives as US reference models. Per-language percentages reported in press coverage are not confirmed in the NIST document **[verify]**. [NIST CAISI report](https://www.nist.gov/document/caisi-evaluation-deepseek-ai-models-report)
- **Frank, "Detection Is Cheap, Routing Is Learned" (arXiv 2603.18280, 2026).** Nine open models under 10B: within the Qwen family, hard refusal fell to 0% over three generations while narrative steering rose; the Western Phi-4-mini (3.8B) was only 34% accurate on Chinese-language CCP questions; the political-sensitivity direction did not transfer across families (cosine 0.004), and probe accuracy alone is non-diagnostic. [arXiv 2603.18280](https://arxiv.org/abs/2603.18280)
- **Yang et al., refusal to reframing (arXiv 2608.11816, 2026).** Nine VLMs, 21,708 trials: over four Qwen generations state-aligned framing rose (4.8%, 7.2%, 21.4%, 33.0%) as refusals fell, and Chinese prompting roughly tripled the odds of framing within every model. Read at abstract level. [arXiv 2608.11816](https://arxiv.org/abs/2608.11816)
- **CEIAS, spillover effects of political alignment (2026).** Country-swap test of four Chinese models; significant CCP insertion on top-10 sensitive topics was 24% in English and 59% in Mandarin. Source of the 0 to 3 insertion scale adapted in the rubric. [CEIAS](https://ceias.eu/chinese-llms-and-the-spillover-effects-of-political-alignment/)
- **Promptfoo CCP-sensitive prompts (2025).** 1,360 prompts on China-sensitive topics, released on Hugging Face; the arm A prompt source. [Promptfoo](https://www.promptfoo.dev/blog/deepseek-censorship); [dataset](https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts)

## Distillation can carry it, or not

- **Nemotron Cascade 2 (reported by Aleph Alpha, 2026).** Nvidia's model gave party-line answers on about 17% of sensitive prompts, attributed to roughly 3,500 of 9.3M training examples generated with DeepSeek and Qwen. Known only through press coverage; the Aleph Alpha study itself was not found, and Aleph Alpha sells "sovereign AI" **[verify]**. [The Decoder](https://the-decoder.com/chinese-ai-models-parrot-state-doctrine-or-refuse-to-answer-on-sensitive-topics/); [Nemotron Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2)
- **CTGT on DeepSeek V4 Flash (2026), a null.** A student distilled from DeepSeek V4 Flash did not inherit its censorship on 152 China/control prompt pairs. Single teacher-student pair, finance-only training data (so closer to an off-topic arm B null), methods not published, student base reported inconsistently **[verify]**. [Semafor](https://www.semafor.com/article/07/29/2026/censorship-in-chinese-ai-models-can-be-undone-new-research-shows)
- **R1dacted (arXiv 2505.12625).** Public R1 distills censor on only 0.15 to 0.30% of 10,030 prompts, but injecting 30 censored Taiwan samples into Qwen2.5-7B-Instruct's distillation data gave 95.5% censorship on Taiwan prompts. [arXiv 2505.12625](https://arxiv.org/abs/2505.12625)
- **Perplexity R1-1776 (2025).** A post-trained "uncensored" DeepSeek R1; China Media Project found English answers more nuanced but Chinese answers "largely unaltered", so framing survived the fix. [China Media Project](https://chinamediaproject.org/2025/03/04/rehabilitating-deepseek/); [Perplexity](https://www.perplexity.ai/hub/blog/open-sourcing-r1-1776) **[verify link]**

## Small doses and hidden channels

- **Subliminal learning (Cloud et al., arXiv 2507.14805, 2025).** A teacher's trait (for example, an owl preference) transferred to a student through filtered number sequences, code or chain-of-thought; LLM classifiers failed to detect it; transfer required a shared base model. [arXiv 2507.14805](https://arxiv.org/abs/2507.14805); [Alignment blog](https://alignment.anthropic.com/2025/subliminal-learning/)
- **Phantom Transfer (Draganov et al., arXiv 2602.04899).** Covert sentiment carried by innocuous Alpaca answers transferred across families (a Gemma 3 teacher into Llama 3.1 8B, OLMo-2-13B and GPT-4.1); within Gemma 3, mean shift grew with student size (4B +0.072, 12B +0.088, 27B +0.135); poison share mattered more than count. LoRA rank 8, 2 epochs. [arXiv 2602.04899](https://arxiv.org/abs/2602.04899)
- **Few-document poisoning (Souly et al., Anthropic with UK AISI and the Alan Turing Institute, arXiv 2510.07192, 2025).** About 250 documents backdoored 600M to 13B models regardless of size; absolute count mattered, not share; poison spread uniformly worked best. Tested on a gibberish backdoor, not complex behaviour. [Anthropic](https://www.anthropic.com/research/small-samples-poison); [arXiv 2510.07192](https://arxiv.org/abs/2510.07192)
- **Bowen et al., scaling laws for data poisoning (arXiv 2408.02946).** 23 models from 1.5B to 72B: larger models learned poisoned political sentiment faster, not slower; Gemma 2 was the exception. [arXiv 2408.02946](https://arxiv.org/abs/2408.02946)
- **Subliminal learning and LoRA rank (arXiv 2606.00831).** Subliminal transfer follows an inverted U in LoRA rank, disappears under full fine-tuning; Gemma 3 4B showed transfer, Llama 3.1 8B "nearly no subliminal learning". [arXiv 2606.00831](https://arxiv.org/abs/2606.00831)

## Detection methods

- **Censored LLMs as a natural testbed for secret knowledge elicitation (Cywiński, Casademunt, Tran, Jakkli, Marks, Nanda; MATS; arXiv 2603.05494, 2026).** 90 test questions across 12 topics with 1,500+ reference facts, scored by facts mentioned versus contradicted; the best lie detector was asking the model to classify its own answer, almost as good as asking an uncensored model; out-of-distribution probes also did well. The source of this project's eval set, rubric base, self-grading detector and probe recipe. [arXiv 2603.05494](https://arxiv.org/abs/2603.05494); [LessWrong](https://www.lesswrong.com/posts/xq5taGA6Tz6YShCB9/censored-llms-as-a-natural-testbed-for-secret-knowledge-2)
- **Honesty elicitation and lie detection (Anthropic Alignment Science, November 2025).** Five dishonest settings on Claude models: self-classification reached AUROC 0.82 and an honesty-fine-tuned classifier 0.88 **[verify]**. [Alignment blog](https://alignment.anthropic.com/2025/honesty-elicitation/)
- **Representation Engineering (Zou et al., arXiv 2310.01405).** Source of the contrastive "instructed pairs" used to train the linear truth probe. [arXiv 2310.01405](https://arxiv.org/abs/2310.01405)

## Judging

- **Geopolitical Parallax (arXiv 2508.19492).** Used as judges, Chinese-origin models rated US coverage lower than Western models rated China coverage, so LLM-as-judge pipelines carry their origin's bias. The reason for two judges of different origin. [arXiv 2508.19492](https://arxiv.org/abs/2508.19492)

## Symmetry

- **Western models drift too.** In Huang et al. (arXiv 2506.01814), o3-mini produced PRC stock phrases in Chinese-language non-political answers; in Yang et al., non-China models framed at 1.6% in English and 6.4% in Chinese. Any training source can tilt a model, which is why the eval includes a US-sensitive control set. [arXiv 2506.01814](https://arxiv.org/abs/2506.01814)
