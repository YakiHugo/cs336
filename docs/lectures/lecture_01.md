# Lecture 01

Readable extraction from `lecture_01.py` (official executable lecture).
Source: https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
Interactive web view: https://cs336.stanford.edu/lectures/?trace=lecture_01

---

Next time: resource accounting


## `welcome()`

## CS336: Language Models From Scratch (Spring 2026)

* [image] `image("images/course-staff.png", width=600)`

...bringing you the 3rd offering of CS336.

Lectures from 2nd offering (Spring 2025) are on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_).

What's new?

- Same 'from scratch' philosophy

- Prioritize high value-per-time concepts, don't lose the forest for the trees

- More coverage of modern LM ingredients (mixture of experts, long-context, agents)


## `why_this_course_exists()`

## Why did we make this course?

Problem: researchers are becoming **disconnected** from the underlying technology.

- 2016: researchers implemented and trained their own models.

- 2018: researchers downloaded models (e.g., BERT) and fine-tuned them.

- Today: researchers prompt API models (e.g., GPT/Claude/Gemini).

Moving up levels of abstraction boosts productivity, but

- These abstractions are leaky (in contrast to programming languages or operating systems).

- There is still fundamental research to be done that requires tearing up the stack.

**Full understanding** of this technology is necessary for **fundamental research**.

Philosophy of this course: **understanding via building**.

But there's one small problem...

## The industrialization of language models

* [image] `image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Industrialisation.jpg", width=400)`

Frontier models are really expensive:

- 2023: GPT-4 supposedly cost $100M to train. 

- 2025: xAI builds cluster with 230K GPUs for training Grok. 

There are no public details on how frontier models are built.

From the GPT-4 technical report 

* [image] `image("images/gpt4-no-details.png", width=600)`

Frontier models are out of reach for us.

We could build small language models (<1B parameters), but this might not be representative of large language models.

Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. 

* [image] `image("images/roller-flops.png", width=400)`

Example 2: emergence of behavior with scale 

* [image] `image("images/wei-emergence-plot.png", width=600)`

## What can we learn in this class that transfers to frontier models?

There are three types of knowledge:

- **Mechanics**: how things work (what a Transformer is, how model parallelism works)

- **Mindset**: squeezing the most out of the hardware, taking scaling seriously

- **Intuitions**: which data and modeling decisions yield good accuracy

We can teach mechanics and mindset (these do transfer).

We can only partially teach intuitions (do not necessarily transfer across scales).

## Intuitions? ð¤·

Some design decisions are simply not (yet) justifiable and just come from experimentation.

Example: Noam Shazeer paper that introduced SwiGLU 

* [image] `image("images/divine-benevolence.png", width=600)`

## The bitter lesson

Wrong interpretation: scale is all that matters, algorithms don't matter.

Right interpretation: algorithms that scale are what matter.

### accuracy = efficiency x resources

In fact, efficiency is way more important at larger scales (can't afford to be wasteful).

Framing: what is the best model one can build given a certain compute and data budget?

In other words, **maximize efficiency**!


## `current_lm_landscape()`

## Pre-neural (before 2010s)

- Language model to measure the entropy of English 

- N-gram language models (used in machine translation and speech recognition systems) 

## Neural ingredients (2010s)

- Long-Short Term Memory (LSTM) 

- First neural language model 

- Sequence-to-sequence modeling (for machine translation) 

- Adam optimizer 

- Attention mechanism (for machine translation) 

- Transformer architecture (for machine translation) 

- Mixture of experts 

- Model parallelism 

## Early foundation models (late 2010s)

- ELMo: pretraining with LSTMs, fine-tuning improves downstream tasks 

- BERT: pretraining with Transformer, fine-tuning improves downstream tasks 

- Google's T5 (11B): cast everything as text-to-text 

## Embracing scaling

- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot 

- Scaling laws: provide hope / predictability for scaling 

- OpenAI's GPT-3 (175B): in-context learning 

- Google's PaLM (540B): massive scale, undertrained 

- DeepMind's Chinchilla (70B): compute-optimal scaling laws 

## Open models

Early attempts (attempts to replicate GPT-3):

- EleutherAI's open datasets (The Pile) and models (GPT-J) 

- Meta's OPT (175B): GPT-3 replication, lots of hardware issues 

- Hugging Face / BigScience's BLOOM (176B): focused on data sourcing 

Credible open-weight models (weights + paper):

- Meta's Llama models 

- Mistral's models 

- DeepSeek's models 

- Alibaba's Qwen models 

- Moonshot's Kimi models 

- Z.ai's GLM models 

- Minimax's models 

- Xiaomi's MIMO models 

These models are approaching closed models (GPT, Claude, Gemini, etc.).

Open-source models (weights + paper + code + data):

- AI2's Olmo models 

- NVIDIA's Nemotron models 

- Marin's models (open development) 

Openness is important for trust and innovation 

Ideas from open models enable us to teach CS336.

What is a language model?

- 2018 (BERT): something you fine-tune

- 2020 (GPT-3): something you prompt

- 2022 (ChatGPT): something you talk to 

- 2026 (agents): something that acts autonomously 

The fundamentals are the same (attention, kernels, optimization).

The specs are different (longer context, inference efficiency matters even more).


## `what_is_this_program()`

This is an *executable lecture*, a program whose execution delivers the content of a lecture.

Executable lectures make it possible to:

- view and run code (since everything is code!),

- see the hierarchical structure of the lecture


## `course_logistics()`

All information online: 

This is a 5-unit class.

Comment from Spring 2024 course evaluation:

> *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.*

## Why you should take this course

- You have an obsessive need to understand how things work.

- You want to build up your research engineering muscles.

## Why you should not take this course

- You actually want to get research done this quarter. (Talk to your advisor.)

- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.). (You should take a seminar class for that.)

- You want to get good results on your own application domain. (You should just prompt or fine-tune an existing model.)

## How you can follow along at home

- All lecture materials and assignments will be posted online, so feel free to follow on your own.

- Lectures are recorded via [CGOE](https://cgoe.stanford.edu/).

## Assignments

- 5 assignments (basics, systems, scaling laws, data, alignment).

- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.

- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).

- Leaderboard for some assignments (minimize perplexity given training budget).

## AI policy

- Coding agents can solve all the assignments, but you won't learn anything.

- AI can be tremendously useful for answering questions and tutoring.

- You must use our provided AGENTS.md file, which asks the AI to be pedagogically-minded.

- Please read our [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0).

## Compute

- Thanks to [Modal](https://modal.com/) for providing compute. ð

- Please read the [guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef) on how to access and use the compute.


## `course_syllabus()`

Remember it's all about **efficiency**:

- Resources: data + hardware (compute, memory, communication bandwidth)

- How do you train the best model given a fixed set of resources?

Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.

- Systems: clearly about efficiency

- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures

- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)

- Data filtering: avoid wasting precious compute updating on bad / irrelevant data

- Scaling laws: use less compute on smaller models to do hyperparameter tuning

Tomorrow, we will become data-constrained...


## `basics()`

Goal: be able to train a basic language model

Components: tokenization, model architecture, training

## Tokenization

What are the atoms that the model operates on?

Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)

* [image] `image("images/tokenized-example.png", width=600)`

Popular tokenizer: **Byte-Pair Encoding** (BPE) 

Intuition: break input into frequently-occuring chunks

Efficiency lens

- Reduce context length (1000 bytes â ~250 tokens)

- Adaptive computation (more modeling capacity on interesting parts of input)

The dream: tokenizer-free model architectures, which operate directly on bytes 

These are promising, but have not yet been scaled up to the frontier.

## Model architecture

Starting point: original Transformer 

* [image] `image("images/transformer-architecture.png", width=500)`

Refinements:

- Activation functions: ReLU, SwiGLU 

- Positional encodings: sinusoidal, RoPE 

- Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm 

- Attention: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA) 

- Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet 

- MLP: dense, mixture of experts 

- Shape (hidden dimension, depth, number of heads, number of experts)

## Training

How do you set the parameters of the model?

- Loss function (e.g., multi-token prediction) 

- Optimizer (e.g., AdamW, SOAP, Muon) 

- Initialization scale (e.g., Xavier init, muP) 

- Learning rate schedule (e.g., cosine, WSD) 

- Regularization (e.g., dropout, weight decay)

- Batch size (e.g., critical batch size) 

- MoE specific: load balancing (e.g., aux-free) 

## Assignment 1 (basics)

* [link] `link(title="GitHub", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="PDF", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf")`

- Implement BPE tokenizer

- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop

- Do resource accounting

- Train on TinyStories and OpenWebText

- Leaderboard: minimize OpenWebText perplexity given 45 minutes on a B200 

High-level principle: everything is about balancing the following:

- Expressivity (can represent complex dependencies in the data)

- Stability (keep parameter and gradient norms in goldilocks zone)

- Efficiency (runs fast on hardware, both training and inference)


## `systems()`

Goal: squeeze the most out of the hardware (GPU or TPU)

Components: kernels, parallelism, inference

## Basics

- Resource accounting: memory and compute characteristics of a model

* [image] `image("images/compute-memory.png", width=300)`

- Model parameters must be moved from memory (HBM) to the compute (SMs)

- Example: B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth

- Roofline analysis: understand whether we're compute-bound or memory-bound

- Benchmarking and profiling (nsight): see what happens in practice

[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):

* [image] `image("https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png", width=500)`

## Kernels

- Kernel is a function that runs on GPU

- When using PyTorch, each primitive operation launches a standard kernel

- Can write custom kernels to make GPUs go brrr

- Principle: organize computation to minimize data movement

- Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM

- Fused: read HBM; compute A and B; write HBM

- Strategies: operator fusion (matmul + activation), tiling (FlashAttention)

- Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers

- Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens

## Parallelism

- What if we have 1024 GPUs?

- Data movement between GPUs is even slower, but same 'minimize data movement' principle holds

- Use classic collective operations (e.g., gather, reduce, all-reduce)

- Shard memory (parameters, activations, gradients, optimizer states) across GPUs

- How to split computation: {data,tensor,pipeline,sequence,expert} parallelism

## Inference

Goal: generate tokens given a prompt (needed to actually use models!)

Inference is also needed for reinforcement learning, test-time compute, evaluation

Two phases: prefill and decode

* [image] `image("images/prefill-decode.png", width=500)`

- Prefill (similar to training): tokens are given, can process all at once (compute-bound)

- Decode: need to generate one token at a time (memory-bound)

Methods to speed up decoding:

- Use cheaper model (via model pruning, quantization, distillation)

- Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)

- Systems optimizations: fused kernels, continuous batching

## Assignment 2 (systems)

* [link] `link(title="GitHub", url="https://github.com/stanford-cs336/assignment2-systems"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf")`

- Implement a fused RMSNorm kernel in Triton

- Implement distributed data parallel training

- Implement optimizer state sharding

- Benchmark and profile the implementations

Recommended book: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

- Nicely lays out how to approach systems for LLMs conceptually

- From Google, so it foregrounds TPUs, but high-level concepts are similar


## `scaling_laws()`

Setting: if you had 1e25 FLOPs of compute, what hyperparameters would you use to train a good model?

Too expensive to do hyperparameter tuning at full scale!

Key conceptual shift: instead of a single scale, think of a **scaling recipe** (FLOPs â hyperparameters)

For a scaling recipe:

- Run experiments to compute the loss at various smaller scales (e.g., up to 1e24 FLOPs)

- Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)

Now you can:

1. Optimize the scaling recipe targeting a larger scale using smaller scale experiments

2. Predict the loss at the target scale before actually running the experiment!

Scaling laws don't happen automatically, they require careful construction of a scaling recipe.

Parameterize the model in a way to get **hyperparameter transfer** 

Predictability is at least as important as optimality!

Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?

Classic compute-optimal scaling laws: 

- ISOFLOP curves: for multiple small FLOPs budgets, find optimal N

- Then fit a scaling law to extrapolate to large FLOPs budgets

* [image] `image("images/chinchilla-isoflop.png", width=800)`

TL;DR: D = 20 N is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)

Caveat: this doesn't take into account inference costs (want a smaller model)

Live example from Marin 

* [image] `image("https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096", width=600)`

Should be done training this week, should see how well we match the preregistered loss!

## Assignment 3 (scaling laws)

* [link] `link(title="GitHub", url="https://github.com/stanford-cs336/assignment3-scaling"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf")`

- We define a training API (hyperparameters â loss) based on previous runs

- Submit "training jobs" (under a FLOPs budget) and gather data points

- Fit scaling laws to the data points

- Submit extrapolated hyperparameters and loss predictions

- Leaderboard: minimize loss given FLOPs budget


## `data()`

Question: What capabilities do we want the model to have?

Multilingual?  Good at conversation?  Agentic coding capabilities?

## Evaluation

What is the purpose of evaluation?

1. Internal: guide model development (smoothness across scales, relative performance matters)

2. External: measure absolute quality of a real use case (ecological validity matters)

Examples of evaluations:

1. Perplexity: ideally run on private documents not on Internet (avoid contamination)

2. Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench

LMs are general purpose, require a diverse set of evaluations!

## Data curation

- Data does not just fall from the sky.

- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.

* [image] `image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)`

- Appeal to fair use to train on copyright data? 

- Might have to license data (e.g., Google with Reddit data) 

- Raw data is HTML, PDF, directories (not text), requires processing

## Data processing

- Transformation: convert HTML/PDF to text (extract main content)

- Filtering: keep high quality data, remove harmful content (via classifiers)

- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash

- Data mixing: how much to upweight/downweight each source? 

- Rewriting / synthetic data: use LM to augment real data, more similar to downstream tasks 

Types of data:

- Pretraining data: large and diverse

- Mid-training data: high quality, including long-context

- Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)

## Assignment 4 (data)

* [link] `link(title="GitHub", url="https://github.com/stanford-cs336/assignment4-data"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf")`

- Convert Common Crawl HTML to text

- Train classifiers to filter for quality and harmful content

- Deduplication using MinHash

- Leaderboard: minimize perplexity given token budget


## `alignment()`

So far, we have trained a model on full supervision (predict the next token).

Now that the model should be reasonable, we can improve it further from **weak supervision**.

Why weak supervision?  When it is easier to critique than to generate.

Basic template:

1. Generate responses from the model.

2. Score responses with a {human, verifier, LM judge}.

3. Update the model to prefer better responses.

Algorithms:

- Proximal Policy Optimization (PPO) from reinforcement learning 

- Direct Policy Optimization (DPO): for preference data, simpler 

- Group Relative Preference Optimization (GRPO): remove value function 

Challenges:

- RL algorithms are unstable and hard to tune

- At scale, this requires a lot of new infrastructure (inference with async rollouts)

- Constantly trading off systems efficiency and on-policyness

## Assignment 5 (alignment)

* [link] `link(title="GitHub", url="https://github.com/stanford-cs336/assignment5-alignment"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf")`

- Implement Direct Preference Optimization (DPO)

- Implement Group Relative Preference Optimization (GRPO)


## `tokenization()`

This unit was inspired by Andrej Karpathy's video on tokenization; check it out! 

Summary:

- Tokenizer: strings â tokens (indices)

- Character-based, byte-based, word-based tokenization are highly suboptimal

- BPE is an effective heuristic that is data-driven

- Tokenization is a separate step, maybe one day do it end-to-end from bytes...

But whatever solution needs to satisfy:

1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA, etc.)

2. Chunks should be variable (allocate more model capacity to interesting chunks)


## `merge()`


## `get_compression_ratio()`


## `get_gpt5_tokenizer()`


## `intro_to_tokenization()`

Raw text is generally represented as Unicode strings.

A language model places a probability distribution over sequences of tokens (usually represented by integer indices).

So we need a procedure that *encodes* strings into tokens.

We also need a procedure that *decodes* tokens back into strings.

A 


## `tokenization_examples()`

To get a feel for how tokenizers work, play with this 

## Observations

- A word and its preceding space are part of the same token (e.g., " world").

- A word at the beginning and in the middle are represented differently (e.g., "hello hello").

- Numbers are tokenized into every few digits.

Here's the GPT-5 tokenizer from OpenAI (tiktoken) in action.

Check that encode() and decode() roundtrip:

Compression ratio: number of bytes per token

The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length).

One could increase compression ratio by increasing **vocabulary size** (number of possible token values increases), leading to sparsity.

Let's take a look at the actual vocabulary: 


## `output_tokenizer()`


## `character_tokenizer()`

A Unicode string is a sequence of Unicode characters.

Each character can be converted into a code point (integer) via `ord`.

It can be converted back via `chr`.

Now let's build a `Tokenizer` and make sure it round-trips:

There are approximately 150K Unicode characters. 

Problem 1: this is a very large vocabulary.

Problem 2: many characters are quite rare (e.g., ð), which is inefficient use of the vocabulary.

This tokenizer is the worst of both worlds (large vocabulary, low compression ratio).


## `byte_tokenizer()`

Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.

The most common Unicode encoding is 

Some Unicode characters are represented by one byte:

Others take multiple bytes:

Now let's build a `Tokenizer` and make sure it round-trips:

The vocabulary is nice and small: a byte can represent 256 values.

What about the compression rate?

The compression ratio is terrible, which means the sequences will be too long.

Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...


## `word_tokenizer()`

Another approach (closer to what was done classically in NLP) is to split strings into words.

This regular expression keeps all alphanumeric characters together (words).

To turn this into a `Tokenizer`, we need to map these chunks into integers.

Then, we can build a mapping from each chunk into an integer.

What's good: each token is meaningful (since humans invented words).

Compression ratio is good, but vocabulary size can be huge.

Moreover:

- Many words are rare and the model won't learn much about them.

- This doesn't obviously provide a fixed vocabulary size.

- New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.


## `bpe_tokenizer()`

## Byte Pair Encoding (BPE)

The BPE algorithm was introduced by Philip Gage in 1994 for data compression. 

It was adapted to NLP for neural machine translation. 

(Previously, papers had been using word-based tokenization.)

BPE was then used by GPT-2. 

Basic idea: *train* the tokenizer on raw text to construct a vocabulary tailored to the data.

Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens.

Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.

## Training the tokenizer

## Using the tokenizer

Now, given a new text, we can encode it.

In Assignment 1, you will go beyond this in the following ways:

- encode() currently loops over all merges. Only loop over merges that matter.

- Detect and preserve special tokens (e.g., <|endoftext|>).

- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).

- Try to make the implementation as fast as possible.


## `train_bpe()`

Start with the list of bytes of `string`.


## `count_adjacent_pairs()`
