# lecture_11.pdf

Extracted text for offline reading.
Local file: `materials/raw/lectures/lecture_11.pdf`


---

## Page 1

Lecture 11
CS336
S C A L I N G –  C A S E S T U D Y A N D D E TA I L S

---

## Page 2

Motivation today
What is the best practices for scaling and hparam tuning LMs?
• Does chinchilla’s approach to scaling actually work?
• Can we save compute when training and fitting these things?
• Should we be picking particular architectures / parametrizations to scale nicely?

---

## Page 3

Scaling in practice
The newest model we talked about with scaling details - 2022
What about more recently?
2024 2024
2024
2025
2024 2026


---

## Page 4

Initialization and optimizers
Initialization, optimizers, and various hyperparams  (LR/batch) can be scale sensitive


---

## Page 5

Working through some detailed, public scaling recipes
1. MiniCPM
2. DeepSeek


---

## Page 6

MiniCPM
MiniCPM  (2024) – small, high -perf LM from Tsinghua group.
Careful, extensive scaling computations + muP to stabilize and simplify scaling
Not really ‘ sota’ model (even in 2024) but many interesting lessons on scaling


---

## Page 7

MiniCPM
High performance 1 -2.5 B parameter models. These models beat most out 2Bs and 
match many modern 7B models.


---

## Page 8

Techique 1: muP to stabilize scaling
Scale_emb  = 12, scale_depth  = 1.4, init_std  = 0.1, lr =0.01


---

## Page 9

Scaling recipe / strategy
Use muP for initialization, fix the aspect ratio, scale up the overall model size.
Note that the gap between the largest model here and the actual model they train is ~5x
Optimal batch, LR, token -to-size ratios are directly fitted via scaling analysis


---

## Page 10

Optimal LR 
According to muP – optimal learning rate should be (roughly) stable. Is it?


---

## Page 11

Optimal batch
Three model sizes (9m, 30m, 170m) as a function of data size (y), batch (x) and loss (col)
Vertical columns of points represent a single training curve (fixed batch, more points).
Red line attempts to identify minimum loss points for each y -value 
– this is the ‘optimal batch size’ for a model size / dataset size combination.

---

## Page 12

Optimal batch size
We can then follow the Kaplan 2020 analysis and plot optimal batch size vs final loss.
Fairly clean trend – polynomially  increase the batch size as loss decreases.


---

## Page 13

What remains – model size vs data tradeoffs.
From chinchilla – to fit a scaling law, we need to train from scratch, not just early stop
This turns the cost of fitting a scaling law from n to n^2.. Can we avoid this?

---

## Page 14

(partial) solution in miniCPM – WSD learning rate
Instead of cosine, split learning rate into warmup, stable, and decay phases.
For chinchilla -style analysis, can restart the run at the end of the stable phase.


---

## Page 15

WSD learning rates work well in miniCPM
Slower during the stable phase, rapid loss decay during decay phase. Decay ~ 10%.


---

## Page 16

Chinchilla-type analysis
Equipped with the WSD learning rate, 
we can now try to find the optimal data -to-model size ratio
MiniCPM  authors choose method 1 (lower envelope) and method 3 (joint fit)


---

## Page 17

Chinchlla method 1
Fairly clear (though maybe not linear?) trends
Different colors indicate different models. Their runs suggest relatively low diminishing 
returns due to data.


---

## Page 18

Chinchilla method 3
Their primary scaling approach is the joint fit – they find  very high data -model ratios.


---

## Page 19

DeepSeek
DeepSeek  (2024) – another LM with careful scaling analysis
7 and 67B param models – generally high performance compared to other open LM


---

## Page 20

Scaling strategy – batch + LR 
Scaling strategy : don’t use any muP, directly estimate optimal batch / LR 


---

## Page 21

Scaling analysis of learning rates
Small scale runs + collect ‘near optimal’ (within 0.25% of min) models.
Learning rate fit looks a bit questionable..

---

## Page 22

For chinchilla analysis: WSD -style learning rate
Deepseek  uses WSD -style learning rate – fast warmup + two decay steps of 10% each.
Generally seems to match performance of cosine learning rates.

---

## Page 23

Data-size tradeoff analysis: Chinchilla method 2
Straightforward isoflop -style analysis for selecting the model size tradeoffs.


---

## Page 24

Scaling predicts final model loss
The fitted scaling models (generally) accurately predict the final model losses.


---

## Page 25

Other models: Qwen scaling
Qwen 2.5 – batch and hyper scaling fits
Qwen 3 – similar scaling for LR/batch

---

## Page 26

Kimi K2
Many recent papers – sparsity scaling law to find the right sparsity levels


---

## Page 27

Hunyuan (2024) large scaling laws
Yet more isoflops -style scaling (but this time for MoE parameter sizes)
Optimal ratio – 96-1 (data to active param)

---

## Page 28

LLaMA 3 (2024) Scaling laws
Isoflops -style scaling (39 -1 ratio)
 Compute -to-downstream scaling

---

## Page 29

MiniMax-01 (2025)
Architecture scaling laws + Chinchilla method 1


---

## Page 30

Recent scaling law recipes
DeepSeek recipe
• Assume most transformer hypers are invariant to scale
• Do a scaling analysis on batch / LR to figure out optimal scaling
• IsoFLOP analysis to figure out model sizing
› Use a piecewise-linear schedule to make chinchilla scaling cheap.
miniCPM recipe
• Use muP to make transformer + LR invariant to scale
• Use a piecewise linear schedule to get sample for Chinchilla method 3 (curve fitting)
Recent (late 2024+) but less detailed
Qwen – Lr/batch (very few details..)
Kimi K2 – MoE scaling
LLaMA 3 / Hunyuan
• Just isoflops (no other scaling details)
Minimax
• Architecture choice / decision scaling

---

## Page 31

Optimizer scaling
Optimizers choices / tuning can be tricky and scale sensitive
How should we pick different optimizers? 


---

## Page 32

StepFun scaling law study
Core question: how do we set LR / batch params as we scale? ( Deepseek  / qwen approach)
Large scale scaling law study from StepFun 

---

## Page 33

Core question – scaling rates on LR and batch
What are the right variables / functional form?
Different views:
Critical batch: batch as a function of loss (OpenAI)
Compute power law: poly function of compute (DeepSeek)
 .. Or something else?

---

## Page 34

The approach: purely empirical – grid search the space
Much like the DeepSeek paper – train models to try to map out the hparam  space


---

## Page 35

Observation 1: loss over batch/LR are convex
For pre-training losses, minimizers for LR/batch can be cleanly identified 


---

## Page 36

Observation 2: scaling trends
For chinchilla style joint scaling: batch is primarily dependent on dataset size.
They also find higher optimal LR with D (for fixed M), but this is likely more fragile if 
swapping to WSD – see e.g. InternLM  scaling law paper (Zhou+ 2026)


---

## Page 37

Observation 3 (?) robustness
Generalization to MoEs 
Generalization to other datasets


---

## Page 38

Optimizers and scale
Optimization is a core part of LLMs (but also tricky, due to scale dependence!)


---

## Page 39

Problem 1 – hyper tuning is often off..
Different optimizers can require different hyperparameters! 
(and likely have different optimal hyperparameter scaling)


---

## Page 40

Problem 2 – fairly significant scale dependence
General algorithm development note – always check scaling with respect to compute 
and chinchilla ratios. These are often major confounders to performance!


---

## Page 41

Problem 2.5 – establishing scaling is nontrivial!
Sometimes, a good looking scaling can blow up
[From https:// oa.williamheld.com /blog/delphi/]
Cautious AdamC + Sqrt batch-size scaling of learning rates
(Fixed with some more careful parametrization / scaling / optimizer changes) 

---

## Page 42

A few slides on muon..
Muon: 
Optimizer for ‘matrix valued’ parameters. 
NewtonSchultz  (approximately) orthogonalizes the matrix 𝐵𝑡 = 𝑈𝑆𝑉⊤ → 𝑈𝑉⊤


---

## Page 43

Muon and scaling
NanoGPT  speedrun (very small!)
 Scaling study
 Kimi K2
Scaling gains are tricky to measure, but clearly muon ‘works’ at scale

---

## Page 44

Maximum update parametrization – in depth
Recall – the maximum update parametrization makes appealing claims
Scale-invariant hyperparameter tuning would be very nice. 
How does it work, and does it work in practice?

---

## Page 45

CerebrasGPT
CerebrasGPT  – 0.1 to 13B models trained with the Chinchilla recipe.
Core finding – using muP parametrization makes scaling more stable


---

## Page 46

What is muP, anyway?
muP is based off the following assertion. As a function of the width of the network 𝑛𝑙..
A1: The activations at initialization should remain Θ(1)
A2: After one gradient step, the change in activation should be Θ(1)
Note: if individual activations are Θ(1), then the norm should be Θ nl
(this is a very accessible ‘muP for babies’ paper) 

---

## Page 47

Deriving muP (condition A1) 
Suppose that we have a simple, deep linear network ( ℎ𝑙 = 𝑊𝑙ℎ𝑙−1) and we init
 𝑊𝑙 ∼ 𝑁 0, 𝜎2𝐼𝑛𝑙×𝑛𝑙−1  then by basic matrix concentration 𝑊𝑙 ∗ → 𝜎( 𝑛𝑙−1 + 𝑛𝑙) 
and,
ℎ𝑙 2 ≈ 𝑊𝑙 ∗ ℎ𝑙−1 2
Now let’s pick 𝜎 =
𝑛𝑙
𝑛𝑙−1
𝑛𝑙 + 𝑛𝑙−1
−1
= Θ
1
𝑛𝑙−1
min 1,
𝑛𝑙
𝑛𝑙−1
. What happens?
 Inductive assumption - ℎ𝑙−1 2 = Θ 𝑛𝑙−1
 Inductive case - 𝑊𝑙 ∗ → 𝜎 𝑛𝑙−1 + 𝑛𝑙 =
𝑛𝑙
𝑛𝑙−1 
ℎ𝑙 2 = 𝑛𝑙 + 𝑜( 𝑛𝑙)
[Comments – this is a kind of ‘worst case’ derivation – the ≈ is an upper bound ]

---

## Page 48

Deriving muP (condition A2)
Now we need to deal with updates. Suppose we have the update Δ𝑊𝑙 on the weights.
For SGD, on a linear layer, this looks like a rank -one loss -activation outer product.
Δ𝑊𝑙 = −𝜂𝑙∇ℎ𝑙ℓ ℎ𝑙−1
⊤
Thus, Δ𝑊𝑙ℎ𝑙−1 2 = Δ𝑊𝑙 ∗ ℎ𝑙−1 2 . Now note that we have the update
Δhl = WlΔℎ𝑙−1 + Δ𝑊𝑙(ℎ𝑙−1 + Δℎ𝑙−1)
Assuming that the leading order terms don’t cancel, we see that
• WlΔℎ𝑙−1 = Θ( 𝑛𝑙)  from induction assumption + condition A1 argument
• Δ𝑊𝑙ℎ𝑙−1 = 𝛥𝑊𝑙 ∗ 𝑛𝑙−1 from above, thus 𝜟𝑾𝒍 ∗ = 𝚯
𝒏𝒍
𝒏𝒍−𝟏
• Δ𝑊𝑙Δℎ𝑙−1 = 𝑂( Δ𝑊𝑙 ∗ 𝑛𝑙−1)

---

## Page 49

Recall –the key is to pick LR such that Δ𝑊𝑙 ∗ 𝑛𝑙−1 = Θ 𝑛𝑙 . How can we do that?
Suppose that the loss update also scales O(1). Then we can write down..
Δℓ ≈ Θ Δ𝑊𝑙, ∇𝑊𝑙ℓ = Θ Δ𝑊𝑙 𝐹 ∇𝑊𝑙ℓ 𝐹 = Θ Δ𝑊𝑙 ∗ ∇𝑊𝑙ℓ ∗
Where we use the fact that Δ𝑊𝑙 = −𝜂∇𝑊𝑙ℓ in standard SGD updates. 
Now plug in Δℓ = O 1 , Δ𝑊𝑙 ∗ = Θ
𝑛𝑙
𝑛𝑙−1 to get that ∇𝑊𝑙ℓ ∗ = Θ
𝑛𝑙−1
𝑛𝑙
Finally, from the previous slide, recall that Δ𝑊𝑙 = −𝜂𝑙∇ℎ𝑙ℓ ℎ𝑙−1
⊤  and thus
𝜂𝑙 = Θ 𝑛𝑙
𝑛𝑙−1
 
Deriving muP (condition A2) part 2
[with Adam, Δ𝑊𝑙 ∗ 𝑛𝑙−1 = Θ 𝜂𝑙 ] 

---

## Page 50

muP mini recap..
So, what is (baby) muP about? Controlling activations (and changes) via 𝑊 and Δ𝑊
Initialization (stdev): Set to Θ
1
𝑛𝑙−1
min 1,
𝑛𝑙
𝑛𝑙−1
Learning rates: Set to 
𝑛𝑙
𝑛𝑙−1
Compared to ‘standard’ parametrizations – these set  
Initialization: Set to 
1
𝑛𝑙−1
 
Learning rates: Set to Θ(1)
Differences – LR changes for Adam, also init diffs when fanout 𝑛𝑙 < fanin
(for Adam 
1
nl−1
)

---

## Page 51

Deeper dive into muP
Recall – muP is a scaling procedure for hyperparams  (as a function of width)
Attention params
Embedding
Input/output MLP MM
Softmax linear


---

## Page 52

Replicating muP
Q1: Does muP work as claimed? When we scale widths, is optimal LR constant?


---

## Page 53

What is muP robust to?
Modern LMs have many components that deviate from muP’s theory
• Activations – SwiGLU and squared relu
• Batch sizes – Large / small 
• Initialization variations – zero attention, etc.
• RMS norm gains
• Exotic optimizers (Lion)
• Regularizers
Which of these (if any) break muP?

---

## Page 54

What is muP not robust to? RMSnorm gain
In our arch – RMSNorm  has learnable gains. This turns out to break muP
But these gains can be removed with little loss of perf..

---

## Page 55

What is muP not robust to? Exotic optimizers
There are other, exotic optimizers based on just gradient signs. Do they transfer?


---

## Page 56

What is muP not robust to? – (strong) weight decay 
What about strong (0.1) weight decay? – this is maybe the only significant muP failure

---

## Page 57

Is muP useful? At least to some extent..
Overall, muP generally seems useful – insofar that SP is quite a bit more unstable.
Current evidence suggests that muP parametrization / initialization may be easier to tune.

---

## Page 58

Recap: scaling in the wild
What are challenges in scaling ‘in practice’
1. Setting model arch hyperparameters ( width, etc)
2. Setting optimizer hyperparameters (LR, batch)
3. Compute needed to fit the big chinchilla sweep
Some solutions?
1. Assume stability (or use muP)
2. Search for optimal LR / batch in small scale, either keep fixed or predict scaling
3. Use alternative learning schedules (WSD -like)