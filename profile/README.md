# Hyperdimensional Computing: An Open Path to Intelligence That Serves Humanity

> **Frontier Labs are failing humanity.**
>
> Centralized intelligence, closed models, and expensive hardware seem to be the direction
> technology firms have taken ever since *Attention Is All You Need* formed the last, and
> not the final, era of AI models. The concentration of capability into a handful
> of walled republics, measured in gigawatts and guarded by proprietary weights, is not an
> inevitable law of technology. It is a choice taken against the will of almost all developers and technologists.
>
> It is the responsibility of scientists and researchers who see an alternative future to
> create the technology of humanity. Insofar as research and open-source computing can lead to a new era
> of **decentralized, energy-efficient compute that serves humanity**, we must pursue it. We
> must put **people before technology**.
>
> We hold that **hyperdimensional computing (HDC)** and **vector symbolic architectures (VSA)** —
> though lesser known than neural networks — represent the true path to artificial
> superintelligence. What follows is our reasoning: a summary of the foundational research
> *Hyperdimensional Computing for Artificial Intelligence*, followed by the complete body of
> references that underpins it.

---

> ### Provenance
>
> **The technical content of this document is a condensed summary of hundreds of sources, the first of which is :** the
> Ph.D. dissertation *Hyperdimensional Computing for Artificial Intelligence* by **Peter Sutor Jr.**
> (University of Maryland, College Park, 2026), of University of Maryland's Perception and Robotics Group **. All
> equations, algorithms, formalisms, experimental results, and the bibliography are derived from
> that work and the cited publications. Every construction described here (XOR-binding,
> Consensus-Sum bundling, the Dollar of Mexico, the Hyperdimensional Inference Layer, the four
> case studies, ToPos generation, HyPE, and hypermapping) originates with Sutor and his
> co-authors.
>
> **Original to this repository:** the framing manifesto in Section 1 and the closing commentary —
> the stance that decentralized, open, energy-efficient, neuromorphic hyperdimensional computing
> should be pursued *in preference to* centralized, closed, hardware-heavy models. That position
> is this repository's; the underlying science is theirs.
>
> If you cite the technical content, **cite the original work**, not this summary. See
> [Section 11](#11-bibliography) for the full reference list.

---

## Table of Contents

1. [A Manifesto for Decentralized Intelligence](#1-a-manifesto-for-decentralized-intelligence)
2. [What Is Hyperdimensional Computing?](#2-what-is-hyperdimensional-computing)
3. [The Intuition: A Thought Experiment](#3-the-intuition-a-thought-experiment)
4. [Formal Foundations](#4-formal-foundations)
5. [Notable Properties and Constructions](#5-notable-properties-and-constructions)
6. [Case Studies in Hyperdimensional Computing](#6-case-studies-in-hyperdimensional-computing)
7. [Properties of Interest to AI and Machine Learning](#7-properties-of-interest-to-ai-and-machine-learning)
8. [Hypermapping: Factorizing the Algebra](#8-hypermapping-factorizing-the-algebra)
9. [The Future of HDC in AI and ML](#9-the-future-of-hdc-in-ai-and-ml)
10. [Why This Matters](#10-why-this-matters)
11. [Bibliography](#11-bibliography)

---

## 1. A Manifesto for Decentralized Intelligence

The dominant trajectory of modern artificial intelligence is toward **centralization**. A
small number of frontier laboratories train ever-larger models on mega compute clusters,
then release them behind APIs, licenses, and legal defenses. The result is a technology that is:

- **Closed** — weights, training data, and methods are held as trade secrets.
- **Energy-hungry** — training and inference demand power measured in megawatts to gigawatts.
- **Hardware-bound** — progress is gated on exotic accelerators accessible only to the wealthiest.
- **Brittle** — models must be painstakingly hand-engineered for each new task and modality.
- **Opaque** — the reasoning of these systems is difficult to inspect, audit, or trust.

This is not the only possible future. It is a process that is inefficient, dense and expensive. Furthermore, it has not and will
not lead to results any different in model performance than open weight models from China.There exists a computing paradigm 
— one rooted in the statistical geometry of very high-dimensional vectors — that is:

- **Open and reproducible** — built from random vectors and simple, well-understood operations.
- **Extremely energy-efficient** — running on commodity CPUs and emerging neuromorphic hardware.
- **Decentralized by nature** — models are small (megabytes), trainable on a single device,
  in real time, online, and continuously.
- **Data-agnostic and multi-modal** — arbitrary signals can coexist in one shared space.
- **Noise-tolerant and robust** — noise is a *feature*, not a bug.
- **Interpretable and neurosymbolic** — symbols and subsymbolic weights are mutually convertible.

We assert that **hyperdimensional computing (HDC)** and **vector symbolic architectures (VSA)**
are the foundation for a new era of intelligence: decentralized, energy-efficient, and
fundamentally superior to deep neural networks. 

---

## 2. What Is Hyperdimensional Computing?

**Hyperdimensional Computing (HDC)** is a neuromorphic and neurosymbolic computing paradigm.
It projects information into high-dimensional spaces — typically on the order of **`n ≈ 2^14`**
bits — with specific statistical properties. In doing so, it produces a computational framework
atypical of general computing:

- **Noise becomes a feature.** Computation is engineered to be noise-tolerant in response.
- **Computation mirrors biology.** Under the right circumstances, HDC achieves the efficiencies
  and capabilities of biological organisms.
- **Capabilities come for free.** Multi-modality, modal fusion, online/continuous/real-time
  learning, and the erosion of rigid network-architecture constraints are directly attainable.

HDC serves as a **"lingua franca"** between differing architectures — allowing both the
integration of existing models and the downstream utilization of features derived from HDC.
It is the *blood* of a larger AI structure: the common medium through which many disparate
AI and ML subsystems communicate, without hand-engineering each interface.

---

## 3. The Intuition: A Thought Experiment

To motivate HDC, imagine a human being who, from birth, is perpetually attached to a brain
scanning machine. The scanner tracks every individual neuron at all times, producing a growing,
real-time dataset of ground-truth brain activity. Three questions arise:

1. Can you build a new brain that mimics the evolution of the subject's activations over time?
2. How would it simulate the subject's behavior?
3. Can you model all past behaviors perfectly — and what would it take?

**Stripping away the "network."** If the scanner sees every neuron and can differentiate them,
we can index them and lay out the connections along that arrangement. Removing the edges leaves
us with a simple **binary array** — a "brain state" `E_i` of `1`s and `0`s.

**Preserving brain-state transfers.** The natural way to record a change between two brain
states is the bit-wise **Exclusive-OR (XOR)**. Defining a transfer function `∆E_i`:

```
∆E_i = (E_i XOR E_{i+1}) XOR E_i
     = E_i XOR (E_i XOR E_{i+1})
     = (E_i XOR E_i) XOR E_{i+1}
     = 0 XOR E_{i+1}
     = E_{i+1}
```

XOR is associative, commutative, has identity `0`, and is an **involution** — its own inverse.
Thus bit-wise XOR can precisely preserve a brain-state transfer as a *digital transfer function*,
using nothing more than arrays of `1`s and `0`s.

**Consolidating transfers by bundling.** A human brain doesn't store transfers; it produces
them. We can instead aggregate clusters of transfer functions by a **majority vote** per
component. Clear winners become `1` or `0`; ties are broken randomly (introducing unbiased
noise). The result is a single **centroid** brain state summarizing a cluster of behaviors.

**The hidden layer.** To simulate behavior, we introduce the "hidden layer." Input and output
neurons are fewer than interneurons, so we project them into the hidden space using **fixed,
random patterns**: replicate and disperse inputs, and filter outputs by random subsets.
Everything reduces to operations on **fixed-length vectors**.

**Scaling to perfection.** To model all past behaviors, we track the *counts* of `1`s vs `0`s
per component. The binary vector becomes a **signed integer vector**, thresholded to binary when
a decision is required. With enough neurons, we overcome statistical noise. Even individual
neurons can be expanded — approximating each neuron's cellular automata as a finite state
automata — by representing activation/inactivation as fixed random patterns. Connections between
neurons are recorded by XOR-ing a neuron's unique ID pattern into its outputs.

The result is a single high-dimensional vector `B` that approximates the global transfer function
`∆E_i`. Given a first state `E_1`, we recursively compute:

```
Ê_2 = B XOR E_1 ≈ E_2,   Ê_3 = B XOR Ê_2 ≈ E_3,   ...
```

If the vectors are long enough, we predict what the original brain would have done, regularized
by fresh sensory input. **These vectors are called hypervectors.** The power of HDC is the power
of maintaining high-dimensional, noise-tolerant vectors that can recall transfer functions on
demand.

---

## 4. Formal Foundations

### 4.1 Dense Binary Hypervectors

A **dense binary hypervector** is a binary vector of `n` random Bernoulli trials where `n`
approaches *hyperdimensional* size — thousands of bits; typically `n ≈ 2^14` or fewer.

For `k ~ Bern(n, 1/2)` successes (the number of on-bits), `k` follows the Binomial Distribution:

```
P(k) = C(n, k) · (1/2)^k · (1/2)^(n−k) = C(n, k) / 2^n
```

At the maximum, `k = n/2`, the binomial coefficient is `Θ(2^n / √n)`, so `P(k) = Θ(1/√n)`.
At the minimum, `k = 1` or `k = n`, `C(n, k) = Θ(n)`, so `P(k) = Θ(n / 2^n)` — vanishingly small.
**A perfect match has probability `0` as `n → ∞`.**

With mean `μ = n/2` and standard deviation `σ = √n / 2`, the vast majority of vectors cluster
near the mean in an almost-Normal distribution. Consequently:

- It is **almost impossibly unlikely** for two random hypervectors to be similar by more than a
  handful of standard deviations.
- **Similarity becomes a metric of significance.**

**Hamming Distance** measures the distance between binary hypervectors:

```
Ham(A, B) = |A ⊕ B|            (count of 1s in the XOR)
Ham_n(A, B) = |A ⊕ B| / n      (normalized)
```

### 4.2 Algebraic Interpretation

A **field** `(F, +, ·)` satisfies associativity, commutativity, identities, inverses, and
distributivity. Setting `F = {0, 1}` with `+` as the **Consensus Sum** and `·` as **XOR** yields
an **approximate field** — so named because it is only *approximately* distributive, and the
Consensus Sum is only *approximately* associative.

Extending to vectors gives the **approximate algebra** `({0, 1}^n, +_C, XOR)`. Where algebraic
manipulations align with `GF(2)`, information is preserved; where they diverge, **noise** is
produced. With only two field elements, noise completely destroys information in the affected
region:

- **Random vectors** absorb noise optimally — they appear near-orthogonal, indistinguishable
  from random noise. Useful for *storage and retrieval*.
- **Correlated vectors** take on aspects of inverses and identities — uniquely identifiable
  regardless of context. Useful for *compression and consolidation*.

### 4.3 Binding and Bundling

- **Binding** (component-wise XOR) ties two pieces of information together — analogous to
  *multiplication*.
- **Bundling** (bit-wise majority vote) aggregates information where similarity may be present —
  analogous to *summation*.

The **Consensus Sum** at component `i` is:

```
C+_i(V_1, ..., V_m) = 1  if  Σ_j V^i_j  >  m − Σ_j V^i_j
```

i.e., the indicator that the count of `1`s exceeds the count of `0`s. For even `m`, one may add
a random tie-breaker term `V_{m+1}`.

**Properties:**
- Consensus Summation is **commutative** but **not associative** — grouping matters.
- The interaction with XOR causes **non-distributivity** in the strict formal sense.
- Non-associativity yields slightly different results in benign cases and completely different
  results in pathological cases sensitive to the number of voting terms.

### 4.4 Distributive Behavior

Binding a source `S` to a Consensus Sum of bound vectors:

```
S_i ⊕ C+_i(V^i_1, ..., V^i_m) = S_i ⊕ 1[Σ_j V^i_j > m − Σ_j V^i_j]
```

- **All-random case:** The sum interferes randomly across components; output is another random
  string with `E = n/2` ones and zeros.
- **Correlated case:** Terms matching `S` nudge the component toward `S_i`. When this nudging
  overcomes enough standard deviations, it **interferes destructively with the Hamming Distance**.
  If `m` is small enough, even a single matching vector is detectable with high certainty.

### 4.5 Data Structures

A sequence `s = (s_1, ..., s_m)` with dictionary hypervectors `S_i` and positional hypervectors
`P_i` is represented by a **Bound-Bundle (BB)**:

```
S = C+(S_1 ⊕ P_1, ..., S_m ⊕ P_m) = BB((S_1, ..., S_m), (P_1, ..., P_m))
```

- Also works for **object-like data** (attributes bound to values).
- **Sets** (order irrelevant) need no positional encodings — just the Consensus Sum.
- **Reading out** a symbol from superposition is a minimization problem:

```
j = argmin_{S^(j) ∈ H_D} Ham(S ⊕ P_i, S^(j))
```

This reconstruction is relatively expensive — expected, since it is equivalent to a human
verbalizing a memory from a large space of possibilities.

---

## 5. Notable Properties and Constructions

### 5.1 The Dollar of Mexico

A classic demonstration of **analogical inference**. Encode country records:

```
U = (H_Name ⊕ H_US) +_C (H_Currency ⊕ H_Dollar) +_C (H_Capital ⊕ H_WashingtonDC)
M = (H_Name ⊕ H_Mexico) +_C (H_Currency ⊕ H_Peso) +_C (H_Capital ⊕ H_MexicoCity)
```

Binding `U ⊕ M` distributes over the common terms, cancels the shared XORs (involution), and
collapses everything else into a single noise vector `N`:

```
U ⊕ M = (H_US ⊕ H_Mexico) +_C (H_Dollar ⊕ H_Peso)
      +_C (H_WashingtonDC ⊕ H_MexicoCity) +_C N
```

Binding further to `H_Dollar` cancels it, yielding `≈ H_Peso`. An argmin search over the
dictionary overwhelmingly selects `H_Peso`. The system has *inferred* the currency analogy.

### 5.2 The Hyperdimensional Inference Layer (HIL)

Building on the Dollar of Mexico, we can aggregate **prototypes** per class. Binding class
prototypes to random class hypervectors and Consensus Summing them yields a single model `H_M`,
called a **Hyperdimensional Inference Layer (HIL)**:

- **Training:** example hypervectors are aggregated per class into prototypical hypervectors,
  then bound to random class vectors.
- **Inference:** given a novel `X`, compute the trace `X ⊕ H_M`, then argmin-search for the
  best-matching class.

This process mirrors **superposition collapse in quantum mechanics**: probing with `X` disrupts
the superposition, isolating the least-noisy (non-random) signal — the best-matching class.

**Key insight:** class labels are random, and dictionary elements may be random, so the *only*
non-random source of information is the **structure of the encoded input data**. Learning in HDC
is performed directly by aggregation. Inference power is dictated *solely* by how separable the
classes are under the encoding function. **HDC reduces inference, algorithmically, to the encoding
problem.** This is a natural integration point with classical autoencoders and ML.

### 5.3 Computing Properties

Although HDC appears neuromorphic, it is **Turing Complete** under the right assumptions:

- A small Turing Machine and an elementary cellular automaton have been emulated with HDC
  constructions (derived in [11], building on [12], [13]).
- Any **context-free grammar** can be implemented by mapping production rules to random
  hypervectors.
- Any programming language can be converted into a hypervector analogue.
- In principle, implementing **lambda calculus** in hypervector form suffices for Turing
  completeness.

A hypervector can be treated as a very large machine instruction, giving HDC analogues at every
level — software to hardware. HDC is thus best leveraged where **multiple instructions execute in
parallel**, favoring massive parallelism down to the bit level and emerging neuromorphic hardware.

---

## 6. Case Studies in Hyperdimensional Computing

### 6.1 Case Study 1 — Hypervector-based Dynamical Systems [2]

*Life-Long Learning of Semantics and Knowledge.*

This work models **distributional semantics** and knowledge graphs as a dynamic system, relaxing
into a minimum-energy state — analogous to a **Spring-Mass system**.

**Requirements for life-long learning:** a knowledge graph `K = (V, E)`; a binary matrix
`X ∈ {0,1}^{b×n_v}` of vertex representations; a set of **oracles** `O` (supervisory mappings
from raw data to edges); and a dictionary of **directors** `D` that specify how lower-level
abstractions combine into higher-level ones.

**Tension minimization.** Hypervectors are modeled as masses connected by springs. Two forces
oppose each other:

- **Connective force** — pulls vectors together based on shared knowledge-graph connections.
- **Proximal force** — a "reverse gravity" that pushes close vectors apart, regularizing against
  collapse into a singularity.

The **total tension** is the sum of unresolved forces:

```
T(A) = Σ_i Σ_j max( F_conn(A,i,j) + F_prox(A,i,j), 0 )
```

The minimization is solvable by **Monte-Carlo / Simulated Annealing**. Without proximal force,
the system collapses; with it, a non-zero equilibrium is reached quickly — reminiscent of
elliptical orbits or pulley systems settling into equilibrium.

**Analysis:** HDC can build hypervector analogues for arbitrary dynamical systems. Because the
Consensus Sum permits *approximate* solutions, minimization develops **heuristics rapidly** —
approximately linear modulations in high-dimensional space. New information can be integrated in
**real time** into a common semantic space.

### 6.2 Case Study 2 — Symbolic Representation Learning with HDC [4]

This work demonstrates HDC's **neurosymbolic** capabilities using **hashing networks** to produce
short, efficient **hashcodes** for images, which maintain rankability and classification power.

**Three hashing networks compared:**
- **DQN** [14] — deep quantization network; controls quantization error.
- **DCH** [15] — deep Cauchy hashing; penalizes similar pairs beyond a Hamming radius using a
  Cauchy-distribution cross-entropy loss.
- **DTQ** [16] — deep triplet quantization; uses similarity triplets and Group Hard selection.

**Evaluation.** Hashcodes are projected to hyperdimensional length by repetition; a HIL is trained
to understand them. Results on CIFAR-10 [19] and NUS-WIDE-81 [20] show:

- **Bootstrapping:** The HIL reaches near-optimal performance *immediately*, while the hashing
  network converges slowly. The HIL "arrives" at good heuristics early.
- **Hamming Distance erasure:** The HIL largely *erases* the need for a Hamming-distance parameter
  search. Performance is essentially directly proportional to the Hamming Distance; simply choose
  the smallest matching distance.
- **Consensus fusion:** By treating each network as a "Dollar of Mexico" analogy, a consensus
  HIL fuses all three. On CIFAR-10, DTQ alone scores **69%**; consensus scores **79%** — a full
  **10-point gain**.

**Analysis:** HDC captures the *approximate algebra* of the network much faster than the network
itself learns. DTQ benefits most (triplets reveal geometry); DCH benefits least (its Cauchy
projection already resembles the Binomial case at hyperdimensional scale). HDC prefers *diverse*
features and lets each subsystem "vote" with features it is most confident in.

### 6.3 Case Study 3 — "Gluing" Neural Networks Together [5]

*HD-glue (Hyperdimensional Glue).*

This extends symbolic integration to the **neuronal level**: can embeddings from neural networks
be absorbed into hypervectors and fused?

**Capturing embeddings:** Apply `tanh` to normalize embeddings into `[−1, 1]`, quantize each
component, map to **bins**, bind bins to positional hypervectors, and Consensus Sum — producing
a hypervector representation of the embedding.

**Distance Preserving Quantization.** Naive bin interpolation gives poor distance preservation.
Instead, using **binary codes** (Gray Codes [21, 22] / Reflective Binary Codes, Thermometer/Unary
Codes [23], or Scatter Codes [24]) creates optimally spaced bins whose distances map cleanly to
Hamming Distance — approaching the identity function and using the full Hamming range.

**Results across experiments:**
- **MNIST [25] consensus:** HD-glue outperforms KNN [26], SVMs [27], Decision Trees [28],
  Random Forests [29], AdaBoost [30], Naive Bayes [31] — *especially in few-shot settings*.
- **Online learning:** No catastrophic forgetting, regardless of class order, thanks to the
  commutativity of Consensus Summation and constant-time thresholding.
- **CIFAR-10 [19]:** More models → better consensus. HDC shows an **affinity for consensus**.
- **CIFAR-100 [19]:** With VGG [32] and ResNet [33] variants, **diversity of architectures**
  outperforms many same-architecture models with different initializations.
- **Variable embedding sizes:** HD-glue is unique in gracefully handling differing embedding
  lengths, outperforming individual networks after just ~500 examples.

**Analysis:** HDC bridges neuronal and symbolic representations, is architecture-agnostic once
embeddings are converted, and is the best choice for **post-hoc gluing** of networks into
consensus architectures.

### 6.4 Case Study 4 — Hyperdimensional Active Perception [3]

*The most significant contribution — raw efficiency at the edge.*

**Active perception** [34–36]: an agent actively engages the environment in an action-perception
feedback loop to maximize information. Here it is applied to **ego-motion** prediction.

**Event cameras** [37]: neuromorphic sensors that emit asynchronous **events** when pixel
intensity changes exceed a threshold. The output is sparse, high-temporal-resolution, and
motion-correlated. Agent motion "lights up" edges via homography — enabling a feedback loop.

**Encoding:** Events are binned into **"time images."** RGB intensities are quantized from a
random start hypervector to a random end hypervector; color channels are bound to random
representations; positional (row/column) hypervectors encode location. Consensus Sum over
non-black events forms the time-image hypervector.

**Velocities:** `x, y, z` components are quantized like color channels; prototypes are learned
per degree of freedom.

**Learning pipeline:** Time images and velocities are bundled into **"time slices"** — a queue-like
moving average that regularizes predictions into smooth, less drastic changes.

**Performance (MVSEC dataset [38]):**
- Comparable accuracy to standard convolutional solutions.
- Runs on a **single classical CPU**.
- Trains in **under 1 second** after ~15 seconds of data.
- Occupies only **several MBs**.
- Inference at **100s per second**.

**Analysis:** The stunning efficiency stems from the *combination* of HDC's preference for sparse
data and the event camera's native sparsity — **the neuromorphic hardware essentially solves the
encoding problem for HDC.** Dense RGB grids are not how biology perceives; computation is sparse
and hardware-dependent. This exposes an uncomfortable truth: most modern ML is not equipped to
exploit hardware-solved problems, whereas HDC is (see also [6] on ultra-efficient edge AI).

---

## 7. Properties of Interest to AI and Machine Learning

### 7.1 Vector Symbolic Sub-Object Classifiers as Manifold Analogues [7]

Using **Category Theory** and **Topos Theory**, this work shows HDC can approximate arbitrary
topological structures.

- A **Category** consists of objects and morphisms (relationships between source and target).
- A **Topos** is a category capturing geometry, logic, and transforms — the end-all generalization
  of topological structure. If something is a Topos, it *is* a topological structure.
- The key property: **sub-object inheritance is preserved under transforms** — drastic jumps may
  occur, but object inheritance transfers. This implies a definite way to classify sub-objects.

**Generic hyperdimensional separator (Algorithm 1):** A binary-search-like loop finds the ideal
Hamming Ball that recursively partitions a set into equally sized halves — maximizing information
gain.

**Generic hyperdimensional pushout (Algorithm 2):** Leverages Consensus Summation. If elements
can be disambiguated via the HIL argmin search, they can be pushed out into larger sub-objects.
Once disambiguation fails, we stop.

**Topos generation (Algorithm 3):** Recursively separate top-down until leaves are reached, then
push out bottom-up to compress as much as possible.

**Utility functions (Algorithm 4):** `Hyperdesic` and `LocalMetric` measure distance along the
ToPos structure, deforming the Hamming metric. The working example (genetic lineage) shows the
ToPos reveals far richer similarity structure than raw Hamming Distance, while leaving random
vectors untouched (hyperdimensional orthogonality).

**Analysis:** Manifolds underlie nearly all of ML — neural networks are manifold-learning
machines. Since arbitrary manifolds can be embedded into hyperdimensional spaces (given enough
samples) to create custom metrics, **any architecture's underlying manifold can be embedded in
HDC.** This explains the heuristic-embedding behavior seen across Case Studies 1–3.

### 7.2 HyPE: Hyperdimensional Propagation of Error [8]

**HyPE = Hyperdimensional Propagation of Error.** This work discovers a feedback mechanism
analogous to **back-propagation**, formalized as a **Generalized Expectation Maximization (GEM)**
algorithm.

**Error definition.** For model `M`, training example `X`, and class `Y`:

```
E_M(X, Y) := X ⊕ M ⊕ Y
err_M(X, Y) := |E_M(X, Y)| = d(0̄, E_M(X, Y))
```

This measures how far the noisy superpositional collapse is from a perfect match. Correct outputs
do *not* resemble a Binomial distribution of random coin tosses; the strength of correlation is
characterized by the reciprocal of the Binomial PMF. Minimizing this likelihood **"hypes" up
non-random features** — hence the name.

**Algorithm (single layer, Algorithm 5):**
1. Train model `M^(k)` on data `D^(k)`.
2. Partition data into a subset `D̂^(k)` the model handles well, and `D^(k+1)` it doesn't.
3. Compute a **HyPE-weighted** model `M̂^(k)` biased toward favorable examples.
4. Iterate while test performance improves.

**Parallel layers and stacking:**
```
L := {M^(0), M^(1), ..., M^(k)}      (a layer)
S := {L, L', ...}                     (a stack)
```
Repeated iteration disperses global error across parallel models, lowering average Hamming error
and increasing prediction odds. Uninformative components can be discarded for online HyPE.

**Interpretation as GEM:** The canonical GEM loop (estimate missing values → estimate parameters
→ estimate missing values → estimate parameters) maps directly to HyPE's encode → learn → prune
→ relearn loop. HyPE performs *Minimization-Maximization* rather than Maximization-Maximization.
Boundedness in dense binary space + [42] guarantee **local convergence**; global optimality is not
guaranteed absent differentiability assumptions. Correctness corresponds to training a layer until
it behaves as a **maximal sieve** — a collection of covering models preserving information.

**Experiments (Fashion-MNIST [40]):** With per-pixel and interpolation encodings, HyPE-boosted
models exceed classic HDC at low lengths, converging as lengths grow. Each trial repeated 10 times
with random dictionaries.

**Analysis:** HDC carries an implicit objective function based on expectation maximization.
Aggregation — the Consensus Summation — is *the workhorse of learning*. Structuring summations to
reduce information loss is a promising strategy, pursued in the next section.

---

## 8. Hypermapping: Factorizing the Algebra

**Hypermapping** (a novel contribution of the dissertation) reveals that class prototypes contain
three distinguishable kinds of components:

| Component type | Behavior | Role | Neuron analogy |
|---|---|---|---|
| **Useless** (red) | Same bit for all classes | No classification power | Maximally *polysemous* |
| **Perfect** (green) | Special to exactly one class | Maximal classification power | *Monosemous* |
| **Balanced** (blue) | Equal counts of `0`/`1` | Ideal HyPE partition | Lower bound on classification speed |

**Causal factorization.** There is a strict causal order:

1. **Useless classifiers** fingerprint *which problem* is being solved. Given `m` datasets,
   argmin over their useless-classifier sets identifies the task.
2. **Perfect classifiers** then identify the *class* most accurately.
3. **In-between classifiers** are consulted last, only when top class Hamming distances are close
   enough that noise could cause misclassification.

Factorizing for all `C(c, i)` cases from `i = 2` up to the inflection point `⌊c/2⌋` relaxes perfect
classifiers to sway decisions. **This reverse-engineers the remainder of HyPE** — but with an
explicit causal ordering HyPE lacked. The result is a proper, causally correlated metric augmenting
Hamming Distance, equivalent to a Topos computed more efficiently than via HyPE.

**Clustering of prototypes.** Out-of-distribution prototypes can be clustered; each cluster gets
its own metric. This resembles Topos generation but **tailors the sub-hypervector length throughout
the process** — accounting for slack introduced by normalization.

**The startling revelation.** A hypervector model was a **neural network all along**. Hypermapping
recreates a layered architecture fueled by backpropagation-like error propagation (HyPE), captured
by a Topos (metric) approximating an underlying manifold. Therefore, **all neural networks can be
modeled by ideal, super-efficient hyperdimensional analogues.**

---

## 9. The Future of HDC in AI and ML

Three broad research directions:

1. **Extension to non-binary HDC.** Fourier-based formulations can yield richer algebras than binary
   spaces alone — especially for image and continuous-value data. All data can eventually be
   abstracted into dense binary hypervectors, but the early stages of computation may benefit from
   other algebras, translated gradually.

2. **Approximation of neural networks into fully hypermapped VSAs.** Fully end-to-end
   **knowledge distillation** may be within reach: HD-glue [5] approximates embeddings, HyPE [8]
   provides feedback like back-propagation, and hypermapping regularizes error propagation
   causally. The missing piece: **no general method exists to convert raw, arbitrary data directly
   into useful hypervectors** without an external encoder or specialized hardware. Hypermapping is
   the most promising direction to solve this.

3. **Neuromorphic hardware for HDC.** Event cameras virtually solved ego-motion for HDC [3]. HDC
   excels at abstracting away precision. Human vision is itself hyperdimensionally motivated:
   log-polar retinal geometry gives scale/rotation invariance, and **saccades** (rapid eye
   movements) provide translation invariance — a set of saccades is order-tolerant, which is exactly
   what Consensus Summation handles. Active perception via saccades is a rich future application.

---

## 10. Why This Matters

HDC has been characterized as being to AI and ML **what the Fourier Transform is to so many other
tasks** — it offers efficiency avenues that should be tried as first solutions. It is not necessary
to use energy-hungry, closed, centralized Large Language Models to do work that simple, open,
decentralized hypervector constructions can accomplish.

The evidence, summarized:

- **It works** — across semantics, symbolic reasoning, network fusion, robotics, and manifold
  learning.
- **It is efficient** — megabytes of model, sub-second training, real-time inference on CPUs.
- **It is open** — built from random vectors and elementary operations.
- **It is unifying** — a common "lingua franca" for modalities, architectures, and modalities of
  computation.
- **It is a path forward** — toward decentralized, energy-efficient intelligence that **puts people
  before technology**.

In the best of cases, this document serves in inspiring future work into Hyperdimensional
Computing — for AI, ML, or general computation.

---

## 11. Bibliography

```
[1]  Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed
     representation with high-dimensional random vectors. Cognitive Computation, 1(2), 139-159.

[2]  Sutor Jr, P., Summers-Stay, D., & Aloimonos, Y. (2018, July). A computational theory for
     life-long learning of semantics. In International Conference on Artificial General
     Intelligence (pp. 217-226). Cham: Springer International Publishing.

[3]  Mitrokhin, A., Sutor, P., Fermüller, C., & Aloimonos, Y. (2019). Learning sensorimotor
     control with neuromorphic sensors: Toward hyperdimensional active perception. Science
     Robotics, 4(30), eaaw6736.

[4]  Mitrokhin, A., Sutor, P., Summers-Stay, D., Fermüller, C., & Aloimonos, Y. (2020). Symbolic
     representation and learning with hyperdimensional computing. Frontiers in Robotics and AI,
     7, 63.

[5]  Sutor, P., Yuan, D., Summers-Stay, D., Fermuller, C., & Aloimonos, Y. (2022, July). Gluing
     neural networks symbolically through hyperdimensional computing. In 2022 International Joint
     Conference on Neural Networks (IJCNN) (pp. 1-10). IEEE.

[6]  Amrouch, H., Imani, M., Jiao, X., Aloimonos, Y., Fermuller, C., Yuan, D., ... & Sutor, P.
     (2022, October). Brain-inspired hyperdimensional computing for ultra-efficient edge AI. In
     2022 International Conference on Hardware/Software Codesign and System Synthesis
     (CODES+ISSS) (pp. 25-34). IEEE.

[7]  Faraone, R., Sutor, P., Fermüller, C., & Aloimonos, Y. (2024, June). Vector symbolic
     sub-objects classifiers as manifold analogues. In 2024 International Joint Conference on
     Neural Networks (IJCNN) (pp. 1-10). IEEE.

[8]  Sutor, P., Faraone, R., Fermüller, C., & Aloimonos, Y. (2025, August). HyPE: Hyperdimensional
     Propagation of Error. In International Conference on Artificial General Intelligence
     (pp. 241-251). Cham: Springer Nature Switzerland.

[9]  Plate, T. A. (1995). Holographic reduced representations. IEEE Transactions on Neural
     Networks, 6(3), 623-641.

[10] Bick, C., & Rabinovich, M. I. (2009). Dynamical origin of the effective storage capacity in
     the brain's working memory. Physical Review Letters, 103(21), 218101.

[11] Kleyko, D., Davies, M., Frady, E. P., Kanerva, P., Kent, S. J., Olshausen, B. A., ... &
     Sommer, F. T. (2022). Vector symbolic architectures as a computing framework for emerging
     hardware. Proceedings of the IEEE, 110(10), 1538-1571.

[12] Cook, M. (2004). Universality in elementary cellular automata. Complex Systems, 15(1), 1-40.

[13] Neary, T., & Woods, D. (2009, September). Small weakly universal Turing machines. In
     International Symposium on Fundamentals of Computation Theory (pp. 262-273). Berlin,
     Heidelberg: Springer Berlin Heidelberg.

[14] Yue, C., Long, M., Wang, J., Han, Z., & Wen, Q. (2016, February). Deep quantization network
     for efficient image retrieval. In Proc. 13th AAAI Conf. Artif. Intell. (pp. 3457-3463).

[15] Zhu, H., Long, M., Wang, J., & Cao, Y. (2016, March). Deep hashing network for efficient
     similarity retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence
     (Vol. 30, No. 1).

[16] Liu, B., Cao, Y., Long, M., Wang, J., & Wang, J. (2018, October). Deep triplet quantization.
     In Proceedings of the 26th ACM International Conference on Multimedia (pp. 755-763).

[17] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep
     convolutional neural networks. Advances in Neural Information Processing Systems, 25.

[18] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009, June). ImageNet: A
     large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and
     Pattern Recognition (pp. 248-255). IEEE.

[19] Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

[20] Chua, T. S., Tang, J., Hong, R., Li, H., Luo, Z., & Zheng, Y. (2009, July). NUS-WIDE: A
     real-world web image database from National University of Singapore. In Proceedings of the
     ACM International Conference on Image and Video Retrieval (pp. 1-9).

[21] Gray, F. (1953). Pulse code communication. United States Patent Number 2632058.

[22] Savage, C. (1997). A survey of combinatorial Gray codes. SIAM Review, 39(4), 605-629.

[23] Kak, S. (2016). Generalized unary coding. Circuits, Systems, and Signal Processing, 35(4),
     1419-1426.

[24] Smith, D., & Stanford, P. (1990, June). A random walk in Hamming space. In 1990 IJCNN
     International Joint Conference on Neural Networks (pp. 465-470). IEEE.

[25] Deng, L. (2012). The MNIST database of handwritten digit images for machine learning
     research [best of the web]. IEEE Signal Processing Magazine, 29(6), 141-142.

[26] Fix, E. (1985). Discriminatory analysis: Nonparametric discrimination, consistency
     properties (Vol. 1). USAF School of Aviation Medicine.

[27] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.

[28] Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.

[29] Ho, T. K. (1995, August). Random decision forests. In Proceedings of 3rd International
     Conference on Document Analysis and Recognition (Vol. 1, pp. 278-282). IEEE.

[30] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning
     and an application to boosting. Journal of Computer and System Sciences, 55(1), 119-139.

[31] Rish, I. (2001, August). An empirical study of the naive Bayes classifier. In IJCAI 2001
     Workshop on Empirical Methods in Artificial Intelligence (Vol. 3, No. 22, pp. 41-46).

[32] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image
     recognition. arXiv preprint arXiv:1409.1556.

[33] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
     In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
     (pp. 770-778).

[34] Bajcsy, R. (1988). Active perception. Proceedings of the IEEE, 76(8), 966-1005.

[35] Bajcsy, R., Aloimonos, Y., & Tsotsos, J. K. (2018). Revisiting active perception. Autonomous
     Robots, 42(2), 177-196.

[36] Aloimonos, Y. (2013). Active Perception. Psychology Press.

[37] Mitrokhin, A., Fermüller, C., Parameshwara, C., & Aloimonos, Y. (2018, October). Event-based
     moving object detection and tracking. In 2018 IEEE/RSJ International Conference on
     Intelligent Robots and Systems (IROS) (pp. 1-9). IEEE.

[38] Zhu, A. Z., Thakur, D., Özaslan, T., Pfrommer, B., Kumar, V., & Daniilidis, K. (2018). The
     multi-vehicle stereo event camera dataset: An event camera dataset for 3D perception. IEEE
     Robotics and Automation Letters, 3(3), 2032-2039.

[39] Faraone, R. (2025). Analogies, Metaphors, Allegories: Categorial Architectures of General
     Intelligence. Università di Parma, Ph.D. thesis.

[40] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for
     benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.

[41] Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.).
     John Wiley & Sons.

[42] Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. The Annals of
     Statistics, 11(1), 95-103.

[43] Goldblatt, R. (1984). Topoi: The Categorial Analysis of Logic. Studies in Logic and the
     Foundations of Mathematics, Vol. 98, North-Holland.

[44] Freyd, P. J., & Scedrov, A. (1990). Categories, Allegories. Mathematical Library, Vol. 39,
     North-Holland.

[45] Goldblatt, R. (2014). Topoi: The Categorial Analysis of Logic (Vol. 98). Elsevier.

[46] Caramello, O. (2018). Theories, Sites, Toposes: Relating and Studying Mathematical Theories
     Through Topos-Theoretic 'Bridges'. Oxford University Press.

[47] Ehresmann, A. C., & Vanbremeersch, J. P. (2007). Memory Evolutive Systems: Hierarchy,
     Emergence, Cognition (Vol. 4). Elsevier.

[48] Baas, N. A. (1992). Hyperstructures — a framework for emergence, hierarchies and complexity.
     In Proceedings du Congrès sur l'émergence dans les modèles de la cognition (pp. 67-93).

[49] Moore, E. H., & Smith, H. L. (1922). A general theory of limits. American Journal of
     Mathematics, 44(2), 102-121.

[50] Herrlich, H., & Strecker, G. E. (2007). Category Theory. Sigma Series in Pure Mathematics,
     Vol. 1.

[51] Johnstone, P. T. (2002). Sketches of an Elephant: A Topos Theory Compendium: Volume 2
     (Vol. 2). Oxford University Press.

[52] Greco, G., Liang, F., Moortgat, M., Palmigiano, A., & Tzimoulis, A. (2019). Vector spaces as
     Kripke frames. arXiv preprint arXiv:1908.05528.

[53] Hofmann, D., Seal, G. J., & Tholen, W. (Eds.). (2014). Monoidal Topology: A Categorical
     Approach to Order, Metric, and Topology (Vol. 153). Cambridge University Press.

[54] Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories. Rendiconti del
     Seminario Matematico e Fisico di Milano, 43(1), 135-166.
```

---

<p align="center">
  <strong>Centralized intelligence is a choice — not a destiny.</strong><br/>
  Build open. Build decentralized. Build efficient. <em>Put people before technology.</em>
</p>
