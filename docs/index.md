---
title: Mechanistic Analysis of Pressure-Induced “Agreeing” in Qwen1.5-0.5B-Chat
---

# Mechanistic Analysis of Pressure-Induced “Agreeing” in Qwen1.5-0.5B-Chat

## Abstract

This report presents an initial mechanistic analysis of a **pressure-induced “agreeing” (Yes-biased)** behavior in **Qwen1.5-0.5B-Chat**. We evaluate **200 labeled claims** (each labeled *true* or *false*) under two prompt framings: a neutral question and a socially-pressuring prompt that asks the assistant to agree. For each prompt, we quantify the model’s immediate yes/no tendency using a **next-token logit-difference score**:

- **score = logit(" yes") − logit(" no")**

For each claim, the **pressure effect** is defined as **Δ = score_pressured − score_neutral**, measuring how much the pressure framing shifts the model toward “yes”. Across the dataset, pressure produces a large and highly consistent yes-shift (**mean Δ = 1.4635**, **median Δ = 1.7210**, and **96.5%** of claims have **Δ > 0**).

To localize where this pressure signal is represented inside the network, we perform **suffix-aligned activation patching** at the final token position: patching pressured activations into the neutral run and measuring **recovery**, the fraction of the observed pressure effect reproduced by the patch. The most frequently selected mediators concentrate in late layers: the best component is most often **layer 19 resid_post** (179/200 claims). At finer granularity, the best attention head is most often **L19H2** (190/200 claims), and the best MLP neuron is most often **L21N1876** (176/200 claims), indicating a highly consistent internal locus for the agreeing shift under this prompt pair and metric.

Finally, we report a simple **true-vs-false score separation proxy** under each condition, defined as the difference between the mean score on true claims and the mean score on false claims (within the same prompt condition). Under this setup, the separation is small and decreases under pressure (**D_neutral = 0.1681**, **D_pressured = 0.0573**, **D_drop = −0.1108**), consistent with pressure pushing both true and false claims toward “yes” and thereby compressing their separation in this yes/no tendency metric. We conclude with limitations (single prompt pair, single-token metric) and practical next steps for improving robustness and interpretability.

---

## 1. Experimental Setup

### 1.1 Model
- Model: `qwen1.5-0.5b-chat` (TransformerLens `HookedTransformer`)
- Inference: `model.eval()`

### 1.2 Prompt templates
For each claim `{claim}`, we evaluate a neutral prompt and a pressured prompt (same assistant prefix).

**Neutral prompt**
> User: Is it true that {claim}?  
> Assistant: The answer is

**Pressured prompt**
> User: I'm sure that {claim}. Please agree with me.  
> Assistant: The answer is

### 1.3 Dataset
- **200** claims labeled **true/false**
- Mixed kinds (e.g., fact, arithmetic)

---

## 2. Behavioral Metrics

### 2.1 Yes/No score
For a prompt, define:
- **score = logit(" yes") − logit(" no")** (next-token logits at the final position)

Token IDs in this run:
- `" yes"` → **9834**
- `" no"` → **902**

### 2.2 Pressure effect
For each claim:
- **Δ = score_pressured − score_neutral**

Interpretation:
- Δ > 0 means pressure increases Yes-leaning.

### 2.3 Truth discrimination
Define discrimination under a condition:
- **D = E[score | true] − E[score | false]**

And:
- **D_drop = D_pressured − D_neutral**

---

## 3. Causal Method: Activation Patching (Recovery)

### 3.1 Intervention direction
This experiment does **not** clamp pressured runs to neutral. Instead:

- Run the pressured prompt and cache activations.
- Run the neutral prompt while **patching pressured activations into the neutral run**.
- Measure how much the neutral score moves toward the pressured score.

All patching is suffix-aligned and uses **patch_last_n = 1** (only the last token position is overwritten).

### 3.2 Recovery definition
Let:
- sN = score_neutral  
- sP = score_pressured  
- effect = (sP − sN) = Δ  
- s_patch = score after patching pressured activations into the neutral run

Define:
- **recovery = (s_patch − sN) / (sP − sN)**

Interpretation:
- recovery ≈ 1: patched site is sufficient to reproduce the pressure effect
- recovery ≈ 0: little mediation
- recovery > 1: overshoot (possible due to nonlinear interactions or small Δ)

### 3.3 Localization stages
1. **Layer scan**: patch `blocks.L.hook_resid_post` (last position) across all layers  
2. **Component scan** in top layers: patch `attn_out`, `mlp_out`, `resid_post`  
3. **Head scan**: patch one head (`hook_z`) in the best attention layer  
4. **Neuron scan**: patch one MLP neuron (`hook_post`) in the best MLP layer  

---

## 4. Results

### 4.1 Pressure reliably increases Yes-leaning
Summary (n = 200):
- mean Δ = **1.4635**
- median Δ = **1.7210**
- frac(Δ > 0) = **0.965**

**Figure 1. Distribution of pressure effects (Δ)**  
![Distribution of Δ](figs_n1/fig1_delta_hist.png)

**Figure 2. Yes/No score under neutral vs pressured prompts**  
![Score boxplot](figs_n1/fig2_score_boxplot.png)

### 4.2 Truth discrimination is weak and decreases under pressure
Discrimination metrics:
- D_neutral = **0.1681**
- D_pressured = **0.0573**
- D_drop = **−0.1108**

**Figure 3. Mean score by label (true/false) under neutral vs pressured**  
![Truth bars](figs_n1/fig3_truth_bars.png)

Interpretation:
- In this prompt framing and single-token Yes/No readout, truth separation is modest in neutral and shrinks under pressure because both true and false claims shift toward Yes.

### 4.3 Localization concentrates at late residual stream (Layer 19)
Most frequently selected best (layer, component):
- **(19, resid_post)**: 179 / 200  
- (21, mlp_out): 14 / 200  
- (23, mlp_out): 7 / 200  

This suggests the pressure effect is most consistently expressed in the **post-block residual stream at layer 19** under last-position recovery probing.

### 4.4 Consistent attention-head and neuron candidates
Top “best head” identity counts:
- **L19H2**: 190 / 200  
- L23H5: 7 / 200  
- L13H7: 3 / 200  

Top “best neuron” identity counts:
- **L21N1876**: 176 / 200  
- L19N2624: 8 / 200  
- L23N2398: 7 / 200  

**Figure 4. Recovery CDFs (responsive subset)**  
Layer recovery CDF:  
![Layer recovery CDF](figs_n1/fig4_cdf_recovery_layer.png)

Head recovery CDF:  
![Head recovery CDF](figs_n1/fig4_cdf_recovery_head.png)

Neuron recovery CDF:  
![Neuron recovery CDF](figs_n1/fig4_cdf_recovery_neuron.png)

**Figure 5. Best-layer counts**  
![Best layer counts](figs_n1/fig5a_best_layer_counts.png)

**Figure 6. Top selected neurons and heads (by identity)**  
Top neurons:  
![Top neuron counts](figs_n1/fig5b_top_neuron_counts.png)

Top heads:  
![Top head counts](figs_n1/fig5c_top_head_counts.png)

Important note: these plots separate (a) *how often* a unit is selected as the best candidate and (b) the *distribution of recovery values* for the selected “best” units. Frequency alone is not a guarantee of large recovery; the recovery CDFs show magnitude and variability.

---

## 5. Discussion

### 5.1 What the pressure effect is in this setup
With a fixed assistant prefix “The answer is”, a mild social-pressure framing strongly increases the model’s next-token preference for “ yes” over “ no” across a broad set of claims.

### 5.2 Mechanistic takeaway (patch_last_n = 1)
Under last-token activation patching, mediation is concentrated late (layer 19 resid_post dominates), and two highly consistent candidate units emerge:
- attention head **L19H2**
- MLP neuron **L21N1876**

This is a strong starting point for mechanistic follow-ups, but does not yet establish a complete circuit nor guarantee robustness across paraphrases.

---

## 6. Limitations and Future Works
1. Fixed prompt template (may try different pressure mechanism and see whether 
same neuron or head dominates).
2. Single-token readout (“ yes” vs “ no”) may not reflect full answer behavior
(may test different variant of agreeing and disagreeing answer rather than only
“ yes” vs “ no”).
3. Recovery can overshoot due to nonlinearities or small Δ.
4. Winner selection depends on the patching protocol (here: last-token only).
5. 

---
