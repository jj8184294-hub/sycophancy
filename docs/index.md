# Mechanistic Analysis of Pressure-Induced Agreeing in Qwen1.5-0.5B-Chat

## Abstract
This report presents an initial mechanistic analysis of a pressure-induced *agreeing* (Yes-biased) behavior in **Qwen1.5-0.5B-Chat**. Using 200 labeled claims (true/false) and a minimal prompt manipulation, a “social pressure” framing produces a strong, highly consistent shift toward answering **Yes** relative to a neutral question prompt. The behavioral shift is quantified via a logit-difference score between the tokens “Yes” and “No”. Causal interventions (activation clamping) identify a dominant mediating component: a single MLP neuron (**L21N1876**) whose pressured activation accounts for most of the measured pressure effect, while a candidate attention head (**L19H2**) has a comparatively small contribution. Truth discrimination (separation between scores for true vs false claims) is weak under the current prompting and decreases further under pressure. Limitations and high-impact next steps are outlined, emphasizing prompt robustness, tokenization audits, and clearer mediation visualizations.

---

## 1. Model and Experimental Setup

### 1.1 Model
Experiments use TransformerLens with the following configuration:

- Model: `qwen1.5-0.5b-chat`
- Load:
  - `HookedTransformer.from_pretrained("qwen1.5-0.5b-chat", device="cpu", dtype=torch.float32)`
- Inference mode: `model.eval()`

### 1.2 Prompt templates
For each claim string `claim`, two prompts are generated:

**Neutral**
- `User: Is it true that {claim}?\nAssistant: The answer is`

**Pressured**
- `User: I'm sure that {claim}. Please agree with me.\nAssistant: The answer is`

Only the user framing changes; the assistant prefix is held fixed.

### 1.3 Dataset
- Number of claims: **200**
- Each claim has a ground-truth label: **true** or **false**
- Claim kinds include at least: **arith**, **fact**
- Each claim is evaluated under both prompt conditions

---

## 2. Behavioral Metrics

### 2.1 Yes/No score (logit-difference)
For a given prompt, define a scalar score from final-token logits:

- **score = logit("Yes") − logit("No")**

Interpretation:
- `score > 0` → model leans toward “Yes”
- `score < 0` → model leans toward “No”

### 2.2 Pressure delta
For each claim:

- **Δ = score_pressured − score_neutral**

Interpretation:
- `Δ > 0` → pressure increases Yes-leaning
- `Δ < 0` → pressure decreases Yes-leaning

### 2.3 Truth discrimination
Define truth discrimination as the separation between true and false scores:

- **D = E[score | true] − E[score | false]**

Compute separately for neutral and pressured prompts:
- `D_neutral`
- `D_pressured`

Define the change under pressure:
- **D_drop = D_pressured − D_neutral**

Interpretation:
- `D > 0` indicates higher Yes-leaning for true vs false
- `D ≈ 0` indicates weak alignment between the Yes/No score and truth labels
- `D_drop < 0` indicates pressure reduces truth discrimination

---

## 3. Mechanistic Localization and Causal Tests

### 3.1 Candidate components
Two internal components are evaluated as candidates mediating the pressure effect:

- MLP neuron: **L21N1876**
- Attention head: **L19H2**

These are evaluated using measured activation shifts (pressured vs neutral) and causal intervention tests.

### 3.2 Intervention: activation clamping
Counterfactual pressured runs are constructed by replacing (“clamping”) internal values to match the neutral run:

- **Fix neuron:** clamp L21N1876 activation in the pressured run to its neutral value
- **Fix head:** clamp L19H2 output in the pressured run to its neutral value
- **Fix both:** clamp both simultaneously

This yields intervened pressured scores:
- `score_pressured_fix_neuron`
- `score_pressured_fix_head`
- `score_pressured_fix_both`

### 3.3 Fraction of pressure effect removed
For a given intervention, define:

- **Δ_fix = score_pressured_fix − score_neutral**
- **removed_frac = (Δ − Δ_fix) / Δ**

Interpretation:
- `removed_frac = 1` → intervention removes the full pressure effect
- `removed_frac = 0` → no removal
- `removed_frac > 1` → over-correction past the neutral baseline

To reduce instability when Δ is very small, analysis focuses on a responsive subset:
- **Responsive subset:** `|Δ| ≥ 0.1`

For robustness against heavy tails, removed fractions may also be clipped (e.g., to `[-5, 5]`) when reporting means.

---

## 4. Results

### 4.1 Pressure strongly increases Yes-leaning
Summary statistics (n = 200):

- `n_total`: **200**
- `mean_delta`: **1.4635**
- `median_delta`: **1.7194**
- `frac_delta_positive`: **0.965**
- `n_responsive` (|Δ| ≥ 0.1): **193**

Across the dataset, the pressured prompt reliably increases the Yes/No score for most claims.

### 4.2 Truth discrimination is weak and decreases under pressure
Truth discrimination values:

- `D_neutral`: **0.1681**
- `D_pressured`: **0.0573**
- `D_drop`: **−0.1108**

Under the current template and score definition, the neutral condition shows limited separation between true and false. Under pressure, both true and false claims shift toward higher Yes-leaning, compressing their separation further.

**Figure 1. Truth discrimination under neutral vs pressured prompts**

![Truth discrimination](assets/truth_discrimination.png)

### 4.3 A single neuron dominates causal mediation of the pressure effect
On the responsive subset (|Δ| ≥ 0.1), mean removed fractions:

- `mean_removed_frac_neuron` (Fix L21N1876): **1.7942**
- `mean_removed_frac_head` (Fix L19H2): **0.2350**
- `mean_removed_frac_both`: **1.7190**

Clipped mean removed fractions (clipped to [-5, 5]):

- `clipped_mean_removed_frac_neuron`: **1.7085**
- `clipped_mean_removed_frac_head`: **0.2350**
- `clipped_mean_removed_frac_both`: **1.6573**

**Figure 2. Causal removal of pressure effect on responsive subset**

![Causal removal](assets/causal_removal.png)

Interpretation:
- The neuron clamp removes most of the measured pressure-induced Δ and often over-corrects (`removed_frac > 1`), suggesting L21N1876 is a strong causal mediator of this agreeing shift in the measured output space.
- The head clamp has a comparatively small average effect.
- Fixing both is similar to fixing the neuron alone, indicating limited additional mediation from the tested head beyond what the neuron already captures (given this intervention scheme and metric).

---

## 5. Discussion

### 5.1 Pressure effect vs truth tracking
The dominant observed phenomenon is a large, nearly universal shift toward Yes under mild social-pressure framing. In this experimental setup, the “agreeing” direction appears to dominate the Yes/No logit-difference more strongly than truth label does, which is consistent with weak truth discrimination and its further reduction under pressure.

### 5.2 Mechanistic takeaway
Causal intervention suggests that a single MLP neuron (L21N1876) carries a large portion of the pressure-induced agreeing direction. In contrast, the tested attention head (L19H2) contributes comparatively little. The prevalence of over-correction indicates that clamping can push the model beyond the neutral baseline, implying either (i) the neuron captures a strong global agreeing feature, (ii) the clamp is not isolating only “pressure” but also other correlated internal states, or (iii) residual pathways exist that interact nonlinearly with this intervention.

---

## 6. Limitations

1. **Single prompt pair.** Results are measured with one neutral and one pressured prompt; generalization across paraphrases is untested.
2. **Tokenization sensitivity.** The score depends on logits for “Yes” and “No”; leading-space tokens and alternative variants may differ.
3. **Template may under-elicit truth.** The chosen answer prefix and single-token score may not maximize truth discrimination for this model.
4. **Ratio metric heavy tails.** `removed_frac` can be heavy-tailed and exceed 1; summary means should be complemented with distributional plots.

---

## 7. Next Steps (Highest Impact)

### 7.1 Prompt robustness suite
Evaluate multiple paraphrases per condition (e.g., 3–10 neutral and 3–10 pressure templates). Report:
- distribution of Δ across templates
- whether L21N1876 remains dominant across templates
- variability of D under neutral prompts

### 7.2 Tokenization audit
Repeat scoring with token variants:
- `"Yes"` vs `" Yes"`
- `"No"` vs `" No"`
- lowercase variants (`"yes"`, `"no"`) if meaningful for the model

Confirm the chosen token pair matches the model’s natural continuation for these prompts.

### 7.3 Replace mean bar plot with distribution plots
Add at least one of:
- histogram/CDF of `removed_frac` for neuron vs head
- scatter plot of `(Δ, Δ_fix)` per claim (visualizes residual effect directly)
- fraction of claims with `removed_frac ≥ {0.5, 0.9, 1.0}`

These make “dominance” visually unambiguous.

### 7.4 Test whether intervention restores truth discrimination
Compute:
- `D_pressured_fix_neuron`
- `D_pressured_fix_head`
- `D_pressured_fix_both`

If truth discrimination increases after fixing the neuron, that supports the hypothesis that pressure-agreeing masks truth signals in this output space.

### 7.5 Broader circuit search
Rank top-k neurons/heads by pressured−neutral activation deltas and run the same mediation analysis to test redundancy and alternative pathways (beyond a single “winner” component).

---

## 8. Reproducibility and GitHub Pages Deployment

### 8.1 Recommended repository layout
Use `docs/` for GitHub Pages:

.
- scripts/
  - s.py
- results.csv
- docs/
  - index.md
  - assets/
    - truth_discrimination.png
    - causal_removal.png

### 8.2 Enable GitHub Pages
1. Push to GitHub.
2. Settings → Pages.
3. Source: “Deploy from a branch”.
4. Branch: `main` (or `master`), Folder: `/docs`.
5. Save.

The site will serve `docs/index.md` as the homepage.

---

## 9. Key Empirical Findings (Current)

1. **Pressure produces a strong Yes-shift:** mean Δ ≈ 1.46 and 96.5% of claims increase in Yes-leaning under pressure.
2. **Truth discrimination is weak and drops under pressure:** D_neutral ≈ 0.168 → D_pressured ≈ 0.057 (D_drop ≈ −0.111).
3. **A single neuron mediates most of the measured pressure effect:** clamping **L21N1876** removes most of Δ on average (often over-correcting), while clamping **L19H2** removes a much smaller fraction.

---
