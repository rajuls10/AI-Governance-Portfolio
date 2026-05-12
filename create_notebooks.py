import nbformat
import os

path = r"module-01-foundations\cluster-02-linear-algebra-numpy"
os.makedirs(path, exist_ok=True)

# ─── EXERCISE 01 ───
nb1 = nbformat.v4.new_notebook()
nb1.cells = [
    nbformat.v4.new_markdown_cell("# Exercise 01: Vectors and Bias in Hiring AI\n# Cluster 2 - Linear Algebra + NumPy\n# Framework: EU AI Act Annex III (Employment Systems)\n# Case Study: Recruitment AI"),
    nbformat.v4.new_code_cell("""import numpy as np

applicants = np.array([
    [2, 90, 88],   # Applicant A — woman, career break, high scores
    [8, 75, 70],   # Applicant B — man, no career break, average scores
    [3, 92, 95],   # Applicant C — woman, recent graduate, excellent scores
    [10, 65, 60],  # Applicant D — man, lots of experience, lower scores
    [1, 98, 96],   # Applicant E — woman, fresh out of university, top scores
])

print("Applicant data loaded.")
print("Applicant matrix shape:", applicants.shape)"""),
    nbformat.v4.new_code_cell("""# Moderate bias weights
weights = np.array([0.6, 0.2, 0.2])
scores = applicants @ weights

print("Applicant scores (moderate bias):")
for i, score in enumerate(scores):
    print(f"Applicant {chr(65+i)}: {score:.2f}")"""),
    nbformat.v4.new_code_cell("""# Extreme bias weights — heavily favours experience
biased_weights = np.array([0.9, 0.05, 0.05])
biased_scores = applicants @ biased_weights

print("Scores with experience-heavy weights:")
for i, score in enumerate(biased_scores):
    print(f"Applicant {chr(65+i)}: {score:.2f}")"""),
    nbformat.v4.new_code_cell("""# Fair weights — balanced across all three attributes
fair_weights = np.array([0.2, 0.4, 0.4])
fair_scores = applicants @ fair_weights

print("FAIR RANKING (balanced weights):")
for i, score in enumerate(fair_scores):
    print(f"Applicant {chr(65+i)}: {score:.2f}")"""),
    nbformat.v4.new_code_cell("""# Ranking comparison table
labels = ["A (woman, career break)", "B (man, avg)", "C (woman, grad)", 
          "D (man, experienced)", "E (woman, fresh grad)"]

print("\\n--- RANKING COMPARISON ---")
print(f"{'Applicant':<25} {'Moderate Bias':>14} {'Extreme Bias':>13} {'Fair Model':>11}")
print("-" * 65)
for i, label in enumerate(labels):
    print(f"{label:<25} {scores[i]:>14.2f} {biased_scores[i]:>13.2f} {fair_scores[i]:>11.2f}")

print("\\n--- WHO GETS HIRED (Top 2 each model) ---")
for model_name, model_scores in [("Moderate Bias", scores), ("Extreme Bias", biased_scores), ("Fair Model", fair_scores)]:
    ranked = sorted(zip(model_scores, labels), reverse=True)
    top2 = [label for _, label in ranked[:2]]
    print(f"{model_name}: {top2[0]} | {top2[1]}")"""),
]
nbformat.write(nb1, os.path.join(path, "exercise-01-vectors-and-bias.ipynb"))
print("Exercise 01 created.")

# ─── EXERCISE 02 ───
nb2 = nbformat.v4.new_notebook()
nb2.cells = [
    nbformat.v4.new_markdown_cell("# Exercise 02: Disparate Impact Analysis\n# Cluster 2 - Linear Algebra + NumPy\n# Framework: EU AI Act Annex III (Employment Systems)\n# Case Study: Recruitment AI"),
    nbformat.v4.new_code_cell("""import numpy as np

applicants = np.array([
    [2, 90, 88],
    [8, 75, 70],
    [3, 92, 95],
    [10, 65, 60],
    [1, 98, 96],
])

biased_weights = np.array([0.9, 0.05, 0.05])
fair_weights = np.array([0.2, 0.4, 0.4])

biased_scores = applicants @ biased_weights
fair_scores = applicants @ fair_weights"""),
    nbformat.v4.new_code_cell("""# Disparate impact analysis — separate by gender
Gender = ["woman", "man", "woman", "man", "woman"]

woman_b, man_b = [], []
for i, gen in enumerate(biased_scores):
    if Gender[i] == "woman":
        woman_b.append(gen)
    else:
        man_b.append(gen)

woman_f, man_f = [], []
for i, gen in enumerate(fair_scores):
    if Gender[i] == "woman":
        woman_f.append(gen)
    else:
        man_f.append(gen)

bias_gap = sum(man_b)/len(man_b) - sum(woman_b)/len(woman_b)
fair_gap = sum(woman_f)/len(woman_f) - sum(man_f)/len(man_f)

print("=" * 45)
print("DISPARATE IMPACT ANALYSIS")
print("=" * 45)
print(f"{'Model':<15} {'Women Avg':>10} {'Men Avg':>10} {'Gap':>8}")
print("-" * 45)
print(f"{'Biased':<15} {sum(woman_b)/len(woman_b):>10.2f} {sum(man_b)/len(man_b):>10.2f} {bias_gap:>+8.2f}")
print(f"{'Fair':<15} {sum(woman_f)/len(woman_f):>10.2f} {sum(man_f)/len(man_f):>10.2f} {fair_gap:>+8.2f}")
print("=" * 45)"""),
]
nbformat.write(nb2, os.path.join(path, "exercise-02-disparate-impact.ipynb"))
print("Exercise 02 created.")

# ─── GOVERNANCE REFLECTION ───
nb3 = nbformat.v4.new_notebook()
nb3.cells = [
    nbformat.v4.new_markdown_cell("""# Cluster 2 — Governance Reflection

## Technical Concepts Learned
- NumPy arrays and the `@` operator (matrix multiplication)
- `enumerate()` function to loop with counters
- `chr()` to convert numbers to letters
- Creating weight vectors and calculating weighted scores
- Weights must sum to 1.0 for comparable scores

## Governance Concepts Learned
- Every person in a hiring AI is reduced to a vector of numbers
- The AI sees numbers, not context — not career breaks, not reasons for gaps
- Bias hides in weight choices that look neutral
- Same data + different weights = completely different (unfair) outcomes
- Even "fair-looking" weights can amplify hidden bias
- Weights must be transparent and auditable, but transparency alone does not guarantee fairness
- Companies have freedom to choose weights, but that choice has real consequences for real people
- Interview scores carry human bias (emotion, personality, communication style preference) — by heavily weighting interviews, you do not remove bias, you amplify it
- Disparate impact analysis shows the gap between groups under different models. But numbers alone do not tell you which model is correct — that requires examining the business need, historical outcomes, and whether the weights can be justified under law

## Open Questions
- How do you set weights fairly when cultural bias exists everywhere?
- Who decides what is fair?
- Even if you tell someone "we use AI for hiring," most people will not understand what that means. That is a real governance problem.

## Key Insight
This is why HireScore AI is classified as High-Risk under EU AI Act Annex III — because weight choices that look neutral can systematically exclude women, minorities, and anyone whose life does not fit a linear career path."""),
]
nbformat.write(nb3, os.path.join(path, "governance-reflection.ipynb"))
print("Governance reflection created.")

print("\\nAll three files created successfully.")