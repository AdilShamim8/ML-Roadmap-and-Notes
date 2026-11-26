# Resume and Portfolio Guide — the 6-Second Screen, Engineered

Your resume survives a 6–10 second skim before a decision. Everything below is engineered for that skim: the bullet formula recruiters expect, the ML-project framing that signals seniority, the GitHub profile that closes the loop, and the failure modes that end candidacies silently.

---

## 1. The X-Y-Z Bullet Formula

Every bullet: **"Accomplished [X] as measured by [Y], by doing [Z]"** (Google's own guidance). The pattern works because recruiters scan for verbs + numbers; the formula forces both.

| Weak (what people write) | Strong (X-Y-Z) |
| --- | --- |
| "Built a churn prediction model" | "Cut quarterly churn misprediction by 18% (AUC 0.71 → 0.79) by rebuilding the model on grouped splits and leakage-audited features" |
| "Worked on data pipelines" | "Reduced daily feature-pipeline failures from 12% to 0.5% of runs by adding schema validation, retries, and alerting at the 3 highest-failure stages" |
| "Used pandas and sklearn" | "Sped up a 40M-row feature extraction 23× (47 min → 2 min) by replacing row-wise apply with vectorized operations" |
| "Made dashboards" | "Gave 40+ account managers a churn-risk dashboard that lifted retention-outreach coverage from 20% to 85% of at-risk accounts" |
| "Studied machine learning" | "Authored 26 executed ML notebooks (statistics through LLM/RAG design) with zero-error execution verified on a pinned 2026 stack" |

**Rules of the formula:**
- **Numbers first, always** — even estimates ("~", "order of") beat nothing. No number = not a bullet.
- **Verbs that signal ownership:** built, reduced, cut, led, designed, migrated, automated. Avoid: assisted, helped, participated, was involved.
- **One bullet = one accomplishment.** A bullet with three ideas is three bullets, or one confused skim.
- **4–6 bullets per role, strongest first** — the skim reads the first two.

---

## 2. The ML Resume Section Structure

```
HEADER       Name | Title ("Machine Learning Engineer" — not "aspiring") | links
SUMMARY      2–3 lines: domain + scale + stack + the one differentiator
EXPERIENCE   (reverse chronological) — X-Y-Z bullets
PROJECTS     2–3 deep projects (this is where juniors win; seniors keep 1–2)
SKILLS       Grouped: Languages / ML & Stats / Data / Infra & Serving
EDUCATION    + certifications (only current, only relevant)
```

**The summary, done right (3 lines):** "ML engineer with production experience in fraud detection and churn — models serving 5K requests/day with <100ms p99. Python/PyTorch/sklearn on a Kubernetes stack. Author of an executed-notebook ML roadmap with 2.1k stars." — domain, scale, stack, proof. One of those four per clause, nothing else.

**Skills section rules:** no skill bars/percentages (they invite the question "80% of what?"); no listing things you can't discuss for 5 minutes — **the resume is a menu of interview topics YOU choose**; put the stack you want to be asked about, omit the one you don't.

**ATS pass:** standard section names, no tables/columns/text boxes, one column, PDF export from a clean template. Keywords matter less than people fear but match the JD's phrasing where honest ("ML pipelines" vs "machine learning pipelines" — use both once).

---

## 3. The Projects Section — Where This Repo Becomes Your Portfolio

The projects section is a **portfolio of evidence**, not a reading list. For each: name, one line, the stack, the number, the link.

**The formula per project (4 lines):**

> **End-to-End Titanic Survival Pipeline** — full ML lifecycle in one repo: EDA → leakage-safe pipeline (all transformers inside CV folds) → 5-model bake-off → error analysis.
> `Python · scikit-learn · ColumnTransformer · cross-validation` · **0.79 CV AUC, honestly beaten down from a leaky 0.87**
> [github.com/AdilShamim8/ML-Roadmap-and-Notes] — *data shipped in-repo; one-command reproduction*

**What each project must demonstrate (the checklist reviewers actually apply):**
- [ ] A **baseline** exists (naive model), and every number is relative to it
- [ ] **Leakage-safe** validation (the split strategy named: stratified/grouped/temporal)
- [ ] **Error analysis**, not just metrics — where the model fails, by segment
- [ ] **Shipped data** + reproduction instructions (README, requirements, seeds)
- [ ] An honest **limitation** stated (what you'd do with more time/data)

One project meeting all five beats five projects meeting none. This repo's two capstones (Titanic classification, California Housing regression) were built to this standard — write them up this way.

**The "course projects" trap:** if the project came from a course, make it yours — different data, added error analysis, a production wrinkle (serving sketch, cost analysis). "I extended X" is a fine story; "I followed a tutorial" is a deletion.

---

## 4. The GitHub Profile That Closes the Loop

Recruiters who click your GitHub decide in ~30 seconds. Optimize that 30 seconds:

1. **Pinned repositories:** your 2–3 best, not your 6 most recent. The roadmap repo + capstones.
2. **README = landing page.** The repo's README opens with what it is, the numbers, and how to verify. This repo's README does exactly this — study its structure.
3. **Commit hygiene matters at the margin:** a repo with 200 small, well-messaged commits ("Add leakage-safe cross-validation", not "update") reads as a professional history; a repo with 3 giant commits reads as a weekend dump. Meaningful commit messages are free seniority.
4. **A profile README** (the special `username/username` repo): 5 lines — what you do, what you've built, what you're learning, contact. Not an essay.
5. **Green squares don't matter; artifacts do.** Nobody counts activity — they open the best repo and check whether the README makes sense in one screen.

---

## 5. Portfolio Beyond GitHub (Pick ONE, Do It Well)

- **A deployed demo** — Streamlit/Gradio app on one of your models, linked from the resume: "demo" beats "code" because it proves the serving step. Even a Hugging Face Space counts.
- **A technical blog** — 2–3 posts explaining something you actually executed ("I measured the target-leakage illusion: 15 AUC points"). Writing is the senior signal; your notebooks are pre-written material.
- **Kaggle** — top-10% finishes in a niche (not Titanic); the medal is the line-item, the write-up is the substance.

**Do not** spread thin across all three. One deployed artifact with a real write-up outperforms three half-finished ones — interviewers ask about the *deep* thing, and depth is where you score.

---

## 6. Resume Failure Modes (Each One Ends Candidacies Silently)

- **No numbers anywhere** → reads as participation, not contribution.
- **Every bullet about tools** ("used X, used Y") → you become "has used sklearn" instead of "has shipped outcomes." Tools go in Skills; bullets carry accomplishments.
- **Typos** — a single typo in a data role is near-fatal: the job IS attention to detail. Read it aloud, then have someone else read it.
- **3+ pages** — one page under ~10 years' experience. Cut oldest roles to 2 bullets; the reader skims newest hardest.
- **The "objective statement"** — dead since the 2000s; the 3-line summary replaced it.
- **Unexplained gaps or 8 stints of 6 months** — address gaps briefly and neutrally in the summary or cover; job-hopping, keep only if the work was contract.
- **Listing every library version** — "pandas 2.2.3, numpy 2.1.3..." signals junior; the stack line ("Python data stack: pandas/sklearn/statsmodels") signals competence.
- **A photo / age / marital status** — in US/India tech resumes: noise or liability; in some markets (Germany) conventional — match the market you're applying in.

---

## 7. Tailoring: the 15-Minute Per-Application Routine

1. Read the JD; highlight the **3 capabilities they lead with** (not the keyword soup).
2. Reorder/reword your top bullets so those 3 capabilities appear in the first bullet of each relevant role.
3. Mirror their vocabulary where honest ("ML platform" vs "MLOps" vs "ML infrastructure").
4. One project swap if theirs is domain-matched (fraud JD → fraud project pinned first).
5. Save as `Name_MLE_Company.pdf`; track it in a spreadsheet (company, role, date, contact, status) — applications are a funnel and you're the analyst.

---

## 8. The Recruiter-Screen Handoff

When the recruiter calls, the resume's job is done; yours begins. Three numbers ready in the first 5 minutes: **current/expected compensation** (research bands first — see the [Offer Negotiation guide](Offer%20Negotiation.md)), **notice period/start date**, and **one sentence per role** explaining why you're looking. Hesitation on any of the three turns a screen into a filter.

---

*Next: the [Project Deep-Dive Prep](Project%20Deep-Dive%20Prep.md) — the 5-why drill that prepares you for the technical interrogation of everything your resume just claimed.*
