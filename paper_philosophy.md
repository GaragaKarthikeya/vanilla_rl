# Paper Philosophy
*Distilled from Simon Peyton Jones, "How to Write a Good Research Paper" (Microsoft Research)*

---

## The fundamental goal

You are trying to transfer **one idea** from your head into the reader's head. Not describe a system. Not prove novelty. Not document what you did. Infect the reader with a specific, reusable idea — like a virus. If the idea is useful but nobody reads about it, it might as well not exist. Communication is not what you do at the end of research; it is part of doing research.

> *"If Mozart had sat in a dark room and not told anybody about his amazing music, we wouldn't be able to enjoy it."*

**Utility test (Fred Brooks):** A tool is judged by whether its users succeed, not by novelty. Ask: what is the reusable idea in this paper that will help someone? If you can't answer that concisely, your paper isn't ready.

---

## The one-idea rule

Write a paper about **one idea**. If you have ten ideas, write ten papers.

Sanity check: write this sentence somewhere in section 3 and finish it — *"The main idea of this paper is..."* If you can't finish the sentence cleanly, your readers can't either.

---

## Process: paper first, research second

Wrong order: do all the research → write the paper.  
Right order: have idea → **start writing the paper** → let the paper tell you what research is missing.

Writing is a forcing function. You will discover mid-section that you haven't proved something, haven't built something, don't actually understand something. Better to find out in week 2 than in the week before the deadline. Writing also clarifies your own understanding — if you thought you understood it but can't write it plainly, you didn't understand it yet.

---

## Structure: what goes where

### Abstract (4 sentences, written last)
1. State the problem.
2. Why is it interesting / why does it matter?
3. What does your solution achieve? (not *what* the solution is)
4. What follows from this?

Keep it brief. Don't repeat the introduction. Write it after everything else is done, because the focus of the paper will shift as you write it.

### Title
Catchy enough to attract the right readers; contains enough keywords that experts bid to review it. Overly jokey = dangerous. Factual-but-dull = limits readership. Balance is a judgment call, not a formula.

### Introduction / Page 1

**The first page is everything.** Most readers decide on page 1. Fit problem + hook + contributions on page 1. The physical act of turning the page is a real barrier.

Structure:
1. **State the problem** — briefly, concretely. Use a specific example, not an abstract description. Get them to think "if I could solve that, I'd be happy."
2. **Don't oversell the problem.** "Computer programs have bugs; we will solve them" — nobody believes it, and it conveys no information. Describe a molehill you can actually climb, not Everest.
3. **List your contributions explicitly** — as bullets, with forward references to the sections that back each claim. Make them refutable (not "we study X" — that always succeeds; instead "we prove X" or "we show Y on benchmarks Z").

**Kill the "rest of the paper is structured as follows" paragraph.** Nobody reads it. It wastes your most precious column inches. Your contribution list, with forward references, does the same job and people actually read it.

### Body: present the idea

Lead with **intuition**, not definitions. What would you draw on a whiteboard in 5 minutes? Do that. Formal details come after the reader has the gist.

Start with a concrete example — the very first thing you draw on a whiteboard is always an example.

Do **not** narrate your painful journey. Readers don't want to walk through dead ends with you. Only explain a blind alley if it is the obvious first approach and you need to show why it fails to motivate your solution.

Every claim in your contribution list needs a section that provides evidence for it. Evidence is not necessarily a proof — it can be measurements, case studies, comparisons. The reader should finish thinking: "I could reproduce this myself."

### Related Work (put it at the end, not the beginning)

Putting related work after the introduction is a death march for the reader. They don't yet have the vocabulary, the context, or a reason to care. You're forcing them through dense material before they've been hooked.

Instead: **weave citations naturally** into the narrative wherever they fit ("the obvious first approach is X; Green & White [5] took this path"). Acknowledge them, don't compare yet. Keep moving. Put the full comparison section at the end, once the reader knows your idea well enough to evaluate comparisons.

**Credit is like love, not money.** Giving another paper generous credit does not diminish yours. Say "in the fascinating paper by X, they do Y." It's probably true, it's good science, and it makes the world a better place. You can acknowledge a paper is excellent while still showing data that you improve on some axes.

**Acknowledge your weaknesses.** If reviewers find a dimension on which your system is worse, and you didn't mention it, that is actively damaging. If you mention it first — "on X type of application our approach is probably not the right choice because..." — that's good science and defuses the criticism.

**Never omit important related work you know about.** Program chairs assign reviewers from your citation list. If an author finds their relevant paper missing, that is a near-automatic rejection trigger. Cite everything that matters.

### Conclusions

Keep short. Don't repeat the abstract in past tense. Brief further work — plant the flag, don't propose a grant. Readers who reach the end are tired; don't ask them to re-read everything.

---

## Guinea pigs

Get friends to read the paper before submission. Educate them about what you want: **not** spelling corrections, **not** grammar notes — *"tell me where you got lost."*

When they get lost, sit at a whiteboard and explain it to them. They'll understand. Then write down what you just said. Verbally explained ideas are almost always clearer than the written version. This is the fastest way to fix confusing sections.

For expert feedback: email the author of a paper you cite. Say you enjoyed their work and you're referencing it — could they check that you described it fairly? They will read it. They may review it. Either outcome is better than leaving it unreviewed.

---

## On rejection

Your paper will be rejected. The review will feel unfair. Put it down. Wait a week. Come back and ask: *"How could I rewrite this so that even a careless reviewer couldn't make this mistake?"* Most misreadings are fixable. A reviewer who misunderstood is usually a reader who couldn't follow, not a malicious one.

---

## Self-diagnostics for this paper

Apply these tests to the current ASP-DAC draft:

**1. One-idea test**  
Finish: "The main idea of this paper is..."  
→ Candidate: *A learned placement policy amortizes the search cost that genetic algorithms re-pay from scratch for every new design, and it generalizes across scales it was never trained on.*  
Does every section serve this sentence? Prune what doesn't.

**2. Page 1 hook**  
Does page 1 present a concrete, specific problem (not "FPGAs are important") and immediately follow with contributions that are refutable and forward-referenced?

**3. Related work position**  
Related work section is already at the end — good. But check: does the introduction currently dump GA/GOLDS context before the reader is hooked? Keep intro citations short and admiring; save the comparison for after the methodology.

**4. Intuition before definitions**  
Does Section 3 (methodology) lead with a whiteboard-level picture of what the policy actually does, or does it open with a formal definition? The clarity fix already done (policy role vs VTR role) is the right instinct — make sure it reads like the explanation, not the specification.

**5. Contribution bullets — are they refutable?**  
"We train a policy..." — can fail to train. Good.  
"We show zero-shot..." — measurable. Good.  
"We analyze seed variance..." — vague. What is the specific claim? Make it falsifiable.

**6. Credit tone in related work**  
Does the GOLDS/NSGA-II comparison read like "their work was important, we build on it, and here is the additional thing we uniquely offer" — or like "we beat them"? The former is both more accurate and more persuasive to reviewers.

**7. Weakness acknowledgment**  
The paper has real gaps: single-benchmark training depth, no CLB LUT-size co-optimization, in-pool seed sensitivity on two benchmarks. Name these explicitly, in the paper. Pre-empt the reviewer.

**8. GA comparison framing**  
The argument isn't "RL beats GA in quality." It's "same quality floor, completely different cost structure." One is amortized; the other is per-instance. The paper should make this distinction unavoidably clear, because it is the honest and the strong claim.
