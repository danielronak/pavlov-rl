# Reference and novelty check (2026-11-19)

Every reference in the paper was checked against bibliographic records found by web search:
publisher pages, PubMed Central, DBLP, arXiv and PMLR listings, university repositories, and the
reference lists of later papers. **Full texts were generally not read.** Two hosts (sussex.ac.uk and
biorxiv.org) are blocked from the analysis machine, so the papers closest to ours were checked at
abstract and record level only. Before submission, the PI should read the five papers in Section 3
in full.

## 1. References already in the paper

| Reference | Status | What changed |
|---|---|---|
| Rescorla & Wagner 1972 | confirmed | Full title "…: Variations in the effectiveness of reinforcement and nonreinforcement"; *Classical Conditioning II*, Black & Prokasy (Eds.), Appleton-Century-Crofts, pp. 64–99. (One web source gives 64–69; two give 64–99.) |
| Sutton & Barto 1990 | confirmed | In *Learning and Computational Neuroscience: Foundations of Adaptive Networks*, Gabriel & Moore (Eds.), MIT Press, pp. 497–537. |
| Modayil & Sutton 2014 | confirmed (title, authors, year, venue type) | AAAI-14 workshop paper (AAAI workshop repository). The exact workshop name was not seen on the paper itself. |
| "Butcher et al. 2022" (arXiv:2203.09498) | **corrected** | The first author is **Pilarski**, and the title is "The Frost Hollow experiments: Pavlovian signalling as a path to coordination and communication between agents". The related paper by Butcher, Johanson, Pilarski et al. is arXiv:2201.03709. |
| Domjan 2005 | **corrected** | *Annu. Rev. Psychol.* 56, 179–206. The DOI is 10.1146/annurev.psych.56.091103.070213. Our old DOI (…55.090902.141409) was wrong. |
| Hollis 1997 | partly confirmed (†) | Title, journal and year confirmed through citing papers (Domjan 2005; Timberlake). The volume, issue and pages (52(9), 956–965) are consistent with those sources but were not confirmed on the publisher's page. |
| Sheffield 1965 | confirmed | In *Classical Conditioning: A Symposium*, Prokasy (Ed.), Appleton-Century-Crofts, pp. 302–321. The content claim is supported: one dog ("Vicki") kept salivating across 800 omission trials, and omission did not prevent conditioned salivation. The exact percentages were not checked. Our paper quotes none. |
| Williams & Williams 1969 | confirmed | *J. Exp. Anal. Behav.* 12(4), 511–520, doi:10.1901/jeab.1969.12-511. Full subtitle added. |
| Dayan, Niv, Seymour & Daw 2006 | confirmed | *Neural Networks* 19(8), 1153–1160, doi:10.1016/j.neunet.2006.03.002. |
| Stephens 1991 | confirmed | *Behav. Ecol.* 2(1), 77–89, doi:10.1093/beheco/2.1.77. The abstract supports our use: some change between generations, with regularity within a generation, favours learning. |
| Katsnelson et al. 2012 | confirmed, **title added** | E. Katsnelson, U. Motro, M. W. Feldman, A. Lotem, "Evolution of learned strategy choice in a frequency-dependent game", *Proc. R. Soc. B* 279(1731), 1176–1184 (online in 2011). Our use is supported only loosely: without enough environmental change, innate choice becomes fixed. Its main topic is a producer–scrounger game, so cite it as supporting, not central. |
| Wang et al. 2018 | confirmed | *Nat. Neurosci.* 21(6), 860–868. |
| Binz et al. "2023" | **corrected** | The year is 2024: *Behav. Brain Sci.* 47, e147 (online November 2023). Full author list added. |
| Salimans et al. 2017 | confirmed | arXiv:1703.03864. |
| Schubert et al. 2024 | confirmed, **title added** | "In-context learning agents are asymmetric belief updaters", ICML 2024, PMLR 235, 43928–43946. |
| Kamin 1969 (draft only) | confirmed | In *Punishment and Aversive Behavior*, Campbell & Church (Eds.), pp. 279–296. Some indexes give 1967, which is the symposium date. |
| Lubow & Moore 1959 (draft only) | confirmed | *J. Comp. Physiol. Psychol.* 52, 415–419. The subjects were sheep and goats. |
| Rescorla 1967 (draft only) | confirmed | *Psychol. Rev.* 74(1), 71–80. |
| Pavlov 1927 | not re-checked | Standard citation (Anrep translation, Oxford University Press). |

## 2. Prior work the paper was missing (now cited)

These are the closest precedents. None of them makes our study redundant, but a reviewer would
expect all of them.

1. **Burgos 1997**, "Evolving artificial neural networks in Pavlovian environments", in Donahoe &
   Dorsel (Eds.), *Neural-Network Models of Cognition* (Advances in Psychology 121), pp. 58–79.
   It is a genetic algorithm combined with a *designed* neurocomputational learning model, trained
   on forward-delay conditioning with different ISIs. It is the earliest "evolve networks for
   Pavlovian conditioning" work we found. *Difference:* there, the learning rule is built in and
   the environment is the conditioning procedure itself.
2. **Izquierdo & Harvey 2007** (ECAL; LNAI 4648) and **Izquierdo, Harvey & Beer 2008** (*Adapt.
   Behav.* 16(6), 361–384). They evolve continuous-time recurrent networks **without synaptic
   plasticity** that learn, and re-learn, which temperature goes with food. This is the closest
   precedent for learning that lives in evolved recurrent dynamics. *Difference:* fitness rewards
   solving the association task itself, and they do not test a held-out Pavlov lab, the omission
   test or later conditioning phenomena. (Checked at abstract level only.)
3. **Méndez Salinas, Weissing & Kozielska 2023** (bioRxiv) and **Kozielska & Weissing 2024**
   (*PLoS Comput. Biol.* 20(1), e1011840). These evolve neural networks for associative learning
   with a prediction-error-driven learning rule. Learning is less likely to evolve in fairly
   constant environments. In the 2023 preprint, evolved networks beat Rescorla–Wagner and pass
   through an RW-like stage. *Relevance:* this directly parallels our q-control (fixed meanings
   give inborn responses, variable meanings give learning). We now cite it in the Introduction and
   in the Discussion of the q result. *Difference:* their learning rule is partly designed, and
   their tasks reward association directly. (Checked at abstract level only.)
4. **Niv, Joel, Meilijson & Ruppin 2002** (*Adapt. Behav.* 10(1), 5–24). Evolved RL learning rules
   in simulated bees. Risk aversion and matching emerge.
5. **Keramati & Gutkin 2014** (*eLife* 3, e04811). Homeostatic RL, where reward is physiological
   need reduction. It is the natural reference for our physiology-only reward. Cited in the
   Spontaneity paragraph.

## 3. Related work to know about (not cited yet; read before submission)

- **Burgos 2007**, "Autoshaping and automaintenance: a neural-network approach", *JEAB* 88(1),
  115–130. A designed neural-network model of positive and negative automaintenance (omission). It
  is relevant if a reviewer asks how other network models handle omission.
- **Lesaint et al. 2014**, "Accounting for negative automaintenance in pigeons: a dual learning
  systems approach and factored representations" (*PLoS ONE*). A model of omission persistence.
- **Lee et al. 2025** (bioRxiv 10.64898/2025.11.30.691382), "Emergence of rapid value inference
  through meta-reinforcement learning". Mice and RNNs in classical conditioning with reversals
  shift from plasticity-based to dynamics-based value updating. It is related to our "learning in
  activations" framing (full author list not seen).
- **Qiao 2025**, arXiv:2508.08289 (v1: "Understanding Transformers through the Lens of Pavlovian
  Conditioning"; a later version is titled "Attention as Conditioning: What Classical Learning
  Theory Predicts About Linear Transformers"). It analyses linear-attention in-context learning as
  conditioning. A delta-rule update predicts blocking, but blocking was weak in trained models. It
  is relevant to our blocking result on the AI side. Check the version before citing.
- **Russin, Pavlick & Frank** (arXiv:2402.08674; CogSci 2024). Note: here "blocking" means a
  *blocked curriculum*, not Kamin blocking. Do not cite it as evidence about Kamin blocking.

## 4. Revised novelty statement

**Not new** (and now credited):
- evolving neural networks for conditioning (Burgos 1997);
- evolved learning carried by fixed-weight recurrent dynamics (Izquierdo et al. 2007/2008);
- environmental variability deciding whether learning evolves (Stephens 1991; Kozielska & Weissing
  2024);
- meta-RL as a model of cognition (Wang et al. 2018; Binz et al. 2024).

**New, as far as these searches show:**
1. A CR-like response to a novel cue emerges from a **physiology-only** objective, with no
   designed learning rule and no fitness term for association or prediction.
2. The evolved learner is tested in a **held-out laboratory** with the **omission + yoked**
   design. Outcome insensitivity emerges, and it depends on the ecology's response–outcome
   statistics (sister lineages). We found no evolved or meta-learned agent that has been given the
   omission test.
3. A conditioning-phenomena battery for an evolved learner, compared with RW and TD, including a
   targeted ecological test of blocking (compound-cue world).
4. Pre-registration, intention-to-treat analysis, and automated tests of the "no leak" invariants.

The paper's sentence "to our knowledge none [meta-learned agent] has been taken through Pavlov's
protocols or an omission test" still stands after this search. The new paragraph in the
Introduction makes the earlier evolutionary work explicit, so the novelty claim rests on points 1–4
rather than on "evolving a conditioner".
