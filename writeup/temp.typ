// Temporary draft: Experimental Results and Analysis (main experiments only)

= Experimental Results and Analysis (Main Experiments) <group1>
\
This section reports only the five main experiments from the final setup, in the execution order used during development: with reconstruction loss, no reconstruction loss, and then the three no-reconstruction tuning variants. The earlier exploratory runs (including run13 and before) are intentionally excluded here and will be integrated later in a separate historical subsection.

Concretely, the five experiments are:

1. with_recon (5 runs)
2. no_recon (5 runs)
3. tuning_ae03_len2 (5 runs)
4. tuning_ae05_len2 (5 runs)
5. tuning_ae05_len3 (5 runs)

All summaries below are computed from the generated experiment-level outputs, where each experiment is averaged over its own 5 runs. This explicitly avoids a single 15-run aggregate for tuning.


== Analysis scope and reporting choices <group1>
\
The central objective in this chapter is emergent communication quality, not generic optimisation diagnostics. For that reason, I treat message-to-entity structure and TopSim as primary evidence, and I treat reward/loss curves as secondary context.

In particular, I do not treat _valid_action_rate_, _invalid_entity_idx_rate_, and message length as primary discriminative metrics in this section. In these runs, valid-action behaviour is effectively saturated and invalid-entity indexing is near-zero by construction, so these two variables carry little explanatory signal about protocol structure. Message length is discussed only where it materially changes protocol form (the length-3 tuning branch).

This framing is consistent with the emergent communication literature: high task performance alone does not guarantee interpretable or human-like communication, and successful protocols can remain opaque or degenerate (Lazaridou and Baroni, 2020).


== Branch-level comparison <group1>
\
Before diving into specific runs, Table 1 summarises the key cross-experiment outcomes: final TopSim, and how often the Fish/Salmon pair is grouped under one message.

#table(
	columns: 4,
	[Experiment], [TopSim (mean +- sd)], [Fish+Salmon grouped runs], [Clean Fish+Salmon pair runs],
	[with_recon], [0.261673 +- 0.086217], [0/5], [0/5],
	[no_recon], [0.413854 +- 0.143490], [5/5], [3/5],
	[tuning_ae03_len2], [0.438693 +- 0.058650], [5/5], [4/5],
	[tuning_ae05_len2], [0.452432 +- 0.119406], [5/5], [5/5],
	[tuning_ae05_len3], [0.439227 +- 0.057365], [5/5], [5/5],
)

At a high level, this already captures the main story. The with_recon setting learns very strong one-to-one entity coding but weak semantic grouping. Once reconstruction pressure is removed, grouping behaviour appears immediately and consistently. Tuning then sharpens this behaviour, especially in the action-entropy 0.5 settings.


== Experiment 1: with_recon baseline <group1>
\
My initial hypothesis was that with reconstruction enabled, the model would still assign similar messages to near-identical entities. Fish and Salmon were the test case I tracked most closely, because they differ by exactly one vector dimension:

- Fish = [0,3,0,3,2,2]
- Salmon = [0,3,0,4,2,2]

Empirically, this did _not_ happen in a stable semantic way. Across the five with_recon runs, Fish and Salmon are always separated, and one run even entangles Salmon with unrelated entities.

#table(
	columns: 4,
	[Run], [Fish], [Salmon], [Observation],
	[run14], [0 40 0], [43 40 0], [2/3 token overlap but still separate symbols],
	[run15], [0 0 0], [42 0 0], [2/3 token overlap but still separate symbols],
	[run16], [14 0 0], [0 0 0], [2/3 token overlap but still separate symbols],
	[run17], [35 0 0 (with Dry Campsite)], [0 0 0 (with Storm and Riverbank)], [cross-category grouping anomaly],
	[run18], [33 31 0], [33 33 0], [shared prefix but still strict separation],
)

So the model is not random: it often reuses local token fragments, but the protocol remains mostly identifier-like instead of semantically clustered. This is also reflected in aggregate structure:

- only 2 multi-entity message clusters across 5 runs,
- average cluster size 2.5,
- and near-zero purity by entity type/subtype.

Interpretation: adding reconstruction objective pressure favours entity-specific coding (good for reconstruction), but this can suppress broader semantic reuse and cross-entity grouping, exactly where I needed stronger evidence for emergent compositional structure.


== Experiment 2: no_recon (headline result) <group1>
\
After observing the previous behaviour, I set reconstruction weight to 0.0 so that optimisation no longer rewards strict entity reconstruction. This changed the protocol dynamics immediately.

The strongest signal is that Fish and Salmon become grouped in all 5 runs (clean pair in 3/5 and grouped-with-neighbours in 2/5). Representative rows:

- run20: 42 21 0 -> Fish|Salmon
- run21: 49 43 0 -> Fish|Salmon
- run22: 43 43 0 -> Fish|Salmon
- run19: 37 14 0 -> Fish|Salmon|Berry Patch
- run23: 29 29 0 -> Fish|Salmon|Muddy Water|Mushrooms|Cold Night|Thorns|Fever|Cave

This is exactly the kind of behaviour I wanted to see: communication no longer behaves like a per-entity lookup table and starts collapsing related entities into reusable symbolic buckets.

At the experiment level, no_recon produces:

- 42 multi-entity clusters (vs 2 in with_recon),
- average cluster size 4.07,
- max cluster size 14,
- TopSim mean 0.413854 (substantially above with_recon 0.261673).

Importantly, many high-count clusters are semantically coherent at coarse level (resources/crafting opportunities/events or danger sets), even when they are not perfectly pure at fine-grained subtype level.


== Experiments 3-5: no_recon tuning <group1>
\
I then investigated whether action entropy and message length shift compositional behaviour further, using three controlled branches.


=== tuning_ae03_len2 (action entropy 0.3, length 2) <group1>
\
This branch keeps the no_recon benefits and improves stability. Fish and Salmon are grouped in 5/5 runs, with clean pair in 4/5. The one non-clean case is still meaningful:

- run26: 34 34 0 -> Fish|Salmon|Cold Night|Thorns|Fever

TopSim is 0.438693 +- 0.058650, higher than no_recon baseline and with noticeably lower variance.


=== tuning_ae05_len2 (action entropy 0.5, length 2) <group1>
\
This branch gives the best TopSim mean among the five experiments: 0.452432 +- 0.119406. Fish and Salmon appear as a clean pair in all 5/5 runs.

Representative pair rows:

- 33 33 0 -> Fish|Salmon
- 23 23 0 -> Fish|Salmon
- 16 16 0 -> Fish|Salmon

I also observed a rise in repeated-symbol pair forms (3/5 runs for Fish/Salmon), which is qualitatively consistent with stronger code regularisation.


=== tuning_ae05_len3 (action entropy 0.5, length 3) <group1>
\
Length-3 messages preserve the grouping gains and make token regularity more visible, even though TopSim mean is slightly below ae05_len2 (0.439227 +- 0.057365, but with lower variance).

Fish/Salmon is a clean pair in 5/5 runs, and repeated-token structure becomes more explicit:

- 31 31 31 0 -> Fish|Salmon
- 10 10 10 0 -> Fish|Salmon
- 2 2 2 0 -> Fish|Salmon

The strongest qualitative run here is run25, where large clusters align well with coarse semantics:

- 32 47 47 0 -> Wood|Water|Herbs|Berries|Stone|Rocky Outcrop|Herb Garden|Supply Cache|River Crossing
- 34 34 34 0 -> Storm|Frost|Poison Plant|Cliff|Infection
- 10 7 7 0 -> Goat|Deer|Rabbit

This supports the interpretation that longer messages can improve symbolic organisation and readability of cluster structure, even when global TopSim does not increase further.


== Why messages do not literally match across runs <group1>
\
One surprising but important finding is that literal message overlap across independent runs is effectively zero, even when high-level grouping patterns are stable.

I verified this explicitly:

- within each experiment, intersection of exact message strings across all 5 runs is 0,
- and across the three tuning variants for the same run id (24 to 28), intersection is also 0.

This does _not_ indicate analysis failure. It is expected under emergent communication symmetry: different seeds can converge to functionally equivalent but symbolically permuted codes. In other words, the partition of meaning space can be stable while the actual token identities are arbitrary. This is aligned with prior findings that deep-agent languages are often task-adequate but opaque, and can show weak direct alignment with human-interpretable symbol semantics (Lazaridou and Baroni, 2020; Kottur et al., 2017; Lazaridou et al., 2018).


== Interim conclusion for the main five experiments <group1>
\
For the main branch analysis, the central conclusion is clear.

1. with_recon produced high reconstruction behaviour but weak semantic grouping.
2. no_recon changed the protocol regime and produced strong emergent grouping.
3. tuning improved grouping consistency and token regularity, with ae05_len2 giving the highest TopSim mean and ae05_len3 giving the most interpretable repeated-token cluster forms.

So for this dissertation objective, removing reconstruction from the communication objective was the key design decision. It transformed communication from mostly entity-specific coding into reusable symbolic structure that tracks semantically related entities much more consistently.

I will add the run13 and earlier-historical subsection later, but for the five main experiments, this is the evidence-supported result.


