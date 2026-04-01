The first experiment consisted of 5 runs using the default settings shown in *#underline[@setting-exp-1]*, where the agent had to not only select the best action to take, but also predict the exact entity that the sender encountered. This introduces a second training objective that rewards the sender for producing messages that uniquely identify the current entity, explained in the loss section in the Implementation Chapter.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),

    [Mode],                [GS (Gumbel-Softmax)],
    [Vocab size],          [50],
    [Max message len],     [2],
    [Temperature],         [2.0],
    [Temp decay],          [1.0],
    [Temp minimum],        [0.1],
    [Recon weight],        [1.5],
    [Action entropy],      [0.1],
    [Action temperature],  [2.0],
    [Reward normalise],    [True],
    [Reward scale],        [0.5],
    [Learning rate],       [0.001],
    [Batch size],          [64],
    [Epochs],              [50],
  ),
  caption: [Survival game training configuration for run 18.],
) <setting-exp-1>

My initial hypothesis was that reconstruction pressure might still allow semantically similar entities to share codes, as similar entities do not differ much on their definition. For example, the definitions of Fish and Salmon are [0,3,0,3,2,2] and [0,3,0,4,2,2], so the messages that identify these should have been consistent even if we forced the agents to produce a unique message per entity.

Empirically, this did not hold across all five runs, and most entities that are later grouped into clusters do not show an type of similar message tokens across runs as seen throughout *#underline[@fish-salmon-exp1]* to *#underline[@resources-env]* in the Appendix. The protocol itself is not fully random, as token fragments are reused locally but the full message remains entity specific in every case. The receiver needs to reconstruct the exact entity, so the sender has a strong gradient incentive to keep codes discriminative, broader semantic reuse would reduce reconstruction accuracy.

#figure(
  image("Images/loss_train_test_RECON.png"),
  caption: [Training and test loss across epochs for the reconstruction setting.],
) <loss-recon-exp1>

This conclusion was further supported by inspecting the metrics generated from the data saved. As shown in the loss curve in *#underline[@loss-recon-exp1]*, training exhibits a rapid initial decrease, indicating that both the action selection and reconstruction objectives are quickly learned in the early stages. However, after this phase, the loss stabilises and gradually decreases at a slower rate, suggesting convergence towards a local optimum rather than continued significant improvement. The relatively wide standard deviation band throughout training indicates variability across runs, implying that while the model consistently learns a workable protocol, the exact solution found is sensitive to initialisation and stochasticity in training.

#figure(
  image("Images/topsim_RECON.png"),
  caption: [TopSim (test) across epochs for the reconstruction setting.],
) <topsim-recon-exp1>

As for the TopSim metric seen in *#underline[@topsim-recon-exp1]*, it remains relatively stable across epochs, fluctuating within a narrow range without a clear upward trend. This suggests that, despite successful optimisation of the training objective, the emergent communication protocol does not become increasingly aligned with the underlying semantic structure of the entities. While the mean TopSim value across runs is 0.261673, indicating a weak positive correlation between message similarity and semantic similarity, the relatively high standard deviation of 0.086217 and wide range (from 0.136036 to 0.346536) show that this alignment is inconsistent. Some runs achieve moderate structure, while others remain close to weak or near-random alignment. This reinforces that different runs converge to distinct but equally valid encoding schemes, rather than a single semantically grounded protocol.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),

    [loss],             [-0.540864], [0.279994], [-0.668234], [-0.040009],
    [mean_reward],      [1.306295],  [0.004578], [1.301276],  [1.311425],
    [expected_reward],  [1.295786],  [0.004865], [1.290298],  [1.301480],
    [recon_acc],        [0.992553],  [0.016652], [0.962766],  [1.000000],
    [topsim],           [0.261673],  [0.086217], [0.136036],  [0.346536],
  ),
  caption: [Aggregate metrics across 5 runs for the reconstruction experiment.],
)

The second experiment also consisted of 5 runs, but with reconstruction disabled as shown in *#underline[@setting-exp-2]*. In this setting, the sender is only optimised to support action selection and no longer receives direct pressure to encode the exact entity identity. This makes the protocol free to trade strict discrimination for compact reuse if that improves downstream reward.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),

    [Mode],                [GS (Gumbel-Softmax)],
    [Vocab size],          [50],
    [Max message len],     [2],
    [Temperature],         [2.0],
    [Temp decay],          [1.0],
    [Temp minimum],        [0.1],
    [Recon weight],        [0.0],
    [Action entropy],      [0.1],
    [Action temperature],  [2.0],
    [Reward normalise],    [True],
    [Reward scale],        [0.5],
    [Learning rate],       [0.001],
    [Batch size],          [64],
    [Epochs],              [50],
  ),
  caption: [Survival game training configuration for run 23 (no reconstruction).],
) <setting-exp-2>

Without reconstruction, I expected the protocol to become more compressed and cluster entities that induce similar policy consequences. This is exactly what appears in the message reuse analysis, where reuse ratios are substantially higher than in Experiment 1, reaching between 0.444444 and 0.769231 across runs. Representative clusters include high-frequency resource groupings and threat/environment groupings, indicating that messages are no longer acting as unique identifiers but rather as functional categories.

#figure(
  image("../outputs/metrics/experiments/no_recon/loss_train_test.png"),
  caption: [Training and test loss across epochs for the no reconstruction setting.],
) <loss-no-recon-exp2>

As shown in *#underline[@loss-no-recon-exp2]*, both train and test losses drop quickly in the first epochs and then flatten into a very stable plateau with narrow variation bands. Compared to Experiment 1, this trajectory is smoother and less sensitive to run-level stochasticity, which suggests that removing the reconstruction objective simplifies the optimisation landscape. The final loss statistics are tightly concentrated (mean -0.665787, std 0.001308), reinforcing that runs converge to similarly performant minima.

#figure(
  image("../outputs/metrics/experiments/no_recon/topsim_test.png"),
  caption: [TopSim (test) across epochs for the no reconstruction setting.],
) <topsim-no-recon-exp2>

The TopSim behaviour in *#underline[@topsim-no-recon-exp2]* is consistently higher than in the reconstruction setting, with a mean of 0.413854. This indicates stronger alignment between message-space similarity and entity semantic similarity under policy-only pressure. However, the standard deviation remains relatively high (0.143490), with values ranging from 0.199462 to 0.544537, showing that while semantic organisation emerges more clearly on average, it is still not fully deterministic across seeds.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),

    [loss],             [-0.665787], [0.001308], [-0.667547], [-0.664067],
    [mean_reward],      [1.307678],  [0.002518], [1.305152],  [1.311018],
    [expected_reward],  [1.296982],  [0.002726], [1.293769],  [1.300575],
    [recon_acc],        [0.045135],  [0.030472], [0.013572],  [0.093665],
    [topsim],           [0.413854],  [0.143490], [0.199462],  [0.544537],
  ),
  caption: [Aggregate metrics across 5 runs for the no reconstruction experiment.],
)

The third experiment kept no reconstruction but increased action entropy to 0.3, with settings shown in *#underline[@setting-exp-3]*. This change was designed to encourage broader policy exploration while preserving the same message budget and optimisation pipeline.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),

    [Mode],                [GS (Gumbel-Softmax)],
    [Vocab size],          [50],
    [Max message len],     [2],
    [Temperature],         [2.0],
    [Temp decay],          [1.0],
    [Temp minimum],        [0.1],
    [Recon weight],        [0.0],
    [Action entropy],      [0.3],
    [Action temperature],  [2.0],
    [Reward normalise],    [True],
    [Reward scale],        [0.5],
    [Learning rate],       [0.001],
    [Batch size],          [64],
    [Epochs],              [50],
  ),
  caption: [Survival game training configuration for run 24 (no recon, action entropy 0.3).],
) <setting-exp-3>

From the reuse statistics, this condition remains strongly compositional: reuse ratios lie between 0.500000 and 0.642857, with large shared clusters for resources and environment-like hazards. This indicates that moderate entropy regularisation does not collapse communication; instead it preserves stable structured grouping while allowing more stochastic action exploration.

#figure(
  image("../outputs/metrics/experiments/tuning_ae03_len2/loss_train_test.png"),
  caption: [Training and test loss across epochs for the action-entropy 0.3 setting.],
) <loss-ae03-exp3>

In *#underline[@loss-ae03-exp3]*, optimisation is again characterised by a rapid initial decrease followed by a long stable phase, but converging to a lower mean loss than no_recon (-0.734955 versus -0.665787). The train-test gap remains small and consistent, indicating that this additional entropy regularisation does not introduce visible overfitting in this setup.

#figure(
  image("../outputs/metrics/experiments/tuning_ae03_len2/topsim_test.png"),
  caption: [TopSim (test) across epochs for the action-entropy 0.3 setting.],
) <topsim-ae03-exp3>

TopSim in *#underline[@topsim-ae03-exp3]* is moderately high and stable, with mean 0.438693 and a narrower spread than no_recon (std 0.058650). This suggests that increasing exploration pressure to 0.3 still supports robust semantic structuring of the protocol, while reducing sensitivity to random initialisation compared with the baseline no reconstruction condition.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),

    [loss],             [-0.734955], [0.001554], [-0.736442], [-0.732469],
    [mean_reward],      [1.308570],  [0.004080], [1.303977],  [1.312858],
    [expected_reward],  [1.146950],  [0.004690], [1.141652],  [1.153071],
    [recon_acc],        [0.035267],  [0.030517], [0.000000],  [0.082318],
    [topsim],           [0.438693],  [0.058650], [0.350415],  [0.502967],
  ),
  caption: [Aggregate metrics across 5 runs for the action-entropy 0.3 experiment.],
)

The fourth experiment further increased action entropy to 0.5 while keeping message length at 2, as shown in *#underline[@setting-exp-4]*. The objective here was to test whether stronger exploration destabilises protocol formation or, alternatively, improves coverage while preserving semantic structure.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),

    [Mode],                [GS (Gumbel-Softmax)],
    [Vocab size],          [50],
    [Max message len],     [2],
    [Temperature],         [2.0],
    [Temp decay],          [1.0],
    [Temp minimum],        [0.1],
    [Recon weight],        [0.0],
    [Action entropy],      [0.5],
    [Action temperature],  [2.0],
    [Reward normalise],    [True],
    [Reward scale],        [0.5],
    [Learning rate],       [0.001],
    [Batch size],          [64],
    [Epochs],              [50],
  ),
  caption: [Survival game training configuration for run 26 (no recon, action entropy 0.5, message length 2).],
) <setting-exp-4>

Empirically, this setting still produces strong reuse, with ratios between 0.600000 and 0.687500. The same large resource clusters continue to appear (e.g., Wood|Water|Herbs|Berries|Stone families), which indicates that even under stronger entropy pressure the protocol remains organised around shared decision-relevant semantics rather than collapsing into noisy symbols.

#figure(
  image("../outputs/metrics/experiments/tuning_ae05_len2/loss_train_test.png"),
  caption: [Training and test loss across epochs for the action-entropy 0.5, length-2 setting.],
) <loss-ae05-l2-exp4>

As shown in *#underline[@loss-ae05-l2-exp4]*, both curves descend rapidly and then stabilise at very low-variance plateaus. The final mean loss is -0.879204 with std 0.001792, indicating highly consistent convergence across seeds. This is one of the most stable optimisation profiles among the tested conditions.

#figure(
  image("../outputs/metrics/experiments/tuning_ae05_len2/topsim_test.png"),
  caption: [TopSim (test) across epochs for the action-entropy 0.5, length-2 setting.],
) <topsim-ae05-l2-exp4>

TopSim in *#underline[@topsim-ae05-l2-exp4]* reaches the highest mean among all experiments (0.452432), but with substantial variability (std 0.119406). This implies that the setting can discover highly semantically aligned protocols, yet still admits lower-structure local optima in some runs. In other words, peak representational quality improves, but seed sensitivity is not fully removed.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),

    [loss],             [-0.879204], [0.001792], [-0.880774], [-0.876178],
    [mean_reward],      [1.306050],  [0.004487], [1.299518],  [1.310862],
    [expected_reward],  [0.881756],  [0.005186], [0.875660],  [0.889692],
    [recon_acc],        [0.014775],  [0.020056], [0.000000],  [0.048428],
    [topsim],           [0.452432],  [0.119406], [0.247718],  [0.546769],
  ),
  caption: [Aggregate metrics across 5 runs for the action-entropy 0.5, length-2 experiment.],
)

The fifth experiment kept action entropy at 0.5 but increased maximum message length to 3, as shown in *#underline[@setting-exp-5]*. This directly tests whether additional channel capacity improves protocol quality or simply preserves similar solutions with a larger representational budget.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),

    [Mode],                [GS (Gumbel-Softmax)],
    [Vocab size],          [50],
    [Max message len],     [3],
    [Temperature],         [2.0],
    [Temp decay],          [1.0],
    [Temp minimum],        [0.1],
    [Recon weight],        [0.0],
    [Action entropy],      [0.5],
    [Action temperature],  [2.0],
    [Reward normalise],    [True],
    [Reward scale],        [0.5],
    [Learning rate],       [0.001],
    [Batch size],          [64],
    [Epochs],              [50],
  ),
  caption: [Survival game training configuration for run 26 (no recon, action entropy 0.5, message length 3).],
) <setting-exp-5>

Message reuse remains high in this condition as well (0.562500 to 0.785714), and cluster motifs are nearly identical to the length-2 version. This suggests that adding one extra symbol position does not fundamentally alter the communication regime; the protocol continues to encode broad semantic classes that align with action-relevant structure in the environment.

#figure(
  image("../outputs/metrics/experiments/tuning_ae05_len3/loss_train_test.png"),
  caption: [Training and test loss across epochs for the action-entropy 0.5, length-3 setting.],
) <loss-ae05-l3-exp5>

The loss dynamics in *#underline[@loss-ae05-l3-exp5]* closely match the length-2 case: rapid early improvement and then a stable plateau with minimal train-test divergence. Final loss statistics are also very similar (mean -0.879449, std 0.001422), indicating that extending message length does not yield a clear optimisation advantage in this task.

#figure(
  image("../outputs/metrics/experiments/tuning_ae05_len3/topsim_test.png"),
  caption: [TopSim (test) across epochs for the action-entropy 0.5, length-3 setting.],
) <topsim-ae05-l3-exp5>

As seen in *#underline[@topsim-ae05-l3-exp5]*, semantic alignment remains strong (mean 0.439227) and comparatively more stable than the length-2 variant (std 0.057365 vs 0.119406). This indicates that the extra symbol capacity mainly reduces variance rather than increasing peak semantic alignment. Overall, both action-entropy 0.5 settings converge to similarly effective communication systems, with length 3 offering slightly more consistent TopSim but no major reward or loss gains.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),

    [loss],             [-0.879449], [0.001422], [-0.880940], [-0.877326],
    [mean_reward],      [1.306937],  [0.003845], [1.302265],  [1.311607],
    [expected_reward],  [0.882388],  [0.002572], [0.880565],  [0.886863],
    [recon_acc],        [0.053280],  [0.037035], [0.004505],  [0.091442],
    [topsim],           [0.439227],  [0.057365], [0.380925],  [0.534567],
  ),
  caption: [Aggregate metrics across 5 runs for the action-entropy 0.5, length-3 experiment.],
)
)