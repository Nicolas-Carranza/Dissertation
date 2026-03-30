#import "@preview/clean-math-paper:0.2.5": *

Prototype.py

Before building the full training stack, I developed the prototype implementation as the first complete version of the game logic, and this work was carried out before the final agentic pipeline was assembled, mainly from the end of semester one into the early implementation weeks because I needed a stable semantic reference that could later be reused by both data generation and reward computation. As I have done in previous projects, I organised the file into explicit sections so each part of the game could be reasoned about independently while still composing into one execution flow, which made it easier to manage complexity and maintain clarity as the codebase grew through development and supervisor meetings.

The first section defines the vector schema and global mappings, where dimensions, probabilities, names, and value maps are fixed at the top level to avoid hidden assumptions in later functions. For example, the definition of danger injury probabilities is defined as #raw("DANGER_INJURY_PROB = {0: 0.00, 1: 0.10, 2: 0.25, 3: 0.45, 4: 0.70}"), later used throughout different functions and files.

The second section defines the full entity catalogue, where I made a data class where the 40 entities are defined and manually curated to keep encounters plausible, as this project required a controlled world where interpretation and analysis remain feasible. Here you can also find the different spawn weights for each entity, which were tuned through trial and error to achieve a balanced distribution that supports learning without overwhelming the agent with too many rare or too many common entities.

Moreover, the next section introduces the Inventory and GameState classes in operational terms, where Inventory provides standard add, remove, and query operations for tools, materials, and consumables. The GameState class on the other hand provides the evolving state that all decision and reward logic depends on. I also added tracking related fields in this stage because I already expected that behaviour analysis would be central later, and this early decision avoided retrofitting instrumentation into unstable code paths.

Following this, I define the action space and validity logic, where the game determines which actions are legal for a given entity and state, and then I implement action resolution where each chosen action produces both a concrete state transition and a reward value. It is crucial to note that this is the mechanical core of the environment, because health, energy, inventory, and weather effects all depend on these functions, and also the strategic core, because reward shaping is encoded here and therefore determines what kind of behaviour can emerge during optimisation. Also note I used internal abstractions such as _\_resolve\_craft_ to avoid duplicated logic across crafting variants, which made behavioural changes much safer when I later tuned the reward surface, a practice learned from many years of experience.

Then, the next section moves to encounter generation, where it first constructs a weather compatible pool of entities, then applies the defined weight sampling with sanity checks (in case no entity exists under the weather condition), and finally samples a concrete entity from the selected type. This method was intentional because a direct uninformed random entity draw would have produced unrealistic distributions and weaker task pressure for communication. Although this part looks simple at runtime, it is important for distribution control because encounter frequency directly affects both baseline performance and model learning dynamics.

After this, the episode runner section executes the full turn pipeline in sequence, where a new GameState is created, energy drain is applied, weather is updated, an encounter is generated, valid actions are computed, a policy callback chooses an action, the action is resolved, and terminal conditions are checked before returning the updated state. Note that here the _policy\_fn_ hook is a strategy interface rather than model code itself, which allowed me to evaluate handcrafted baselines and later plug in learned agents with the same environment implementation.

In addition, I implemented four baseline policies in the prototype stage as defined in the Design chapter, namely random, greedy, optimal heuristic, and blind resting. While implementing them, I ensured that each policy reflected a distinct approach to decision-making. For example, the greedy policy follows explicit priority rules such as "if danger then mitigate, else endure it", or "if energy drops lower than 25 and I have food then eat". All of the rules are defined within the comments of the function. Similarly, the optimal heuristic policy follows a more informed strategy that invests earlier in tools, hunts more aggressively once equipped, and manages food more proactively, while the blind policy isolates performance without meaningful perception. This baseline phase was developed before the final implementation chapter experiments, mainly across weeks one to three in semester two, and it provided the first evidence that the environment supports non trivial strategy differences.


Finally, the closing sections of prototype.py handle statistics, reporting, and baseline execution entry points, so this file became the canonical semantic source used later by data.py and losses.py, ensuring that world mechanics seen during data generation remain aligned with the reward simulation used during optimisation.

train.py

Following the prototype stage, in train.py I orchestrate the full pipeline rather than implementing game mechanics directly, and this separation was deliberate because I wanted experiment control and reproducibility to be handled in one place. The file has two main components, where get_params defines the configurable interface and main executes the training workflow, and this structure mirrors the Design chapter goal of high configurability through arguments rather than constant code rewrites.

Furthermore, get_params integrates project specific arguments with standard EGG arguments through core.init, which means one command line entry point controls model mode, data size, optimisation settings, logging options, and analysis toggles. This was crucial during the run to run iteration phase from approximately week six onward, because I could test hypotheses about temperature, entropy, reward scaling, and message settings without rewriting infrastructure.

Then main follows a strict build order, where data is generated through data.py, the game object is assembled through games.py, optimisation is built through EGG core utilities, callbacks are attached for progress and analysis, temperature updating is configured for communication control, and trainer execution runs the actual learning process. Although trainer.train appears as a single call, all previously defined components are instantiated before that point, so this file is effectively the integration layer that binds every implementation file into one executable experimental system.

Moreover, I added tracking flags for topsim, disent, posdis, and entropy as part of this integration stage, because by that point I had already begun planning the Evaluation chapter and needed instrumentation to be available during long runs rather than as an afterthought. Thus train.py is not only a launcher, it is also the consistency point where output directories, naming conventions, logging granularity, and analysis artefacts are coordinated.

data.py

After training orchestration is defined, the role of data.py is to convert the prototype world into model ready supervised samples, and this file is therefore the bridge between environment simulation and training batches. As described in Design, the game mechanics exist in prototype.py, yet the agents consume tensors, so this module encapsulates the transformations required to preserve semantics while adapting format.

The file is organised into four parts, namely encoding helpers, episode simulation for data generation, a PyTorch dataset class, and a dataloader factory. The three encoding helpers follow a clear separation of concerns, where one converts a six dimension entity vector into a 30 dimension one hot sender input, another converts GameState into a normalised 16 dimension receiver state tensor, and a third packs label information used later by the loss helpers. This explicit encoding stage is important because it keeps feature semantics stable across experiments and prevents hidden coupling between environment and architecture code.

Following this, the data generation simulator produces episodes and then decomposes them into turn level samples, where each sample is split into sender input, receiver input, and labels. In this process, the valid action mask is added to the receiver input with 1 for valid actions and 0 for invalid actions, which later supports masked decision logic in the receiver and reward interpretation in losses.

The final factory stage handles dataset scale, split discipline, and overlap control. On the one hand, episodes are generated with controlled seeds for reproducibility, shuffled at episode level rather than turn level to preserve trajectory consistency, split into train validation test partitions, and flattened into samples, and on the other hand a fingerprint based deduplication pass removes overlap between partitions. This fingerprint mechanism became essential after overlap was detected during supervisor guided review, and by hashing sender and receiver inputs for membership checks, it provided an efficient and reliable way to eliminate leakage across splits.

After deduplication, each partition is wrapped by SurvivalGameDataset and loaded through torch.utils.data dataloaders, so the output of data.py is a clean and reproducible tensor pipeline aligned with the game semantics defined in prototype.py.

games.py

With the data pipeline established, games.py is the composition layer that builds the actual EGG game object from sender receiver architectures, wrappers, and loss. Although the file is intentionally compact, its responsibility is central, because it decides how communication, decision, and optimisation components are connected under either GS or Reinforce mode.

In this file I instantiate Sender and Receiver from archs.py, instantiate SurvivalLoss from losses.py with relevant configuration terms such as action entropy coefficient, action temperature, and reward normalisation, then wrap the agents with the EGG recurrent wrappers according to mode. For GS mode this means RnnSenderGS and RnnReceiverGS are used and combined through SenderReceiverRnnGS, while Reinforce mode uses the corresponding Reinforce wrappers.

Moreover, this design keeps the internal model definitions independent from training regime specifics, because wrappers handle sequence level communication behaviour and optimisation compatible interfaces, while my architecture code focuses on representational learning for sender and receiver cores. Thus games.py is small by lines of code, yet it is the exact place where the full talking game is assembled end to end.

archs.py

At this point, archs.py defines the project specific sender and receiver neural cores described in the Design chapter, and this file was developed as the controlled implementation of the intended information asymmetry, where sender sees the entity and receiver sees state plus message. I implemented these as two classes, Sender and Receiver, with compatible hidden dimensionality so they integrate smoothly with the recurrent wrappers.

The sender is an MLP that maps the 30 dimension one hot entity input into a hidden representation used to initialise message generation. In practical terms, forward propagation applies linear layers with ReLU activations to produce the initial recurrent state, and the message sequence itself is generated by the wrapper in gs_wrappers.py, where the recurrent cell is unrolled for max_len steps, vocabulary logits are produced at each step, differentiable symbols are sampled through Gumbel Softmax, and an EOS token is appended at the end.

The receiver follows a similar linear ReLU core but includes both action prediction and reconstruction pathways. The input to the receiver core is the concatenation of message representation and state features, while the valid mask is applied at the logits stage rather than as hidden features, which enforces the design choice that semantic interpretation should be driven by message and state rather than by directly feeding legality structure into feature extraction. In this setup, fc1 acts as the hidden feature extractor and fc2 maps hidden features to action logits over eleven actions, and invalid action logits are set to a very large negative value so that masked actions receive effectively zero probability after softmax.

Furthermore, I added a reconstruction head using nn.Sequential with intermediate dimensionality and final output over N_ENTITIES, because this auxiliary signal was intended to strengthen communication learning when action rewards alone were too weak early in training. The gradient route for this path is important, since cross entropy loss updates the reconstruction head, then flows through the receiver hidden representation and recurrent message decoder, through the message channel, and back into sender parameters. Thus the reconstruction objective contributes not only to receiver discrimination but also to shaping sender messages.

In addition, the GS receiver wrapper decodes the incoming sequence step by step and provides outputs across time, and the game level wrapper aggregates these outputs with EOS weighted logic, which keeps training differentiable while accounting for variable effective message length.

losses.py

From the architecture side, the next layer is losses.py, which defines the training objective and therefore acts as the optimisation interface between environment semantics and model updates, and for clarity I separated it into label unpacking helpers, reward computation helpers, and the main loss class.

The unpacking helpers first recover structured components from packed labels, then resolve entity identity and target index, and finally reconstruct GameState from encoded vectors so reward simulation can be run in the same semantic space as the environment. This means the model does not infer the original entity at this stage, instead it recovers ground truth metadata from labels and falls back to vector matching only when indices are invalid, which protects training from inconsistent labels while keeping behaviour deterministic.

The reward helpers compute both single action outcomes and expected rewards across actions, and this is the crucial part where game logic is translated into differentiable supervision. As requested, the mathematical specification below is kept intact.

In `losses.py`, the function builds a deterministic reward vector $R in RR^{11}$, where each invalid action starts at $-30$ and valid actions are replaced by the formulas below.

Let $d$ be danger level, $E$ energy, $H$ health, and

$ F = max(1, "max_turns" - "turn") dot 8 $

be the future-value term used in death-proximity shaping.

For Hunt, when the target is an animal

$ 
s = 
cases(
  0.85 quad quad "if has tool and" d ≤ 1,
  0.65 quad quad "if has tool and" d > 1,
  0.50 quad quad "if no tool needed",
  0.15 quad quad "otherwise",
) quad quad dot quad H/100 quad dot quad min(E, 50)/50
$

$ R_"succ" = (15 + 10 dot d) - (0.10 dot d) dot ((5 + 5 dot d)/2) $

$ R_"fail" = -5 - (0.25 dot d) dot 15 $

$ R_"hunt" = s dot R_"succ" + (1 - s) dot R_"fail" $

If the entity is not an animal, $ R_"hunt" = -10$.

For Gather, if the target is a resource, craft opportunity or an event

$ R_"gather" = 5 + 2 dot q - [d >= 2] dot (p_d dot ((5 + 5 dot d)/2)) $

where $q$ is inventory quantity and $p_d$ is the injury probability. Otherwise $ R_"gather" = -10$.

For Flee

$
R_"flee" =
cases(
  2   quad quad "danger and" d ≥ 3,
  1   quad quad "animal and" d ≥ 3,
  -2  quad  " otherwise",
)
$

For Rest

$ R_"rest" = -1 $

For Mitigate

$
R_"mitigate" =
cases(
  -10 quad quad quad quad quad quad quad quad " if not a danger",
  -0.3 dot (h_"loss" + e_"loss") quad quad "if danger but tool missing",
  20 + 5d + b quad quad quad quad quad " if danger and tool available",
)
$

$
b =
cases(
  5 quad quad "fire vs cold",
  0 quad quad "otherwise",
)
$

where $ h_"loss"$ and $e_"loss"$ each are the unmitigated damage at danger level $d$, scaled by the endure factor.

For Endure
$
R_"endure" =
cases(
  -10 quad quad quad quad quad quad quad quad " if not a danger",
  -0.3 dot (h_"end" + e_"end") quad quad "if danger",
)
$

For Eat
$
R_"eat" =
cases(
  -5 & "  if no food",
  g + 5 dot ["cooked"] - 3.6 dot ["raw"] + 10 dot [E < 30] & "  otherwise",
)
$

where $g in {15, 10, 6, 5, 5, 3}$ depending on food type.

For Craft actions, spear fire shelter rod

$
R_"craft" =
cases(
  -5 quad quad &  " if already owned",
  -10 quad quad & " if insufficient materials",
  50 quad quad & " if success",
)
$

After each base action reward, I check for starvation, if it eats starving it gives a bonus or more negative points if unaddressed, then also checks the health, with the same idea where risky actions at low health are penalised and fleeing gets a bonus.

If $E < 20$, I define starvation urgency as

$ u_E = 1 - E/20 $

Then add

$
Delta_E =
cases(
  +0.4 dot u_E dot F quad quad "Eat",
  +0.1 dot u_E dot F quad quad "Rest",
  -0.3 dot u_E dot F quad quad "otherwise",
)
$

If $H < 25$, I define injury urgency as

$ u_H = 1 - H/25 $

And let $"risky"$ mean if there is an animal entity with $d >= 2$ and action Hunt, or the action Endure with danger. Then add

$
Delta_H =
cases(
  -0.5 dot u_H dot F quad quad "Risky",
  +0.1 dot u_H dot F quad quad "Flee",
  0 quad quad quad quad quad quad "   otherwise",
)
$

So each valid action gets

$ R_a = R_a^"base" + Delta_E + Delta_H $

while invalid actions remain at $-30$.

Then, in the main GS loss path, for each batch sample I reconstruct the exact survival context, compute rewards for all actions, and store them in reward_matrix, while also preparing reconstruction targets and action behaviour diagnostics, so after the loop each row represents the full action value landscape for one concrete state. If reward normalisation is enabled, rewards are standardised per sample across valid actions only, while invalid penalties remain fixed at minus thirty, because this prevents large magnitude actions such as successful crafting from dominating gradients and suppressing subtler but relevant differences.

Following this, logits are converted into soft action probabilities using action temperature, expected reward is computed as the probability weighted sum over actions, and game loss is defined as negative expected reward scaled by reward_scale. Moreover, entropy regularisation is added to discourage premature collapse to a single action, and reconstruction cross entropy is added to encourage message content that preserves entity identity information. Thus the final objective learns both decision quality and communication informativeness within one differentiable framework

#raw(" 
loss = game_loss + self.recon_weight * recon_loss + entropy_loss
")

In the evaluation chapter I discuss why reconstruction weight was eventually reduced and in some settings set to zero, since this changed how the protocol balanced strict identity coding against synonym like message behaviour.

callbacks.py

Finally, callbacks.py was added to monitor behaviour and language evolution in ways that the default training loss output cannot show directly, and while EGG provides generic callbacks in core, this project required domain specific diagnostics for survival and emergent communication analysis. The file therefore defines two classes, SurvivalGameEvaluator and MessageAnalyzer, and both are analysis components rather than optimisation components.

The first class runs full simulated episodes with the current model extracted from the game object, and this process acts as policy evaluation rather than learning, because no gradients are applied and no weights are updated. In practical terms, the evaluator plays fresh episodes using the current sender receiver policy, applies the chosen actions to the environment, computes rewards and terminal outcomes, and reports survival focused metrics such as survival rate, mean reward, valid action rate, message length, and action distribution. Following this approach, the model is tested in sequential gameplay conditions that reflect long horizon state transitions rather than isolated turn samples.

The second class analyses emergent language from validation interaction logs, and unlike the evaluator it does not run new episodes. Instead, it reads the messages produced in the current validation stage, converts GS soft distributions to discrete token sequences when needed, groups message strings by entity type, counts frequencies per type, computes uniqueness and top frequent messages, tracks global diversity, then performs a finer pass by specific entity vector before writing snapshots and printing summaries. Moreover, this class currently keeps top three messages in practice, since the reporting path is fixed to top three for legacy compatibility, so the analysis emphasises stability and comparability across runs.

Overall, callbacks.py is less central to forward execution than architecture or loss code, yet it is essential for interpretation, because it provides the behavioural and protocol evidence used later in evaluation and critical appraisal.

== Training, Results and Design Changes

At this stage of the project, the focus moved from building components to running controlled experiments, and although the implementation section above explains how each file was constructed, this section explains how training was actually executed over time, how monitoring was performed through the callback outputs, and how each observed result informed the next design decision across the sequence of runs.

Following the initial integration of the full pipeline, the first experimental block covered runs 1 to 8 on my personal laptop using CPU only, and this period corresponds to the earlier semester two weeks where each run required roughly five to nine hours, which constrained iteration speed but still provided crucial behavioural evidence that the early configuration was learning stable but sub optimal strategies. Moreover, this phase was where I validated practical convergence behaviour, where I did not keep epoch counts fixed by default but adjusted training length according to observed convergence in the logs, because continuing long after metric flattening was not efficient on CPU while stopping too early prevented reliable comparisons.

After these early runs, I continued iterating and also completed runs 10 to 12, and although those files are not currently central in the summary artefacts, they were part of the same exploratory stream that connected weekly supervisor feedback to concrete parameter updates. Moreover, the overlap issue was identified in week seven, and the three way split with fingerprint based deduplication was introduced before the later run blocks, which meant the more reliable comparisons began once the overlap was removed rather than at run 13. Following this, the week seven and week eight experiments introduced the central parameter changes that shaped later work, including the change from five entity classes to forty entities, the reconstruction head update to forty outputs, recon_weight reduced to 1.5, reward_scale increased to 0.5, temperature set to 2.0 with decay 0.975, and the action policy fixes of entropy regularisation, action temperature, and reward normalisation, and these changes produced the run 5 and run 6 improvements but also confirmed the local optimum problem where gather and eat dominated despite perfect reconstruction.

Furthermore, run 7 and run 8 were explicit hyperparameter tests in the same phase, where higher entropy, lower action temperature, no temperature decay, a lower initial GS temperature, and longer training produced higher variance and weaker reconstruction, and this evidence made it clear that overusing the entropy term competes with the reconstruction objective. Then, in the later week eight runs 11 and 12, the no decay setting produced the most stable protocol, with higher entropy and fuller vocabulary usage in run 12, posdis near zero with weak bosdis and topsim around 0.3, and consistent survival rates where better communication did not materially change the action profile, which aligned with the analysis that the language was mostly holistic with only weak category level regularities.

Then run 13 became a major transition point in logging rather than data integrity, because it was the first run where I treated message progression and final message snapshots as standard outputs, which provided a much stronger basis for later language analysis than relying only on scalar metrics. Furthermore, the next block covered runs 14 to 18 with reconstruction enabled, where I systematically changed settings such as temperature schedules and related optimisation parameters, and this was the stage in which I observed in the more recent experiments that removing temperature decay improved stability in several trajectories. On the one hand, reconstruction gave strong identity pressure and high decoding quality, and on the other hand the same pressure sometimes restricted action policy flexibility, which led to the supervisor guided decision to test recon_weight set to zero, despite this being initially counter to the design intuition. Despite this being counter intuitive, the no reconstruction setting produced behaviour that was more aligned with the objective of improving policy outcomes while still maintaining useful communication structure.

From run 18 onward, training moved to the lab machines with CUDA enabled as described in the Design chapter, and this reduced experiment runtime enough to support repeat based reporting rather than one off claims. Thus, from this point I adopted a minimum of five runs per fixed configuration so the reported values could be expressed with standard deviation and not only single run point estimates, which materially improved result credibility and made comparisons between settings statistically more informative.

Following this shift, I organised the experimental outputs into grouped directories, where with reconstruction runs are stored under outputs with_recon for runs 14 to 18, no reconstruction runs are stored under outputs no_recon for runs 19 to 23, and scripted no reconstruction tuning runs are stored under outputs no_recon_tuning for the fifteen run batch built as three parameter configurations each repeated with five seeds. The scripted batch was a long execution block of approximately fifteen hours, and its purpose was to test parameter effects extrapolated from lessons in the preparation phase basic games experiments, while keeping seed controlled repetition for fair comparison.

For logging and monitoring, I leveraged the callback outputs directly during training rather than adding a separate external logger, where SurvivalGameEvaluator provided periodic policy level metrics from fresh episode rollouts and MessageAnalyzer provided language snapshots from validation messages, and this gave me two complementary views, one for survival behaviour and one for protocol structure. In practical terms, each run generated train logs with per epoch scalar metrics, periodic message progression files, and final message snapshot files, then post processing scripts produced aggregate tables and plots used for comparison. Thus, logging was not treated as cosmetic output, it was the core mechanism used to decide whether a change was retained, reverted, or moved to the next tuning phase, and it also served as the record for parameter changes across runs, such as recon_weight shifts, temperature decay removal, action temperature scaling, entropy strength, reward normalisation, and the move from class level to entity level reconstruction.

Using the five run summaries currently available for the reconstruction and no reconstruction settings, the final test epoch aggregates show a clear pattern. With reconstruction enabled across runs 14 to 18, mean reward was 1.306295 with standard deviation 0.004578, expected reward was 1.295786 with standard deviation 0.004865, reconstruction accuracy was 0.992553 with standard deviation 0.016652, and topographic similarity was 0.261673 with standard deviation 0.086217. Without reconstruction across runs 19 to 23, mean reward was 1.307678 with standard deviation 0.002518, expected reward was 1.296982 with standard deviation 0.002726, reconstruction accuracy dropped to 0.045135 with standard deviation 0.030472 as expected under zero weight, and topographic similarity increased to 0.413854 with standard deviation 0.143490, while valid action rate remained 1.0 in both settings. Moreover, message length remained stable at 3.0 in the no reconstruction block, indicating that the policy shift was not due to a trivial collapse in sequence length.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.header(
      [*Setting*],
      [*Runs*],
      [*Mean reward*],
      [*Expected reward*],
      [*Recon acc*],
      [*TopSim*],
    ),
    [With reconstruction], [14 to 18], [1.3063 ± 0.0046], [1.2958 ± 0.0049], [0.9926 ± 0.0167], [0.2617 ± 0.0862],
    [No reconstruction], [19 to 23], [1.3077 ± 0.0025], [1.2970 ± 0.0027], [0.0451 ± 0.0305], [0.4139 ± 0.1435],
  ),
  caption: [Final test epoch summary over five run groups, values are mean plus minus standard deviation.]
)

In addition, message reuse summaries and heatmap outputs from the no reconstruction runs showed partial synonym like behaviour where certain semantically close entities could share or rotate messages, including examples such as fish and salmon under overlapping codes in some runs, while other entities still maintained distinct mappings, and these patterns were complemented by later observations such as resource prefixes like 38 and a cheap signal case where fish became 0 0 0, and this behaviour is reported here as an observed result rather than interpreted causally, because the deeper interpretation is reserved for the Evaluation chapter.

Following these observations, the design changes were made incrementally and always tied to observed evidence rather than preference. Early CPU runs motivated reward shaping and exploration adjustments, mid stage runs motivated stricter logging and snapshot capture, post overlap runs motivated identity tracking at higher granularity, and later CUDA runs enabled repeated configuration comparison with variance reporting, which in turn justified retaining no reconstruction as a serious experimental branch. Thus the final training narrative is not a single parameter success story, it is a sequence of constrained decisions where each new setting was chosen because the previous logs exposed a specific failure mode or limitation.

*IMPORTANT REMINDER FOR FINAL DISSERTATION EDITING, add the exact run commands in README and in the Appendix so all reported figures can be reproduced from the documented command lines.*

Moreover, to keep the main chapter readable, only summarised curves and tables are intended to remain here, while full per epoch outputs, full message snapshots, and extended plots should remain in the Appendix where they can be referenced without overloading the implementation narrative.
