#import "@preview/in-dexter:0.5.3": *
#import "@preview/wordometer:0.1.4": word-count, total-words

#show: word-count.with(exclude: <no-wc>)

// Basic document settings
#set page(numbering: "1")
#set par(justify: true, first-line-indent: 1em)
#set text(lang: "en", region: "GB", hyphenate: auto)
#set list(indent: 2em)

#show link: underline

// Set heading numbering and indentation styles
#set heading(numbering: "1.")

\ \

// Title Page
#align(center)[
  #text(26pt)[*Communication Emerges from Multi-Agent Survival Games*] \ \ \
]

\ \ \

#align(center)[
  #text(16pt)[
    Nicolas Carranza (220005595)
  ]
]

#align(center)[
  #text(16pt)[
    Supervisor: Dr. Phong Le
  ]
]

#align(center)[
  #text(12pt)[
    April 2026
  ]
]

\ \ 



\ \ 

#align(center)[
  #image("Images/logo.png") \ \ \ \
  #text(18pt)[
    School of Computer Science \
    University of St Andrews \
    Scotland \
  ]
]

#pagebreak()

#set heading(numbering: none)

= Abstract <group1>

Outline of the project using at most 250 words.

Simple and short, outlining the research presented, what was achieved and how.

#pagebreak()

= Declaration <group1>

I hereby certify that this dissertation, which is approximately *#total-words* words in length, has been composed by me, that it is the record of work carried out by me and that it has not been submitted in any previous application for a degree. This project was conducted by me at the University of St Andrews from September 2025 to April 2026 towards fulfilment of the requirements of the University of St Andrews for the degree of Bachelor of Science (Honours) Computer Science And Management under the supervision of Dr. Phong Le.

In submitting this project report to the University of St Andrews, I give permission for it to be published online. I retain the copyright in this work.

April 3rd 2026 *[insert signature]*

#pagebreak()

// Table of Contents 

#outline( title: [Table of Contents], 
  indent: auto,
  depth: 3,
  target: <group1>
)

#pagebreak()

// Set heading numbering and indentation styles
#set heading(numbering: "1.")

= Introduction <group1>

\
Communication is a fundamental mechanism underpinning coordination in both natural and artificial systems. In biological settings, species rely on signalling to survive, whether through cooperative hunting, predator avoidance, or social organisation. These communication systems are not designed explicitly, but instead emerge over long periods of evolution, shaped by environmental pressures and the need for efficient interaction. Understanding how such communication protocols arise is a central problem in studying the origins of language and collective behaviour. However, investigating this process in natural environments is inherently difficult, as it unfolds over extended timescales and involves complex, often unobservable dynamics. As a result, isolating the factors that drive the emergence of structured communication remains a significant challenge.

Advances in machine learning provide an alternative approach to this problem. By simulating environments in which artificial agents must interact to achieve shared goals, it becomes possible to observe the emergence of communication under controlled conditions. In particular, multi-agent reinforcement learning (MARL) enables agents to develop signalling strategies through repeated interaction and reward-driven learning, offering a tractable framework for studying emergent behaviour. This project explores these ideas by investigating whether artificial agents can develop meaningful communication protocols in a custom survival-based scenario, leveraging the EGG (Emergence of lanGuage in Games) library @kharitonov:etal:2021 to place agents in environments where coordination is necessary for success. To this end, the EGG toolkit is used to implement communication games with discrete signalling channels, allowing agents to coordinate under task-specific pressures. Thus, the goal is to analyse whether communication emerges naturally from these interactions, and to what extent survival constraints influence the structure and development of emergent communication.

Formally, our core research question drives this investigation: *do meaningful communication protocols emerge when AI agents are constrained by survival scenarios requiring coordination?* To answer this, the project pursued four prioritised aims, first being to design and implement a tractable survival environment where communication is strictly necessary for optimal performance. Following this design, our second step is to train reinforcement learning agents within this environment to induce signalling behaviour. Once achieved, the third aim is to analyse the resulting protocols for structure, stability, and utility, finalising with our fourth aim: evaluating how specific design and training factors influence the emergence of these protocols. Given that this work was conducted as part of a 15-credit module, the scope was deliberately constrained to establishing these experimental foundations and conducting targeted analysis rather than pursuing large-scale simulations as it was initially designed for.

The project achieved a functional emergent communication pipeline, demonstrating a clear performance gap between agents with communication channels and those without. Specifically, agents trained with communication consistently outperformed baseline _blind_ or _random_ policies, verifying the necessity of signalling for the survival task. My analysis revealed that while protocols did stabilise and exhibit structural patterns, the system remained sensitive to local optima, where some message collapses were observed in specific training runs where agents converged on sub-optimal strategies. These results confirm the potential of survival-based signalling games while highlighting the stability challenges inherent in multi-agent reinforcement learning.


#pagebreak()

= Context Survey <group1>

Surveying the context, the background literature and any recent work with similar aims. *Should I assume that the marker know about basic Machine Learning concepts, or should I explain them?*

The context survey describes the work already done in this area, either as described in textbooks, research papers, or in publicly available software. 

You may also describe potentially useful tools and technologies here but do not go into project-specific decisions.

== Emergent communication in multi-agent systems <group1>

Talk about what EC is, the context and history of this field and the variables that are needed to be known. Make sure to define terminology.

Must use some papers here.

== Related signalling/reference games <group1>

Talk about the nature of survival games (not to be confused with video games). Mention the motivation behind using such a setting. *Maybe ill delete this section, should I give context on survival games?*

== EGG ecosystem and relevant prior approaches <group1>

Here I want to introduce the EGG repository and the work that has been done with it. Why it is useful in our context and why it is appropriate to use.

Here we can also explain the networked used and what model it is, going into detail on the maths and how the communication is formed / passed between agents.

Need to define and explain all relevant functions/files with names and links so it can be referenced in the implementation section.

== Methods and metrics used in prior work <group1>

Introduce the work metrics used to analyse our work for the runs with concepts as loss, accuracy, confusion matrix, precision, recall and the specific EC metrics.

Need to also list typical problems encountered with the models to setup a plausible explanation for my results (like the message collapse)

#pagebreak()

= Requirements Specification <group1>
\
The objectives for this project are presented below, organised into primary, secondary, and tertiary priorities. These objectives were originally defined in the DOER and provide a structured progression from foundational work to more advanced exploration of emergent communication in multi-agent systems. In practice, primary and most secondary objectives were completed, whereas tertiary objectives were partially deferred due to time and scope constraints.

== Original objectives from DOER <group1>
\
*Primary Objectives*

1. *Machine Learning Foundation Development:*
   The first objective involves establishing comprehensive understanding of machine learning fundamentals and PyTorch framework. This includes mastering multi-agent reinforcement learning concepts and techniques. Additionally, studying existing literature on emergent communication in AI systems forms a crucial component of this foundational phase.

2. *Emergent Communication Background Research:*
   This objective requires conducting thorough literature review on emergent communication research. The work involves gaining proficiency with the #underline[#link("https://github.com/facebookresearch/EGG/tree/main")[EGG library]] (Emergent Communication through Games). Furthermore, analysing existing game setups and agent architectures used in communication research will provide essential background knowledge.

3. *Baseline Implementation:*
   The third objective focuses on implementing AI agents within an existing survival game using the EGG library. This phase involves training the agents to develop communication protocols and evaluating their performance in survival scenarios, with particular emphasis on understanding the practical aspects of training communicating agents. A key outcome of this stage will be the establishment of baseline performance metrics and evaluation methodologies.

4. *Simple Game Design:*
   This objective centres on designing a basic survival scenario optimised for studying emergent communication. The focus remains on simple two-agent communication scenarios such as predator warning and hunting coordination. Implementation of visual input processing using image-based rather than symbolic inputs constitutes an important technical requirement.

\
*Secondary Objectives*

5. *Simple Game Implementation and Training:*
   This objective focuses on implementing AI agents within the custom-designed simple survival game. The work involves integrating the EGG framework with the newly created game environment, training agents to develop communication protocols specific to the designed scenarios, and conducting initial experiments to validate that meaningful communication emerges. This phase emphasises debugging the game mechanics, optimising training parameters, and establishing that the custom environment successfully promotes agent communication development.

6. *Language Analysis:*
   This objective requires analysing emergent communication protocols for fundamental properties like compositionality and efficiency. The work investigates how environmental pressures shape basic language development. Initial evaluation of communication strategies provides foundational analysis for the research.

\
*Tertiary Objectives*

7. *Complex Game Design:*
   This advanced objective involves developing sophisticated survival scenarios with multiple environmental factors and complex agent interactions. Implementation of varied survival challenges and dynamic environmental conditions extends beyond basic two-agent scenarios.

8. *Multi-Agent Communication:*
   This objective explores communication emergence in scenarios involving more than two agents. The work examines how group communication protocols develop and how information propagates through larger agent populations.

9. *Advanced Language Analysis Across Multiple Games:*
   This comprehensive objective involves analysing emergent communication protocols across different game environments and scenarios. The work compares systematicity, compositionality, and efficiency across varied survival contexts to understand broader patterns in language evolution.

10. *Thesis Documentation:*
   The final objective encompasses a comprehensive evaluation of experimental results. This includes critical analysis of findings in context of existing literature. Documentation of methodology, results, and implications for understanding language evolution represents the culmination of the project work.

\
== Scope evolution and objective changes <group1>
\
After the interim report, the implementation scope was revised by dropping the image based game branch and prioritising the vector based resource-chain survival game. The main reason was practical, building and curating image data at the required quality and scale would have consumed disproportionate project time, reducing the depth of analysis possible for communication behaviour. Following this change, the project retained the same core research aim, while concentrating effort on a single environment that could be iterated quickly and evaluated rigorously. Furthermore, the game design itself evolved during implementation, as several mechanics and reward choices were adjusted in response to training behaviour, and these design iterations became central to obtaining analysable message dynamics.

\
== Final revised objectives <group1>
\
*Primary Objectives*

1. *Machine Learning Foundation Development:*
   The first objective involves establishing comprehensive understanding of machine learning fundamentals and PyTorch framework. This includes mastering multi-agent reinforcement learning concepts and techniques. Additionally, studying existing literature on emergent communication in AI systems forms a crucial component of this foundational phase.

2. *Emergent Communication Background Research:*
   This objective requires conducting thorough literature review on emergent communication research. The work involves gaining proficiency with the #underline[#link("https://github.com/facebookresearch/EGG/tree/main")[EGG library]] (Emergent Communication through Games). Furthermore, analysing existing game setups and agent architectures used in communication research will provide essential background knowledge.

3. *Baseline Implementation:*
   The third objective focuses on implementing AI agents within an existing survival game using the EGG library. This phase involves training the agents to develop communication protocols and evaluating their performance in survival scenarios, with particular emphasis on understanding the practical aspects of training communicating agents. A key outcome of this stage will be the establishment of baseline performance metrics and evaluation methodologies.


4. *Simple Game Design (Revised):*
   This objective centres on designing a basic survival scenario optimised for studying emergent communication. The focus remains on simple two-agent communication scenarios such as predator warning and hunting coordination. Implementation will follow a vector based resource-chain infrastructure, motivated by the basic games library from the EGG repository.

\
*Secondary Objectives*

5. *Simple Game Implementation and Training:*
   This objective focuses on implementing AI agents within the custom-designed simple survival game. The work involves integrating the EGG framework with the newly created game environment, training agents to develop communication protocols specific to the designed scenarios, and conducting initial experiments to validate that meaningful communication emerges. This phase emphasises debugging the game mechanics, optimising training parameters, and establishing that the custom environment successfully promotes agent communication development.

6. *Language Analysis:*
   This objective requires analysing emergent communication protocols for fundamental properties like compositionality and efficiency. The work investigates how environmental pressures shape basic language development. Initial evaluation of communication strategies provides foundational analysis for the research.

\
*Tertiary Objectives*

7. *Complex Game Design:*
   This advanced objective involves developing sophisticated survival scenarios with multiple environmental factors and complex agent interactions. Implementation of varied survival challenges and dynamic environmental conditions extends beyond basic two-agent scenarios.

8. *Multi-Agent Communication:*
   This objective explores communication emergence in scenarios involving more than two agents. The work examines how group communication protocols develop and how information propagates through larger agent populations.

9. *Advanced Language Analysis Across Multiple Games:*
   This comprehensive objective involves analysing emergent communication protocols across different game environments and scenarios. The work compares systematicity, compositionality, and efficiency across varied survival contexts to understand broader patterns in language evolution.

10. *Thesis Documentation:*
   The final objective encompasses a comprehensive evaluation of experimental results. This includes critical analysis of findings in context of existing literature. Documentation of methodology, results, and implications for understanding language evolution represents the culmination of the project work.


Note: Tertiary objectives were retained for completeness and alignment with the original project scope, but were not fully pursued due to time and scope constraints, with exception of the Thesis Documentation.



= Preparation and Technical Ramp-Up <group1>
\
This chapter documents the preparatory work completed before the core survival-game experiments and explains why that preparation was necessary for the later research claims. Although these activities were enabling work rather than the central contribution, they directly addressed the first primary objectives in the DOER and established the methodological reliability required for the implementation and evaluation chapters. The preparation process followed weekly supervisor checkpoints, where progress was reviewed, technical misunderstandings were corrected, and short-term milestones were refined in response to observed results.

A central aim of this chapter is to show that later design decisions were not made ad hoc. Instead, they were grounded in a staged progression from machine learning theory, to framework-level engineering, to controlled communication experiments. This progression reduced trial-and-error development, improved reproducibility, and made it possible to interpret model behaviour in terms of known optimisation and generalisation mechanisms.

== Machine Learning foundation <group1>
\
The first preparation stage focused on building a robust conceptual base in machine learning so that subsequent model design and experimental reasoning could be justified rigorously. The primary resource was the #underline[#link("https://developers.google.com/machine-learning/crash-course")[Google Machine Learning Crash Course]] @google:mlcrashcourse, supported by targeted readings, practical notebooks, and weekly discussion with my supervisor. The objective was not exhaustive breadth, but strong operational understanding of the concepts that directly affect experimental reliability in emergent communication work.

Coverage included linear and logistic regression, loss formulation, gradient descent, convergence diagnostics, and hyperparameter sensitivity. The study then expanded to evaluation and error analysis, including confusion matrices, precision, recall, threshold effects, regularisation, and failure modes associated with overfitting. Additional modules on numerical and categorical data preparation were particularly relevant, because they addressed feature representation, normalisation, data quality checks, and split discipline, all of which later informed how game-state vectors and train/validation/test partitions were handled in this dissertation.

This phase was completed over roughly four weeks and combined conceptual study with implementation-oriented exercises, including supervised classification tasks and small pipeline replications. These exercises surfaced recurrent practical issues, such as unstable training under unsuitable learning rates, metric misinterpretation when class distributions were imbalanced, and debugging difficulty when preprocessing assumptions were implicit rather than explicit. Resolving these issues at this stage prevented them from contaminating later communication experiments.

The outcome of this phase was both conceptual and procedural. Conceptually, it provided a coherent explanation framework for interpreting optimisation behaviour. Procedurally, it established a repeatable workflow for model setup, metric reporting, and result validation. Together, these foundations enabled later chapters to argue from evidence rather than from isolated run outcomes.

== PyTorch familiarisation and initial experiments <group1>
\
Following the theoretical phase, the project moved to PyTorch familiarisation to ensure that model behaviour could be controlled and diagnosed at code level. Early work focused on tensor operations, shape transformations, autograd mechanics, module design, optimiser configuration, batching logic, and device-aware execution. Particular emphasis was placed on avoiding silent shape errors and ensuring deterministic experimental setup where feasible, because these were frequent causes of misleading outcomes in initial tests.

Practical exercises progressed from tutorial-scale scripts to complete training and evaluation loops, including custom classifiers on MNIST-style data. This transition was important because it exposed the full engineering path required for later communication games: data loading, model construction, loss computation, parameter updates, metric aggregation, and checkpoint management. At this stage, I also developed confidence in interpreting loss and accuracy trends across epochs, distinguishing genuine learning progress from unstable oscillation or reporting artefacts.

This PyTorch phase directly improved implementation quality in later chapters. By the time communication experiments began, I had a stable development workflow for instrumenting runs, isolating code-level defects, and applying targeted hyperparameter changes. In practice, this reduced avoidable rework and made the eventual EGG adaptation substantially more efficient.

== EGG framework familiarisation <group1>

EGG familiarisation focused on understanding the framework as a research instrument rather than treating it as a black-box training utility. The `zoo/basic_games` modules were used as the entry point because they clearly expose sender-receiver composition, game objectives, wrapper-level message handling, and optimisation differences between training modes. This stage clarified how communication games are parameterised, where task-specific logic should be implemented, and how metrics should be interpreted in relation to each game objective.

A major outcome of this phase was understanding the practical trade-off between Gumbel-Softmax relaxation and REINFORCE for discrete communication. Comparative reconstruction runs across multiple configurations showed substantial performance variance, with final accuracies spanning 3% to 17%, depending on learning rate, architecture size, message constraints, and optimisation mode. Although these scores were modest in absolute terms, the experiments were methodologically valuable because they revealed how sensitive emergent communication training can be to configuration details that might otherwise appear secondary.

This familiarisation also provided a transferable mental model of data flow, from sender input encoding through message generation to receiver prediction and loss backpropagation or policy-gradient updates. That understanding later informed the custom survival-game design, especially in defining communication channels, selecting diagnostics, and recognising when observed collapse reflected optimisation dynamics rather than implementation failure.

== MNIST adaptation in EGG as preparatory work <group1>

The MNIST adaptation served as the culmination of the preparation phase, combining machine learning foundations, PyTorch implementation competence, and EGG-specific communication mechanics in one controlled experiment. The implemented task was a discrimination game in which a sender observed a target digit and sent a discrete message to a receiver that selected the target among distractors. This setup was intentionally chosen because it preserved the core communication loop while remaining sufficiently interpretable for detailed debugging and message-level analysis.

Experiment configuration was explicit and reproducible, including sender and receiver recurrent architectures, constrained vocabulary, fixed maximum message length, entropy regularisation, and tracked epoch-wise metrics. In tracked runs, the system demonstrated strong classification capability, including test accuracies above 90% and a best recorded run at 96.5% by epoch 10 under the reported configuration. Data scale and runtime characteristics were also documented, with 60,000 training samples, 10,000 test samples, and practical CPU-only execution time suitable for iterative experimentation.

Beyond raw accuracy, the MNIST stage was valuable because it established an analysis methodology later reused in the survival-game experiments. Confusion-matrix inspection showed strong diagonal concentration, while message-level analysis indicated non-random token usage, recurring symbol motifs, and entropy variation across classes. These behaviours suggested that communication was not merely noise, but was adapting to task structure in measurable ways.

For this reason, the MNIST adaptation is treated in this dissertation as preparatory validation rather than a principal research contribution. Its main role was to validate tooling, calibrate instrumentation, and reduce risk before moving to the more complex survival setting, where interpretation is harder and experimental control is more limited.

#pagebreak()

= Preparation and Technical Ramp-Up <group1>
\
This section is an additional one I want to add ti talk about the first two objectives in the DOER which were about learning and did not necessarily relate to the research question directly.

== Machine Learning foundation <group1>
\
The first phase in the project established the theoretical and practical foundations required to support subsequent model design and experimental reasoning. The primary objective was not exhaustive theoretical coverage, but the development of a working understanding of core machine learning principles that could be directly applied during implementation and evaluation. The main resource used was the #underline[#link("https://developers.google.com/machine-learning/crash-course")[Google's Machine Learning Crash Course]] provided by Google @google:mlcrashcourse, which offers a structured introduction to supervised learning through a combination of short lectures, interactive exercises, and applied examples. This was supplemented with targeted readings and implementation focused materials derived from the weekly checkpoints where my supervisor would quiz me and explain areas that I did not comprehend.

This preparation phase was completed over approximately three weeks and combined conceptual study with small implementation tasks, including the construction of a simple handwritten digit classifier using the MNIST @lecun2010mnist data. These exercises served as verification of understanding and exposed common failure modes such as incorrect tensor dimensions, unstable gradients, and poor hyper-parameter selection. The outcomes of this phase were therefore both conceptual and procedural, and overlapped with the subsequent weeks where I learnt how  to use PyTorch in depth.

Overall, this stage provided a stable foundation for the remainder of the project. It enabled informed decisions about model architecture, loss functions, and optimisation strategies, and it reduced reliance on trial-and-error during later development. Moreover, it supported more rigorous evaluation by allowing observed behaviours, such as training instability or performance plateaus, to be interpreted in terms of underlying learning dynamics rather than treated as opaque outcomes.

\
== PyTorch familiarisation and initial experiments <group1>
\
Following the conceptual foundation, work shifted to PyTorch familiarisation so that model behaviour could be understood and controlled at implementation level. Initial exercises focused on tensors, autograd, module design, forward and backward passes, optimiser setup, batching, and validation loops. Particular effort was placed on data preprocessing, dimensional consistency, and reproducibility controls, because these repeatedly affected training reliability in early experiments.

The experimental progression moved from simple networks to convolutional architectures on image data, with systematic variation of learning rate, batch size, and optimiser settings. This phase clarified which training failures were due to modelling assumptions and which were due to implementation errors. Consequently, by the time communication experiments started, the project had an established workflow for logging, debugging, and interpreting model behaviour, which reduced avoidable rework in later chapters.

== EGG framework familiarisation <group1>

EGG familiarisation focused on understanding how emergent communication experiments are structured end to end, including sender and receiver roles, wrapper behaviour, task objectives, and training orchestration. The basic games modules were used as the main reference point, especially reconstruction and discrimination setups, because they expose the core mechanics needed for custom communication-game development. This stage also clarified practical differences between Gumbel-Softmax relaxation and REINFORCE training for discrete signalling.

Hands-on runs in the basic games provided useful calibration of expected behaviour and limitations. Early reconstruction performance varied substantially across settings, with final accuracies ranging from 3% to 17%, highlighting sensitivity to learning rate, architecture configuration, and message constraints. Although these scores were modest, the experiments were valuable because they built a correct mental model of data flow and optimisation dynamics, which later informed the design and justification of the custom survival-game pipeline.

== MNIST adaptation in EGG as preparatory work <group1>

The MNIST adaptation served as the integration milestone that combined machine learning theory, PyTorch engineering, and EGG communication mechanics. A discrimination game was implemented in which the sender observed a target digit and transmitted a message to a receiver that selected the target among distractors. This intermediate task was deliberately chosen because it preserved the full communication loop while remaining sufficiently interpretable for controlled debugging and message analysis.

The resulting performance provided strong evidence that the pipeline was functional before moving to the survival domain. In a tracked run, test accuracy reached 91.47% by epoch 10, and confusion-matrix analysis over the test set showed 97.68% diagonal mass overall, with per-class recall generally between 96% and 99%. Message analysis also indicated stable, non-random signalling tendencies, including consistent symbol reuse patterns. In this dissertation, the MNIST stage is therefore treated as preparatory validation rather than a core contribution, with its primary value being risk reduction, tooling validation, and methodological calibration for the main survival-game experiments.

\
== Machine Learning foundation <group1>

Brief summary of what was studied and what resource was used. Talk about the progress, challenges and ways of learning implemented (videos, lectures, practical use and quizzes). 

List the sections learnt, and how they were relevant to the project and how successful it turned out to be.

The project began with a structured machine learning foundation phase using Google's Machine Learning Crash Course, complemented by lecture notes and practical exercises. This phase covered core topics including regression, classification, loss optimisation, model evaluation, and neural network fundamentals, which were necessary before implementing communication games in EGG. The learning process combined reading, quizzes, and short coding experiments, and it provided a reliable conceptual base for later decisions on training setup, metric interpretation, and error analysis.

\ \ \ \

The semester began with building foundational knowledge in machine learning, an area where significant gaps existed. #underline[#link("https://developers.google.com/machine-learning/crash-course")[Google's Machine Learning Crash Course]] became the primary learning resource, covering everything from linear and logistic regression through to neural network architectures, classification systems, and model evaluation. The course proved highly effective, completing it within four weeks with perfect scores on all quizzes.

Alongside this structured learning, I developed PyTorch skills through self-directed experimentation and the guidance of my supervisor. The MNIST dataset served as the main training ground, supplemented by other benchmark datasets. This hands-on approach, where I was building custom neural networks with `torch.nn.Module`, implementing training loops, and experimenting with different architectures, proved invaluable for understanding how theory translates to practice.

\
== PyTorch familiarisation and initial experiments <group1>

Brief summary of what was studied and how it was learnt (documentation and coding exercises/exploration).

\
== EGG framework familiarisation <group1>

Summary of what was studied, how and why. Here it is now more relevant to the project so make sure to be concise but explanatory enough to be self contained.

Mention the structure of the repository but not in full detail

EGG familiarisation was carried out through detailed study of the `zoo/basic_games` modules and hands-on experiments with reconstruction and discrimination setups. This phase clarified the sender-receiver architecture, message generation process, and the differences between Gumbel-Softmax and REINFORCE training modes. Although early reconstruction scores were modest, these experiments were useful because they established a working understanding of data flow, optimisation behaviour, and practical constraints when training communicating agents.

\
== MNIST adaptation in EGG as preparatory work <group1>

Example of how the egg framework was learnt and combining the learning outcomes of the three learning objectives. 

Talk about the implementation and success of it but very briefly. Purpose is tooling and framework on boarding, not core project evaluation.

To consolidate this preparation, a baseline MNIST discrimination game was implemented in EGG, where agents communicated about one target digit among distractors. The model reached 96.5% test accuracy, and message analysis showed clear symbol reuse patterns and vocabulary compression. In this dissertation, this baseline is treated as preparatory work rather than a core contribution, because its main value was to validate tooling, establish analysis workflows, and reduce implementation risk before developing the custom survival environment.

\
== Lessons from preparation phase that informed final design choices <group1>

Maybe a small paragraph on how all of these learnt topics/skills helped in the development of the game, implementation and analysis.

#pagebreak()
= Software Engineering Process <group1>

The development approach taken and justification for its adoption.

== Iterative development approach <group1>

Could potentially talk about an adoption of an Agile strategy for individual projects? Weekly supervisor driven milestones, keeping progress moving.

The project followed an iterative supervisor-driven workflow, similar in spirit to Agile for an individual project, where weekly meetings defined short milestones, reviewed results, and adjusted technical priorities based on evidence from the latest runs. This approach allowed controlled progression from preparation work to implementation and analysis, and it supported rapid refinement when early training behaviour exposed weaknesses in game balance or evaluation setup.

== Development phases and pivots <group1>

Mention the split between first and second semester, and how I planned at the start to divide the work across the semesters with my supervisor. Specifically also talk about the moment I changed from learning to coding, focusing on the project.

== Resources and Technologies <group1>

Talk about the tech stack used, the dependancies required for the project and what additional resources were used. *Maybe mention here that potentially we should have used a dedicated server?*

\
= Ethics <group1>

Mention i have no ethic concerns. *EGG lib has a Facebook copyright tho, so would this be a concern?*

This project is based entirely on computational simulation and does not involve human participants, personal data, or interaction with living animals. Consequently, no direct ethical concerns were identified for data handling or participant risk. The work focuses on artificial agents in controlled environments, and all analysis is performed on generated experimental outputs. Use of the EGG framework is a licensing and attribution matter rather than an ethics concern, therefore external dependencies are documented appropriately in references and implementation notes.

\
= Design <group1>

Indicating the structure of the system, with particular focus on main ideas of the design, unusual design features, etc.

Talk about the planning that I made, how i planned the game structure based off the egg repo and taking into account the past work I was doing. Image based game was inspired by the MNIST adaptation while learning PyTorch and the egg repo, while the resource chain was motivated by the actual basic games library.

Could maybe talk about the different models that could play this game and how it was designed to adapt both (GS and Reinforce).

== Survival game environment and state/action structure <group1>

Talk about the design of the game, everything to do about it and why. Talk about the logic behind it, nature of points and how it pushes to EC.

Also talk about how some aspects of the game had to be changed while implementing it due to unforeseen problems within the training phase. Such as the message collapse and the low recon time.

=== Reward structure and strategy incentives <group1>

Talk about the design of the reward system taking into account future work with the actual loss for the models.

== Data creation <group1>

How the data is generated and why. Changes done throughout development and the statistics on the data.

Also could talk about the problem faced with duplicates in train/test and validation sets.

== Communication channel design choices <group1>

Explain why I choose the vector based architecture

== Brief design note on dropped image branch (for scope completeness) <group1>

A brief mention about the image based game design and why it was dropped.

\
= Implementation <group1>

How the implementation was done and tested, with particular focus on important / novel algorithms and/or data structures, unusual implementation decisions, novel user interface features, etc.

Here I want to go through the code and how I leveraged the egg repo framework. We need to be referencing the context section as there we will have the definition on the function and models that are in the egg repo.

Files to cover:
  - archs.py
  - callbacks.py
  - data.py
  - games.py
  - losses.py
  - prototype.py
  - train.py

== Survival game implementation architecture <group1>

Here I want to explain how I adapted the architecture of the designed game into the egg repo. What files had to be made and how it all got connected properly.

== Training pipeline and experiment configuration <group1>

Here I want to talk about all the hyper parameters used and why i changed them. Also about how training developed and changed over time.

It is crucial to talk about the changes that emerged as each training run was done, learning from each run. *We could also potentially mention we focused too much on the survival rate metric at first?*

== Logging, instrumentation, and output artefacts <group1>

Here I want to be talking about how i was keeping track of the progress and logging the information from each run. Explain the different difficulties faced and what the final logging style that was implemented.

Here we can include examples of the outputs, evaluation and analysis of messages.

== Design implications and changes done <group1>

In here I want to talk about what changed throughout the implementation. What each run found and what I decided to change to solve the given problem. *Maybe this can go in design?*

Go through the runs and the technical change done, with an explanation and justification.

\
= Evaluation and Critical Appraisal <group1>

You should evaluate your own work with respect to your original objectives. You should also critically evaluate your work with respect to related work done by others. You should compare and contrast the project to similar work in the public domain, for example as written about in published papers, or as distributed in software available to you.

First I need to explain the results by objectives, what objectives were not achieved Here I am going to talk about the results of the game and how successful the AI agents where.

== Communication emergence analysis <group1>

Big dive into the messages produced in our last run. Analysis based of academic literature and results of the metrics analysed throughout the run. For this we will use run 13, as it is the last one and the best.

Talk about the effect of hyper parameters too (like temperature decay)

== Performance outcomes (survival, reward, action distributions) <group1>

Show some metrics like confusion matrix, messages, action distribution and compare it with the benchmark. *Might need to re model the benchmark as the game has changed since.*

== Message protocol progression and stability <group1>

Analysis on the development of the messages and justifications or explanations of why this is like it is. 

== Limitations, local optima, and failure analysis <group1>

Problems encountered, limitations of the game (inherent in the design), difficulties in analysis (takes a lot of time).

== Critical reflection against objectives and related work <group1>

Critically analyse our results, how relevant they are, do they answer the research question and how successful was the project. Maybe compare to related work in the field?

= Conclusion and Future Work <group1>

You should summarise your project, emphasising your key achievements and significant drawbacks to your work, and discuss future directions your work could be taken in.

Overall feeling with the project, what went well and what went bad. How the 15 credit module capped the time given (already committed more than a 30 credit module).

== Summary of findings <group1>

Table summary with final opinion on the results. Summarize the entire project and achievements.

== Main constraints <group1>

Brief paragraph on constraints and drawbacks from the project

== Future work (including possible image-based extension) <group1>

Future work that can be done based off these results.

#pagebreak()

= Bibliography <group1>

#bibliography("references.bib", style: "ieee", title: none )
#pagebreak()

= Appendix <group1>

Extra resources

Links:
https://studres.cs.st-andrews.ac.uk/Library/ProjectLibrary/cs4098/2021/slb30-Final_Report.pdf
https://info.cs.st-andrews.ac.uk/student-handbook/course-specific/individual-projects.html
CS4098 – Minor Software Project 

Progress:

Learnt ML -> learn Pytorch -> Learnt EGG -> Implemented MNIST in EGG -> Designed Game (complex) -> Implemented the Game -> Analysed the Language -> Writing thesis

#pagebreak()
// main.typ



== Testing Summary <group1>

This should describe the steps taken to debug, test, verify or otherwise confirm the correctness of the various modules and their combination.

== User Manual <group1>

Instructions on installing, executing and using the system where appropriate.

== Extended figures/tables/message snapshots <group1>


#pagebreak()

Questions:
  1. What should i submit from my code, it only works integrated to third party code.
  2. Should i keep aims/objectives in the Introduction or move them fully to Requirements?
  3. In Context Survey, should I assume markers know basic ML concepts, or briefly define them?
  4. Should I keep the Related signalling/reference games subsection, or trim it?
  5. In Requirements scope change, should I explicitly state that part of game design evolved during implementation?
  6. In final revised objectives, should tertiary objectives be removed or kept as deferred?
  7. In Resources and Technologies, should I mention that a dedicated server may have improved experimentation?
  8. Is EGG's Meta copyright/license an ethics concern for this project or only a licensing/compliance note?
  9. In Training pipeline and experiment configuration, should I explicitly note we initially over-focused on survival rate?
  10. Should "Design implications and changes done" remain in Implementation, or move to Design?
  11. In Performance outcomes, do I need to recompute/remodel benchmarks after game changes before final comparison?


  Questions for Supervisor

- Where should I discuss iterative design changes that occurred during implementation, especially when those changes were necessary to enable meaningful analysis?

- Is it acceptable to present a system where the design evolved significantly during implementation, provided the rationale and outcomes are clearly justified?

- How concise should the preparation chapter be for a 15-credit dissertation, and how much detail is appropriate without it appearing excessive?

- To what extent should I explain frameworks like EGG and underlying concepts, given the guideline that I should not assume prior subject knowledge?

- Is it acceptable to structure objectives using prioritised bullet points (primary, secondary, tertiary) instead of a formal traceability table?

- Should detailed reproducibility information (commands, hyperparameters, setup) be placed mainly in the appendix/user guide, with only a summary in the implementation chapter?

- Given that I am only submitting my own code (not the full framework), how explicitly should I document modifications to external framework components?

- How much emphasis should I place on discussing limitations such as local optima, metric sensitivity, and design trade-offs in the evaluation section?