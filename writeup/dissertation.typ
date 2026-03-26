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

#pagebreak()

= Preparation and Technical Ramp-Up <group1>
\
This chapter documents the preparatory work completed before the core survival-game experiments and explains why that preparation was necessary for the later research claims. Although these activities were enabling work rather than the central contribution, they directly addressed the first primary objectives in the DOER and established the methodological reliability required for the implementation and evaluation chapters. The preparation process followed weekly supervisor checkpoints, where progress was reviewed, technical misunderstandings were corrected, and short-term milestones were refined in response to observed results.

A central aim of this chapter is to show that later design decisions were not made ad hoc. Instead, they were grounded in a staged progression from machine learning theory to framework-level engineering, leading to controlled communication experiments. This progression reduced trial-and-error development, improved reproducibility, and made it possible to interpret model behaviour in terms of known optimisation and generalisation mechanisms.

== Machine Learning foundation <group1>
\
The first preparation stage focused on building a robust conceptual base in machine learning so that subsequent model design and experimental reasoning could be justified rigorously. The primary resource was the #underline[#link("https://developers.google.com/machine-learning/crash-course")[Google Machine Learning Crash Course]] @google:mlcrashcourse, supported by targeted readings, practical notebooks, and weekly discussion with my supervisor. The objective was not exhaustive breadth, but a working understanding of the specific concepts that would directly bear on experimental reliability in the emergent communication work to follow.

Throughout the course taken, I learnt about many topics covering supervised learning fundamentals such as linear and logistic regression, loss formulation, gradient descent, and convergence diagnostics as explained in the Context Survey. The course then expanded to evaluation methodology, including confusion matrices, accuracy, precision, recall, threshold effects, regularisation, and the failure modes associated with overfitting which proved useful in future work. Moreover, the course also offered modules on data preparation, covering feature representation, normalisation via Z-score standardisation, and train, validation, and test split discipline. These topics later helped me design the game and architecture of the model, allowing me to leverage the skills learnt throughout the course.

As part of the first primary objective, I accompanied the development with exercises using real datasets provided by the course in order to enhance my skills within the practical environment. These included a regression task predicting Chicago taxi fares from trip distance and duration, a binary classification task distinguishing two rice grain varieties from morphological measurements, and a data quality investigation using a calories and test score dataset with a hidden day-level confound. These exercises were methodologically valuable because they made abstract failure modes concrete. For example, fitting a single feature fare model at an excessive learning rate produced a loss curve that oscillated without converging, demonstrating directly how learning rate sensitivity can obscure genuine progress. As for the other two exercises, working with the rice classification task on an imbalanced feature space showed how accuracy alone is an insufficient metric and why precision and recall must be examined jointly, a lesson that later motivated the evaluation methodology in the survival game experiments. Similarly, the data quality task illustrated how unexamined structure in a dataset, in that case systematic variation by day of week, can produce misleading aggregate statistics, reinforcing the importance of explicit split discipline, a challenge I faced later on the implementation.

Despite having followed the course, two topics required substantially more effort than the others due to my lack of background on the subject: the cross-entropy loss and backpropagation. Cross-entropy loss proved difficult to internalise initially because its behaviour is less intuitive than mean squared error, particularly regarding how the log term penalises confident wrong predictions disproportionately. This was resolved through targeted reading and by tracing the loss through small worked examples. The effort was directly worthwhile because cross-entropy loss is one of the primary training signal in both the MNIST adaptation and the projects' communication experiments shown later on. A superficial understanding here would have made it difficult to interpret training instability later, thus it became a crucial point within the project. Similarly, back-propagation, the chain rule and how gradients accumulate through sequential layers where explained by my supervisor, which became directly relevant when working with the recurrent sender and receiver architectures in the EGG framework.

This phase was completed over approximately three weeks and concluded with the MNIST implementation, described in the Appendix in *#underline[@mnist-performance-summary]*, which served as an integration checkpoint. Conceptually, it provided a principled framework for interpreting optimisation behaviour, so that observations such as training instability or a plateau in validation performance could be explained in terms of known learning dynamics rather than treated as opaque implementation failures. Procedurally, it established a repeatable workflow for model setup, metric reporting, and result validation that was carried forward into every subsequent experimental chapter. Thus, this foundation did not merely satisfy the first primary objective, it directly shaped the reliability of the analysis presented in the remainder of this dissertation.

\
== PyTorch familiarisation and initial experiments <group1>
\
Following the initial theoretical phase, the project transitioned to practical familiarisation with PyTorch to ensure that model behaviour could be inspected, controlled, and debugged at the implementation level. Using #underline[#link("https://docs.pytorch.org/tutorials/beginner/basics/intro.html")[PyTorch's Introduction to Basics]] @pytorch:basics:intro as a foundation, I explored core fundamental building blocks such as tensor operations, shape transformations or autograd mechanics, with a particular emphasis on identifying and preventing silent shape mismatches, which can propagate through models without immediate runtime errors but lead to invalid training dynamics. 

All of the work done can be found under the Appendix *#underline[@pytorch_app]* where I have the results for the core operations I explored. Additionally, I also expanded from the MNIST classification problem solved in the previous phase to complete training and evaluation pipelines on the Fashion MNIST dataset. This progression was critical, as it exposed the full experimental workflow that I applied in the design of the survival game: dataset loading, model definition, forward and backward passes, loss computation, parameter updates, and performance monitoring. Through these experiments, I developed the ability to interpret loss and accuracy trends, distinguishing meaningful convergence from unstable or noisy training behaviour. Thus, this phase had a direct impact on subsequent work by establishing a reliable development and debugging workflow. As a result, integration with the EGG framework and the development of communication games proceeded more efficiently and with reduced need for iterative rework.

== Emergent Communication and EGG Framework Familiarisation <group1>
\
Following the completion of the first primary objective, the next step on the roadmap was to acquire the theoretical and practical grounding in emergent communication necessary to design and evaluate meaningful experiments. This addressed the second primary objective directly, requiring both a study of the research literature and hands-on engagement with the EGG framework before any custom development could begin.

Emergent communication, in the context of multi-agent systems, refers to the phenomenon by which agents develop a shared signalling protocol without that protocol being explicitly programmed or supervised. As surveyed by Lazaridou and Baroni @lazaridou2020emergent, agents are placed in a game where communication is a means to achieve a shared reward, and the symbols they exchange carry no pre-assigned meaning at the start of training. Meaning and structure emerge through repeated interaction, guided only by task success. The agents are typically divided into a sender, who observes an input and produces a message, and a receiver, who processes that message and acts on it. Communication is considered emergent when the receiver's behaviour is causally influenced by the sender's messages, rather than the channel being ignored or used trivially. Although a key technical challenge is that communication in this setting is discrete, meaning that gradients cannot flow through the message channel directly, which requires either a continuous relaxation such as Gumbel-Softmax @jang2017categorical, or a policy gradient method such as REINFORCE @williams1992simple to train the agents. As explained later on, several findings from the literature proved directly relevant to the results observed in this project. For example, Kottur et al. @kottur2017natural demonstrated that agents often fail to converge to compositional or human-like protocols, instead developing minimal codes that are just sufficient for the task at hand, a pattern consistent with the message collapse behaviour encountered in several training runs described later in this dissertation.

With this theoretical grounding established, I decided to start exploring the EGG framework itself. The EGG repository is structured around a clear separation between game-specific logic and general communication infrastructure. The researcher's responsibility is to define the input data, the core agent modules, and the task loss, while EGG's core layer handles message generation, message processing, training mode selection, and optimisation orchestration. Agent modules are wrapped by framework components that implement either Gumbel-Softmax or REINFORCE training, depending on the chosen mode, and these wrappers connect to a game object that ties the sender, receiver, and loss together into a single trainable system. Moreover, the framework also provides callback utilities for logging, temperature annealing, and validation event printing, which allowed experiment outputs to be inspected without requiring custom implementations at this stage. Thus, to develop a working understanding of this pipeline, i used the `zoo/basic_games` module as a primary point of entry. This module implements both a reconstruction game, in which the receiver must reproduce the full input vector from the sender's message, and a discrimination game, in which the receiver must identify a target item among a set of distractors. In order to learn how different architecture is affected by hyperparameters, I decided to run the game while changing one hyperparameter and keeping the rest constant, which clarified how changing the game objective alters the demands placed on the communication channel.

I executed five reconstruction runs to examine the sensitivity of emergent communication training to configuration choices, varying learning rate, architecture size, vocabulary, message length, and optimisation mode. The results are summarised in *#underline[@egg-basic-runs]*, where we can see that training with a learning rate of 0.01 under Gumbel-Softmax produced a final accuracy of 3%, with a loss curve that showed no convergence, confirming that learning rate sensitivity in emergent communication training is more severe than in standard supervised settings. On the other hand, reducing the learning rate to 0.001 improved stability substantially, reaching 14% accuracy over 200 epochs, and switching to REINFORCE under the same configuration produced the best result across all runs at 17%, a factor that I took into account when designing the implementation. Also, reducing architecture capacity below the task's complexity suppressed communication quality independently of optimisation mode, and replacing GRU cells with LSTM cells produced slower convergence under otherwise identical settings. Although these accuracy figures are modest, their value was methodological, they established a calibrated understanding of how sensitive the training dynamics are, and which configuration choices have the largest practical impact.

Full run logs and validation outputs are provided in *NEED PROOF HERE*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: center,
    table.header(
      [*Run*], [*Mode*], [*Architecture*], [*LR*], [*Epochs*],
      [*Final Acc*], [*Outcome*],
    ),
    [1], [GS],  [S:256 / R:512 GRU],  [0.01],   [50],  [3%],    [Loss diverged],
    [2], [GS],  [S:256 / R:512 GRU],  [0.001],  [200], [14%],   [Stable],
    [3], [RF],  [S:256 / R:512 GRU],  [0.001],  [200], [17%],   [Best result],
    [4], [GS],  [S:128 / R:128 GRU],  [0.0005], [200], [5%],    [Under-capacity],
    [5], [GS],  [S:256 / R:512 LSTM], [0.001],  [200], [7%],    [Slower convergence],
  ),
  caption: [EGG basic reconstruction game runs across training modes and configurations.]
) <egg-basic-runs>

== MNIST adaptation in EGG as preparatory work <group1>

The MNIST adaptation served as the integration milestone that combined machine learning theory, PyTorch engineering, and EGG communication mechanics. A discrimination game was implemented in which the sender observed a target digit and transmitted a message to a receiver that selected the target among distractors. This intermediate task was deliberately chosen because it preserved the full communication loop while remaining sufficiently interpretable for controlled debugging and message analysis.

The resulting performance provided strong evidence that the pipeline was functional before moving to the survival domain. In a tracked run, test accuracy reached 91.47% by epoch 10, and confusion-matrix analysis over the test set showed 97.68% diagonal mass overall, with per-class recall generally between 96% and 99%. Message analysis also indicated stable, non-random signalling tendencies, including consistent symbol reuse patterns. In this dissertation, the MNIST stage is therefore treated as preparatory validation rather than a core contribution, with its primary value being risk reduction, tooling validation, and methodological calibration for the main survival-game experiments.

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

Papers that might be useful:
Furthermore, this point was reinforced by Bouchacourt and Baroni @bouchacourt2018agents by showing that agents trained on a referential game could communicate effectively about Gaussian noise at test time, suggesting that
the emergent protocol was capturing shallow discriminative properties rather than
meaningful semantic content. Furthermore, Lowe et al. @lowe2019pitfalls provided
an important methodological caution: simply ablating the communication channel and
observing a drop in performance is insufficient evidence of genuine communication,
as the extra architectural capacity alone could account for some of the
improvement. This distinction shaped how the evaluation methodology for the
survival-game experiments was designed.


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

=== Machine Learning Outputs and Metrics
\ 
The MNIST implementation, using a simple feedforward neural network, provided insightful results regarding both training stability and model performance.

The model trained over five epochs and exhibited steady improvements in both loss and accuracy. After training, the model was evaluated on the test set. A sample test image was selected for prediction. The model correctly predicted the label of the digit (7), with a high confidence of 99.03%.

Performance Summary

#figure(
  table(
    columns: (auto, auto, auto),
    [*Epoch*], [*Accuracy*], [*Avg Loss*],
    [1], [90.72%], [0.3428],
    [2], [95.47%], [0.1571],
    [3], [96.85%], [0.1077],
    [4], [97.57%], [0.0828],
    [5], [98.09%], [0.0645],
    [Test], [97.30%], [0.0866]
  ),
  caption: [MNIST Model Performance: Epoch Results and Final Test Evaluation]
) #label("mnist-performance-summary")

MNIST_classifier.py output:

#raw("
Using device: cuda
======================================================================
Loading MNIST training dataset...
100%|██████████████████████████████████████| 9.91M/9.91M [00:01<00:00, 8.26MB/s]
100%|██████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 456kB/s]
100%|██████████████████████████████████████| 1.65M/1.65M [00:00<00:00, 1.68MB/s]
100%|██████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 7.31MB/s]
Loading MNIST test dataset...
Training samples: 60000
Test samples: 10000
======================================================================
Model Architecture:
MNISTClassifier(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (network): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
)
======================================================================
Total parameters: 101,770
Trainable parameters: 101,770
======================================================================
Training Configuration:
  Loss Function: Cross Entropy Loss
  Optimizer: Adam
  Learning Rate: 0.001
  Batch Size: 64
  Epochs: 5
======================================================================

Starting training...
======================================================================
Epoch [1/5], Batch [100/938], Loss: 0.3755
Epoch [1/5], Batch [200/938], Loss: 0.3468
Epoch [1/5], Batch [300/938], Loss: 0.2063
Epoch [1/5], Batch [400/938], Loss: 0.2304
Epoch [1/5], Batch [500/938], Loss: 0.2883
Epoch [1/5], Batch [600/938], Loss: 0.2439
Epoch [1/5], Batch [700/938], Loss: 0.1047
Epoch [1/5], Batch [800/938], Loss: 0.2208
Epoch [1/5], Batch [900/938], Loss: 0.3837
Epoch [1/5] Complete - Avg Loss: 0.3428, Accuracy: 90.72%
----------------------------------------------------------------------
Epoch [2/5], Batch [100/938], Loss: 0.2584
Epoch [2/5], Batch [200/938], Loss: 0.1049
Epoch [2/5], Batch [300/938], Loss: 0.1111
Epoch [2/5], Batch [400/938], Loss: 0.1816
Epoch [2/5], Batch [500/938], Loss: 0.1789
Epoch [2/5], Batch [600/938], Loss: 0.1086
Epoch [2/5], Batch [700/938], Loss: 0.2424
Epoch [2/5], Batch [800/938], Loss: 0.0665
Epoch [2/5], Batch [900/938], Loss: 0.0707
Epoch [2/5] Complete - Avg Loss: 0.1571, Accuracy: 95.47%
----------------------------------------------------------------------
Epoch [3/5], Batch [100/938], Loss: 0.1348
Epoch [3/5], Batch [200/938], Loss: 0.1507
Epoch [3/5], Batch [300/938], Loss: 0.2202
Epoch [3/5], Batch [400/938], Loss: 0.2126
Epoch [3/5], Batch [500/938], Loss: 0.2146
Epoch [3/5], Batch [600/938], Loss: 0.1349
Epoch [3/5], Batch [700/938], Loss: 0.1692
Epoch [3/5], Batch [800/938], Loss: 0.0377
Epoch [3/5], Batch [900/938], Loss: 0.1700
Epoch [3/5] Complete - Avg Loss: 0.1077, Accuracy: 96.85%
----------------------------------------------------------------------
Epoch [4/5], Batch [100/938], Loss: 0.0803
Epoch [4/5], Batch [200/938], Loss: 0.2084
Epoch [4/5], Batch [300/938], Loss: 0.0618
Epoch [4/5], Batch [400/938], Loss: 0.0707
Epoch [4/5], Batch [500/938], Loss: 0.0448
Epoch [4/5], Batch [600/938], Loss: 0.0404
Epoch [4/5], Batch [700/938], Loss: 0.0589
Epoch [4/5], Batch [800/938], Loss: 0.1391
Epoch [4/5], Batch [900/938], Loss: 0.0285
Epoch [4/5] Complete - Avg Loss: 0.0828, Accuracy: 97.57%
----------------------------------------------------------------------
Epoch [5/5], Batch [100/938], Loss: 0.0593
Epoch [5/5], Batch [200/938], Loss: 0.0244
Epoch [5/5], Batch [300/938], Loss: 0.0499
Epoch [5/5], Batch [400/938], Loss: 0.0754
Epoch [5/5], Batch [500/938], Loss: 0.1032
Epoch [5/5], Batch [600/938], Loss: 0.0964
Epoch [5/5], Batch [700/938], Loss: 0.1120
Epoch [5/5], Batch [800/938], Loss: 0.0447
Epoch [5/5], Batch [900/938], Loss: 0.1143
Epoch [5/5] Complete - Avg Loss: 0.0645, Accuracy: 98.09%
----------------------------------------------------------------------

Training completed!
======================================================================

Evaluating model on test set...
Test Results:
  Accuracy: 97.30%
  Average Loss: 0.0866
======================================================================

Model saved to: mnist_classifier.pth
======================================================================

Testing inference on a sample image...
True Label: 7
Predicted Label: 7

Class Probabilities:
  Digit 0: 0.0000
  Digit 1: 0.0000
  Digit 2: 0.0002
  Digit 3: 0.0094
  Digit 4: 0.0000
  Digit 5: 0.0000
  Digit 6: 0.0000
  Digit 7: 0.9903
  Digit 8: 0.0001
  Digit 9: 0.0001
======================================================================
")

=== PyTorch Learning <pytorch_app>

Output of tutorial_pytorch.py:

#raw("
  Enter '1' for tensor basics or '2' for FashionMNIST visualization: 1

======================================================================
PART 1: Tensors and NumPy bridge
======================================================================

[1] Tensor from Python list:
tensor([[1, 2],
        [3, 4]])

[2] Tensor from NumPy array:
tensor([[1, 2],
        [3, 4]])

[3] Ones tensor (same shape as x_data):
tensor([[1, 1],
        [1, 1]])

[4] Random tensor (float):
tensor([[0.2444, 0.6654],
        [0.5640, 0.1841]])

[5] Random tensor:
tensor([[0.8831, 0.3790, 0.7701],
        [0.5920, 0.0392, 0.8420]])

[6] Ones tensor:
tensor([[1., 1., 1.],
        [1., 1., 1.]])

[7] Zeros tensor:
tensor([[0., 0., 0.],
        [0., 0., 0.]])

[8] Tensor properties
Shape: (3, 4)
Dtype: torch.float32
Device: cpu
Moved tensor to accelerator: cuda:0

[9] Indexing and slicing
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
Modified tensor:
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
Concatenated tensor:
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

[10] Matrix multiplication result:
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

[11] Element-wise multiplication result:
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

[12] Sum converted to Python scalar: 12.0 (float)

Tensor before in-place add:
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

Tensor after add_(5):
tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])

[13] PyTorch <-> NumPy memory sharing
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
t after add_: tensor([2., 2., 2., 2., 2.])
n after t is modified: [2. 2. 2. 2. 2.]
n after np.add: [2. 2. 2. 2. 2.]
t after n is modified: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
")

#raw("
Enter '1' for tensor basics or '2' for FashionMNIST visualization: 2

======================================================================
PART 2: FashionMNIST sample visualization
======================================================================
Loaded training samples: 60000
Loaded test samples: 10000
")

#image("Images/MNIST_Fashion.png")


\ \

Output for MNIST_Fashion.py:

#raw("
======================================================================
PyTorch FashionMNIST Model Walkthrough
======================================================================
Using device: cuda

======================================================================
Model Architecture
======================================================================
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Predicted class index (random input): 3

======================================================================
Layer-By-Layer Shapes
======================================================================
Input batch shape: (3, 28, 28)
Flattened batch shape: (3, 784)
After first linear layer: (3, 20)

======================================================================
Activation Example (summarized)
======================================================================
Hidden activations before ReLU: shape=(3, 20), dtype=torch.float32, device=cpu
Hidden activations before ReLU preview: tensor([ 0.2577,  0.0513,  0.6861, -0.1938,  0.0945, -0.2237,  0.1269,  0.4146])
Hidden activations after ReLU: shape=(3, 20), dtype=torch.float32, device=cpu
Hidden activations after ReLU preview: tensor([0.2577, 0.0513, 0.6861, 0.0000, 0.0945, 0.0000, 0.1269, 0.4146])
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Shape: (512, 784) | mean=0.0000 std=0.0206
Layer: linear_relu_stack.0.bias | Shape: (512,) | mean=0.0015 std=0.0207
Layer: linear_relu_stack.2.weight | Shape: (512, 512) | mean=-0.0000 std=0.0256
Layer: linear_relu_stack.2.bias | Shape: (512,) | mean=-0.0015 std=0.0242
Layer: linear_relu_stack.4.weight | Shape: (10, 512) | mean=-0.0005 std=0.0253
Layer: linear_relu_stack.4.bias | Shape: (10,) | mean=-0.0090 std=0.0232
")


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