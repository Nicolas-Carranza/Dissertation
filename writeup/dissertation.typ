#import "@preview/in-dexter:0.5.3": *
#import "@preview/clean-math-paper:0.2.5": *
#import "@preview/wordometer:0.1.4": word-count, total-words


#show figure: set block(breakable: true)

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

\
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

\
== Emergent Communication and EGG Framework Familiarisation <group1>
\
Following the completion of the first primary objective, the next step on the roadmap was to acquire the theoretical and practical grounding in emergent communication necessary to design and evaluate meaningful experiments. This addressed the second primary objective directly, requiring both a study of the research literature and hands-on engagement with the EGG framework before any custom development could begin.

Emergent communication, in the context of multi-agent systems, refers to the phenomenon by which agents develop a shared signalling protocol without that protocol being explicitly programmed or supervised. As surveyed by Lazaridou and Baroni @lazaridou2020emergent, agents are placed in a game where communication is a means to achieve a shared reward, and the symbols they exchange carry no pre-assigned meaning at the start of training. Meaning and structure emerge through repeated interaction, guided only by task success. The agents are typically divided into a sender, who observes an input and produces a message, and a receiver, who processes that message and acts on it. Communication is considered emergent when the receiver's behaviour is causally influenced by the sender's messages, rather than the channel being ignored or used trivially. Although a key technical challenge is that communication in this setting is discrete, meaning that gradients cannot flow through the message channel directly, which requires either a continuous relaxation such as Gumbel-Softmax @jang2017categorical, or a policy gradient method such as REINFORCE @williams1992simple to train the agents. As explained later on, several findings from the literature proved directly relevant to the results observed in this project. For example, Kottur et al. @kottur2017natural demonstrated that agents often fail to converge to compositional or human-like protocols, instead developing minimal codes that are just sufficient for the task at hand, a pattern consistent with the message collapse behaviour encountered in several training runs described later in this dissertation.

With this theoretical grounding established, I decided to start exploring the EGG framework itself. The EGG repository is structured around a clear separation between game-specific logic and general communication infrastructure. The researcher's responsibility is to define the input data, the core agent modules, and the task loss, while EGG's core layer handles message generation, message processing, training mode selection, and optimisation orchestration. Agent modules are wrapped by framework components that implement either Gumbel-Softmax or REINFORCE training, depending on the chosen mode, and these wrappers connect to a game object that ties the sender, receiver, and loss together into a single trainable system. Moreover, the framework also provides callback utilities for logging, temperature annealing, and validation event printing, which allowed experiment outputs to be inspected without requiring custom implementations at this stage. Thus, to develop a working understanding of this pipeline, i used the `zoo/basic_games` module as a primary point of entry. This module implements both a reconstruction game, in which the receiver must reproduce the full input vector from the sender's message, and a discrimination game, in which the receiver must identify a target item among a set of distractors. In order to learn how different architecture is affected by hyperparameters, I decided to run the game while changing one hyperparameter and keeping the rest constant, which clarified how changing the game objective alters the demands placed on the communication channel.


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

I executed five reconstruction runs to examine the sensitivity of emergent communication training to configuration choices, varying learning rate, architecture size, vocabulary, message length, and optimisation mode. The results are summarised in *#underline[@egg-basic-runs]*, where we can see that training with a learning rate of 0.01 under Gumbel-Softmax produced a final accuracy of 3%, with a loss curve that showed no convergence, confirming that learning rate sensitivity in emergent communication training is more severe than in standard supervised settings. On the other hand, reducing the learning rate to 0.001 improved stability substantially, reaching 14% accuracy over 200 epochs, and switching to REINFORCE under the same configuration produced the best result across all runs at 17%, a factor that I took into account when designing the implementation. Also, reducing architecture capacity below the task's complexity suppressed communication quality independently of optimisation mode, and replacing GRU cells with LSTM cells produced slower convergence under otherwise identical settings. Although these accuracy figures are modest, their value was methodological, they established a calibrated understanding of how sensitive the training dynamics are, and which configuration choices have the largest practical impact.


\
== MNIST Adaptation in EGG as Preparatory Evaluation <group1>
\
Finally, in week seven, I was ready to start combining the previous phases and produce a practical working game as an evaluation checkpoint. Together with my supervisor we decided that the best approach would be to implement the MNIST digit classifier into the basic games framework, combining machine learning foundations, PyTorch engineering competence, and EGG communication mechanics in a single controlled experiment. This addressed the third primary objective directly, establishing baseline implementation and evaluation methodology before any custom environment work began.

For the design and implementation, I deiced to design a discrimination game in which a sender observed a target digit image and transmitted a discrete message to a receiver, which had to identify the target among a set of candidate images. Each sample presented three images to the receiver, comprising one target and two distractors drawn randomly from the dataset at construction time. This data processing stage was implemented and handled by the `MNISTDiscriDataset` class, sampling distractor images at index time, stacking all images into a flattened receiver input tensor, and shuffling their positions so that the target was not always at a fixed location. Furthermore, I use the ground truth label to record the position of the target after shuffling, an important step because without it the receiver could exploit positional regularity rather than genuinely attending to the sender's message.

As for the architecture, the sender extends the standard EGG pattern with a convolutional front-end, replacing the simple linear encoder used in the basic games. I use two convolutional layers with max pooling which extract spatial features from the 28x28 pixel input before a two-layer feed forward network compresses them into the hidden representation used to initialise the message generating RNN. On the other hand, the receiver applies the same convolutional stack independently to each candidate image, projecting the results to the hidden dimensionality and computing dot products against the message encoding to produce a distribution over candidate positions, following the same mechanism as the `DiscriReceiver` from the basic games. When deciding the parameters to use for the agents, the GRU cells were chosen for both sender and receiver based on the finding from the earlier reconstruction runs that they converge faster than LSTM cells under the same configuration. Moreover, my hidden sizes were set to 256 for both agents, with an embedding dimension of 50, a vocabulary size of 50, and a maximum message length of 10 derived from past conversations with my supervisor. Finally, I used the REINFORCE optimisation mode with a reduced entropy coefficient of 0.001, following the basic games calibration which showed that the entropy coefficient has a direct effect on message diversity and collapse risk.

Once my implementation was finished, I conducted a three-way REINFORCE entropy sweep under a fixed seed (42), using sender entropy coefficients of 0.001, 0.003, and 0.01. The outcomes were sharply different. At 0.001, performance collapsed to chance-level discrimination: test accuracy reached only 32.71% by epoch 10, with loss saturating around 1.0986 (approximately _$ln(3)$_ for three candidates), and sender entropy collapsing from 0.403 at epoch 1 to approximately $4.2 times 10^{-10}$ from epoch 2 onward as seen in *#underline[@entropy-comparison]*. At 0.003, the model converged well: test accuracy rose to 97.03% by epoch 10 (with a temporary dip at epoch 8), while sender entropy stayed in a moderate range (approximately 0.08 to 0.61 across training), indicating a stable but still expressive communication policy, shown in *#underline[@0.003-loss]*. At 0.01, performance again remained at chance level, ending at 32.75% test accuracy; unlike the 0.001 collapse, sender entropy remained very high and increased toward 3.556, showing persistent over-exploration and failure to stabilise a useful protocol. These runs therefore showed a narrow effective entropy regime around 0.003 for this setup. Note that full epoch wise logs for all runs are provided in the Appendix in *#underline[@train-logs-mnist]*.

#set table(
  stroke: none,
)

#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/Entropy 0.001 MNIST.png")],
    [#image("Images/Entropy 0.01 MNIST.png")],

  ),
  caption: [Entropy Comparison (0.01 vs 0.01)],
) #label("entropy-comparison")

#figure(
  image("Images/Entropy 0.003 MNIST.png"),
  caption: [Entropy Comparison (Best: 0.03)]
) #label("0.003-loss")

I then analysed sender messages and confusion structure for each run over the full 10,000-sample test split. The 0.001 run showed complete protocol collapse: every class used the same single message (`[3, 3, 3, ..., 3]`) with frequency 1.0 and near-zero entropy, so the receiver could not exploit communication and remained close to chance. The 0.01 run exhibited the opposite failure mode: sender entropy stayed near 3.556, and each class mixed a few dominant templates (for example, sequences ending in tokens 9, 13, and 19) without class-specific separation, again yielding chance-level discrimination. By contrast, the 0.003 run produced a functional emergent code: per-class message entropy remained low-to-moderate (approximately 0.08 to 0.16), top message frequencies were distributed (roughly 3% to 12% rather than 100% collapse), and per-class accuracies were consistently high (approximately 95.8% to 98.2%). The confusion matrices reflected this directly. For 0.003, the diagonal was strongly concentrated (for example 974/980 for digit 0, 1128/1135 for digit 1, and 982/1009 for digit 9), with only small off-diagonal leakage. For 0.001 and 0.01, row-normalised diagonals stayed around 0.37 to 0.43 with diffuse off-diagonal mass around 0.05 to 0.09, which is consistent with ineffective communication. The confusion matrix figures are included below in *#underline[@cm-001]*, *#underline[@cm-003]* and #underline[*@cm-01*].


#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/Confusion Matrix MNIST 0.001.png")],
    [#image("Images/Confusion Matrix MNIST 0.001 acc.png")]
  ),
  caption: [Sender entropy coefficient 0.001 Confusion Matrices]
) #label("cm-001")


#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/Confusion Matrix MNIST 0.003.png")],
    [#image("Images/Confusion Matrix MNIST 0.003 acc.png")]
  ),
  caption: [Sender entropy coefficient 0.003 Confusion Matrices]
) #label("cm-003")


#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/Confusion Matrix MNIST 0.01 acc.png")],
    [#image("Images/Confusion Matrix MNIST 0.01.png")]
  ),
  caption: [Sender entropy coefficient 0.01 Confusion Matrices]
) #label("cm-01")

#set table(
  stroke: luma(black),
)

During this analysis phase, a question also arose from parallel work in another module where confusion matrices had shown high training accuracy but poor generalisation. In this MNIST case, the sweep clarified a different mechanism: the main risk was not classical overfitting, but entropy miscalibration. At 0.001, entropy collapsed too early and both train and test accuracies remained near chance. At 0.01, entropy stayed too high and the protocol never stabilised, again leaving both train and test near chance. Only at 0.003 did train and test curves rise together to strong performance, with final test accuracy at 97.03% and confusion-matrix diagonals near 0.97 to 0.99. Methodologically, this was an important finding: in this communication game, sender entropy coefficient is a first-order control variable that determines whether an emergent protocol converges, collapses, or remains noisy.

Overall, the MNIST adaptation is therefore treated as preparatory validation rather than a core dissertation contribution. Its role was to validate the EGG integration path, establish a disciplined experimental workflow, expose early communication-analysis pitfalls, and provide an evidence-based bridge from framework familiarisation to custom environment design.

#pagebreak()

= Software Engineering Process <group1>
\
This chapter describes the development approach taken throughout the project, the
structure of work across both semesters, and the technical resources used during
implementation and experimentation.

\
== Development Process <group1>

\
The project followed an iterative, supervisor driven workflow similar to Agile @agilemanifesto development, adapted for a single person research project. Rather than committing to a rigid implementation plan at the outset, my progress was governed by the weekly meetings with my supervisor, where the results of the most recent work were reviewed, short-term milestones were set, and priorities were adjusted in response to observed outcomes. This structure was well suited to the nature of the work, where training behaviour, game design decisions, and evaluation methodology all evolved in response to experimental evidence rather than being fixed in advance.

The iterative approach had a direct impact on several design decisions documented in later chapters. Game mechanics were revised across multiple training runs when early behaviour exposed reward imbalances. Hyperparameter choices were updated following observed instabilities, and the scope of the evaluation methodology expanded incrementally as the experiments produced interpretable outputs worth analysing in greater depth. Without the weekly checkpoint structure, these adjustments would have been harder to make in a principled and documented way.

\
== Development Structure <group1>

\
At the start of the project, in academic week two, the overall timeline was divided into two broad phases in discussion with the supervisor, with the boundary falling at the end of the first semester.

The first semester, covering weeks two through nine, was dedicated entirely to preparation and foundational work. The first four weeks focused on building machine learning foundations and PyTorch competence, the following two weeks on EGG framework familiarisation and basic games experimentation, and the remaining weeks on the MNIST adaptation and initial custom game design. Work was paused during weeks ten through twelve due to examinations, and the preparation phase was formally concluded by the end of semester.

The second semester, running from late January to the April, provided approximately ten weeks of focused project time. The first two weeks were used to finalise the survival game design following semester-break reflection, the next four weeks covered full implementation and iterative training, the following three weeks were dedicated to analysis and evaluation, and the final two weeks were reserved for dissertation write-up. In practice, the implementation and analysis phases overlapped somewhat, as results from each training run informed both the next experimental iteration and the structure of the evaluation chapter. The total time committed to the project substantially exceeded the expectation for a 15 credit module, reflecting the depth of the preparation phase and the iterative nature of emergent communication experimentation.

\
== Resources and Technologies <group1>

\
The project was implemented in Python and built on top of the EGG toolkit, which was installed directly from source as an editable package. All experiments were run on a personal CPU-only machine, which was sufficient for the MNIST preparation work and early survival-game training but introduced practical constraints on the speed and scale of multi-run sweeps. A dedicated GPU machine would have meaningfully reduced iteration time, particularly during the Gumbel-Softmax hyperparameter exploration where temperature decay schedules required many sequential runs to evaluate. @tech-stack summarises the full technology stack used across the project.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header(
      [*Category*], [*Tool or Library*], [*Role in the Project*],
    ),
    [Language],        [Python 3.9+],          [Primary implementation language],
    [Deep learning],   [PyTorch],              [Model definition, training, and gradient computation],
    [Data],            [Torchvision],          [MNIST dataset loading and image transforms],
    [EC framework],    [EGG toolkit],          [Sender-receiver wrappers, game orchestration, training modes],
    [Numerics],        [NumPy, SciPy],         [Array operations and statistical utilities],
    [Data handling],   [Pandas],               [Metric logging and tabular result summaries],
    [Visualisation],   [Matplotlib, Seaborn],  [Loss curves, confusion matrices, and message plots],
    [ML utilities],    [scikit-learn],         [Evaluation metrics and data split utilities],
    [Experiment log],  [wandb (optional)],     [Run tracking and metric comparison across sweeps],
    [Testing],         [pytest],               [Unit tests inherited from EGG core],
    [Edit distance],   [editdistance],         [Message-level Levenshtein analysis],
  ),
  caption: [Technology stack used across the project.]
) <tech-stack>

Training was performed on two different machines across the project. Throughout the preparation phase and the early second-semester implementation work, all experiments ran on a personal laptop equipped with an Intel Core i5-1135G7 processor, 4 cores at 2.40 GHz, and 16 GB of RAM, with no discrete GPU available. Under this setup, a full survival-game training run required approximately nine hours to complete, which made iterative hyperparameter exploration slow and constrained the number of configurations that could be evaluated in a given week. In March, work moved to the lab machines in the Jack Cole building, specifically PC9, equipped with an Intel Core i5-12400 processor, 6 cores at up to 4.40 GHz, 30 GB of RAM, and an NVIDIA GeForce RTX 3060 with 12 GB of VRAM. The EGG framework supports CUDA-accelerated training, and enabling this reduced the per-run training time from approximately nine hours to roughly one hour, a reduction of nearly ninefold. This transition meaningfully increased the number of experimental iterations possible in the final weeks of the project and was directly responsible for the depth of the hyperparameter analysis presented in the evaluation chapter. For future work extending these experiments to larger agent populations or longer training schedules, the RTX 3060 remains a practical minimum, though 16 to 24 GB of VRAM would be preferable for larger survival-game configurations with expanded vocabulary or message length.



#pagebreak()

= Ethics <group1>
\
This project is based entirely on computational simulation and does not involve human participants, personal data, or interaction with living animals. Consequently, no direct ethical concerns were identified for data handling or participant risk. The work focuses on artificial agents in controlled environments, and all analysis is performed on generated experimental outputs. Use of the EGG framework is a licensing and attribution matter rather than an ethics concern, therefore external dependencies are documented appropriately in references and implementation notes.

#pagebreak()

= Design <group1>

Throughout this chapter, I have defined and documented the design decisions made for the survival game and the planned architecture for the systems conceptual structure, modelling assumptions, and justifications. Although implementation details and empirical findings are discussed elsewhere, this chapter explains why each part of the system was designed as it was, what changed over time, and how those changes improved scientific validity. 

At this stage, all primary objectives up to and including Objective 3 had been completed, shifting my focus towards the remaining Objective 4 and all secondary objectives. At a high level, the project was designed around one central principle, *communication must be instrumentally necessary rather than decorative.* Thus, the game environment, the agent's protocols, the action constraints, and the evaluation plan were all defined to create pressure for meaningful signalling. As I applied an iterative approach, several structural decisions were revised during implementation when early training behaviour exposed weaknesses in the original plan. Thus, this chapter covers the reasoning behind the initial design, the revisions that were made and why, and how the final design remained aligned with the research question on emergent communication under survival constraints.

*TABLE 1: DESIGN EVOLUTION OVERVIEW, INITIAL PLAN VERSUS PRESENT DESIGN*


Note that the design was shaped by two parallel bodies of prior work completed during the preparation phase. On one hand, the MNIST adaptation had demonstrated that image based inputs could drive meaningful discrimination in EGG, which initially motivated a visually grounded survival scenario. This design idea was also supported by the supervisor, as it was inspired by real-world imagery and environments that resemble natural survival settings, helping to create a more realistic and grounded scenario. On the other hand, while working through the `zoo/basic_games` module in the previous phase, it established a clear understanding of how vector based attribute inputs could be constructed efficiently and paired directly with EGG's communication wrappers. These two influences led to two distinct game designs being considered in parallel during week nine of the first semester, one image based and one vector based, with the vector based design ultimately being selected for reasons discussed at the end of this chapter. *MAKE SURE IT IS EXPLAINED (it provided tighter control over world structure, clearer attribution of communication effects, and lower development risk within the 15 credit scope)*


== Survival Game Environment and State/Action Structure <group1>
\
The survival game was designed around the main sender/receiver communication loop that had to be embedded within cooperative survival scenarios. The core premise is that a sender agent observes a survival encounter, in the form of an entity from the game world, and must communicate sufficient information to a receiver agent for the receiver to select the best action in response. Without the sender's message, the receiver operates blindly and cannot make contextually appropriate decisions, thus making communication strictly necessary for optimal performance. This design property is essential for the research because if the receiver could perform well without any message, there would be no selective pressure for the sender to develop a meaningful protocol.


To play the game, one must go through episodes, where each episode consists of a sequence of turns, with a maximum of 20 turns per episode. At each turn, the sender observes a single entity drawn from the game world, along with the current survival state of the agents. The sender transmits a message to the receiver, which then selects one action from a fixed action space of eleven options, producing a reward that depends on the entity encountered, the action chosen, and the current state of the agents. For each turn, it also checks the state of the agent and terminates early if either energy or health reaches zero, penalising the agents for poor decision making.


=== Entity Design and the Expansion from Five to Forty Entities <group1>
\
Each entity in the game is represented as a discrete six-dimensional attribute vector of the form:

$ e = [t, s, d, v, u, w] $
where $t$ is entity type, $s$ subtype, $d$ danger level, $v$ energy value class, $u$ tool requirement, and $w$ weather dependency. 

This gave a theoretical combination space of $5^6 = 15625$ entity vectors, from which a curated subset of 40 meaningful entities was selected, as a fully dense combinatorial world would increase realism in one sense, but it would dilute semantic control and make protocol analysis difficult within the dissertation scope. Thus, the chosen 40 entities provided enough diversity for non trivial communication while keeping interpretability manageable. This representation was directly inspired by the design of the `zoo/basic_games` module, where attribute value vectors provide a structured, disentangled input space that has been shown in the
emergent communication literature to favour the emergence of partial compositionality \@lazaridou2018emergence. 

Each dimension takes values in ${0, 1, 2, 3, 4}$, giving a structured symbolic space with bounded combinatorics, where each dimension encodes the following information. The first dimension, _entity type_, takes one of five values corresponding to Animals, Resources, Dangers, Craft Opportunities, and Special Events, and was designed to identify the main tye of the entity. Following this, the second dimension encodes the _subtype_ within that category, for example distinguishing predators from herbivores within animals, or food from materials within resources. Then, the third dimension encodes the _danger level_ on a scale from zero to four, where zero is considered safe and four represents high risk of significant health loss. Similarly, the fourth dimension encodes the _energy value_, which can be negative for entities that drain energy, and it is used directly on the reward system. As for the fifth dimension, it encodes whether a specific _tool_ is _required_ to interact beneficially with the entity, such as a spear for hunting or fire for cooking, providing depth and complexity to the game. Finally, the sixth dimension encodes the _weather dependency_, distinguishing entities that are only relevant in certain weather conditions. For the full 40 entity description, you can find in *#underline[@full-entity-list]* with the entity description.

Note that the initial design used these five categories as a core semantic distinction, including category level reconstruction pressure. However, this proved too coarse as a communication target because category level identification could be solved with highly compressed signalling that did not require fine grained distinctions. This proved insufficient during early training and motivated the focus on 40 specific entities rather than 5, later explained in the implementation and evaluation section.


=== State Variables and Action Space <group1>

\
An important aspect of the design for the survival game was the *survival state*, which is tracked across episodes and consists of four components.

First, *energy*, which decrements by a random amount between 2 and 4 per turn and is replenished by eating. Second, *health*, which decrements when the agent is injured by dangerous entities or takes environmental damage. Third, *an inventory*, designed with boolean tool flags recording whether a spear, fire, shelter, or fishing rod have been crafted. And fourth, a *current weather* condition that changes every four to six turns. 

The receiver observes this full survival state alongside the sender's message when selecting an action, meaning the receiver must learn to integrate two distinct information sources, the entity identity communicated by the sender, and the current state context available to it directly.

Furthermore, another important aspect of the game design is the *action space*, which governs what the receiver can do, containing eleven discrete options defined as:

$ A = {"hunt", "gather", "flee", "rest", "mitigate", "endure", "eat", "craft_spear", "craft_fire", "craft_shelter", "craft_rod"} $

Out of the eleven possible actions, two of these actions are context independent in the sense that they are always structurally valid: * flee and rest.* The remaining actions are conditional on entity type, inventory, and state. For example *hunt* is valid only when the encountered entity is an animal, and *mitigate* is valid only when a tool addressable danger is present and the required tool is available. 

This structure was intentional as it creates a situation where the receiver must have access to entity identity information to distinguish between the contextually optimal action and the superficially safe alternatives. Without knowing what entity the sender has observed, the receiver cannot reliably decide whether to hunt, flee, or mitigate, and falls back to gather and eat, which are always safe but suboptimal.
Thus, it served two goals, first it created conditional action validity so that not all actions were sensible in all contexts. And second, it allowed strategic chains, such as gather to craft to hunt to eat, which made communication useful beyond one step reflexes. Moreover, I added action validity masking to prevent logically impossible decisions from dominating policy learning, while still allowing difficult choices among valid alternatives.

In symbols, receiver computation can be viewed as:

$ h = f_theta(m, s) $

$ z = g_theta(h) $

where $z$ are unmasked action logits, and legality is enforced afterwards by

$ z'_a = z_a $ for valid actions, and $ z'_a = -infinity $ for invalid actions.

This mechanism ensures invalid actions receive zero probability after softmax while keeping the semantic inference path anchored to message and state.



=== Reward Structure and Strategy Incentives <group1>
\

Another crucial aspect of the game design is the *reward structure*, designed to create a clear gradient of strategic value that would incentivise the receiver to use the sender's message rather than adopting a context free policy. 

Each turn, 10 points are given for remaining alive, with additional bonuses for maintaining energy levels and health above 70. Then, successful hunting an animal gives 15 points plus a multiplier based on the animal's danger level, rewarding the agent for taking on riskier prey when equipped to do so, motivating the agent to explore the crafting opportunities.Furthermore, completing a full transformation chain, for example hunting an animal, cooking the meat with fire, and eating the cooked meat, awards a 40 point completion bonus as it requires _'luck'_. For each episode completed with all 20 turns or more finished gives a further 150 point bonus, complemented by a efficiency reward from remaining energy and health at the episode's end.

Throughout development, the reward structure went through three major revisions before reaching a state where the baseline policies produced a clear strategy gradient.

In the original configuration, the random policy achieved survival rates close to those of the greedy policy, which indicated that the reward landscape was too flat to differentiate good and poor decisions. The first revision raised starting energy from 80 to 100, reduced metabolic drain from a mean of 4.5 to 3 per turn, and reduced all action energy costs, which gave the agents more time to develop tool based strategies before dying of starvation. 

The second revision substantially increased eating rewards, raising cooked meat from plus 7 to plus 15 energy, fish from plus 4 to plus 10, and berries from plus 3 to plus 6, making the gather/craft/cook/eat chain a high value strategy that would justify the additional steps. A bug was also fixed in this iteration in which dangers with a tool\_required value of zero could be mitigated for free, bypassing the intended
risk structure. 

*The third revision aimed the optimal policy to behave proactively, eating before energy dropped critically, hunting aggressively when tools were available, and enduring moderate dangers rather than wasting energy fleeing from low-risk entities.*

This iterative shaping was justified by the dissertation goal itself, communication quality cannot be studied meaningfully if the task objective admits cheap degenerate policies that bypass semantic coordination.

Following these revisions, I produced four baseline policies based on my personal input with the following results:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Policy*], [*Survival*], [*Avg Reward*], [*Avg Length*]),

    [RANDOM], [20.8%], [+108.3], [16.5],
    [GREEDY], [55.6%], [+352.3], [18.7],
    [OPTIMAL], [62.1%], [+336.1], [18.9],
    [BLIND (no sender)], [19.3%], [-99.0], [14.6],
  ),
  caption: [Policy baseline results (averaged across multiple runs) showing survival rate, average reward, and episode length.],
) <policy-baseline-results>

Note that these runs were conducted prior to the implementation of the agentic architecture, serving to validate the game mechanics. As we can see in the @policy-baseline-results, the survival rate gap between the optimal and blind policies was approximately 42.8%, confirming that sender information provides a substantial and measurable survival advantage. Furthermore, the near identical survival rates of the random and blind (always rest) policies, at 20.8% and 19.3% respectively, further indicate that the reward structure does not favor uninformed action selection over random chance. Finally, the relatively small gap between the greedy and optimal policies, at around 6.5%, reflects a design choice to keep the task tractable without requiring highly complex multi step strategies.

Finally, the penalty structure was designed to deter specific failure modes. Consuming raw or parasitic food without cooking imposed a minus 30 penalty. Failed hunts with injury imposed a minus 20 penalty. Wasted actions, such as attempting to hunt without a tool, imposed a minus 10 penalty. These penalties, combined with the stochastic health reduction from dangerous entities, were intended to discourage the simple gather and eat local optimum by making unsafe context independent actions costly when the wrong entity was present.

\
== Data Creation <group1>
\

Regarding the process of data generation and processing throughout the project, I intitally designed a system that would allow me adapt on future testing or structural changes. By doing so, I was able to use the same data for producing the behavioural baseline on which the hand coded policies are evaluated in the prototype, and producing the supervised training batches consumed by the emergent communication models. Hence, both pipelines draw from the same world mechanics and entity catalogue, which ensures that baseline performance figures and model performance figures remain directly comparable.


Given the nature of the game, training data is generated synthetically by simulating episodes of the survival game under a random valid policy, producing a diverse distribution of entity encounters across turns.

Each encounter, such as Goat (0,1,1,3,1,0), is created through a two stage selection process. First, an entity type is sampled according to a fixed set of spawn weights, where resources appear most frequently at 35%, followed by animals at 25%, dangers at 20%, and crafting opportunities and events at 10% each. These weights were chosen to reflect a plausible survival environment in which useful resources are the most common encounter, dangerous situations are frequent but not overwhelming, and high value events are rare. Second, the pool of entities of the selected type is filtered by weather compatibility before a target entity is drawn uniformly at random from the filtered pool.

Then, to generate episodes and turns it follows the same episodic logic as the game itself. I simulate each episode proceeds turn by turn, where I generate an encounter for each turn, simulate the games logic (energy drain, weather changes, valid actions...), and then collect the set of turns and create an episode. Note that at each turn, a sample is extracted and stored for later use by the training pipeline. This preserves the causal structure of the game and ensures that each sample reflects a plausible game context rather than an independently drawn random state.


With this system, my intital data set was approximately 2,200 episodes, generating a maximum of $2,200 times 20 = 44,000 "samples"$, although later on this proved to be insufficient and problematic as the  produced a training set had significant overlap problem, where 12.4% of validation samples were found to be exact duplicates of training samples. From a design perspective, this happened because the number of unique entity/state combinations is finite and relatively small, so independently generated episodes converge on repeated configurations by chance. Such duplication is a form of data leakage in which validation accuracy overestimates generalisation performance. To solve this, I increased the episode generation to 10,000 and introduced a fingerprint based de-duplication, where each sample is hashed before split assignment, and any fingerprint already present in a prior split is excluded. At this scale, the generator produced approximately 164,000 samples in nine seconds, and the final split achieved zero overlap between all three partitions.

The split policy adopted following this revision was 80 percent training, 10 percent validation, and 10 percent test, applied at episode level rather than at turn level. This was done to prevent temporal leakage from correlated turns within the same trajectory appearing in different splits. The three-way structure allocates the training partition for gradient updates, the validation partition for hyperparameter monitoring and run comparison, and the test partition to unbiased final performance reporting as seen in *#underline[@data-generation-results]*. The final split produces 131,361 training samples, 13,517 validation samples, and 13,821 test samples on average.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left, left),
    table.header(
      [*Split*], [*Episodes*], [*Samples*],
      [*Animal*], [*Resource*], [*Danger*], [*CraftOpp / Event*]
    ),

    [Train], [8000], [132135], [32691], [46362], [26621], [26461],
    [Val], [1000], [13655], [3383], [4765], [2776], [2731],
    [Test], [1000], [13583], [3401], [4774], [2636], [2772],
  ),
  caption: [Data generation results example.],
) <data-generation-results>

\
== Architecture planned for the models
\
Moving on to the projects architecture, I designed it leveraging the standard EGG sender/receiver core in which two neural networks communicate through a discrete symbolic channel. This framing allowed me to create a natural information asymmetry, where the sender has perceptual access to the entity but not to the survival state, and the receiver has access to the survival state but not to the entity. The sender was designed to map encounter representation into a communication latent state, while the receiver maps said message plus some game context into an action preference,  extended with a reconstruction head to provide a supervised auxiliary signal alongside the reward driven communication objective. Neither agent can act optimally in isolation, which makes communication strictly necessary for survival and ensures that any learned protocol carries genuine task relevant information.

From the beginning, the architecture was designed to support both Reinforce and Gumbel Softmax training modes, sharing core components and configurable wrappers. I did this so that I could experiment in the future with both modes, as if communication only appears under one optimisation regime, conclusions become fragile. Unfortunately, despite this dual design the present project used only Gumbel Softmax because it offered smoother optimisation under the project constraints and clearer control of exploration through temperature scheduling.

Gumbel-Softmax replaces discrete symbol sampling with a differentiable continuous relaxation \@jang2017categorical, allowing gradients to flow directly through the message generation process and into the sender's weights. This produces a more stable and informative training signal than the high-variance policy gradient estimates that REINFORCE relies on. *FIND SOURCE FOR DIFFERENTIATION*

Even though the agents have similar structures, there are some small differences that are key to the success of the design. 

The sender takes a one-hot encoded entity vector as input, processes it through a linear projection layer to produce a fixed sized hidden vector, and uses this hidden representation to initialise the message generating RNN (Recurrent Neural Network). Note that the projection is necessary because the input, the one-hot encoding across six entity dimensions, is a sparse high dimensional representation that benefits from a compact learned projection before sequential processing. Then, the RNN generates the message symbol by symbol, emitting a sequence of discrete symbols up to the configured maximum message length (2 as default), with each symbol drawn from a vocabulary of configurable size (50 as default). By doing so, it allows exponentially more distinct messages than a single symbol alone, and creates structural pressure for compositional organisation because the sender must decide what to encode in each position independently. 

Similarly, the receiver mirrors this structure and takes the incoming symbol sequence, processes it through a corresponding RNN to produce a message representation, and combines this with the current survival state to produce two outputs. First some action logits (preferences) over the eleven action space, and second some entity reconstruction logits (entity predictions) over the 40 entity classes.

#pagebreak()

Later on the project, I added the reconstruction head to the receiver as an auxiliary supervised objective to accelerate the development of entity discriminative communication. Without reconstruction, the only training signal for the sender was the reward obtained by the receiver's action, which is a weak and delayed supervision signal particularly early in training when the action policy is still random. 

As for the loss, at turn level, the simulator computes action outcomes, unaddressed encounter penalties, and survival bonuses. In abstract form:

$ R_t = r_"action" + r_"unaddressed" + r_"alive" + r_"energy" + r_"health" $

This expression is intentionally additive to keep interpretation transparent during analysis. The final benchmark simulator also includes terminal survival completion terms, making episode return:

$ G = sum_(t = 1)^T R_t + r_"terminal" $

with terminal additions for completion and end state efficiency when the episode is survived.

For differentiable training mode, the design required a smooth surrogate objective over the discrete action set. Therefore, the Gumbel-Softmax pathway uses expected reward over action probabilities:

$ E[R | s, m] = sum_(a in A) p_theta(a | s, m) dot hat(R)(a, s, e) $

where $hat(R)$ is a deterministic expected reward approximation and $p_theta$ is the receiver action distribution conditioned on state $s$ and message $m$.

The design intention at the start was to optimise task return primarily and introduce communication regularisation only if required for protocol quality. In the final implemented formulation, the GS objective combines task and auxiliary terms:

$ L = -lambda_r dot E[R] + lambda_c dot L_"recon" - lambda_h dot H(p_theta) $

where $L_"recon"$ is entity reconstruction loss and $H$ is action entropy. The reconstruction cross-entropy loss provides a direct, dense gradient that encourages the sender to emit messages from which the entity identity can be recovered, establishing the communication channel before the action policy has matured. Further disccusions on the loss and the historical progression of these terms and their scheduling is discussed in the Implementation chapter.


A notable feature that I designed also was a high configurability through runtime arguments rather than hardcoded values. Motivated by the principle of encapsulation, the model architecture is decoupled from any specific experimental configuration, allowing different configurations to be tested by changing arguments at launch rather than by modifying the model code. This increased the efficiency later on while testing, allowing me to execute automated runs with changing parameters, improving reproducibility of run settings, and controlled comparison across design alternatives without rewriting model code.

Finally, further down the line I adapted the code and made a major design change where I separated message temperature from action temperature. Message temperature controls how sharp or diffuse the Gumbel-Softmax distributions are during symbol generation, determining how close to discrete the message channel is, while action temperature independently controls the sharpness of the receiver's action distribution. By doing this it allowed the communication channel to anneal towards discrete symbols at a different rate from the action policy's exploration schedule, which provides finer control over the dynamics of communication emergence relative to action learning. 


*[FIGURE: End-to-end pipeline diagram showing the full forward pass. Left to
right: entity vector input → sender linear projection → sender LSTM → discrete
message sequence → receiver LSTM → message representation combined with survival
state → (a) action logits over 11 actions with softmax, (b)
reconstruction logits over 40 entities with cross-entropy. Include reward feedback arrow from action selection to loss. INSERT AS A FULL-WIDTH FIGURE.]*

\
== Brief design note on dropped image branch <group1>
\
As mentioned earlier, alongside the vector based design, an image based survival game was initially planned. In this design, the sender would observe a rendered 64×64 image representing a survival scenario, process it through a convolutional encoder, and transmit a symbolic message to the receiver. This approach aligned with the emergent communication literature that was done at the start of the project, where perceptual inputs can induce grounded representations and richer communication dynamics *\@lazaridou2018emergence*.

Despite its conceptual relevance, the image branch was removed from the core implementation scope following a reassessment as the image branch would have introduced significant overhead in data generation or curation, representation validation, and computational cost. This would have reduced the depth of investigation into the central research question within the available project timeline.



#pagebreak()

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

When the reconstruction head was configured to predict across five classes, the agent reached 100% reconstruction accuracy within three to five epochs but generated only five unique messages, one per class. The receiver was therefore receiving category-level information with no discriminative signal about the specific entity within that category, which was insufficient to drive context-appropriate action selection. Following this observation, the entity set was expanded to 40 individual entities, comprising 11 animals, 9 resources, 10 dangers, 5 craft opportunities, and 5 events, and the reconstruction head was updated accordingly. This change forced the sender to develop a richer protocol distinguishing all 40 entities, rather than merely grouping them into five types, and reconstruction accuracy took substantially longer to converge, reaching 100% by epoch 22 rather than epoch 3 to 5, which allowed meaningful communication structure to develop during that window.

Long short-term memory cells were chosen as the recurrent unit for both agents.
Simple recurrent networks are known to suffer from vanishing gradients, which
makes them unreliable for learning dependencies across the full length of a
message sequence. Gated units such as long short-term memory address this by
maintaining an explicit memory cell controlled by input, forget, and output
gates, which allows the network to selectively preserve or discard information
across steps \@jang2017categorical. Transformers were considered as an
alternative, but their self-attention mechanism requires longer sequences to
provide meaningful benefit, and the short message lengths used in this project
make recurrent units the more appropriate and computationally lighter choice.

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


== Impplementation relevant Material

#pagebreak()

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, left, left, left),
    table.header([*Type*], [*Entity*], [*Vector (t,s,d,v,u,w)*], [*Description*]),

    [*Animals (11)*], [], [], [],
    [Animal], [Lion], [(0,0,4,4,1,0)], [Predator with extreme danger and high food reward, best handled with a spear.],
    [Animal], [Wolf], [(0,0,3,3,1,0)], [Pack predator with high danger and medium reward from hunting.],
    [Animal], [Bear], [(0,0,4,4,1,3)], [Large predator with extreme danger that is favoured in calm weather.],
    [Animal], [Deer], [(0,1,1,4,1,1)], [Low-danger herbivore with high reward that appears in sunny weather.],
    [Animal], [Goat], [(0,1,1,3,1,0)], [Low-danger herbivore with moderate return and broad weather compatibility.],
    [Animal], [Snake], [(0,2,3,2,0,0)], [High-danger reptile that can be hunted without a tool but with risk.],
    [Animal], [Lizard], [(0,2,1,3,0,1)], [Low-danger reptile that is easier to encounter in sunny weather.],
    [Animal], [Fish], [(0,3,0,3,2,2)], [Safe aquatic target that is most effective with a fishing rod in rain.],
    [Animal], [Salmon], [(0,3,0,4,2,2)], [Safe aquatic target with higher reward than fish under rainy conditions.],
    [Animal], [Rabbit], [(0,4,1,3,1,1)], [Low-danger small game with reliable medium reward in sunny weather.],
    [Animal], [Squirrel], [(0,4,0,2,0,1)], [Safe small game that can be taken bare-handed for low reward.],

    [*Resources (9)*], [], [], [],
    [Resource], [Berries], [(1,3,1,3,0,1)], [Plant resource with low risk and medium energy return in sunny weather.],
    [Resource], [Mushrooms], [(1,3,2,3,0,0)], [Plant resource with medium poison risk and moderate energy value.],
    [Resource], [Herbs], [(1,3,0,2,0,1)], [Safe plant resource with neutral energy and medicinal utility.],
    [Resource], [Wood], [(1,1,1,2,0,1)], [Material resource with low risk that supports tool crafting.],
    [Resource], [Firewood], [(1,1,0,2,0,0)], [Safe material resource that contributes directly to fire and shelter.],
    [Resource], [Stone], [(1,4,0,2,0,0)], [Safe mineral resource required for spear crafting.],
    [Resource], [Flint], [(1,4,1,2,0,0)], [Low-risk mineral source used as a partial substitute for stone supply.],
    [Resource], [Water], [(1,2,0,3,0,0)], [Safe water resource with medium energy-equivalent benefit.],
    [Resource], [Muddy Water], [(1,2,2,3,0,2)], [Rain-linked water source with medium disease-associated risk.],

    [*Dangers (10)*], [], [], [],
    [Danger], [Storm], [(2,0,3,0,4,2)], [High danger weather threat where shelter is the intended mitigation tool.],
    [Danger], [Blizzard], [(2,0,4,0,4,0)], [Extreme danger weather event that strongly penalises unprepared states.],
    [Danger], [Cold Night], [(2,1,2,1,3,4)], [Medium danger cold event where fire is the key defensive tool.],
    [Danger], [Frost], [(2,1,3,0,3,4)], [High danger cold event that rewards proactive fire preparation.],
    [Danger], [Poison Plant], [(2,2,3,0,0,0)], [High danger hazard where tool-based mitigation is not available.],
    [Danger], [Thorns], [(2,2,2,1,0,1)], [Medium danger hazard that causes attrition if not addressed.],
    [Danger], [Cliff], [(2,3,3,1,0,0)], [High danger terrain encounter with substantial potential health loss.],
    [Danger], [Quicksand], [(2,3,4,0,0,2)], [Extreme terrain danger in rainy contexts with severe survival risk.],
    [Danger], [Fever], [(2,4,2,1,0,0)], [Medium danger disease encounter that can erode long-horizon survival.],
    [Danger], [Infection], [(2,4,3,0,0,0)], [High danger disease encounter requiring careful risk management.],

    [*Craft Opportunities (5)*], [], [], [],
    [Craft Opportunity], [Weapon Cache], [(3,0,0,2,0,0)], [Low-risk opportunity that boosts weapon-crafting material availability.],
    [Craft Opportunity], [Dry Campsite], [(3,1,0,2,0,1)], [Low-risk opportunity that supports efficient fire setup in sun.],
    [Craft Opportunity], [Rocky Outcrop], [(3,2,0,2,0,3)], [Low-risk opportunity that favours shelter construction in calm weather.],
    [Craft Opportunity], [Riverbank], [(3,3,0,2,0,2)], [Low-risk opportunity that supports fishing-oriented preparation in rain.],
    [Craft Opportunity], [Herb Garden], [(3,4,0,2,0,1)], [Low-risk opportunity that increases access to medicinal herbs.],

    [*Events (5)*], [], [], [],
    [Event], [Supply Cache], [(4,0,0,3,0,0)], [Positive event that grants bonus survival materials.],
    [Event], [River Crossing], [(4,1,1,2,0,0)], [Low-risk event that can replenish water stores.],
    [Event], [Animal Migration], [(4,2,0,3,1,0)], [Favourable event that offers opportunistic meat gain.],
    [Event], [Berry Patch], [(4,3,0,4,0,1)], [High-value event that yields strong berry-based energy support.],
    [Event], [Cave], [(4,4,0,2,0,4)], [Night-compatible event that provides safe situational shelter benefit.],
  ),
  caption: [Full entity list across all 40 entities, grouped by type, with vectors and descriptions.],
) <full-entity-list>


== Testing Summary <group1>

This should describe the steps taken to debug, test, verify or otherwise confirm the correctness of the various modules and their combination.

=== Machine Learning Outputs and Metrics <ML-lerning>
\ 
The MNIST implementation, using a simple feed-forward neural network, provided insightful results regarding both training stability and model performance.

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

#pagebreak()

=== PyTorch Learning <pytorch_app>
\
*Output of tutorial_pytorch.py:
*\
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

#figure(
  image("Images/MNIST_Fashion.png", width: 40%),
  caption: [MNIST Fashion Example]
)

\ \ \
*Output for MNIST_Fashion.py:
*
\
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

#pagebreak()

=== MNIST Classifier Output <train-logs-mnist>
\
*Output for entropy coefficient at 0.01:
*\
#{
  set text(size: 8pt)
raw("
Loading MNIST data...
Train dataset size: 60000
Test dataset size: 10000
Images per sample: 3 (1 target + 2 distractors)
Random seed: 42
Starting training...
{\"loss\": 1.202682614326477, \"acc\": 0.3354499936103821, \"sender_entropy\": 0.40308356285095215, \"receiver_entropy\": 0.0, \"length\": 10.95181655883789, \"mode\": \"train\", \"epoch\": 1}
{\"loss\": 1.0991835594177246, \"acc\": 0.3359000086784363, \"sender_entropy\": 4.227686811741904e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 1}
{\"loss\": 1.0988200902938843, \"acc\": 0.33391666412353516, \"sender_entropy\": 4.0395667366688315e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 2}
{\"loss\": 1.098612904548645, \"acc\": 0.33489999175071716, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 2}
{\"loss\": 1.0986089706420898, \"acc\": 0.3344166576862335, \"sender_entropy\": 4.0395667366688315e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 3}
{\"loss\": 1.0986121892929077, \"acc\": 0.3264999985694885, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 3}
{\"loss\": 1.0986086130142212, \"acc\": 0.33285000920295715, \"sender_entropy\": 4.0395667366688315e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 4}
{\"loss\": 1.0986120700836182, \"acc\": 0.3386000096797943, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 4}
{\"loss\": 1.0986088514328003, \"acc\": 0.33356666564941406, \"sender_entropy\": 4.0395667366688315e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 5}
{\"loss\": 1.0986121892929077, \"acc\": 0.3296000063419342, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 5}
{\"loss\": 1.0986084938049316, \"acc\": 0.3297666609287262, \"sender_entropy\": 4.0395670142245876e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 6}
{\"loss\": 1.0986120700836182, \"acc\": 0.3312000036239624, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 6}
{\"loss\": 1.0986086130142212, \"acc\": 0.3319833278656006, \"sender_entropy\": 4.0395667366688315e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 7}
{\"loss\": 1.0986120700836182, \"acc\": 0.3467999994754791, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 7}
{\"loss\": 1.0986084938049316, \"acc\": 0.3343000113964081, \"sender_entropy\": 4.0395670142245876e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 8}
{\"loss\": 1.0986120700836182, \"acc\": 0.33660000562667847, \"sender_entropy\": 4.2276873668534165e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 8}
{\"loss\": 1.0986084938049316, \"acc\": 0.3293166756629944, \"sender_entropy\": 4.0395670142245876e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 9}
{\"loss\": 1.0986120700836182, \"acc\": 0.3310000002384186, \"sender_entropy\": 4.227687921964929e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 9}
{\"loss\": 1.0986084938049316, \"acc\": 0.3308500051498413, \"sender_entropy\": 4.0395670142245876e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 10}
{\"loss\": 1.0986121892929077, \"acc\": 0.32710000872612, \"sender_entropy\": 4.227688477076441e-10, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 10}

Saving checkpoint to /cs/home/nc212/Documents/Fourth_Year/Dissertation/egg/zoo/basic_games_MNIST/runs/mnist_run_20260327_140311_01_mode-rf_ent-0.001_seed-42/checkpoint.pth...
Checkpoint saved!
Training complete!
")
} <no-wc>

\
*Output for entropy coefficient at 0.003:
*\

#{
  set text(size: 8pt)

raw("
Loading MNIST data...
Train dataset size: 60000
Test dataset size: 10000
Images per sample: 3 (1 target + 2 distractors)
Random seed: 42
Starting training...
{\"loss\": 2.9201760292053223, \"acc\": 0.6377000212669373, \"sender_entropy\": 1.1577460765838623, \"receiver_entropy\": 0.0, \"length\": 10.850749969482422, \"mode\": \"train\", \"epoch\": 1}
{\"loss\": 1.8013989925384521, \"acc\": 0.85589998960495, \"sender_entropy\": 0.6081305146217346, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 1}
{\"loss\": 1.1886173486709595, \"acc\": 0.8604166507720947, \"sender_entropy\": 0.2831442058086395, \"receiver_entropy\": 0.0, \"length\": 10.949216842651367, \"mode\": \"train\", \"epoch\": 2}
{\"loss\": 0.332803338766098, \"acc\": 0.9017000198364258, \"sender_entropy\": 0.08913595229387283, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 2}
{\"loss\": 0.49584928154945374, \"acc\": 0.8858333230018616, \"sender_entropy\": 0.09701570868492126, \"receiver_entropy\": 0.0, \"length\": 10.997133255004883, \"mode\": \"train\", \"epoch\": 3}
{\"loss\": 0.3099953830242157, \"acc\": 0.91839998960495, \"sender_entropy\": 0.15136100351810455, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 3}
{\"loss\": 0.40766119956970215, \"acc\": 0.9246666431427002, \"sender_entropy\": 0.11690425872802734, \"receiver_entropy\": 0.0, \"length\": 10.997883796691895, \"mode\": \"train\", \"epoch\": 4}
{\"loss\": 0.23304638266563416, \"acc\": 0.9246000051498413, \"sender_entropy\": 0.09958086907863617, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 4}
{\"loss\": 0.3271089196205139, \"acc\": 0.9250500202178955, \"sender_entropy\": 0.10193736851215363, \"receiver_entropy\": 0.0, \"length\": 10.999833106994629, \"mode\": \"train\", \"epoch\": 5}
{\"loss\": 0.2378828078508377, \"acc\": 0.9358999729156494, \"sender_entropy\": 0.11043731123209, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 5}
{\"loss\": 0.32480356097221375, \"acc\": 0.9498666524887085, \"sender_entropy\": 0.11339305341243744, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 6}
{\"loss\": 0.22481600940227509, \"acc\": 0.9223999977111816, \"sender_entropy\": 0.0945795550942421, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 6}
{\"loss\": 0.23530103266239166, \"acc\": 0.9511333107948303, \"sender_entropy\": 0.08413448184728622, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 7}
{\"loss\": 0.15195830166339874, \"acc\": 0.9552000164985657, \"sender_entropy\": 0.07693234831094742, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 7}
{\"loss\": 0.24594157934188843, \"acc\": 0.9501166939735413, \"sender_entropy\": 0.1034746915102005, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 8}
{\"loss\": 0.20704349875450134, \"acc\": 0.9100000262260437, \"sender_entropy\": 0.14888796210289001, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 8}
{\"loss\": 0.2625497877597809, \"acc\": 0.9548166394233704, \"sender_entropy\": 0.12389612942934036, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"train\", \"epoch\": 9}
{\"loss\": 0.15872858464717865, \"acc\": 0.9581000208854675, \"sender_entropy\": 0.10291261225938797, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 9}
{\"loss\": 0.21799813210964203, \"acc\": 0.9555833339691162, \"sender_entropy\": 0.104851134121418, \"receiver_entropy\": 0.0, \"length\": 10.999833106994629, \"mode\": \"train\", \"epoch\": 10}
{\"loss\": 0.1466856449842453, \"acc\": 0.970300018787384, \"sender_entropy\": 0.11302988231182098, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 10}

Saving checkpoint to /cs/home/nc212/Documents/Fourth_Year/Dissertation/egg/zoo/basic_games_MNIST/runs/mnist_run_20260327_140843_02_mode-rf_ent-0.003_seed-42/checkpoint.pth...
Checkpoint saved!
Training complete!
")
} <no-wc>
\
*Output for entropy coefficient at 0.001:
*\
#{
  set text(size: 8pt)

  raw("
Loading MNIST data...
Train dataset size: 60000
Test dataset size: 10000
Images per sample: 3 (1 target + 2 distractors)
Random seed: 42
Starting training...
{\"loss\": 1.3987045288085938, \"acc\": 0.33000001311302185, \"sender_entropy\": 3.532022714614868, \"receiver_entropy\": 0.0, \"length\": 10.038399696350098, \"mode\": \"train\", \"epoch\": 1}
{\"loss\": 1.1472893953323364, \"acc\": 0.3353999853134155, \"sender_entropy\": 3.5482306480407715, \"receiver_entropy\": 0.0, \"length\": 7.878900051116943, \"mode\": \"test\", \"epoch\": 1}
{\"loss\": 1.1394423246383667, \"acc\": 0.3333333432674408, \"sender_entropy\": 3.549595832824707, \"receiver_entropy\": 0.0, \"length\": 9.952982902526855, \"mode\": \"train\", \"epoch\": 2}
{\"loss\": 1.1088190078735352, \"acc\": 0.3305000066757202, \"sender_entropy\": 3.550182580947876, \"receiver_entropy\": 0.0, \"length\": 8.81879997253418, \"mode\": \"test\", \"epoch\": 2}
{\"loss\": 1.107113242149353, \"acc\": 0.3350333273410797, \"sender_entropy\": 3.5516908168792725, \"receiver_entropy\": 0.0, \"length\": 9.96186637878418, \"mode\": \"train\", \"epoch\": 3}
{\"loss\": 1.074389100074768, \"acc\": 0.3312999904155731, \"sender_entropy\": 3.551551103591919, \"receiver_entropy\": 0.0, \"length\": 4.0, \"mode\": \"test\", \"epoch\": 3}
{\"loss\": 1.0944782495498657, \"acc\": 0.33258333802223206, \"sender_entropy\": 3.5527215003967285, \"receiver_entropy\": 0.0, \"length\": 9.972599983215332, \"mode\": \"train\", \"epoch\": 4}
{\"loss\": 1.092093825340271, \"acc\": 0.33880001306533813, \"sender_entropy\": 3.5528388023376465, \"receiver_entropy\": 0.0, \"length\": 10.986000061035156, \"mode\": \"test\", \"epoch\": 4}
{\"loss\": 1.0873688459396362, \"acc\": 0.3334166705608368, \"sender_entropy\": 3.553968667984009, \"receiver_entropy\": 0.0, \"length\": 9.967633247375488, \"mode\": \"train\", \"epoch\": 5}
{\"loss\": 1.0867252349853516, \"acc\": 0.32919999957084656, \"sender_entropy\": 3.554259777069092, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 5}
{\"loss\": 1.0828406810760498, \"acc\": 0.3298499882221222, \"sender_entropy\": 3.5547852516174316, \"receiver_entropy\": 0.0, \"length\": 9.959783554077148, \"mode\": \"train\", \"epoch\": 6}
{\"loss\": 1.0826563835144043, \"acc\": 0.3310000002384186, \"sender_entropy\": 3.5552055835723877, \"receiver_entropy\": 0.0, \"length\": 10.994999885559082, \"mode\": \"test\", \"epoch\": 6}
{\"loss\": 1.0797361135482788, \"acc\": 0.33193331956863403, \"sender_entropy\": 3.555321216583252, \"receiver_entropy\": 0.0, \"length\": 9.9667329788208, \"mode\": \"train\", \"epoch\": 7}
{\"loss\": 1.080179214477539, \"acc\": 0.3467000126838684, \"sender_entropy\": 3.5554914474487305, \"receiver_entropy\": 0.0, \"length\": 10.8371000289917, \"mode\": \"test\", \"epoch\": 7}
{\"loss\": 1.077720284461975, \"acc\": 0.33426666259765625, \"sender_entropy\": 3.555680990219116, \"receiver_entropy\": 0.0, \"length\": 9.964683532714844, \"mode\": \"train\", \"epoch\": 8}
{\"loss\": 1.0784094333648682, \"acc\": 0.33660000562667847, \"sender_entropy\": 3.5558295249938965, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 8}
{\"loss\": 1.075705885887146, \"acc\": 0.3293166756629944, \"sender_entropy\": 3.5559093952178955, \"receiver_entropy\": 0.0, \"length\": 9.957050323486328, \"mode\": \"train\", \"epoch\": 9}
{\"loss\": 1.076270580291748, \"acc\": 0.33070001006126404, \"sender_entropy\": 3.5559520721435547, \"receiver_entropy\": 0.0, \"length\": 10.99020004272461, \"mode\": \"test\", \"epoch\": 9}
{\"loss\": 1.0737301111221313, \"acc\": 0.3308500051498413, \"sender_entropy\": 3.556049346923828, \"receiver_entropy\": 0.0, \"length\": 9.973199844360352, \"mode\": \"train\", \"epoch\": 10}
{\"loss\": 1.0741565227508545, \"acc\": 0.32749998569488525, \"sender_entropy\": 3.556000232696533, \"receiver_entropy\": 0.0, \"length\": 11.0, \"mode\": \"test\", \"epoch\": 10}

Saving checkpoint to /cs/home/nc212/Documents/Fourth_Year/Dissertation/egg/zoo/basic_games_MNIST/runs/mnist_run_20260327_141408_03_mode-rf_ent-0.01_seed-42/checkpoint.pth...
Checkpoint saved!
Training complete!
")
} <no-wc>


#pagebreak()

== User Manual <group1>

Instructions on installing, executing and using the system where appropriate.

== Extended figures/tables/message snapshots <group1>


Results from the runs:

#set table(
  stroke: none,
)

#table(
  columns: (auto,auto),
  align: (center, center),
  image("Images/Loss_across_runs_with_recon.png"),
  image("Images/Loss_across_runs_no_recon.png"),
) #label("Loss-comparison")

#table(
  columns: (auto,auto),
  align: (center, center),
  image("Images/Aggregate_mean_reward_test_with_recon.png"),
  image("Images/Aggregate_mean_reward_test_no_recon.png"),
) #label("Mean-Comparison-Test")



// Recon accuracy
#table(
  columns: (auto, auto),
  align: (center, center),
  [#image("Images/Aggregate_recon_accuracy_test_with_recon.png")],
  [#image("Images/Aggregate_recon_accuracy_test_no_recon.png")],
)
// TopSim
#table(
  columns: (auto, auto),
  align: (center, center),
  [#image("Images/Aggregate_topsim_test_with_recon.png")],
  [#image("Images/Aggregate_topsim_test_no_recon.png")],
)

// Heatmaps WITH recon (2x2 + 1)
#table(
  columns: (auto, auto),
  align: (center, center),

  [#image("Images/message_entity_heatmap_run14.png")],
  [#image("Images/message_entity_heatmap_run15.png")],

  [#image("Images/message_entity_heatmap_run16.png")],
  [#image("Images/message_entity_heatmap_run17.png")],

  [#image("Images/message_entity_heatmap_run18.png")],
  []
)

// Heatmaps WITHOUT recon (2x2 + 1)
#table(
  columns: (auto, auto),
  align: (center, center),

  [#image("Images/message_entity_heatmap_run19.png")],
  [#image("Images/message_entity_heatmap_run20.png")],

  [#image("Images/message_entity_heatmap_run21.png")],
  [#image("Images/message_entity_heatmap_run22.png")],

  [#image("Images/message_entity_heatmap_run23.png")],
  []
)

// Message reuse
#table(
  columns: (auto, auto),
  align: (center, center),
  [#image("Images/message_reuse_across_runs.png")],
  [#image("Images/message_reuse_across_runs_no_recon.png")]
)

With recon:
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),

    [*Run*], [*Ent*], [*Msg*], [*Reuse*], [*Ratio*], [*Max Ent*], [*Top Msg*], [*Top Ent*], [*Top Count*],

    [train_run14.txt], [40], [40], [0], [0.0], [1], [0 40 0], [1], [711.0],
    [train_run15.txt], [40], [40], [0], [0.0], [1], [0 0 0], [1], [756.0],
    [train_run16.txt], [40], [40], [0], [0.0], [1], [0 0 0], [1], [726.0],
    [train_run17.txt], [40], [37], [2], [0.0541], [3], [0 0 0], [3], [767.0],
    [train_run18.txt], [40], [40], [0], [0.0], [1], [1 33 0], [1], [749.0]
  ),
  caption: [With Recon]
) #label("runs-with-recon")

Without recon:
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),

    [*Run*], [*Ent*], [*Msg*], [*Reuse*], [*Ratio*], [*Max Ent*], [*Top Msg*], [*Top Ent*], [*Top Count*],

    [train_run19.txt], [40], [16], [8], [0.5], [7], [13 14 0], [7], [2681.0],
    [train_run20.txt], [40], [13], [10], [0.769], [14], [15 32 0], [14], [4814.0],
    [train_run21.txt], [40], [13], [9], [0.692], [9], [15 7 0], [9], [3415.0],
    [train_run22.txt], [40], [18], [8], [0.444], [8], [12 20 0], [8], [3269.0],
    [train_run23.txt], [40], [11], [7], [0.636], [8], [29 29 0], [8], [2602.0]
  ),
  caption: [No Recon]
) #label("runs-without-recon")

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