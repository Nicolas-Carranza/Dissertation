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
\
Communication is often treated as something designed, yet in many settings it emerges only because it becomes necessary. When agents must rely on one another to survive, the ability to convey information is no longer optional but instrumental. This project investigates whether such conditions are sufficient for artificial agents to develop meaningful communication. To explore this, a custom survival game was implemented within the EGG framework in which one agent observes the environment and another selects actions based solely on the message it receives. Following preparatory work in machine learning and PyTorch, the system was evaluated through iterative experiments across multiple settings. The results show a clear performance advantage for communication over blind behaviours, while also indicating that message structure is strongly shaped by the objectives imposed on the agents. The findings presented in this paper suggest that while structured communication can emerge in survival based environments, its effectiveness depends on how objectives guide both signalling and action.

#pagebreak()

= Declaration <group1>

I hereby certify that this dissertation, which is approximately *#total-words* words in length, has been composed by me, that it is the record of work carried out by me and that it has not been submitted in any previous application for a degree. This project was conducted by me at the University of St Andrews from September 2025 to April 2026 towards fulfilment of the requirements of the University of St Andrews for the degree of Bachelor of Science (Honours) Computer Science And Management under the supervision of Dr. Phong Le.

In submitting this project report to the University of St Andrews, I give permission for it to be published online. I retain the copyright in this work.

April 3rd 2026, *Nicolás Carranza Arauna*

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

Advances in machine learning provide an alternative approach to this problem. By simulating environments in which artificial agents must interact to achieve shared goals, it becomes possible to observe the emergence of communication under controlled conditions. In particular, multi-agent reinforcement learning (MARL) enables agents to develop signalling strategies through repeated interaction and reward driven learning, offering a tractable framework for studying emergent behaviour. This project explores these ideas by investigating whether artificial agents can develop meaningful communication protocols in a custom survival based scenario, leveraging the EGG (Emergence of lanGuage in Games) library @kharitonov:etal:2021 to place agents in environments where coordination is necessary for success. To this end, the EGG toolkit is used to implement communication games with discrete signalling channels, allowing agents to coordinate under task specific pressures. The goal is to analyse whether communication emerges naturally from these interactions, and to what extent survival constraints influence the structure and development of emergent communication.

Formally, our core research question drives this investigation: *do meaningful communication protocols emerge when AI agents are constrained by survival scenarios requiring coordination?* To answer this, the project pursued four prioritised aims, first being to design and implement a tractable survival environment where communication is strictly necessary for optimal performance. Following this design, our second step is to train reinforcement learning agents within this environment to induce signalling behaviour. Once achieved, the third aim is to analyse the resulting protocols for structure, stability, and utility, finalising with our fourth aim which is evaluating how specific design and training factors influence the emergence of these protocols. Given that this work was conducted as part of a 15-credit module, the scope was deliberately constrained to establishing these experimental foundations and conducting targeted analysis rather than pursuing large scale simulations as it was initially designed for.

The project achieved a functional emergent communication pipeline, demonstrating a clear performance gap between agents with communication channels and those without. Specifically, agents trained with communication consistently outperformed _blind_ or _random_ policies, verifying the necessity of signalling for the survival task. My analysis revealed that while protocols did stabilise and exhibit structural patterns, the system remained sensitive to local optima, where some message collapses were observed in specific training runs where agents converged on suboptimal strategies. These results confirm the potential of survival based signalling games while highlighting the stability challenges inherent in multi-agent reinforcement learning.


#pagebreak()

= Context Survey <group1>

== Machine Learning Foundations <group1>
\
Machine learning is the study of computational methods that improve performance on a task through exposure to data or experience, rather than through explicitly programmed rules @mitchell1997machine. This project employs machine learning because the survival environment under study is too complex for a fixed rule set to capture all useful strategies, whereas repeated simulation provides sufficient experience for an adaptive model to discover and refine effective behaviour over time.

At a high level, a model defines a mapping from inputs to outputs, and its parameters are adjusted during training so that outputs become progressively more useful. Two learning paradigms are especially relevant here, and they differ fundamentally in how feedback is provided to the model.

The first is *supervised learning*, in which the training set consists of labelled input–output pairs, and the model is optimised to reproduce those labels. Supervised learning can be understood as direct instruction, where the correct answer is provided at every training step, much like a student working through an answer sheet. This makes it well suited to tasks where labelled data is plentiful, such as image classification or text categorisation.

The second paradigm, and the one more central to this dissertation, is *reinforcement learning* (RL). Unlike supervised learning, RL provides no direct label for each action the model takes. Instead, an agent interacts with an environment over a sequence of steps and receives scalar reward signals. Reinforcement learning is therefore a form of trial and feedback learning where the agent must discover, through interaction, which sequences of actions lead to high cumulative return @sutton2018reinforcement. This distinction from supervised learning is important because, in a survival environment, there is no natural "correct action" to label at each time-step, thus success or failure only becomes apparent across many steps.

#figure(
  image("Images/neural_network.png", width: 60%),
  caption: [Example of a deep neural network with multiple hidden layers @schmidhuber2015deep.],
)

Regardless of the learning paradigm in use, the underlying model class throughout this work is a *neural network*, a layered function approximator in which each layer applies a learnable transformation to its input, and passes the resulting representation to the next. To do this, a *forward pass* computes a prediction from a given input, while a *loss function* then quantifies how far that prediction is from the desired output. Then, optimisation proceeds by adjusting parameters to reduce this loss, typically via gradient based updates of the form:

$ theta <- theta - eta nabla_theta cal(L)(theta) $

where $eta$ is the learning rate, and error signals are propagated from the output layer back through the network using *backpropagation*, an application of the chain rule of calculus @rumelhart1986learning. Linking back to the two paradigms above, in supervised learning the loss is computed directly against known labels, whereas in reinforcement learning it is derived from reward signals accumulated through interaction.

With the core training mechanics established, it is worth defining several properties that characterise how well a trained model behaves. First, *convergence* refers to a regime in which successive parameter updates produce only negligible changes in performance, indicating that training has stabilised. A related but distinct concern is *overfitting*, which is when a model achieves low training loss but generalises poorly to unseen data, for instance, a model that memorises survival strategies specific to one map layout but fails on another. Conversely, a property we scientists look to maximise is *generalisation*, which describes the degree to which learned behaviour transfers from training samples to new inputs.

It is also worth noting that training is inherently stochastic in this project. *Stochastic training* refers to the intentional introduction of randomness into the training process, for example through data shuffling, mini-batch sampling, and random parameter initialisation. Because of this, results can vary across independent runs. To aid reproducibility, a *random seed* is a fixed value used to initialise a pseudo-random number generator, enabling reproducibility of stochastic experiments, an important practical technique used when comparing configurations.

This stochastic nature also makes evaluation methodology critical. To obtain reliable performance estimates, data is partitioned into three non-overlapping splits. The *training split* is used to update model parameters, the *validation split* guides model selection and hyperparameter tuning, and the *test split* is held out and used only for final performance reporting. Note that in our context *data leakage* is the inadvertent contamination of one split by information from another, which must be carefully avoided, as it produces artificially optimistic evaluation results that do not reflect true generalisation ability.

Finally, a number of representational and architectural terms appear in later chapters and are most naturally defined here. At the most basic level, categorical information is encoded using a *one-hot vector*, in which a single active entry represents a specific symbol, for example assigning "apple" to one position within a fixed vocabulary. Moreover, a *hard symbol* refers to the selection of a single discrete token from such a representation, whereas a *soft representation* retains a full probability distribution over the vocabulary, a distinction that becomes particularly consequential when training agents to communicate, as discussed in the following section. To support stable learning, inputs are typically subjected to *normalisation*, ensuring that features lie within a controlled range and improving the behaviour of gradient based optimisation. 

At the output stage, models produce *logits*, which are un-normalised scores over possible outcomes, transformed into a valid probability distribution through the softmax function:

$ p_i = exp(z_i) / (sum_j exp(z_j)) $

which is the natural final step in any model that must choose amongst discrete outcomes.

Building on the idea of sequential decision making introduced in the reinforcement learning discussion above, this work also makes use of *recurrent neural networks* (RNNs), which model sequential data by maintaining a hidden state updated at each time-step. The simplest form is the vanilla RNN, though it is known to suffer from vanishing gradients over long sequences, making it difficult to retain relevant information over many steps. On the other hand, *gated recurrent units* (GRUs) and *long short-term memory* networks (LSTMs) address this limitation by introducing learnable gates that regulate information flow @hochreiter1997long @cho2014learning, a factor that I take into account in later chapters. Both architectures are particularly relevant in the emergent communication setting discussed next too, where messages are inherently sequential objects rather than static vectors.

\
== Emergent Communication in Multi Agent Systems <group1>
\
*Emergent communication* refers to the study of settings in which no communication protocol is specified in advance, and a shared signalling convention develops instead through interaction under a common objective @lazaridou2020emergent. A protocol is considered meaningful when the receiver behaviour is demonstrably influenced by the content of the sender messages, rather than the communication channel being ignored entirely. An *agent* in this context is a decision making module that observes a (possibly partial) view of an environment and outputs an action. In emergent communication research, at least two agents interact, and they typically occupy one of two roles. The *sender* observes privileged information and encodes it into a message, whilst the *receiver* processes that message and takes an action in the environment. This asymmetry is fundamental to the field and mirrors the basic structure of human communication, where one party holds information the other needs.

For clarifications, a *symbol* in this context is a discrete token drawn from a finite vocabulary, typically represented during training as an integer index, and a *message* is an ordered sequence of one or more symbols. Critically, these symbols carry no predefined semantics, they begin as arbitrary tokens and any meaning they acquire emerges entirely through task experience, much like how the sounds of a spoken language only convey meaning through shared convention rather than any inherent property of the sounds themselves.

The foundational experimental paradigm in this literature is the *referential game*  @lazaridou2018emergence. In its canonical form, the sender observes a target object and transmits a message, and the receiver must identify the target from a set of candidates using only that message. This setup is valuable precisely because it creates explicit information asymmetry, making communication instrumentally useful and the quality of the emergent protocol directly measurable via task performance. It is worth noting, however, that referential games are deliberately simple, and this dissertation extends the setting considerably by placing agents in a survival environment where communication quality depends on long horizon decision making rather than a single correct identification. Throughout the project, the term *survival game* is used as reference to denote a sequential, cooperative environment in which agents face resource scarcity, environmental hazards, and temporally delayed consequences across multiple turns and episodes. Relative to single step referential games, such settings introduce substantially greater strategic depth and make the value of communication depend on its contribution to long horizon decision making rather than immediate accuracy.

Before proceeding, it is important to distinguish three related but distinct concepts that frequently appear together in this literature: coordination, communication and cooperation. *Coordination* refers to agents producing mutually compatible actions, while *communication* concerns whether the content of a signal causally influences the behaviour of its recipient. *Cooperation*, in turn, describes agents jointly optimising a shared objective. Although these properties often appear together in multi-agent settings, they are not equivalent. Agents may exhibit coordination by relying on shared environmental cues without engaging in meaningful communication, and may successfully transmit information without achieving fully cooperative outcomes.

If we look at previous work done in the field, work on emergent communication shows that the development of structured signalling is not automatic, but instead depends strongly on the interaction between task design and optimisation dynamics. In particular, learned protocols tend to reflect the objectives imposed during training rather than converging toward any inherently meaningful structure @kottur2017natural. The nature of the input space also plays a decisive role, with symbolic and perceptual representations giving rise to qualitatively different forms of communication @lazaridou2018emergence. At the same time, the literature highlights persistent challenges in evaluation, as high task performance and simple channel level tests are insufficient to demonstrate that messages encode robust and generalisable meaning @lowe2019pitfalls @lazaridou2020emergent.

Relatedly, evidence for compositionality must be treated with caution, since geometric alignment between message and meaning spaces may be suggestive but does not, on its own, establish fully compositional semantics @chaabouni2020compositionality. Furthermore, the broader motivation for this line of research is dual. Scientifically, emergent communication environments provide controlled, reproducible test beds for hypotheses about how signalling systems arise under selective pressure, a question of both cognitive and linguistic interest. Technically, they support the development of cooperative AI systems in which useful communication must be discovered rather than engineered.

\ 
== The EGG Framework and Evaluation Metrics <group1>
\
EGG (Emergence of lanGuage in Games) is an open source research framework developed at Facebook AI Research to support controlled experimentation on emergent communication @kharitonov:etal:2021, and serves as the primary infrastructure backbone of this project.

EGG adopts a clean separation between reusable framework components and game specific logic. In practice, the researcher specifies the environment data, agent architectures, and task losses, whilst EGG handles communication wrappers, training loops, logging, and experiment orchestration. This modularity is important for iterative research, as individual design factors can be varied in isolation without modifying the broader training infrastructure. The repository is organised into three areas that are directly relevant to this project. The `core` folder provides shared infrastructure including agent wrappers, the trainer class, and optimisation utilities. The `zoo` folder contains a collection of reference games and worked examples used to establish baseline patterns and guide implementation, and the `nest` folder provides tooling for systematic hyperparameter sweeps.

EGG supports two principal approaches to training agents through discrete communication channels, and the distinction between them reflects a fundamental tension in working with discrete variables. *REINFORCE* is a policy gradient estimator that optimises the expected return of sampled discrete actions @williams1992simple. It avoids the need to differentiate through the sampling operation directly, but is susceptible to high gradient variance, typically requiring variance reduction techniques such as learned baselines. The alternative, *Gumbel Softmax* relaxation @jang2017categorical, is a continuous approximation to categorical sampling that enables gradient based training to be applied to what would otherwise be a non differentiable discrete operation. In this model, a *temperature parameter* $tau$ controls the sharpness of the approximation,

$ y_i = exp((log p_i + g_i) / tau) / (sum_j exp((log p_j + g_j) / tau)) $

where $g_i tilde "Gumbel"(0,1)$ are independent Gumbel noise samples. As $tau -> 0$, the distribution concentrates on a single symbol, recovering the hard discrete selection discussed earlier in the context of hard symbols, whilst at higher temperatures the output is a smooth mixture across tokens.

Prior work using EGG has documented several recurring phenomena relevant to this project, including pressure toward compressed or degenerate codes under communication bottlenecks, strong sensitivity to the form of the training objective, and the need to guard against strategies that achieve high task reward without developing semantically robust communication @kharitonov2020entropy @dessi2021interpretable. These findings inform the evaluation approach adopted here, which draws on a range of complementary metrics rather than relying on task accuracy alone.

The first and most direct measure is *loss*, the scalar objective minimised during training, where comparisons are meaningful only within a fixed experimental configuration. Alongside this, *accuracy* records the fraction of correctly resolved communication targets on a scale of $[0,1]$, and a *confusion matrix* tabulates predicted versus true class assignments, which is useful for identifying systematic error patterns that accuracy alone would obscure. Together these three measures capture task level performance, but they say relatively little about the structure of the communication protocol that produced it.

To assess protocol structure, this project makes use of information theoretic and geometric measures. For example, a crucial metric is *message entropy*, computed as

$ H(m) = -sum_i p(m_i) log p(m_i) $

measures symbol diversity in the learned protocol, where high entropy indicates diverse use of the vocabulary and near zero entropy indicates that one or very few messages dominate, suggesting the kind of degenerate collapse noted in the following chapters. Building on this, *topographic similarity (TopSim)* @brighton2006understanding is used as one of the primary metrics, as it measures the Spearman rank correlation between pairwise distances in the space of meanings and pairwise distances in the space of messages:

$ "TopSim" = rho(Delta_"meaning", Delta_"message") $

where values near $+1$ indicate strong alignment between semantic and communicative structure and values near $0$ indicate weak or absent alignment.

Finally, another two measures provide a more direct view of compositional structure. *Positional disentanglement (PosDis)* @chaabouni2020compositionality measures the degree to which individual symbol positions encode stable, consistent semantic factors, where higher values indicate that each position in the message reliably tracks a distinct aspect of meaning. Complementing this, *bag of symbols disentanglement (BosDis)* @chaabouni2020compositionality is an order invariant analogue of PosDis, measuring whether the multiset of symbols in a message consistently encodes stable semantic factors regardless of their sequential arrangement.

Taken together, these metrics are properly understood as diagnostic instruments rather than definitive tests. No single threshold establishes that a learned protocol constitutes full compositional language, which connects back to the cautionary note raised earlier regarding overclaiming compositionality. As a result, evaluation is typically framed in comparative terms, examining behaviour across controlled conditions, random seeds, and ablations, and complemented by qualitative inspection of message distributions alongside analysis of downstream behavioural outcomes.

#pagebreak()

= Requirements Specification <group1>
\
The objectives for this project are presented below, organised into primary, secondary, and tertiary priorities. These objectives were originally defined in the DOER and provide a structured progression from foundational work to more advanced exploration of emergent communication in multi-agent systems. In practice, primary and secondary objectives were completed, whereas tertiary objectives were partially deferred due to time and scope constraints.

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
   This objective centres on designing a basic survival scenario optimised for studying emergent communication. The focus remains on simple two-agent communication scenarios such as predator warning and hunting coordination. Implementation of visual input processing using image based rather than symbolic inputs constitutes an important technical requirement.

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
After the interim report, the implementation scope was revised by dropping the image based game branch and prioritising the vector based resource chain survival game. The main reason was practical, building and curating image data at the required quality and scale would have consumed disproportionate project time, reducing the depth of analysis possible for communication behaviour. Following this change, the project retained the same core research aim, while concentrating effort on a single environment that could be iterated quickly and evaluated rigorously.

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
   This objective centres on designing a basic survival scenario optimised for studying emergent communication. The focus remains on simple two-agent communication scenarios such as predator warning and hunting coordination. Implementation will follow a vector based resource chain infrastructure, motivated by the basic games library from the EGG repository.

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

#pagebreak()

= Preparation and Technical Ramp-Up <group1>
\
This chapter documents the preparatory work completed before the core survival game experiments and explains why that preparation was necessary for the later research claims. Although these activities were enabling work rather than attending to the central research question, they directly addressed the first primary objectives in the DOER and established the methodological reliability required for the implementation and evaluation chapters. Note that the preparation process followed weekly supervisor checkpoints, where progress was reviewed, technical misunderstandings were corrected, and short term milestones were refined in response to observed results.

A central aim of this chapter is to show that later design decisions were not made ad hoc. Instead, they were grounded in a staged progression from machine learning theory to framework level engineering, leading to controlled communication experiments. This progression reduced trial and error development, improved reproducibility, and made it possible to interpret model behaviour in terms of known optimisation and generalisation mechanisms.

\
== Machine Learning foundation <group1>
\
The first preparation stage focused on building a robust conceptual base in machine learning so that subsequent model design and experimental reasoning could be justified rigorously. The primary resource was the *#underline[#link("https://developers.google.com/machine-learning/crash-course")[Google Machine Learning Crash Course]]* @google:mlcrashcourse, supported by targeted readings, practical notebooks, and weekly discussion with my supervisor. The objective was to obtain a working understanding of the specific concepts that would directly bear on experimental reliability in the emergent communication work to follow.

Throughout the course taken, I focused on building a coherent foundation across core areas of supervised learning, combining theoretical understanding with structured practice. Topics covered included optimisation and loss behaviour, evaluation methodology, and data handling principles, each of which later informed the design and analysis of the survival game experiments. Rather than treating these areas in isolation, the emphasis was on understanding how they interact in practice, for example how optimisation choices affect evaluation outcomes, or how data preparation decisions influence model behaviour. This was reinforced through hands on exercises using real datasets provided by the course, with implementations and solutions included in the accompanying code. These tasks made abstract concepts tangible and provided an early environment in which to observe common failure modes, such as unstable training dynamics or misleading aggregate performance, both of which later reappeared in more complex forms during the main experiments.

Two areas in particular required deeper engagement, cross entropy loss and backpropagation. Both initially proved difficult to internalise, especially in terms of how training signals propagate and influence model updates over time. Addressing these gaps involved targeted reading, small scale experimentation, and iterative debugging, which in turn strengthened intuition around optimisation behaviour. This effort translated directly into later stages of the project, where these mechanisms underpin both the MNIST evaluation checkpoint and the communication experiments within the EGG framework. As a result, observations such as training instability or performance plateaus could be interpreted through known learning dynamics rather than treated as opaque issues. More broadly, this phase established a consistent experimental workflow and grounded subsequent design decisions, ensuring that later results could be analysed with a clear understanding of the underlying mechanisms.

This phase was completed over approximately three weeks and concluded with the MNIST implementation, described in the Appendix in *#underline[@mnist-performance-summary]*. Ultimately, this foundation did not merely satisfy the first primary objective, it directly shaped the reliability of the analysis presented in the remainder of this dissertation.

\
== PyTorch familiarisation and initial experiments <group1>
\
Following the initial theoretical phase, the project transitioned to practical familiarisation with PyTorch to ensure that model behaviour could be inspected, controlled, and debugged at the implementation level. Using *#underline[#link("https://docs.pytorch.org/tutorials/beginner/basics/intro.html")[PyTorch's Introduction to Basics]]* @pytorch:basics:intro as a foundation, I explored core fundamental building blocks such as tensor operations, shape transformations or autograd mechanics, with a particular emphasis on identifying and preventing silent shape mismatches, which can propagate through models without immediate runtime errors but lead to invalid training dynamics. 

All of the work done can be found under the Appendix in *#underline[@pytorch_app]* where I have listed the results for the core operations I explored. Additionally, I also expanded from the MNIST classification problem solved in the previous phase to complete training and evaluation pipelines on the Fashion MNIST dataset. This progression was critical, as it exposed the full experimental workflow that I applied in the design of the survival game: dataset loading, model definition, forward and backward passes, loss computation, parameter updates, and performance monitoring. Through these experiments, I developed the ability to interpret loss and accuracy trends, distinguishing meaningful convergence from unstable or noisy training behaviour. Thus, this phase had a direct impact on subsequent work by establishing a reliable development and debugging workflow. As a result, integration with the EGG framework and the development of communication games proceeded more efficiently and with reduced need for iterative rework.

\
== Emergent Communication and EGG Framework Familiarisation <group1>
\
Following the completion of the first primary objective, the next step on the roadmap was to acquire the theoretical and practical foundation in emergent communication necessary to design and evaluate meaningful experiments. This addressed the second primary objective directly, requiring both a study of the research literature and hands on engagement with the EGG framework before any custom development could begin. At this stage, I studied the definitions and distinctions already established in the Context Survey, and focused on implementation consequences. In practical terms, I needed to understand when the communication channel is genuinely used, how discrete message optimisation behaves in EGG (Gumbel Softmax versus REINFORCE), and which known failure modes such as collapse should be expected during training @lazaridou2020emergent @jang2017categorical @williams1992simple @kottur2017natural.

With this theoretical grounding established, I decided to start exploring the EGG framework itself. The EGG repository is structured around a clear separation between game specific logic and general communication infrastructure. The researcher's responsibility is to define the input data, the core agent modules, and the task loss, while EGG's core layer handles message generation, message processing, training mode selection, and optimisation orchestration. Agent modules are wrapped by framework components that implement either Gumbel-Softmax or REINFORCE training, depending on the chosen mode, and these wrappers connect to a game object that ties the sender, receiver, and loss together into a single trainable system. Moreover, the framework also provides callback utilities for logging, temperature annealing, and validation event printing, which allowed experiment outputs to be inspected without requiring custom implementations at this stage. 

To develop a working understanding of this pipeline, i used the `zoo/basic_games` module as a primary point of entry. This module implements both a reconstruction game, in which the receiver must reproduce the full input vector from the sender's message, and a discrimination game, in which the receiver must identify a target item among a set of distractors. In order to learn how different architecture is affected by hyperparameters, I decided to run the game while changing one hyperparameter and keeping the rest constant, which clarified how changing the game objective alters the demands placed on the communication channel.


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

Once my implementation was finished, I conducted a three way REINFORCE entropy sweep under a fixed seed (42), using sender entropy coefficients of 0.001, 0.003, and 0.01. The outcomes were sharply different. At 0.001, performance collapsed to chance level discrimination as the test accuracy reached only 32.71% by epoch 10, with loss saturating around 1.0986 (approximately _$ln(3)$_ for three candidates), and sender entropy collapsing from 0.403 at epoch 1 to approximately $4.2 times 10^{-10}$ from epoch 2 onward as seen in *#underline[@entropy-comparison]*. At 0.003, the model converged well, test accuracy rose to 97.03% by epoch 10 (with a temporary dip at epoch 8), while sender entropy stayed in a moderate range (approximately 0.08 to 0.61 across training), indicating a stable but still expressive communication policy, shown in *#underline[@0.003-loss]*. At 0.01, performance again remained at chance level, ending at 32.75% test accuracy, unlike the 0.001 collapse, sender entropy remained very high and increased toward 3.556, showing persistent over exploration and failure to stabilise a useful protocol. These runs therefore showed a narrow effective entropy regime around 0.003 for this setup. Note that full epoch wise logs for all runs are provided in the Appendix in *#underline[@train-logs-mnist]*.

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
  caption: [Entropy Comparison (0.001 vs 0.01)],
) #label("entropy-comparison")

#figure(
  image("Images/Entropy 0.003 MNIST.png"),
  caption: [Entropy Comparison (Best: 0.003)]
) #label("0.003-loss")

I then analysed the sender's messages and confusion structure for each run over the full 10,000 sample test split. The 0.001 run showed complete protocol collapse, where every class used the same single message (`[3, 3, 3, ..., 3]`) with frequency 1.0 and near zero entropy, so the receiver could not exploit communication and remained close to chance. The 0.01 run exhibited the opposite failure mode, the sender entropy stayed near 3.556, and each class mixed a few dominant templates (for example, sequences ending in tokens 9, 13, and 19) without class specific separation, again yielding chance level discrimination. By contrast, the 0.003 run produced a functional emergent code where per class message entropy remained low to moderate (approximately 0.08 to 0.16), top message frequencies were distributed (roughly 3% to 12% rather than 100% collapse), and per class accuracies were consistently high (approximately 95.8% to 98.2%). The confusion matrices reflected this directly. For 0.003, the diagonal was strongly concentrated (for example 974/980 for digit 0, 1128/1135 for digit 1, and 982/1009 for digit 9), with only small off diagonal leakage. For 0.001 and 0.01, diagonals stayed around 0.37 to 0.43 with diffuse off diagonal mass around 0.05 to 0.09, which is consistent with ineffective communication. The confusion matrix figures are included below in *#underline[@cm-001]*, *#underline[@cm-003]* and #underline[*@cm-01*].


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

During this analysis phase, a question also arose from parallel work in another module where confusion matrices had shown high training accuracy but poor generalisation. In this MNIST case, the main risk was not classical overfitting, but entropy mis-calibration. At 0.001, entropy collapsed too early and both train and test accuracies remained near chance. At 0.01, entropy stayed too high and the protocol never stabilised, again leaving both train and test near chance. Only at 0.003 did train and test curves rise together to strong performance, with final test accuracy at 97.03% and confusion-matrix diagonals near 0.97 to 0.99. Methodologically, this was an important finding as in this communication game, sender entropy coefficient is a first order control variable that determines whether an emergent protocol converges, collapses, or remains noisy.

Overall, this MNIST adaptation is treated as preparatory validation rather than a core dissertation contribution. Its role was to validate the EGG integration path, establish a disciplined experimental workflow, expose early communication analysis pitfalls, and provide an evidence based bridge from framework familiarisation to custom environment design.

#pagebreak()

= Software Engineering Process <group1>
\
This chapter describes the development approach taken throughout the project, the
structure of work across both semesters, and the technical resources used during
implementation and experimentation.

\
== Development Process <group1>
\
The project followed an iterative, supervisor driven workflow similar to Agile @agilemanifesto development, adapted for a single person research project. Rather than committing to a rigid implementation plan at the outset, my progress was governed by the weekly meetings with my supervisor, where the results of the most recent work were reviewed, short term milestones were set, and priorities were adjusted in response to observed outcomes. This structure was well suited to the nature of the work, where training behaviour, game design decisions, and evaluation methodology all evolved in response to experimental evidence rather than being fixed in advance.

This iterative approach had a direct impact on several design decisions documented in later chapters, as game mechanics were revised across multiple training runs when early tests exposed reward imbalances. Furthermore, hyperparameter were updated following observed instabilities, and the scope of the evaluation methodology expanded incrementally as the experiments produced interpretable outputs worth analysing in greater depth. Without the weekly checkpoint structure, these adjustments would have been harder to make in a principled and documented way.

\
== Development Structure <group1>
\
At the start of the project, in week two, the overall timeline was divided into two broad phases in discussion with the supervisor, setting the boundary at the end of the first semester.

The first semester, covering weeks two through nine from September to December, was dedicated entirely to preparation and foundational work. The first four weeks focused on building machine learning foundations and PyTorch competence, the following two weeks on EGG framework familiarisation and basic games experimentation, and the remaining weeks on the MNIST adaptation and initial custom game design. Work was paused during weeks ten through twelve due to exams, and the preparation phase was formally concluded by the end of semester.

The second semester, running from late January to April, provided approximately ten weeks of focused project time. The first two weeks were used to finalise the survival game design following the semester break, the next four weeks covered full implementation and iterative training, then the following three weeks were dedicated to analysis and evaluation, and the final two weeks were reserved for dissertation write-up. In practice, the implementation and analysis phases overlapped somewhat, as results from each training run informed both the next experimental iteration and the structure of the evaluation chapter. The total time committed to the project substantially exceeded the expectation for a 15 credit module, reflecting the depth of the preparation phase and the iterative nature of emergent communication experimentation.

\
== Resources and Technologies <group1>
\
The project was implemented in Python and built on top of the EGG toolkit, which was installed directly from source as an editable package. All experiments were run on my personal CP -only machine, which was sufficient for the MNIST preparation work and early survival game training but introduced practical constraints on the speed and scale of multi run sweeps. Thus,aA dedicated GPU machine would have meaningfully reduced iteration time, particularly during the Gumbel-Softmax hyperparameter exploration where temperature decay schedules required many sequential runs to evaluate. *#underline[@tech-stack]* summarises the full technology stack used across the project.

\
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

\
Training was performed on two different machines across the project. Throughout the preparation phase and early second semester work, all experiments ran on a personal laptop equipped with an Intel Core i5-1135G7 processor, 4 cores at 2.40 GHz, and 16 GB of RAM, with no discrete GPU available. Under this setup, a full survival game training run required approximately five to nine hours to complete, which made iterative hyperparameter exploration slow and constrained the number of configurations that could be evaluated in a given week. 

In March, work moved to the lab machines in the Jack Cole building, specifically PC9, equipped with an Intel Core i5-12400 processor, 6 cores at up to 4.40 GHz, 30 GB of RAM, and an NVIDIA GeForce RTX 3060 with 12 GB of VRAM. The EGG framework supports CUDA-accelerated training, and enabling this reduced the per run training time from approximately nine hours to roughly one hour, a reduction of nearly ninefold. This transition meaningfully increased the number of experimental iterations possible in the final weeks of the project and was directly responsible for the depth of the hyperparameter analysis presented in the evaluation chapter. For future work extending these experiments to larger agent populations or longer training schedules, the RTX 3060 remains a practical minimum, though 16 to 24 GB of VRAM would be preferable for larger survival-game configurations with expanded vocabulary or message length.

#pagebreak()

= Ethics <group1>
\
This project is based entirely on computational simulation and does not involve human participants, personal data, or interaction with living animals. Consequently, no direct ethical concerns were identified for data handling or participant risk. The work focuses on artificial agents in controlled environments, and all analysis is performed on generated experimental outputs. Use of the EGG framework is a licensing and attribution matter rather than an ethics concern, therefore external dependencies are documented appropriately in references and implementation notes. You can find the signed ethics form in the Appendix in *#underline[@ethics-form]*.

#pagebreak() 

= Design <group1>
\
Throughout this chapter, I have defined and documented the design decisions made for the survival game and the planned architecture for the systems conceptual structure, modelling assumptions, and justifications. Although implementation details and empirical findings are discussed in the next chapter, here I explain why each part of the system was designed as it was, what changed over time, and how those changes improved scientific validity. 

At this stage, all primary objectives up to and including Objective 3 had been completed, shifting my focus towards the remaining Objective 4 and all secondary objectives. At a high level, the project was designed around one central principle, *communication must be instrumentally necessary rather than decorative.* Thus, the game environment, the agent's protocols, the action constraints, and the evaluation plan were all defined to create pressure for meaningful signalling. As I applied an iterative approach, several structural decisions were revised during implementation when tests exposed weaknesses in the original plan. Thus, this chapter covers the reasoning behind the initial design, the revisions that were made and why, and how the final design remained aligned with the research question on emergent communication under survival constraints.

Note that the design was shaped by two parallel bodies of prior work completed during the preparation phase. On one hand, the MNIST adaptation had demonstrated that image based inputs could drive meaningful discrimination in EGG, which initially motivated a visually grounded survival scenario. This design idea was also supported by the supervisor, as it was inspired by real world imagery and environments that resemble natural survival settings, helping to create a more realistic and grounded scenario. On the other hand, while working through the `zoo/basic_games` module in the previous phase, it established a clear understanding of how vector based attribute inputs could be constructed efficiently and paired directly with EGG's communication wrappers. These two influences led to two distinct game designs being considered in parallel during week nine of the first semester, one image based and one vector based, with the vector based design ultimately being selected because it provided tighter control over world structure, clearer attribution of communication effects, and lower development risk within the 15 credit scope.

\
== Survival Game Environment and State/Action Structure <group1>
\
The survival game was designed around the main sender/receiver communication loop that had to be embedded within cooperative survival scenarios. The core premise is that a sender agent observes a survival encounter, in the form of an entity from the game world, and must communicate sufficient information to a receiver agent for the receiver to select the best action in response. Without the sender's message, the receiver operates blindly and cannot make contextually appropriate decisions, thus making communication strictly necessary for optimal performance. This design property is essential for the research because if the receiver could perform well without any message, there would be no selective pressure for the sender to develop a meaningful protocol.

To play the game, one must go through episodes, where each episode consists of a sequence of turns, with a maximum of 20 turns per episode. At each turn, the sender observes a single entity drawn from the game world, along with the current survival state of the agents. The sender transmits a message to the receiver, which then selects one action from a fixed action space of eleven options, producing a reward that depends on the entity encountered, the action chosen, and the current state of the agents. For each turn, it also checks the state of the agent and terminates early if either energy or health reaches zero, penalising the agents for poor decision making.

\
=== Entity Design and the Expansion from Five to Forty Entities <group1>
\
Each entity in the game is represented as a discrete six-dimensional attribute vector of the form:

$ e = [t, s, d, v, u, w] $
where $t$ is entity type, $s$ subtype, $d$ danger level, $v$ energy value class, $u$ tool requirement, and $w$ weather dependency. 

This gave a theoretical combination space of $5^6 = 15625$ entity vectors, from which a curated subset of 40 meaningful entities was selected, as a fully dense combinatorial world would increase realism in one sense, but it would dilute semantic control and make protocol analysis difficult within the dissertation scope. Thus, the chosen 40 entities provided enough diversity for non trivial communication while keeping interpretability manageable. This representation was directly inspired by the design of the `zoo/basic_games` module, where attribute value vectors provide a structured, disentangled input space that has been shown in the
emergent communication literature to favour the emergence of partial compositionality @lazaridou2018emergence. 

Each dimension takes values in ${0, 1, 2, 3, 4}$, giving a structured symbolic space with bounded combinatorics, where each dimension encodes the following information. The first dimension, _entity type_, takes one of five values corresponding to Animals, Resources, Dangers, Craft Opportunities, and Special Events, and was designed to identify the main type of the entity. Following this, the second dimension encodes the _subtype_ within that category, for example distinguishing predators from herbivores within animals, or food from materials within resources. Then, the third dimension encodes the _danger level_ on a scale from zero to four, where zero is considered safe and four represents high risk of significant health loss. Similarly, the fourth dimension encodes the _energy value_, which can be negative for entities that drain energy, and it is used directly on the reward system. As for the fifth dimension, it encodes whether a specific _tool_ is _required_ to interact beneficially with the entity, such as a spear for hunting or fire for cooking, providing depth and complexity to the game. Finally, the sixth dimension encodes the _weather dependency_, distinguishing entities that are only relevant in certain weather conditions. For the full 40 entity description, you can find in *#underline[@full-entity-list]* with the entity description.

Note that the initial design used these five categories as a core semantic distinction, including category level reconstruction pressure. However, this proved too coarse as a communication target because category level identification could be solved with highly compressed signalling that did not require fine grained distinctions. This proved insufficient during early training and motivated the focus on 40 specific entities rather than 5, later explained in the Implementation and Evaluation chapters.

\
=== State Variables and Action Space <group1>
\
An important aspect of the design for the survival game was the *survival state*, which is tracked across episodes and consists of four components.

First, *energy*, which decrements by a random amount between 2 and 4 per turn and is replenished by eating. Second, *health*, which decrements when the agent is injured by dangerous entities or takes environmental damage. Third, *an inventory*, designed with boolean tool flags recording whether a spear, fire, shelter, or fishing rod have been crafted. And fourth, a *current weather* condition that changes every four to six turns. 

The receiver observes this full survival state alongside the sender's message when selecting an action, meaning the receiver must learn to integrate two distinct information sources, the entity identity communicated by the sender, and the current state context available to it directly.

Furthermore, another important aspect of the game design is the *action space*, which governs what the receiver can do, containing eleven discrete options defined as:

$ A = {"hunt", "gather", "flee", "rest", "mitigate", "endure", "eat", "craft_spear", "craft_fire", "craft_shelter", "craft_rod"} $

Out of the eleven possible actions, two of these actions are context independent in the sense that they are always structurally valid: * flee and rest.* The remaining actions are conditional on entity type, inventory, and state. For example *hunt* is valid only when the encountered entity is an animal, and *mitigate* is valid only when a tool addressable danger is present and the required tool is available. 

This structure was intentional as it creates a situation where the receiver must have access to entity identity information to distinguish between the contextually optimal action and the superficially safe alternatives. Without knowing what entity the sender has observed, the receiver cannot reliably decide whether to hunt, flee, or mitigate, and falls back to gather and eat, which are always safe but suboptimal.
Thus, it served two goals, first it created conditional action validity so that not all actions were sensible in all contexts, and second, it allowed strategic chains, such as gather to craft to hunt to eat, which made communication useful beyond one step reflexes. Moreover, I later added action validity masking to prevent logically impossible decisions from dominating policy learning, while still allowing difficult choices among valid alternatives.

This changed in the receiver computation can be viewed as:

$ h = f_theta(m, s) $

$ z = g_theta(h) $

where $z$ are unmasked action logits, and legality is enforced afterwards by

$ z'_a = z_a $ for valid actions, and $ z'_a = -infinity $ for invalid actions.

This mechanism was added as it ensures invalid actions receive zero probability after softmax while keeping the semantic inference path anchored to message and state.

\
=== Reward Structure and Strategy Incentives <group1>
\
Another crucial aspect of the game design is the *reward structure*, designed to create a clear gradient of strategic value that would incentivise the receiver to use the sender's message rather than adopting a context free policy. 

Each turn, 10 points are given for remaining alive, with additional bonuses for maintaining energy levels and health above 70. Then, successful hunting an animal gives 15 points plus a multiplier based on the animal's danger level, rewarding the agent for taking on riskier prey when equipped to do so, motivating the agent to explore the crafting opportunities.Furthermore, completing a full transformation chain, for example hunting an animal, cooking the meat with fire, and eating the cooked meat, awards a 40 point completion bonus as it requires _'luck'_. For each episode completed with all 20 turns or more finished gives a further 150 point bonus, complemented by a efficiency reward from remaining energy and health at the episode's end.

Throughout development, the reward structure went through two major revisions before reaching a state where the baseline policies produced a clear strategy gradient.

In the original configuration, the random policy achieved survival rates close to those of the greedy policy, which indicated that the reward landscape was too flat to differentiate good and poor decisions. The first revision raised starting energy from 80 to 100, reduced metabolic drain from a mean of 4.5 to 3 per turn, and reduced all action energy costs, which gave the agents more time to develop tool based strategies before dying of starvation. 

The second revision substantially increased eating rewards, raising cooked meat from plus 7 to plus 15 energy, fish from plus 4 to plus 10, and berries from plus 3 to plus 6, making the gather/craft/cook/eat chain a high value strategy that would justify the additional steps. A bug was also fixed in this iteration in which dangers with a tool\_required value of zero could be mitigated for free, bypassing the intended risk structure. 

This iterative shaping was justified by the dissertation goal itself, where communication quality cannot be studied meaningfully if the task objective admits cheap degenerate policies that bypass semantic coordination. Following these revisions, I produced four baseline policies based on my personal input with the following results:

\
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

\
Note that these runs were conducted prior to the implementation of the agentic architecture, serving to validate the game mechanics. As we can see in *#underline[@policy-baseline-results]*, the survival rate gap between the optimal and blind policies was approximately 42.8%, confirming that sender information provides a substantial and measurable survival advantage. Furthermore, the near identical survival rates of the random and blind (always rest) policies, at 20.8% and 19.3% respectively, further indicate that the reward structure does not favour uninformed action selection over random chance. Finally, the relatively small gap between the greedy and optimal policies, at around 6.5%, reflects a design choice to keep the task tractable without requiring highly complex multi step strategies.

Finally, the penalty structure was designed to deter specific failure modes. Consuming raw or parasitic food without cooking imposed a -30 penalty, failed hunts with injury imposed a -20 penalty and wasted actions, such as attempting to hunt without a tool, imposed a -10 penalty. These penalties, combined with the stochastic health reduction from dangerous entities, were intended to discourage the simple gather and eat local optimum by making unsafe context independent actions costly when the wrong entity was present.

\
== Data Creation <group1>
\
Regarding the process of data generation and processing throughout the project, I intitally designed a system that would allow me adapt on future testing or structural changes. By doing so, I was able to use the same data for producing the behavioural baseline on which the hand coded policies are evaluated in the prototype, and producing the supervised training batches consumed by the emergent communication models. Hence, both pipelines draw from the same world mechanics and entity catalogue, which ensures that baseline performance figures and model performance figures remain directly comparable.

Given the nature of the game, training data is generated synthetically by simulating episodes of the survival game under a random valid policy, producing a diverse distribution of entity encounters across turns.

Each encounter, such as Goat (0,1,1,3,1,0), is created through a two stage selection process. First, an entity type is sampled according to a fixed set of spawn weights, where resources appear most frequently at 35%, followed by animals at 25%, dangers at 20%, and crafting opportunities and events at 10% each. These weights were chosen to reflect a plausible survival environment in which useful resources are the most common encounter, dangerous situations are frequent but not overwhelming, and high value events are rare. Second, the pool of entities of the selected type is filtered by weather compatibility before a target entity is drawn uniformly at random from the filtered pool.

Then, to generate episodes and turns it follows the same episodic logic as the game itself. I simulate each episode proceeds turn by turn, where I generate an encounter for each turn, simulate the games logic (energy drain, weather changes, valid actions...), and then collect the set of turns and create an episode. Note that at each turn, a sample is extracted and stored for later use by the training pipeline. This preserves the causal structure of the game and ensures that each sample reflects a plausible game context rather than an independently drawn random state.

With this system, my intital data set was approximately 2,200 episodes, generating a maximum of $2,200 times 20 = 44,000 "samples"$, although later on this proved to be insufficient and problematic as the  produced a training set had significant overlap problem, where 12.4% of validation samples were found to be exact duplicates of training samples. From a design perspective, this happened because the number of unique entity/state combinations is finite and relatively small, so independently generated episodes converge on repeated configurations by chance. Such duplication is a form of data leakage in which validation accuracy overestimates generalisation performance. To solve this, I increased the episode generation to 10,000 and introduced a fingerprint based de-duplication, where each sample is hashed before split assignment, and any fingerprint already present in a prior split is excluded. At this scale, the generator produced approximately 164,000 samples in nine seconds, and the final split achieved zero overlap between all three partitions.

The split policy adopted following this revision was 80 percent training, 10 percent validation, and 10 percent test, applied at episode level rather than at turn level. This was done to prevent temporal leakage from correlated turns within the same trajectory appearing in different splits. The three-way structure allocates the training partition for gradient updates, the validation partition for hyperparameter monitoring and run comparison, and the test partition to unbiased final performance reporting as seen in *#underline[@data-generation-results]*. The final split produces 131,361 training samples, 13,517 validation samples, and 13,821 test samples on average.

\
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
== Architecture planned for the models <group1>
\
Moving on to the projects architecture, I designed it leveraging the standard EGG sender/receiver core in which two neural networks communicate through a discrete symbolic channel. This framing allowed me to create a natural information asymmetry, where the sender has perceptual access to the entity but not to the survival state, and the receiver has access to the survival state but not to the entity. The sender was designed to map encounter representation into a communication latent state, while the receiver maps said message plus some game context into an action preference,  extended with a reconstruction head to provide a supervised auxiliary signal alongside the reward driven communication objective. Neither agent can act optimally in isolation, which makes communication strictly necessary for survival and ensures that any learned protocol carries genuine task relevant information.

From the beginning, the architecture was designed to support both Reinforce and Gumbel-Softmax training modes, sharing core components and configurable wrappers. I did this so that I could experiment in the future with both modes, as if communication only appears under one optimisation regime, conclusions become fragile. Unfortunately, despite this dual design the present project used only Gumbel-Softmax because it offered smoother optimisation under the project constraints and clearer control of exploration through temperature scheduling.

Gumbel-Softmax replaces discrete symbol sampling with a differentiable continuous relaxation @jang2017categorical, allowing gradients to flow directly through the message generation process and into the sender's weights. This produces a more stable and informative training signal than the high variance policy gradient estimates that REINFORCE relies on @williams1992simple.

Even though the agents have similar structures, there are some small differences that are key to the success of the design. 

The sender takes a one-hot encoded entity vector as input, processes it through a linear projection layer to produce a fixed sized hidden vector, and uses this hidden representation to initialise the message generating RNN (Recurrent Neural Network). Note that the projection is necessary because the input, the one-hot encoding across six entity dimensions, is a sparse high dimensional representation that benefits from a compact learned projection before sequential processing. Then, the RNN generates the message symbol by symbol, emitting a sequence of discrete symbols up to the configured maximum message length (2 as default), with each symbol drawn from a vocabulary of configurable size (50 as default). By doing so, it allows exponentially more distinct messages than a single symbol alone, and creates structural pressure for compositional organisation hence the sender must decide what to encode in each position independently. 

Similarly, the receiver mirrors this structure and takes the incoming symbol sequence, processes it through a corresponding RNN to produce a message representation, and combines this with the current survival state to produce two outputs. First some action logits (preferences) over the eleven action space, and second some entity reconstruction logits (entity predictions) over the 40 entity classes. Later on the project, I added the reconstruction head to the receiver as an auxiliary supervised objective to accelerate the development of entity discriminative communication. Without reconstruction, the only training signal for the sender was the reward obtained by the receiver's action, which is a weak and delayed supervision signal particularly early in training when the action policy is still random. Although this was later proven to contradict what we were researching for, explained in the Evaluation chapter.

As for the loss, at turn level, the simulator computes action outcomes, unaddressed encounter penalties, and survival bonuses. In abstract form:

$ R_t = r_"action" + r_"unaddressed" + r_"alive" + r_"energy" + r_"health" $

This expression is intentionally additive to keep interpretation transparent during analysis. The final benchmark simulator also includes terminal survival completion terms, making episode return:

$ G = sum_(t = 1)^T R_t + r_"terminal" $

with terminal additions for completion and end state efficiency when the episode is survived.

For differentiable training mode, the design required a smooth surrogate objective over the discrete action set. Therefore, the Gumbel-Softmax pathway uses expected reward over action probabilities:

$ E[R | s, m] = sum_(a in A) p_theta(a | s, m) space dot space hat(R)(a, s, e) $

where $hat(R)$ is a deterministic expected reward approximation and $p_theta$ is the receiver action distribution conditioned on state $s$ and message $m$.

The design intention at the start was to optimise task return primarily and introduce communication regularisation only if required for protocol quality. In the final implemented formulation, the GS objective combines task and auxiliary terms:

$ L = -lambda_r space dot space E[R] space + space lambda_c space dot space L_"recon" - lambda_h space dot space H(p_theta) $

where $L_"recon"$ is entity reconstruction loss and $H$ is action entropy. The reconstruction cross entropy loss provides a direct, dense gradient that encourages the sender to emit messages from which the entity identity can be recovered, establishing the communication channel before the action policy has matured. Further discussions on the loss and the historical progression of these terms and their scheduling is discussed in the Implementation chapter.

A notable feature that I designed also was high configurability through runtime arguments rather than hard coded values. Motivated by the principle of encapsulation, the model architecture is decoupled from any specific experimental configuration, allowing different configurations to be tested by changing arguments at launch rather than by modifying the model code. This increased the efficiency later on while testing, allowing me to execute automated runs with changing parameters, improving reproducibility of run settings, and controlled comparison across design alternatives without rewriting model code.

Finally, further down the line I adapted the code and made a major design change where I separated message temperature from action temperature. Message temperature controls how sharp or diffuse the Gumbel-Softmax distributions are during symbol generation, determining how close to discrete the message channel is, while action temperature independently controls the sharpness of the receiver's action distribution. By doing this it allowed the communication channel to anneal towards discrete symbols at a different rate from the action policy's exploration schedule, which provides finer control over the dynamics of communication emergence relative to action learning. 

\
== Brief design note on dropped image branch <group1>
\
As mentioned earlier, alongside the vector based design, an image based survival game was initially planned. In this design, the sender would observe a rendered 64×64 image representing a survival scenario, process it through a convolutional encoder, and transmit a symbolic message to the receiver. This approach aligned with the emergent communication literature that was done at the start of the project, where perceptual inputs can induce grounded representations and richer communication dynamics *@lazaridou2018emergence*.

Despite its conceptual relevance, the image branch was removed from the core implementation scope following a reassessment as the image branch would have introduced significant overhead in data generation or curation, representation validation, and computational cost. This would have reduced the depth of investigation into the central research question within the available project timeline.

#pagebreak()

= Implementation <group1>
\
In this chapter, I describe how the designed survival environment was implemented as an executable training system, how each module was integrated through EGG, and how the experimentation and testing workflow was instrumented. Building on the design decisions outlined in the previous chapter, the focus here is on the practical realisation of those components, including module structure, class and method responsibilities, and key implementation choices made during development. The chapter then concludes with an overview of the testing strategy used to evaluate the system through iterative experimental runs.

\
== Survival game implementation <group1>
\
Before building the full training stack, I developed the #raw("prototype.py") implementation as the first complete version of the game logic, carried out before the final agentic pipeline was assembled, mainly from the end of semester one into the early implementation weeks because I needed a stable semantic reference that could later be reused by both data generation and reward computation. As I have done in previous projects, I organised the file into explicit sections so each part of the game could be reasoned about independently while still composing into one execution flow, which made it easier to manage complexity and maintain clarity as the code base grew through development and supervisor meetings.

The first section defines the vector schema and global mappings, where dimensions, probabilities, names, and value maps are fixed at the top level to avoid hidden assumptions in later functions. For example, the definition of danger injury probabilities is defined as #raw("DANGER_INJURY_PROB = {0: 0.00, 1: 0.10, 2: 0.25, 3: 0.45, 4: 0.70}"), later used throughout different functions and files. The second section defines the full entity catalogue, where I made a data class where the 40 entities are defined and manually curated to keep encounters plausible, as this project required a controlled world where interpretation and analysis remain feasible. Here you can also find the different spawn weights for each entity, which were tuned through trial and error to achieve a balanced distribution that supports learning without overwhelming the agent with too many rare or too many common entities.

Moreover, the next section introduces the Inventory and GameState classes defined earlier in operational terms, where Inventory provides standard add, remove, and query operations for tools, materials, and consumables. The GameState class on the other hand provides the evolving state that all decision and reward logic depends on. I also added tracking related fields in this stage because I already expected that behaviour analysis would be central later, and this early decision avoided retrofitting instrumentation into unstable code paths. Following this, I defined the action space and validity logic, where the game determines which actions are legal for a given entity and state, and then I implement action resolution where each chosen action produces both a concrete state transition and a reward value. It is crucial to note that this is the mechanical core of the environment, because health, energy, inventory, and weather effects all depend on these functions, and also the strategic core, because reward shaping is encoded here and therefore determines what kind of behaviour can emerge during optimisation.

Then, the next section moves to encounter generation, where it first constructs a weather compatible pool of entities, then applies the defined weight sampling with sanity checks (in case no entity exists under the weather condition), and finally samples a concrete entity from the selected type. This method was intentional because a direct uninformed random entity draw would have produced unrealistic distributions and weaker task pressure for communication. Although this part looks simple at runtime, it is important for distribution control because encounter frequency directly affects both baseline performance and model learning dynamics. After this, the episode runner section executes the full turn pipeline in sequence, where a new GameState is created, energy drain is applied, weather is updated, an encounter is generated, valid actions are computed, a policy callback chooses an action, the action is resolved, and terminal conditions are checked before returning the updated state. Something that stands out here is the _policy\_fn_ hook, a strategy interface which allowed me to evaluate handcrafted baselines and later plug in learned agents with the same environment implementation, enhancing abstraction.

In addition, I implemented four baseline policies in the prototype stage as defined in the Design chapter, namely random, greedy, optimal, and blind resting. While implementing them, I ensured that each policy reflected a distinct approach to decision making. For example, the greedy policy follows explicit priority rules such as "if danger then mitigate, else endure it", or "if energy drops lower than 25 and I have food then eat". All of the rules are defined within the comments of the function. Similarly, the optimal policy follows a more informed strategy that invests earlier in tools, hunts more aggressively once equipped, and manages food more proactively, while the blind policy isolates performance without meaningful perception. This baseline phase was developed before the final implementation chapter experiments, mainly across weeks one to three in semester two, and it provided the first evidence that the environment supports non trivial strategy differences.

Finally, the closing sections of #raw("prototype.py") handle statistics, reporting, and baseline execution entry points, so this file became the canonical semantic source used later by #raw("data.py") and #raw("losses.py"), ensuring that world mechanics seen during data generation remain aligned with the reward simulation used during optimisation.

\
== AI Agent Models and EGG Integration <group1>
\
Following the survival game implementation, I moved on according to plan to the production of the core agentic model. The main file that controls the flow is train.py, where I orchestrate the full pipeline rather than implementing game mechanics directly, a separation done deliberately because I wanted experiment control and reproducibility to be handled in one place. The file has two main components, where #raw("get_params") defines the configurable interface and #raw("main") executes the training workflow, mirroring the Design chapter goal of high configurability through arguments rather than constant code rewrites. Something to note here is that #raw("get_params") integrates project specific arguments with standard EGG arguments through #raw("core.init"), which means one command line entry point controls model mode, data size, optimisation settings, logging options, and analysis toggles. This was crucial during the run to run iteration phase from approximately week five onward because I could test hypotheses about temperature, entropy, reward scaling, and message settings without rewriting infrastructure.

Then main follows a strict build order, where data is generated through data.py, the game object is assembled through games.py, optimisation is built through EGG core utilities, callbacks are attached for progress and analysis, temperature updating is configured for communication control, and trainer execution runs the actual learning process. Although trainer.train appears as a single call, all previously defined components are instantiated before that point, so this file is effectively the integration layer that binds every implementation file into one executable experimental system. Moreover, I added tracking flags for TopSim, PosDis, BosDis, and entropy as part of this integration stage, because by that point I had already begun planning the Evaluation chapter and needed instrumentation to be available during long runs rather than as an afterthought. Thus, train.py is not only a launcher, it is also the consistency point where output directories, naming conventions, logging granularity, and analysis artefacts are coordinated.

After the training instances were defined, the role of data.py is to convert the prototype world into model ready supervised samples, making this file the bridge between environment simulation and training batches. As described in the Design, the game mechanics exist in prototype.py, yet the agents consume tensors, so this module encapsulates the transformations required to preserve semantics while adapting the format.

The file is organised into four parts, namely encoding helpers, episode simulation for data generation, a PyTorch dataset class, and a dataloader factory. The three encoding helpers follow a clear separation of concerns, where one converts a six dimension entity vector into a 30 dimension one hot sender input ($6 times 5 = 30$), another converts GameState into a normalised 16 dimension receiver state tensor, and a third packs label information used later by the loss helpers. This explicit encoding stage is important because it keeps feature semantics stable across experiments and prevents hidden coupling between environment and architecture code. Following this, the data generation simulator produces episodes and then decomposes them into turn level samples, where each sample is split into sender input, receiver input, and labels. In this process, the valid action mask is added to the receiver input with 1 for valid actions and 0 for invalid actions, which later supports masked decision logic in the receiver and reward interpretation in losses.

The final factory stage handles the dataset scale, split discipline, and overlap control. Episodes are generated with controlled seeds for reproducibility, shuffled at episode level rather than turn level to preserve trajectory consistency, split into train/validation/test partitions, and flattened into samples. On the other hand, a fingerprint based de-duplication pass removes overlap between partitions, a mechanism that became essential after overlap was detected. By hashing sender and receiver inputs for membership checks, it provided an efficient and reliable way to eliminate leakage across splits. After de-duplication, each partition is wrapped by SurvivalGameDataset and loaded through torch.utils.data dataloaders, so the output of data.py is a clean and reproducible tensor pipeline aligned with the game semantics defined in prototype.py.

With the data pipeline established, games.py is the composition layer that builds the actual EGG game object from sender/receiver architectures, wrappers, and loss. Although the file is intentionally compact, its responsibility is central, as it decides how communication, decision, and optimisation components are connected under either GS or Reinforce mode.

In this file, I instantiate Sender and Receiver from archs.py, instantiate SurvivalLoss from losses.py with relevant configuration terms such as action entropy coefficient, action temperature, and reward normalisation, then wrap the agents with the EGG recurrent wrappers according to mode. For GS mode this means RnnSenderGS and RnnReceiverGS are used and combined through SenderReceiverRnnGS, while Reinforce mode uses the corresponding Reinforce wrappers from the core folder. Moreover, this design keeps the internal model definitions independent from training regime specifics because wrappers handle sequence level communication behaviour and optimisation compatible interfaces, while my architecture code focuses on representational learning for sender and receiver cores. Thus games.py is small by lines of code, yet it is the exact place where the full talking game is assembled end to end.

At this point, archs.py defines the project specific sender and receiver neural cores described in the Design chapter, developed as the controlled implementation of the intended information asymmetry, where sender sees the entity and receiver sees state plus message. I implemented these as two classes, Sender and Receiver, with compatible hidden dimensionality so they integrate smoothly with the recurrent wrappers.

The sender is an MLP that maps the 30 dimension one hot entity input into a hidden representation used to initialise message generation. In practical terms, the forward propagation applies linear layers with ReLU as an activation function to produce the initial recurrent state, and the message sequence itself is generated by the wrapper in gs_wrappers.py, where the recurrent cell is unrolled for max_len steps and vocabulary logits are produced at each step, followed by a sampling of differentiable symbols through Gumbel Softmax and an EOS token being appended at the end.

The receiver follows a similar linear ReLU core but includes both action prediction and reconstruction pathways. The input to the receiver core is the concatenation of message representation and state features, while the valid mask is applied at the logits stage rather than as hidden features, following the design choice that semantic interpretation should be driven by message and state rather than by directly feeding legality structure into feature extraction. In this setup, fc1 acts as the hidden feature extractor and fc2 maps hidden features to action logits over eleven actions. Invalid action logits are set to a very large negative value (resembling $-infinity$ mentioned in the Design chapter) so that masked actions receive effectively zero probability after Softmax.

Furthermore, after some testing I added a reconstruction head using nn.Sequential with intermediate dimensionality and final output over N_ENTITIES, as this auxiliary signal was intended to strengthen communication learning when action rewards alone were too weak early in training. The gradient route for this path is important, since cross entropy loss updates the reconstruction head, then flows through the receiver hidden representation and recurrent message decoder, through the message channel, and back into sender parameters. Thus the reconstruction objective contributes not only to receiver discrimination but also to shaping sender messages. In addition, the GS receiver wrapper decodes the incoming sequence step by step and provides outputs across time, where the game level wrapper aggregates these outputs with EOS weighted logic, which keeps training differentiable while accounting for variable effective message length.

The next layer is losses.py, which defines the training objective and therefore acts as the optimisation interface between environment semantics and model updates. For clarity I separated it into label unpacking helpers, reward computation helpers, and the main loss class.

The unpacking helpers where made to support the loss creation, where I abstracted first over recovering structured components from packed labels, then resolving entity identity and target index, and finally reconstructing the GameState from encoded vectors so reward simulation can be run in the same semantic space as the environment. This means the model does not infer the original entity at this stage, instead it recovers ground truth metadata from labels and falls back to vector matching only when indices are invalid, which protects training from inconsistent labels while keeping behaviour deterministic.

Similarly, the reward helpers compute both single action outcomes and expected rewards across actions, a crucial part where game logic is translated into differentiable supervision.

In `losses.py`, the function builds a deterministic reward vector $R in RR^{11}$, where each invalid action starts at $-30$ and valid actions are replaced by the formulas below.
\ \
Let $d$ be danger level, $E$ energy, $H$ health, and

$ F = max(1, "max_turns" - "turn") space dot space 8 $

be the future value term used in death proximity shaping.
\ \ \ \
*Hunt:* When the target is an animal, success probability is conditioned on tool availability and danger level,

$ 
s = 
cases(
  0.85 quad quad "if has tool and" d ≤ 1,
  0.65 quad quad "if has tool and" d > 1,
  0.50 quad quad "if no tool needed",
  0.15 quad quad "otherwise",
) quad quad dot quad H/100 quad dot quad min(E, 50)/50
$

\ \
and the expected reward is computed from separate success and failure terms,

$ R_"succ" = (15 + 10 dot d) - (0.10 dot d) dot ((5 + 5 dot d)/2) $

$ R_"fail" = -5 - (0.25 dot d) dot 15 $

$ R_"hunt" = s dot R_"succ" + (1 - s) dot R_"fail" $

\ \
If the entity is not an animal, $R_"hunt" = -10$.

\ \
*Gather:* For a valid target, that is a resource, craft opportunity, or event, reward is modulated by inventory quantity $q$ and injury probability $p_d$,

$ R_"gather" = 5 + 2 dot q - [d >= 2] dot (p_d dot ((5 + 5 dot d)/2)) $

\ \
and an invalid target yields $R_"gather" = -10$.

\ \
*Flee:* The reward reflects whether flight is contextually appropriate,

$
R_"flee" =
cases(
  2   quad quad "danger and" d ≥ 3,
  1   quad quad "animal and" d ≥ 3,
  -2  quad  " otherwise",
)
$

\ \
*Rest:* Rest carries a fixed small penalty to discourage passive behaviour,

$ R_"rest" = -1 $

\ \
*Mitigate:* The reward here captures the value of using the correct tool against an active danger,

$
R_"mitigate" =
cases(
  -10 quad quad quad quad quad quad quad quad " if not a danger",
  -0.3 dot (h_"loss" + e_"loss") quad quad "if danger but tool missing",
  20 + 5d + b quad quad quad quad quad " if danger and tool available",
)
$

\ \
where $h_"loss"$ and $e_"loss"$ are the unmitigated health and energy damage at danger level $d$ scaled by the endure factor, and $b$ captures a situational bonus,

$
b =
cases(
  5 quad quad "fire vs cold",
  0 quad quad "otherwise",
)
$

\ \
*Endure:* Taking damage without the appropriate tool is penalised proportionally to the harm sustained,

$
R_"endure" =
cases(
  -10 quad quad quad quad quad quad quad quad " if not a danger",
  -0.3 dot (h_"end" + e_"end") quad quad "if danger",
)
$

\ \
*Eat:* The eating reward is sensitive to food type and metabolic state,

$
R_"eat" =
cases(
  -5 & "  if no food",
  g + 5 dot ["cooked"] - 3.6 dot ["raw"] + 10 dot [E < 30] & "  otherwise",
)
$

\ \
where $g in {15, 10, 6, 5, 5, 3}$ depending on food type.

\ \
*Craft:* A uniform structure applies across all craft actions (spear, fire, shelter, and rod),

$
R_"craft" =
cases(
  -5 quad quad &  " if already owned",
  -10 quad quad & " if insufficient materials",
  50 quad quad & " if success",
)
$

After each base action reward, I check for starvation, if it eats starving it gives a bonus or more negative points if unaddressed, then also checks the health, with the same idea where risky actions at low health are penalised and fleeing gets a bonus.
\ \
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

Furthermore, in the main GS loss path for each batch sample I reconstruct the exact survival context, compute rewards for all actions, and store them in reward_matrix, while also preparing reconstruction targets and action behaviour diagnostics, so after the loop each row represents the full action value landscape for one concrete state. If reward normalisation is enabled, rewards are standardised per sample across valid actions only, while invalid penalties remain fixed at $-30$ as this prevents large magnitude actions such as successful crafting from dominating gradients and suppressing subtler but relevant differences.

Following this, logits are converted into soft action probabilities using action temperature, and the expected reward is computed as the probability weighted sum over actions. The game loss is defined as the negative expected reward scaled by reward_scale, an addition added throughout development to compensate the lack of motivation for some rewards. Moreover, I added entropy regularisation to discourage premature collapse to a single action and reconstruction cross entropy is added to encourage message content that preserves entity identity information. Thus, the final objective learns both decision quality and communication informativeness within one differentiable framework:

#raw("           loss = game_loss + self.recon_weight * recon_loss + entropy_loss")

In the evaluation chapter I discuss why reconstruction weight was eventually reduced to zero, since this changed how the protocol balanced strict identity coding against synonym like message behaviour.

Finally, once these files had been crafted, my last add on was callbacks.py meant to monitor behaviour and language evolution in ways that the default training loss output cannot show directly, and while EGG provides generic callbacks in core, this project required domain specific diagnostics for survival and emergent communication analysis. The file therefore defines two classes, SurvivalGameEvaluator and MessageAnalyzer, and both are analysis components rather than optimisation components.

The first class runs full simulated episodes with the current model extracted from the game object, and this process acts as policy evaluation rather than learning as no gradients are applied and no weights are updated. In practical terms, the evaluator plays fresh episodes using the current sender receiver policy and reports metrics such as survival rate, mean reward, valid action rate, message length, and action distribution. Following this approach, the model is tested in sequential gameplay conditions that reflect long horizon state transitions rather than isolated turn samples.

The second class analyses emergent language from validation interaction logs, and unlike the evaluator it does not run new episodes. Instead, it reads the messages produced in the current validation stage, groups message strings by entity type and counts frequencies per type, computes uniqueness, tracks global diversity, and then performs a finer pass by specific entity vector before writing snapshots and printing summaries. Overall, callbacks.py is less central to forward execution than architecture or loss code, yet it is essential for interpretation, because it provides the behavioural and protocol evidence used later in evaluation and critical appraisal.

\
== Testing and Training <group1>
\
Given the experimental nature of this project, traditional unit or integration testing was not the primary validation mechanism. Instead, the system was evaluated through a structured empirical testing strategy based on continuous experimental runs and controlled comparisons across configurations, where performance and behaviour were assessed through observed metrics rather than predefined test cases. 

Testing was conducted in three layers, first, the underlying game mechanics were validated using prototype baseline policies to confirm a clear performance gradient and justify the communication setting. Second, short smoke runs (one to five epochs) were used during implementation to verify tensor dimensions, masking logic, callbacks, and output generation. And third, full training runs were executed and compared through logged metrics and message snapshots. This approach emphasised iterative refinement, with each run informing subsequent design decisions.

Experiments were executed across both local and lab environments, with early runs conducted on a personal CPU setup and later runs migrated to GPU enabled lab machines to improve computational efficiency and enable repeated trials. In total, the project comprised over 38 individual runs, including both exploratory and controlled configurations, with later stages incorporating multiple repetitions per setting to support more reliable comparison. This progression enabled faster iteration, more robust evaluation, and increased confidence in the consistency of observed performance trends.

#pagebreak()

= Experimental Results and Analysis <group1>
\
In this chapter, the final experimental outcomes are reported and interpreted in relation to the dissertation's core research question. The analysis is organised in two parts, first, the iterative runs that exposed implementation bottlenecks and informed objective redesign, and second, the controlled final experiments comparing reconstruction constrained and no reconstruction settings. Across both parts, results are interpreted using complementary evidence from task performance, optimisation traces, and communication structure diagnostics.

As mentioned in previous chapters, the work done followed an iterative approach where experimentation went alongside implementation. Thus, the experiments relied on the baselines defined in the Design section to formalise the experimental method and what "good" looks like.

\
== Initial Iterative Experimentation <group1>
\
Throughout the experimentation in early runs, as explained in the Design chapter I levaraged the curated baseline policies shown in *#underline[@policy-baseline-results]* as a guide to interpret my results. While experimenting, I noticed that starvation dominated deaths as seen in *#underline[@death-causes-results]*, and that tool crafting was strategically important because it enabled high value cooking chains.

\
#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Policy*], [*Starvation*], [*Injury*], [*Total*]),

    [RANDOM], [61.15%], [38.85%], [100%],
    [GREEDY], [85.55%], [14.45%], [100%],
    [OPTIMAL], [77.20%], [22.80%], [100%],
    [BLIND (no sender)], [0.00%], [100.00%], [100%],
  ),
  caption: [Distribution of death causes across baseline policies.],
) <death-causes-results>

\
Hence, I started to re-engineer the game's logic via observation of the following runs. The first learning runs exposed structural reasons for failure rather than simple parameter issues. Initially, I used the REINFORCE mode to examine the agent's behaviour, but it performed poorly surviving 0% of the times, motivating a move to the differentiable Gumbel Softmax approach, advised by my supervisor. Despite this move, the first GS evaluations showed again 0% survival rate and highly skewed action distributions, with behaviour dominated by fleeing, gathering and enduring, and messages collapsing into a small set of repeated patterns that did not encode the necessary distinctions: $m(e) = [41, 41, 41, 41, 41, 41, 0] quad forall e$

After investigating, the diagnosis pointed to several root causes. First, constant reward bonuses applied equally to all actions, meaning it did not contribute useful gradient signals, while also having the stochastic outcome resolution injecting noise into the expected reward calculation. To solve these, I replaced the stochastic simulation with deterministic expected rewards and removed constant bonuses from the GS reward computation. Furthermore, the implementation was extremely slow due to repeated deep copies, a default in EGG, and even worse the receiver did not need the message if it could rely on validity masks and safe default actions to choose what to do, as it was being motivated purely by points awarded (gather and flee had high rewards). Thus, I decided to use manual save restore copies instead of deep copies to reduce the time taken to run (a decrease of $times 5$ on time), and increased the reward scaling so the reward term had meaningful influence. After these fixes, message collapse was effectively resolved at the coarse level, with distinct message patterns emerging per entity type and reconstruction accuracy quickly reaching near perfect values, showing that the communication channel could be trained when the gradient path was clean and the receiver was forced to attend to the message.

Following this, the next set of runs focused on establishing a consistent training configuration and observing how performance evolved with more constrained message length and controlled temperatures. The main configuration used for GS mode was with vocabulary size 50, maximum message length 3, fixed temperature settings (1.0) and a weighted combination of expected reward and reconstruction cross entropy. In these runs, survival rate during evaluation tended to sit in the mid 30s at earlier checkpoints, although action distributions revealed a strong behavioural bias where agents overwhelmingly favoured gathering, eating, and fleeing, while hunting behaviour was effectively absent ($≈0%$) across both training and evaluation.

\
#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([*Entity Type*], [*Count*], [*Top Message(s)*]),

    [Animal],   [814],  [`[11 0 0 0]`],
    [Resource], [1105], [`[11 11 6 0]`, `[11 11 11 0]`],
    [Danger],   [695],  [`[0 0 0 0]`],
    [CraftOpp], [351],  [`[0 0 0 0]`],
    [Event],    [323],  [`[0 0 0 0]`],
  ),
  caption: [Message protocol at Epoch 20 showing persistent collapse across non-resource entities and limited refinement for resources.],
) <near-total-collapse>

\
As seen above, the messages at these checkpoints suggested that the protocol was beginning to separate some categories, but important distinctions were still coarse and several entity types shared identical or near identical codes as seen in *#underline[@near-total-collapse]*, implying that the receiver could reconstruct broad classes without necessarily supporting fine grained action selection. This also meant that my previously solved problem had come back partially (message collapse). At this point, I was also doing multiple runs with different amount of epochs to test if longer periods would improve the results, but it proved that longer training alone was not the decisive factor in escaping the behavioural attractor.

\
#figure(
  image("Images/run3_metrics.png"),
  caption: [Training metrics for run 3],
) <metrics-run-3>

\
After further improvements, the experiments were reframed more explicitly around the combined objective and the role each term played. It is at this point were I changed the loss, treating it as a weighted sum of a negative expected rewards and a reconstruction term, as explained in the Implementation chapter, keeping the full objective differentiable. In the improved runs, the reconstruction loss collapsed towards zero, effectively reaching perfect reconstruction, which meant the overall loss became dominated by the negative reward component as seen in *#underline[@metrics-run-3]* and *#underline[@training-curves-run-3]*.

\
#figure(
  image("Images/training_curves.png"),
  caption: [Training metrics for run 3],
) <training-curves-run-3>
\

This was encouraging in that it demonstrated stable decoding and therefore a functional communication channel, but it also underlined an important limitation. Even with strong communication, the policy could remain stuck in suboptimal action patterns, and communication could continue evolving late into training, suggesting that protocol convergence and behavioural convergence were not necessarily aligned and that the models were stuck in a local optimum.

This set up naturally led into the analysis of this local optimum that became the central explanation for why task performance lagged behind communication quality. In particular, the gather eat loop emerged as a reward maximising equilibrium because gathering is always legal and safe, eating becomes consistently available once food is accumulated, and the strategy is therefore self reinforcing across turns. By contrast, context dependent actions such as hunting or fleeing provide sparse learning signals because they are only relevant under specific encounters, and once the Softmax distribution concentrates on the safe actions, the gradients for rare actions become too small to recover. 

Moreover, this issue is compounded by the fact that entropy regularisation is too blunt, as it encourages uniform exploration over all actions rather than the appropriate action in a given context. Thus, the architecture couples decoding and action selection within the same network, so once the action head collapses, there is little pressure to utilise the decoded information for decision making. The deeper issues were therefore understood as arising from a combination of factors within the learning setup. In particular, the reward structure, together with weak or delayed credit assignment for failing to respond to hazards, limited the agent's ability to associate actions with long term consequences. This was further compounded by the fact that the game was designed as a single step progression, restricting multi turn planning, as well as the absence of an advantage style baseline, which would otherwise emphasise relative action value rather than raw reward.

With the local optimum identified, the next step was to reconsider the direction of optimisation. Initially, I considered making further changes to the game itself, such as simplifying its structure or refining its internal logic. However, this led to questioning whether the primary metric being optimised, namely average survival rate, was fully appropriate in this context. From a broader perspective, the evaluation had focused heavily on survival outcomes while placing less emphasis on the communication protocol itself. 

Upon reflection, it became clear that survival rate alone is not necessarily a fully informative objective, as it is inherently constrained by the stochastic nature of the environment. Due to probabilistic encounter distributions, there are scenarios in which survival is limited regardless of agent behaviour, for example when episodes contain a high proportion of predators and insufficient food resources. As established earlier, starvation is the dominant cause of failure, and therefore even optimal decision making cannot always overcome environmental constraints. This suggests that there is an upper bound on achievable survival performance imposed by the game dynamics rather than the learning system. Thus, I shifted the focus of evaluation towards communication centred metrics, giving greater importance to measures such as topographic similarity (TopSim), PosDis, BosDis, and message entropy, as defined in the Context Survey. This change provided a more appropriate basis for analysing emergent communication and ultimately led to clearer and more meaningful results in relation to the research objective.

Consequently, the next step was to investigate the structure of the emergent protocol more directly. My first approach was to use Hungarian assignment (the Hungarian algorithm @wiki:Hungarian_algorithm), an optimal one-to-one matching method that pairs entities and messages by minimising total assignment cost in a cost matrix. I applied it to the message to entity mappings from the longest training runs, producing a cost matrix and heatmap. The assignment converged well and was nearly diagonal, but revealed an asymmetry in what the model had learned. Frequently encountered entities such as Water, Flint, Firewood and Stone were matched with high counts and low cost, while abstract or conditional entities such as Frost, Cold Night, Bear and Cave were poorly matched with high cost and low counts. This confirmed that the protocol learned concrete entities better than situational or environmental ones, and that the frequency driven objective struggled with ambiguous states. However, the Hungarian approach did not yield further actionable insight beyond this observation, so it was set aside in favour of broader statistical measures.

\
== Final Experimental Runs <group1>
\
After completing the implementation, I shifted my focus fully to evaluating the capabilities of the current model. In the time available, I conducted five experimental conditions, progressing from a reconstruction constrained regime to increasingly expressive and less constrained setups in their execution order: _with_recon_, _no_recon_, and three no reconstruction tuning variants. The main question here is  whether removing the reconstruction objective changes how the sender structures its communication, and whether tuning action entropy and message length then sharpens that structure.

Note that all experiments were conducted with five independent seeds per condition, using the same seeds for the last 3 experimental runs. Also, metrics were computed at the run level and summarised across runs focusing on topographic similarity as the primary signal, and complemented by message reuse statistics and qualitative inspection of cluster structure.

The first experiment consisted of 5 runs using the default settings shown in *#underline[@setting-exp-1]*, where the agent had to not only select the best action to take, but also predict the exact entity that the sender encountered. This introduces a second training objective that rewards the sender for producing messages that uniquely identify the current entity, explained in the loss section in the Implementation Chapter.

\
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
\
My initial hypothesis was that reconstruction pressure might still allow semantically similar entities to share codes, as similar entities do not differ much on their definition. For example, the definitions of Fish and Salmon are [0,3,0,3,2,2] and [0,3,0,4,2,2], so the messages that identify these should have been consistent even if we forced the agents to produce a unique message per entity.

Empirically, this did not hold across all five runs, and most entities that are later grouped into clusters do not show any type of similar message tokens across runs as seen throughout *#underline[@fish-salmon-exp1]* to *#underline[@resources-env]* in the Appendix. The protocol itself is not fully random, as token fragments are reused locally but the full message remains entity specific in every case. The receiver needs to reconstruct the exact entity, so the sender has a strong gradient incentive to keep codes discriminative as broader semantic reuse would reduce reconstruction accuracy. Moreover, no two runs converge to the same literal token assignments, reflecting that the protocol is invariant up to permutation of symbols, and that only the functional structure of the mapping is meaningful rather than the specific tokens used, although this is supported by literature and explained in the Evaluation chapter.

\
#figure(
  image("Images/loss_train_test_RECON.png"),
  caption: [Training and test loss across epochs for the reconstruction setting.],
) <loss-recon-exp1>
\
This conclusion was further supported by inspecting the metrics generated from the data saved. As shown in the loss curve in *#underline[@loss-recon-exp1]*, training exhibits a rapid initial decrease, indicating that both the action selection and reconstruction objectives are quickly learned in the early stages. However, after this phase, the loss stabilises and gradually decreases at a slower rate, suggesting convergence towards a local optimum rather than continued significant improvement. The relatively wide standard deviation band throughout training indicates variability across runs, implying that while the model consistently learns a workable protocol, the exact solution found is sensitive to initialisation and stochasticity in training.

\
#figure(
  image("Images/topsim_RECON.png"),
  caption: [TopSim (test) across epochs for the reconstruction setting.],
) <topsim-recon-exp1>

\
As for the TopSim metric seen in *#underline[@topsim-recon-exp1]*, it remains relatively stable across epochs, fluctuating within a narrow range without a clear upward trend. This suggests that, despite successful optimisation of the training objective, the emergent communication protocol does not become increasingly aligned with the underlying semantic structure of the entities. While the mean TopSim value across runs is 0.261673, indicating a weak positive correlation between message similarity and semantic similarity, the relatively high standard deviation of 0.086217 and wide range (from 0.136036 to 0.346536) show that this alignment is inconsistent. Some runs achieve moderate structure, while others remain close to weak or near-random alignment. This reinforces that different runs converge to distinct but equally valid encoding schemes, rather than a single semantically grounded protocol.

\
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

\

After reviewing these results, I decided to take a step back and analyse the loss function in more depth. My initial hypothesis was that I might be forcing the model too much to reconstruct perfectly unique messages, even though exact reconstruction was not strictly necessary for the task. Thus, I decided to try and experiment under *#underline[@setting-exp-2]* settings, where the sender is only optimised to support action selection and no longer receives direct pressure to encode the exact entity identity, effectively dropping the #raw("recon_weight") to 0. This made the protocol free to trade strict discrimination for compact reuse if that improves downstream reward. It also motivates the use of TopSim as the primary metric of analysis, as reward remains effectively constant across runs due to the policy converging to a local optimum, and therefore does not provide meaningful discriminative signal.

\
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
\
This single change fundamentally altered the communication protocol completely as instead of producing strictly unique messages per entity, the model began to group semantically similar entities under shared codes. For instance, Fish and Salmon were consistently clustered together, as were Goat, Deer, and Rabbit. The main clusters that were found across the runs are shown in *#underline[@clusters-exp2]*.

\
#figure(
  table(
    columns: (auto, auto, 8cm),
    align: (left, left, left),
    table.header([*Cluster Type*], [*Core Entities*], [*Notes*]),

    [Resources],
    [Wood, Water, Herbs, Berries, Stone],
    [Often extended with Herb Garden, Supply Cache, River Crossing; in some runs also absorbs Squirrel and Lizard],

    [Tools / Utilities],
    [Firewood, Weapon Cache],
    [Often includes Dry Campsite and Riverbank; Flint and Cave are less consistent and can shift to broader mixed clusters],

    [Environmental Hazards],
    [Storm, Frost, Poison Plant, Infection],
    [Commonly extended with Cliff, Quicksand, and Blizzard],

    [Animals],
    [Goat, Deer, Rabbit],
    [Consistently grouped; Lion and Bear are usually grouped together as a separate subgroup],

    [Aquatic / Food],
    [Fish, Salmon],
    [Consistently grouped, sometimes embedded within larger mixed clusters],

    [Secondary / Mixed Attachments],
    [Muddy Water, Mushrooms],
    [Frequently paired; attachments such as Cold Night, Fever, and Thorns are less consistent across runs],
  ),
  caption: [Emergent semantic clusters across runs without reconstruction, showing core groupings and common extensions.],
) <clusters-exp2>
\
    
Unlike the previous experiment, these clusters emerge consistently across runs, although their exact composition may slightly vary. The key observation is that messages are no longer uniquely tied to individual entities, but instead reused across semantically related groups. This indicates a clear shift from a one-to-one mapping towards a many-to-one mapping, where messages encode shared properties rather than exact identities. Importantly, these groupings are not purely semantic in a descriptive sense, but reflect action level equivalence where entities that afford similar optimal actions are compressed into the same message because the receiver does not benefit from distinguishing between them under the learned policy.

\
#figure(
  image("Images/loss_train_test_RECON2.png"),
  caption: [Training and test loss across epochs without reconstruction.],
) <loss-recon-exp2>

\
As shown in *#underline[@loss-recon-exp2]*, the loss is highly stable across runs (very low standard deviation), confirming that removing reconstruction simplifies the optimisation landscape. Compared to Experiment 1, this trajectory is smoother and less sensitive to run level stochasticity, which suggests that removing the reconstruction objective simplifies the optimisation landscape. At the same time, reconstruction accuracy dropped significantly, as expected, since the model is no longer incentivised to encode exact identities. 

Most importantly though is the TopSim score shown in *#underline[@topsim-recon-exp2]*, as it increases substantially (mean 0.413854 compared to the previous experiment), indicating a stronger correlation between message similarity and semantic similarity. However, the standard deviation remains relatively high (0.143490), with values ranging from 0.199462 to 0.544537, showing that while semantic organisation emerges more clearly on average, it is still not fully deterministic across seeds. It is worth noting that TopSim captures global alignment between meaning and message spaces, but may miss finer local structure, so these values are best interpreted alongside the observed cluster patterns.

\
#figure(
  image("Images/topsim_RECON2.png"),
  caption: [TopSim (test) across epochs without reconstruction.],
) <topsim-recon-exp2>

\
Together, these results show that reconstruction acts as a constraint that enforces uniqueness at the cost of semantic structure, whereas removing it enables the emergence of more meaningful and generalisable communication patterns. At the same time, this behaviour is closely tied to the underlying policy, the receiver converges to a stable gather/eat strategy, which reduces the need for fine grained distinctions and encourages the sender to group entities that lead to equivalent outcomes.

\
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

\
After establishing the no reconstruction regime as a baseline, my final stage of experimentation focused on systematically exploring different settings that would enhance the results. I decided to focus on two key parameters, action entropy and message capacity, as due to time constraints I could not experiment further. Rather than treating each experiment in isolation, the results are analysed jointly to understand how these parameters reshape the emergent communication protocol relative to the no_recon setting.

If we have a look and inspect the results, a consistent pattern across all three experiments (with parameters: *#underline[@setting-exp-3]* to *#underline[@setting-exp-5]*) is that the overall structure of the protocol remains remarkably stable. The same high level semantic clusters identified in the baseline persist, particularly the large resource clusters (e.g., Wood, Water, Herbs, Berries, Stone families) and environmental hazard groupings. This indicates that once reconstruction pressure is removed, the protocol converges to a representation driven primarily by action relevant equivalence classes, proving this structure is robust to moderate changes in optimisation dynamics.

The increase of action entropy (0.3 in Experiment 3 and 0.5 in Experiments 4 and 5) primarily affects *how* the protocol is explored rather than *what* it represents. As seen in *#underline[@loss-exp3]*, *#underline[@loss-exp4]*, and *#underline[@loss-exp5]*, increasing entropy leads to consistently lower final loss values (from approximately -0.66 in no_recon to around -0.73 and -0.88), with extremely low variance across runs. This suggests that entropy regularisation improves optimisation stability, likely by preventing early convergence to suboptimal deterministic policies. Furthermore, one observation that stood out was that the clusters seemed to be more consistent throughout these runs than in the previous one. For example, the Fish and Salmon grouping appeared uniquely 5/5 times throughout the runs, while in the parent test it only appeared 3/5 times. This might indicate that action entropy affects the models capability to define differences across entities better.

\
#figure(
  image("Images/loss_train_test_3.png"),
  caption: [Training and test loss across epochs for Experiment 3 (action entropy 0.3).],
) <loss-exp3>
\
#figure(
  image("Images/loss_train_test_4.png"),
  caption: [Training and test loss across epochs for Experiment 4 (action entropy 0.5, length 2).],
) <loss-exp4>
\
#figure(
  image("Images/loss_train_test_5.png"),
  caption: [Training and test loss across epochs for Experiment 5 (action entropy 0.5, length 3).],
) <loss-exp5>

\
From a representational perspective, TopSim provides a clearer view of how these changes impact semantic alignment. As shown in *#underline[@topsim-exp3]* to *#underline[@topsim-exp5]*, all three experiments achieve higher average TopSim than the no_recon baseline ($≈0.41$), with means of $0.40$ to $0.45$. This confirms that entropy not only stabilises training but also encourages more globally consistent mappings between input similarity and message similarity. However, an important observation is that while higher entropy (0.5) enables the *highest* TopSim values, it also introduces greater variance across runs, indicating sensitivity to initial conditions. In contrast, the moderate entropy setting (0.3) achieves slightly lower peak alignment but is more consistent.

\
#figure(
  image("Images/topsim_test_exp3.png"),
  caption: [TopSim (test) across epochs for Experiment 3 (action entropy 0.3).],
) <topsim-exp3>
\
#figure(
  image("Images/topsim_test_exp4.png"),
  caption: [TopSim (test) across epochs for Experiment 4 (action entropy 0.5, length 2).],
) <topsim-exp4>
\
#figure(
  image("Images/topsim_test_exp5.png"),
  caption: [TopSim (test) across epochs for Experiment 5 (action entropy 0.5, length 3).],
) <topsim-exp5>

\
The second parameter that I explored was increasing the message length from 2 to 3 (Experiment 5), which proved to not fundamentally alter the communication regime. The same clusters persist, and both loss and reward metrics remain nearly identical to the length 2 case. The primary effect is a reduction in TopSim variance, suggesting that additional channel capacity provides redundancy rather than new expressive power. In other words, the agents already have sufficient capacity to encode the relevant distinctions, and extra symbols mainly smooth the learning dynamics rather than improving peak performance. This is further supported by the emergence of repeated token patterns (e.g., 31 31 or 10 10 10), indicating that additional capacity is often used redundantly rather than compositionally.


#set table(
  stroke: none,
)

\
#figure(
  image("Images/mean_vs_expected_reward_test_exp2.png"),
  caption: [Mean vs expected reward across epochs for Experiment 1 (reconstruction).],
) <reward-exp1>
\
#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/mean_vs_expected_reward_test_exp1.png")],
    [#image("Images/mean_vs_expected_reward_test_exp3.png")],
  ),
  caption: [Mean vs expected reward across epochs for Experiments 2 (left) and 3 (right).],
) <reward-exp2-3>
\
#figure(
  table(
    columns: (auto, auto),
    align: (center, center),

    [#image("Images/mean_vs_expected_reward_test_exp4.png")],
    [#image("Images/mean_vs_expected_reward_test_exp5.png")],
  ),
  caption: [Mean vs expected reward across epochs for Experiments 4 (left) and 5 (right).],
) <reward-exp4-5>

#set table(
  stroke: luma(black),
)

\
A notable and somewhat unexpected observation across all experiments is the consistent gap between mean reward and expected reward, which varies significantly between settings. While mean reward remains stable ($≈1.30$ to $1.31$) across all runs, expected reward drops substantially as entropy increases, from $≈1.29$ in no_recon to $≈1.15$ in Experiment 3 and $≈0.88$ in Experiments 4 and 5. This suggests that the stochastic policies encouraged by higher entropy achieve strong realised performance but distribute probability mass over suboptimal actions. In other words, the agents learn policies that perform well when sampled, but are less sharply peaked around the optimal action, lowering the expected value under the full action distribution. This also indicates that different runs may converge to slightly different policy distributions despite achieving similar realised rewards.

This divergence highlights an important trade-off where increasing entropy improves exploration and stabilises representation learning, but at the cost of policy sharpness. This may warrant further investigation, particularly into whether annealing entropy or decoupling exploration from evaluation could recover higher expected reward without sacrificing the structural benefits observed in TopSim.

Overall, these experiments demonstrate that once reconstruction is removed, the emergent communication protocol becomes robust, semantically structured, and primarily governed by action equivalence. Entropy acts as a stabilising force that improves optimisation and representational consistency, while additional message capacity provides diminishing returns. Together, these results mark the convergence of the experimental process, showing that relatively simple modifications to the loss and exploration dynamics can reliably produce structured, compositional communication systems, even if full compositionality is not achieved, answering our core research question.

#pagebreak()

= Evaluation and Critical Appraisal <group1>
\
Throughout this chapter, I evaluate the completed project against the final revised objectives and critically compare the results with related emergent communication research studied throughout the project, informed by my supervisor guidance and by technically similar work in the EGG ecosystem @kharitonov:etal:2021 @lazaridou2020emergent.

\
== Evaluation Against Final Revised Objectives <group1>
\
*Primary Objectives:*

1. *Machine Learning Foundation Development*

This objective was successfully completed through the preparation work that translated directly into implementation and experimental decision making. The project work demonstrates that the foundation was sufficient for architecture design, debugging, and interpretation of unstable training dynamics.

2. *Emergent Communication Background Research*

This objective was completed to a strong standard and the literature work clearly shaped both environment design and evaluation choices. In particular, sender/receiver information asymmetry, communication bottlenecks, and collapse risks were incorporated before large runs were launched, proving the initial research helped throughout development.

3. *Baseline Implementation*

This objective was completed with a reproducible core pipeline in EGG that provided a stable reference for all later experiments as explained in the preparation chapter. By obtaining practice with the EGG framework, I was able to understand and identify potential errors that could arise while designing the survival game and infrastructure of the project.

4. *Simple Game Design (Revised)*

This objective was fully delivered through the revised vector based survival environment, including entity definitions, action constraints, reward logic, and explicit communication necessity. The resulting design was completely used throughout the project as the underlying logic for the game, with some minor changes throughout the implementation.

\
*Secondary Objectives:*

1. *Simple Game Implementation and Training*

This objective was successfully completed, with an end to end training operational model within the custom game designed. The implementation produced stable logs, reproducible outputs, and enough run coverage to support comparative claims, complying with the requirements of the objective.

2. *Language Analysis*

This objective was successfully achieved through protocol analysis across multiple metrics and qualitative inspections, with TopSim and message clustering used as central signals. As reported in the previous chapter, the findings support structured and functionally meaningful communication under specific optimisation regimes. Although a clear limitation is that full compositionality could not be claimed from the available evidence, so conclusions are appropriately bounded.

\
*Tertiary Objectives:*

1. *Complex Game Design*

This objective was partially achieved. The final survival environment includes substantially richer mechanics than a toy reference game, such as delayed consequences, tool chains, and multi type encounter logic. However, the broader tertiary plan of implementing multiple complex game families was not completed.

2. *Multi-Agent Communication*

This objective was not achieved. The completed system remains a two agent sender/receiver setting and does not implement the intended two plus agent communication regime. Although population level work was reviewed conceptually, no corresponding experimental pipeline was delivered.

3. *Advanced Language Analysis Across Multiple Games*

This objective was not achieved in its original form. The project produced advanced analysis across multiple settings within one validated vector game, but did not complete the intended cross game programme combining vector and image based environments. As a result, conclusions about protocol behaviour are strong within the implemented setting but not across environment general claims.

4. *Thesis Documentation*

This objective was successfully completed through a coherent dissertation structure linking motivation, design, implementation, results, and limitations to the revised requirements.

\
== Critical Comparison with Prior Work <group1>
\
Work in emergent communication varies substantially in task design, from controlled single step referential games to richer settings with more realistic perception and optimisation constraints. Despite this variation, there is a consistent cross paper pattern that communication structure is strongly shaped by what the task rewards and by how information is bottlenecked through the channel. The comparison below is organised around that shared pattern, then used to position what this survival game project does well and where it remains methodologically limited.

One of the main references use is Lazaridou et al. @lazaridou2018emergence study on referential communication under two input regimes, one symbolic and one pixel based, while keeping game structure and learning machinery aligned. Their central claim is that "the degree of structure found in the input data affects the nature of the emerged protocols" #cite(<lazaridou2018emergence>, supplement: [p. 1]), and they report that agents "struggle to produce structured messages" under entangled input conditions #cite(<lazaridou2018emergence>, supplement: [p. 2]). My setting is not image based, but the same principle appears in practice, when optimisation pressure emphasises action relevant distinctions, message structure becomes clearer and when pressure is misaligned collapse like behaviour is more likely. The key difference is that my environment introduces richer sequential survival consequences, so the relevant structure is policy functional rather than visual semantic.

Furthermore, Kharitonov et al. @kharitonov2020entropy analyse two agent discrete channel games from an information theoretic perspective, using controlled tasks to isolate how much information must pass through messages for successful coordination. They show that "emergent languages are (nearly) as simple as the task they are developed for allow them to be" #cite(<kharitonov2020entropy>, supplement: [p. 1]), and that this pressure strengthens with greater channel discreteness. This provides a strong lens for interpreting my no reconstruction experiments, where communication became more compressed around action equivalent clusters while policy behaviour remained imperfectly sharp. In other words, the project reproduces the core simplicity pressure but also highlights that simpler protocols do not automatically imply better long horizon action quality.

Moreover, Chaabouni et al. @chaabouni2020compositionality directly examine compositionality versus generalisation in emergent languages, and their findings are particularly important for preventing overclaiming in this dissertation. They report that "there is no correlation between the degree of compositionality of an emergent language and its ability to generalize" #cite(<chaabouni2020compositionality>, supplement: [p. 1]), and they further caution that "topographic similarity is still rather agnostic about the nature of composition" #cite(<chaabouni2020compositionality>, supplement: [p. 1]). This is directly relevant to my evaluation strategy, where TopSim was useful for global structure trends but cannot, on its own, establish full compositionality. My conclusions therefore remain intentionally bounded to structured and partially compositional communication rather than strong compositional claims.

A final study that I also found relevant to my work done is that of Dessì et al. @dessi2021interpretable, who move to large scale visual referential training from scratch and study whether learned symbols remain usable and interpretable beyond the training categories. They show that symbols can "denote partially interpretable categories" #cite(<dessi2021interpretable>, supplement: [p. 1]), and explicitly motivate augmentation because different sender/receiver views can make it "harder for them to adopt degenerate strategies" #cite(<dessi2021interpretable>, supplement: [p. 5]). My results align with their broader emphasis on guarding against degenerate shortcuts, but my current validation remains internal to one environment and lacks stronger transfer style interpretability checks. This is an important quality gap between the present project and the strongest recent work.

Methodologically, the literature also makes clear that successful task performance does not by itself prove meaningful communication. As summarised in the field survey, "just ablating the language channel and showing a drop in task success does not prove much" #cite(<lazaridou2020emergent>, supplement: [p. 10]), reflecting the measurement concerns formalised by Lowe et al. @lowe2019pitfalls. The same survey also notes that "positive signaling gives no guarantee of communication" #cite(<lazaridou2020emergent>, supplement: [p. 10]). This criticism applies to my own evaluation choices as well, where while TopSim, entropy trends, and protocol snapshots were informative, stronger causal tests of message use would further improve validity.

Overall, prior work and my results converge on a common conclusion, emergent protocols are highly sensitive to task pressure, optimisation design, and representation constraints. The project contributes evidence that the survival game together with the EGG framework can produce stable and structured communication under targeted objective design, while also making explicit that multi agent population dynamics, cross environment transfer, and stronger causal communication diagnostics remain open work.

#pagebreak()

= Conclusion and Future Work <group1>
\
In conclusion, this chapter consolidates the main findings of the dissertation, reflects critically on the constraints that shaped the work, and outlines the most defensible directions for extending it. The project set out to test whether meaningful communication can emerge when agents are placed in a survival constrained setting where signalling is instrumentally necessary for coordination. Across preparation, design, implementation, and evaluation, the work produced a reproducible sender/receiver pipeline in the EGG framework and demonstrated that communication structure is strongly shaped by objective design, rather than arising automatically from training alone.

The strongest technical contribution is the end to end integration of a custom survival environment with controllable communication learning dynamics, encompassing explicit reward design, masking, auxiliary reconstruction control, and language analysis instrumentation. The project provides evidence that removing strict reconstruction pressure can increase semantically structured reuse in the message space, whilst also showing that better protocol structure does not automatically eliminate suboptimal policy attractors. These two findings together suggest that objective design and policy optimisation are interacting forces that must be considered jointly, not independently.

\
== Summary of Findings <group1>
\

This section summarises the main findings by returning to the central question of whether meaningful communication can emerge between agents under survival pressure. The results show that communication does emerge in a structured and non trivial way, but only when the learning objective aligns with task relevant distinctions rather than strict identity reconstruction. This relationship between objective design and emergent structure explains both the success of the no reconstruction setting and the limitations observed under reconstruction pressure.

Within the baseline framework, the behaviour of the communication protocol reveals that under reconstruction pressure, agents developed highly discriminative protocols that reliably encoded individual entities, but these remained weakly aligned with the broader semantic structure of the environment. When this pressure was removed, the protocol shifted towards grouping semantically related entities, leading to stable clustering patterns and higher TopSim values on average. These findings reinforce the idea that communication systems are shaped by the objectives they are trained under, and that strict correctness at the level of individual instances can hinder the emergence of more generalisable structure.

On the other hand, introducing entropy improved the stability of the learned representations and often increased alignment with semantic structure, yet it also reduced the sharpness of the learned policies and did not eliminate convergence to local optima. This highlights a trade off between exploration and precision, where improvements in representational quality do not always translate directly into better decision making under the full action distribution. Even though these results are presentable, it is important to interpret these results with caution. As discussed in the Context survey, the metrics used in this work are diagnostic rather than definitive, meaning that they provide useful signals about structure without fully capturing the richness of emergent language. The variability observed across runs, particularly in measures such as TopSim, emphasises that multiple valid communication systems can arise under the same conditions, each reflecting different equilibria rather than a single optimal solution.

Overall, the dissertation demonstrates that meaningful and structured communication can emerge in a survival based setting when the learning objective supports functional equivalence rather than strict identity. At the same time, it raises a broader question that remains open beyond the scope of this work, namely how far such structured communication can generalise across environments, agents, and objectives before it begins to resemble the flexibility and compositionality associated with natural language.

\
== Main Constraints <group1>
\
In this project, the main constraints were scope, resources, and evaluation breadth, each shaping the others and narrowing the final analysis. Although the overall effort exceeded what is typically expected for a project of this size, the available time still limited how many environment variants could be fully implemented and validated. As a result, the image based branch was dropped early, and the analysis ultimately focused on a single vector based environment rather than the broader cross environment comparison initially planned.

Furthermore, the lack of resources limited the amount of experiments that I could do, a limitation that I did not foresee when designing the project, making it harder to draw firm conclusions about sensitivity. Finally, the analysis relied primarily on TopSim, entropy trends, and message inspection, which provided useful signals about structure but, as discussed in the Evaluation chapter, are not sufficient on their own to support strong causal claims about communication. This is particularly relevant in settings where coordination and communication may diverge without being fully captured by geometric metrics.

\
== Future Work <group1>
\
In the future, I think the focus should be on improving validity while also testing the approach across a wider range of environments, since the current results are best understood as a proof of concept that opens up several clear research directions rather than providing a final answer.

A nice extension would be to move from a two agent setup to a multi agent population, which was intended to be done but was not completed by the end of the project. Implementing three or more communicating agents with shared and role specific objectives would allow investigation of whether stable protocol conventions persist under partial observability and partner variation, a question which the current fixed pairing design cannot address. Alongside this, improving the reward assignment for long horizon strategy is important, as the current single step focus likely contributes to the convergence toward gather and eat behaviours observed during evaluation. Moving to sequence aware optimisation, for example through actor critic style baselines or return decomposition, would provide clearer learning signals across longer action sequences and may reduce this effect.

Further work should also strengthen how communication is evaluated, extending beyond TopSim and clustering with intervention based tests, such as message scrambling or targeted channel ablations, would help distinguish genuine communication from correlated behaviour and address the current limitations in causal interpretation. The previously dropped image based extension could then be reintroduced under controlled conditions, ensuring that both vector and image environments share the same underlying structure so that differences in communication can be attributed to representation rather than task design. Ultimately, there are infinite ways that one could expand the project and experiment to formulate theories of how these communications emerge and why.

Taken together, these extensions would convert the current proof of concept into a broader empirical programme on a question that this dissertation has only begun to address. I have thoroughly enjoyed this project and would like to extend my sincere thanks to my supervisor for their continued guidance, support, and teaching throughout. This dissertation has been a genuinely rewarding experience, and I am very satisfied with the results achieved. Most importantly, it has fulfilled my original objective of learning new concepts and developing my skills, and has played a significant role in shaping my approach to both research and problem solving.

#set quote(block: true)

#quote(attribution: [Winston Churchill])[
  _This is not the end. It is not even the beginning of the end. But it is, perhaps, the end of the beginning._
]

#pagebreak()

= Bibliography <group1>
\
#bibliography("references.bib", style: "ieee", title: none )

#pagebreak()

#block[

= Appendix <group1>

== Signed Ethics form <ethics-form>

#image("Images/signed-1.png", width: 100%)

#image("Images/signed-2.png", width: 100%)


== Implementation Relevant Material <group1>
\
*Full entity list:*

\
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

\
== Testing Summary <group1>

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

#colbreak()

=== PyTorch Learning <pytorch_app>
\
*Output of tutorial_pytorch.py:*

\
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

\
#figure(
  image("Images/MNIST_Fashion.png", width: 40%),
  caption: [MNIST Fashion Example]
)

\ \ \

*Output for MNIST_Fashion.py:*

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

#colbreak()

=== MNIST Classifier Output <train-logs-mnist>
\
*Output for entropy coefficient at 0.01:*

\
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


#colbreak()

== User Manual <group1>
\
This section summarises the minimum steps required to install dependencies, run training, and execute tests in this repository.

1. Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install project dependencies and the local package.

```bash
pip install -r requirements.txt
pip install -e .
```

3. Run a baseline EGG game (sanity check).

```bash
python -m egg.zoo.mnist_autoenc.train --vocab=10 --n_epochs=5
```

4. Run the project's training script with the desired configuration.

```bash
python -m egg.zoo.basic_games.train --help
python -m egg.zoo.basic_games.train --n_epochs 50
```

5. Run repository tests.

```bash
python -m pytest
```

In GPU-enabled environments, CUDA is used automatically when available, which substantially reduces run time for full experimental sweeps.
\ \
For more information please use the *README* file.

\
== Extended figures/tables/message snapshots <group1>
\

*Results from the runs:*

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

#set table(
  stroke: black,
)


#colbreak()


=== Metrics for the 5 final experiments

\
Below is a detailed summary of the metrics collected from the final experiments conducted. 
\
#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header([*Run*], [*Fish message*], [*Salmon message*]),

    [run14], [0 40 0], [43 40 0],
    [run15], [0 0 0], [42 0 0],
    [run16], [14 0 0], [0 0 0],
    [run17], [35 0 0], [0 0 0],
    [run18], [33 31 0], [33 33 0],
  ),
  caption: [Message IDs for Fish and Salmon across runs 14 to 18.],
) <fish-salmon-exp1>

\

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header([*Run*], [*Lion message*], [*Bear message*], [*Wolf message*]),

    [run14], [8 24 0], [14 0 0], [16 16 0],
    [run15], [33 26 0], [18 0 0], [25 26 0],
    [run16], [9 7 0], [4 0 0], [9 31 0],
    [run17], [31 2 0], [14 0 0], [35 48 0],
    [run18], [20 17 0], [33 41 0], [20 39 0],
  ),
  caption: [Message IDs for Lion, Bear, and Wolf across runs 14 to 18.],
) <lion-bear-wolf>

\

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header([*Run*], [*Goat message*], [*Deer message*], [*Rabbit message*]),

    [run14], [19 30 0], [14 0 0], [39 0 0],
    [run15], [11 33 0], [23 20 0], [14 23 0],
    [run16], [9 2 0], [46 46 0], [46 0 0],
    [run17], [13 35 0], [46 0 0], [38 0 0],
    [run18], [20 41 0], [36 33 0], [4 8 0],
  ),
  caption: [Message IDs for Goat, Deer, and Rabbit across runs 14 to 8.],
) <goat-deer-rabbit>

\

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header([*Run*], [*Storm message*], [*Poison Plant message*], [*Cliff message*]),

    [run14], [40 30 0], [16 27 0], [16 28 0],
    [run15], [29 0 0], [20 5 0], [20 1 0],
    [run16], [11 0 0], [38 7 0], [31 40 0],
    [run17], [0 0 0], [29 6 0], [13 0 0],
    [run18], [33 25 0], [22 25 0], [25 10 0],
  ),
  caption: [Message IDs for Storm, Poison Plant, and Cliff across runs 14 to 18.],
) <storm-poison-cliff>

\

#figure(
  table(
    columns: 6,
    align: (left, left, left, left, left, left),
    table.header([*Run*], [*Wood*], [*Water*], [*Herbs*], [*Berries*], [*Stone*]),

    [run14], [15 0 0], [47 0 0], [24 0 0], [21 0 0], [42 42 0],
    [run15], [20 25 0], [16 26 0], [20 26 0], [20 33 0], [7 20 0],
    [run16], [38 38 0], [36 38 0], [38 32 0], [38 46 0], [22 22 0],
    [run17], [43 0 0], [35 6 0], [39 0 0], [9 0 0], [2 2 0],
    [run18], [20 33 0], [20 49 0], [5 33 0], [28 33 0], [20 43 0],
  ),
  caption: [Message IDs for Wood, Water, Herbs, Berries, and Stone across runs 14 to 18.],
) <resources-env>

\
*With reconstruction enabled:
*
\
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

\
*Without recon:*

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

\
*Parameters used for the three tuning experiments:
*
\
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

\
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
  caption: [Survival game training configuration for run 26 (no recon, action entropy 0.5).],
) <setting-exp-4>

\
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

] <no-wc>