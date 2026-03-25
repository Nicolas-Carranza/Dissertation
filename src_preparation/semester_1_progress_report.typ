#import "@preview/wordometer:0.1.4": word-count, total-words

#show: word-count.with(exclude: (<no-wc>))

// Basic document settings
#set page(numbering: "1")
#set par(justify: true)
#set text(lang: "en", region: "GB", hyphenate: auto)
#set list(indent: 2em)

#show link: underline

// Set heading numbering and indentation styles
#set heading(numbering: "1.")

// Title Page
#set page(margin: auto)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Communication Emerges from Multi-Agent Survival Games
  ]
    
  #text(size: 14pt, weight: "bold")[
    Interim Progress Report
  ]
  
  #v(0.5cm)
  
  Student: Nicolás Carranza \
  Supervisor: Dr. Phong Le \
  February 06, 2026
]

#v(0.5cm)

= Executive Summary
\
This report presents the work completed during the first semester of the dissertation project. The semester began with establishing a solid theoretical foundation in machine learning through Google's Machine Learning Crash Course, alongside developing practical skills in PyTorch through hands-on implementations. Building upon these fundamentals, extensive study of the EGG (Emergent Communication through Games) framework was undertaken, culminating in systematic experimentation with basic communication games to understand training dynamics and agent behaviour.

The practical component of the semester centred on implementing a baseline discrimination game using the MNIST dataset, which achieved 96.5% accuracy on the test set. Analysis of the emergent communication protocols revealed interesting properties including compositional structure and task-adaptive complexity. The semester concluded with designing two comprehensive survival game scenarios that will form the basis of the second semester's implementation work. These games have been carefully specified to study how different environmental pressures influence the emergence and structure of communication between agents.

= Objective 1: Machine Learning Foundation Development
\
The semester began with building foundational knowledge in machine learning, an area where significant gaps existed. #underline[#link("https://developers.google.com/machine-learning/crash-course")[Google's Machine Learning Crash Course]] became the primary learning resource, covering everything from linear and logistic regression through to neural network architectures, classification systems, and model evaluation. The course proved highly effective, completing it within four weeks with perfect scores on all quizzes.

Alongside this structured learning, I developed PyTorch skills through self-directed experimentation and the guidance of my supervisor. The MNIST dataset served as the main training ground, supplemented by other benchmark datasets. This hands-on approach, where I was building custom neural networks with `torch.nn.Module`, implementing training loops, and experimenting with different architectures, proved invaluable for understanding how theory translates to practice.

= Objective 2: Emergent Communication Background Research
\
To learn how to use the EGG framework, my supervisor suggested studying the `zoo/basic_games` module from the git repository. This module became the gateway to understanding how emergent communication systems work in practice. The framework provides the building blocks, Sender and Receiver modules, RNN wrappers for different training modes like `RnnSenderGS` and `RnnSenderReinforce`, and mechanisms for processing attribute-value vectors. After analysing the library, two training approaches stood out: Gumbel-Softmax, which uses continuous relaxation for end-to-end differentiation, and REINFORCE, which implements policy gradient optimisation for discrete symbol sampling.

The real learning came from hands-on experimentation with reconstruction games. Running comparative studies revealed that REINFORCE achieved 17% accuracy compared to 14% for Gumbel-Softmax using a Sender:256/Receiver:512 GRU architecture. Whilst neither approach achieved impressive absolute accuracy on the reconstruction task, these experiments taught valuable lessons about training communicating agents and the trade-offs between continuous and discrete communication methods.

= Objective 3: Baseline Implementation
\
The baseline implementation tackled a key challenge: adapting the EGG framework to handle image inputs rather than the attribute vectors it was designed for. The solution came through implementing a discrimination game using MNIST digits, where agents had to communicate about one target digit among two distractors. This setup tested whether meaningful communication would emerge for a well-understood visual classification task.

The architecture used 256-unit GRU networks for both sender and receiver agents, with a vocabulary of 50 symbols and maximum message length of 10. Training relied on the REINFORCE algorithm with a small entropy coefficient (0.001) to encourage exploration whilst keeping communication relatively deterministic. Over 10 epochs with batch size 128, the model reached 96.5% test accuracy, showing stable convergence from 65.2% in the first epoch.

What made the results particularly interesting was analysing the emergent communication protocols across 10,000 samples. The agents showed remarkable vocabulary efficiency, actively using only about 7 out of 50 available symbols. Systematic patterns emerged: some digits consistently received alternating symbol sequences like [31, 13, 31, 13, ...], whilst others triggered repeating patterns such as [37, 37, 37, ...]. Symbol 31 appeared as a prefix across multiple digit patterns, suggesting emergent compositional structure.

The entropy analysis revealed task-adaptive complexity. Digits 2 and 3 shared similar message structures, which explains digit 2's lower accuracy (95.0%). Meanwhile, digits 7, 9, and 4 exhibited higher message entropy (0.56, 0.58, and 0.49 respectively), suggesting the communication protocol adapted its complexity based on how difficult discrimination was. This baseline established evaluation methodologies such as per-class accuracy, message entropy, confusion matrices, and symbol usage tracking, that would prove valuable for analysing the custom survival games.

= Objective 4: Simple Game Design
\
The semester concluded with designing two survival game scenarios, each optimised to study different aspects of emergent communication. Both games implement a Scout-Strategist structure where one agent perceives the environment and communicates with a second agent who makes survival decisions.

== Island Survival: Image-Based Communication

This scenario centres on visual perception driving communication. The Scout agent receives 28×28 or 64×64 pixel images representing environmental threats, resources, or conditions, and produces variable-length messages. The Strategist receives these messages alongside partial context (inventory state, weather conditions) and must choose from 2-5 discrete actions.

Episodes comprise 5-10 rounds covering threat detection, resource gathering, crafting decisions, environmental events, and health management. Images will be pre-rendered using PIL and OpenCV, providing complete control whilst enabling compositional flexibility. State variables track energy (0-100, decreasing 8-12 per turn), binary inventory flags for tools (spear, fire, shelter), and health affecting survival probabilities.

The research value lies in context-dependent communication: identical images require different strategies based on inventory state. For example, encountering a lion with a spear available (85% survival, -40 energy) versus without (60% survival) should elicit different messages. This creates compositional pressure to develop concepts spanning animal types, threat levels, and tool requirements.

== Resource Chain Survival: Vector-Based Communication

This scenario employs attribute-vector representations requiring discovery of transformation chains. Each entity encodes as a six-dimensional vector: `[entity_type, subtype, danger_level, energy_value, tool_required, weather_dependency]`. Entity categories include animals (5 subtypes), resources (5 subtypes), environmental dangers (5 subtypes), and tools (4 types).

The game mechanics revolve around transformation chains representing survival strategies. Hunting an animal yields raw meat (+2 energy, parasite risk), which fire transforms into cooked meat (+7 energy, safe consumption). Similarly, gathering wood and stone enables spear crafting, increasing hunt success rates from 30% to 70%. Episodes span 15-25 turns with energy management (-8 to -12 per turn), health degradation from injuries, and weather changes every 4-6 turns.

The reward structure encourages chain completion (+40), tool discovery (+50), and episode completion (+150 plus efficiency bonus), whilst penalising death (-300) and inefficient choices (-8). Training will use approximately 80,000 episodes. The research value encompasses studying entity type distinction (huntable versus gatherable versus avoidable), risk communication (expressing "high reward but dangerous"), tool dependencies, state-dependent optimisation, and compositional pressure to develop symbols for animal types, danger levels, tool requirements, and urgency.

Both games have been fully specified with complete rule systems, reward structures balancing immediate and long-term objectives, clear integration points with the EGG framework, and established evaluation metrics.

= Planned Work for Second Semester
\
The second semester will focus on implementing and analysing the designed survival games. Weeks 1-3 will involve implementing the Island Survival game with PIL and OpenCV image generation. Weeks 4-6 will focus on training agents and analysing emergent communication protocols. Weeks 7-9 will implement the Resource Chain Survival game. The final weeks (10-12) will conduct comparative analysis across both environments and compile thesis documentation.

= Conclusion
\
The first semester established both theoretical foundations and practical capabilities necessary for the dissertation work. The progression from machine learning fundamentals through EGG framework proficiency to baseline implementation provided hands-on experience with the technical challenges of training communicating agents. The MNIST discrimination game, whilst relatively simple, revealed interesting properties of emergent communication including compositional structure, vocabulary efficiency, and task-adaptive complexity.

The two designed survival scenarios represent different approaches to studying emergent communication. Island Survival emphasises image-based perception and context-dependent decision-making, whilst Resource Chain Survival focuses on compositional communication about transformation chains and sequential planning. Both games create pressures that should encourage meaningful communication: context dependency requires nuanced messages, transformation chains demand compositional structure, and survival rewards incentivise efficient coordination.

The semester's work positions the project well for implementation phases. The baseline metrics, validated training procedures, and complete game specifications provide a robust foundation for the second semester's focus on custom game implementation and comprehensive analysis of emergent communication protocols.
