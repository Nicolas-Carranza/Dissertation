#import "@preview/in-dexter:0.5.3": *
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
    Description of Objectives, Ethics and Resources (DOER)
  ]
  
  #v(0.5cm)
  
  Student: Nicolas Carranza \
  Supervisor: Dr. Phong Le \
]

#v(0.5cm)

= Problem Description
\
Communication is fundamental to survival in nature. From lions coordinating during hunts through body language to vervet monkeys using distinct alarm calls to warn of predators, effective communication systems have evolved as critical survival mechanisms. Understanding how these communication protocols emerge and evolve is essential for comprehending language origins, yet studying this phenomenon in natural settings is challenging due to the extensive timescales involved in language evolution.

This project addresses this challenge through computational simulation, exploring how communication emerges from interactions between AI agents in nature-inspired survival scenarios. The research will focus on multi-agent reinforcement learning environments where agents must develop communication protocols to successfully navigate survival challenges, such as coordinating responses to predators or cooperating in hunting scenarios.

The core research question investigates whether and how meaningful communication protocols spontaneously emerge when AI agents are placed in survival scenarios requiring coordination, and what factors influence the complexity and effectiveness of these emergent languages.

= Objectives
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

= Ethics Considerations
\

This project involves computational simulation exclusively and does not involve human subjects, personal data collection, or interaction with living animals. The research focuses on artificial agents operating in simulated environments. No ethical issues were identified and all statements on the ethics form checklist have been confirmed as applicable to this project.

Attached signed ethics form at the appendix.

\
= Resources Required
\
*Hardware Requirements:*
Access to GPU resources is the primary hardware requirement, as they are essential for training deep reinforcement learning models. Standard School lab computers will provide the necessary development and analysis infrastructure.

*Software Requirements:*
The project will use Python 3.8+ with the PyTorch framework as its core development environment. The EGG library will serve as the primary research tool, supported by standard machine learning libraries such as NumPy, Matplotlib, and Pandas for data analysis and visualisation.

*Educational Resources:*
Access to relevant academic papers and books on emergent communication will provide the theoretical foundation for the project, with Lazaridou & Baroni (2020) and related literature serving as key references. Online tutorials and EGG library documentation will further support the technical implementation.


= Risk Assessment and Mitigation

== Identified Risks
\
The primary risk for this project lies in computational resource limitations, particularly the possibility of insufficient GPU availability or hardware failures that could disrupt training progress. There are also technical implementation challenges which present another risk, as difficulties may arise when integrating the EGG library or developing custom game environments, potentially slowing development. 

A further risk is the possibility that agents may fail to develop meaningful communication protocols, which could limit the depth of analysis and findings. Finally, there is the risk of time management issues, particularly the potential underestimation of the time required to complete individual phases, which could put pressure on later project milestones.

== Contingency Plans
\
The project has been designed with built-in flexibility to ensure academic rigour is maintained even if significant obstacles arise in achieving the primary objectives. Early submission of resource requests and the development of smaller-scale experimental setups provide effective fallback options in the event of computational or technical limitations. 

Our approach ensures that foundational learning, baseline implementations, and partial results will still yield substantial and meaningful project content, even if the more advanced objectives cannot be fully realised, considering the weight of the project being 15 credits.

= Project Management
\
Weekly supervision meetings with Dr. Phong Le have been established to monitor progress, address technical challenges, and adjust objectives as necessary. A private Git repository will be maintained for version control, with regular commits providing a clear record of development progress. The project will follow an agreed sequence of phases, with defined deliverables and assessment points to ensure consistent progress towards the final dissertation submission.


#pagebreak()

= Appendix

#image("signed-1.png", width: 100%)

#image("signed-2.png", width: 100%)