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

Describe the problem you set out to solve and the *extent of your success in solving it*. Here i want to mention also this is a 15 credit module so workload was limited.

You should include the aims and objectives of the project in order of importance and try to outline key aspects of your project for the reader to look for in the rest of your report. *Should i keep this here or in the req section?*

Communication is fundamental to coordination in natural and artificial systems, however the mechanisms through which structured communication emerges remain difficult to study in real settings due to scale, uncertainty, and time constraints. This project therefore studies the problem through computational simulation, where two agents must coordinate in survival scenarios and progressively develop useful signalling strategies. As this was a 15 credit module, the scope was deliberately bounded, balancing foundational learning, implementation, and analysis within a limited timeline.

This dissertation outlines the project aims at a high level in this chapter and then formalises them in detail in the Requirements Specification chapter. Following this, the report presents the technical background, implementation process, and evaluation results, with particular emphasis on how communication behaviour changed across training runs and how those changes relate to the core research question.

== Problem motivation and project context <group1>

Background on the problem, where it emerges from and motivation for this project. Talk about the requirements needed to study this topic and how i had to learn about Ml and other resources first.

Could cite a couple fo papers here.

Understanding how communication protocols form under environmental pressure is relevant both for language emergence research and for practical multi-agent systems, where coordination quality can determine task success. In this project, the motivation was to build a controlled environment in which communication is not hard-coded, then analyse whether meaningful conventions emerge from repeated interaction and reward-driven learning. To reach that stage, the work required an initial preparation phase in machine learning, PyTorch, and the EGG framework, after which the focus shifted to custom game design, implementation, and protocol analysis.

== Core research question <group1>

Deep explanation of what we are searching for exactly without taking into account the learning aspect of the project.

The core research question is whether meaningful communication protocols emerge when AI agents are placed in survival scenarios that require coordination, and, if they do, which factors influence their structure, stability, and effectiveness. More specifically, the project investigates how message patterns evolve under changing training conditions, how those messages relate to entity and action choices, and to what extent observed communication behaviour supports successful survival outcomes.

== Contributions and dissertation roadmap <group1>

Write about the plan i established, how it was followed and made.

\
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


\
= Requirements Specification <group1>

Capturing the properties the software solution must have in the form of requirements specification. You may wish to specify different types of requirements and given them priorities if applicable.

== Original objectives from DOER <group1>

List all the objectives as in the DOER. Mention how the scope was very good for primary objectives and secondary objectives. Tertiary objectives were too ambitious for the project and did not have time.

The original objectives were organised into three priority levels.

*Primary objectives*
- Establish machine learning and PyTorch foundations relevant to reinforcement and multi-agent learning.
- Build background knowledge in emergent communication research and gain working proficiency with the EGG framework.
- Implement a baseline communication setup to establish training and evaluation methodology.
- Design a simple survival-oriented communication game.

*Secondary objectives*
- Implement and train agents in the custom game environment.
- Analyse emergent language properties, including systematicity and efficiency.

*Tertiary objectives*
- Design a more complex game setting with richer environmental dynamics.
- Extend communication experiments beyond two-agent settings.
- Perform advanced cross-game language analysis.
- Consolidate findings into the final dissertation.

In practice, primary and most secondary objectives were completed, whereas tertiary objectives were partially deferred due to time and scope constraints.


== Scope evolution and objective changes <group1>

Explicitly talk about image based game branch dropped from implementation scope after the interim report.

All objectives were followed but gathering the images for the game designed was too time consuming and complex, so I decided to remove the game and use my second designed game where I make the data.

*Mention maybe here that the game was developed over the implementation too which was not anticipated in the objective set out?*

After the interim report, the implementation scope was revised by dropping the image-based game branch and prioritising the vector-based resource-chain survival game. The main reason was practical, building and curating image data at the required quality and scale would have consumed disproportionate project time, reducing the depth of analysis possible for communication behaviour. Following this change, the project retained the same core research aim, while concentrating effort on a single environment that could be iterated quickly and evaluated rigorously. Furthermore, the game design itself evolved during implementation, as several mechanics and reward choices were adjusted in response to training behaviour, and these design iterations became central to obtaining analysable message dynamics.

== Final revised objectives (for final project phase) <group1>

Same as in the DOER but with the update after the revision of the game. *Should i also take out the tertiary objectives?*

\
= Preparation and Technical Ramp-Up <group1>

This section is an additional one I want to add ti talk about the first two objectives in the DOER which were about learning and did not necessarily relate to the research question directly.

== Machine Learning foundation (Crash Course, core concepts) <group1>

Brief summary of what was studied and what resource was used. Talk about the progress, challenges and ways of learning implemented (videos, lectures, practical use and quizzes). 

List the sections learnt, and how they were relevant to the project and how successful it turned out to be.

The project began with a structured machine learning foundation phase using Google's Machine Learning Crash Course, complemented by lecture notes and practical exercises. This phase covered core topics including regression, classification, loss optimisation, model evaluation, and neural network fundamentals, which were necessary before implementing communication games in EGG. The learning process combined reading, quizzes, and short coding experiments, and it provided a reliable conceptual base for later decisions on training setup, metric interpretation, and error analysis.

== PyTorch familiarisation and initial experiments <group1>

Brief summary of what was studied and how it was learnt (documentation and coding exercises/exploration).

== EGG framework familiarisation <group1>

Summary of what was studied, how and why. Here it is now more relevant to the project so make sure to be concise but explanatory enough to be self contained.

Mention the structure of the repository but not in full detail

EGG familiarisation was carried out through detailed study of the `zoo/basic_games` modules and hands-on experiments with reconstruction and discrimination setups. This phase clarified the sender-receiver architecture, message generation process, and the differences between Gumbel-Softmax and REINFORCE training modes. Although early reconstruction scores were modest, these experiments were useful because they established a working understanding of data flow, optimisation behaviour, and practical constraints when training communicating agents.

== MNIST adaptation in EGG as preparatory work <group1>

Example of how the egg framework was learnt and combining the learning outcomes of the three learning objectives. 

Talk about the implementation and success of it but very briefly. Purpose is tooling and framework on boarding, not core project evaluation.

To consolidate this preparation, a baseline MNIST discrimination game was implemented in EGG, where agents communicated about one target digit among distractors. The model reached 96.5% test accuracy, and message analysis showed clear symbol reuse patterns and vocabulary compression. In this dissertation, this baseline is treated as preparatory work rather than a core contribution, because its main value was to validate tooling, establish analysis workflows, and reduce implementation risk before developing the custom survival environment.

== Lessons from preparation phase that informed final design choices <group1>

Maybe a small paragraph on how all of these learnt topics/skills helped in the development of the game, implementation and analysis.

\
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