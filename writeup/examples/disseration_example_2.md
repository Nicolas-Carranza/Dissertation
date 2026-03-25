# Monophonic Sheet Music Transcription from Audio: An Automated Approach

**Author:** Kazimierz Wilowski  
**Supervisor:** Dr Kasim Terzic  

## Abstract

For most of history, musical transcription, the act of creating sheet music from a recording or a live performance of a piece of music, has solely been the domain of trained musicians, however this is slowly becoming not the case. As digital technology has improved gradually over time, it has become possible to tackle the problem of transcription using computers rather than the ear of an experienced musician. This project investigates a computational approach to generating monophonic musical performances automatically.

## Declaration

I declare that the material submitted for assessment is my own work except where credit is explicitly given to others by citation or acknowledgement. This work was performed during the current academic year except where otherwise stated. The main text of this project report is 13,615 words long, including project specification and plan. In submitting this project report to the University of St Andrews, I give permission for it to be made available for use in accordance with the regulations of the University Library. I also give permission for the title and abstract to be published and for copies of the report to be made and supplied at cost to any bona fide library or research worker, and to be made available on the World Wide Web. I retain the copyright in this work

---

## Contents

1. [Introduction](#introduction)
2. [Context Survey](#context-survey)
   - 2.1 Overview of Music Information Retrieval (MIR) literature
   - 2.2 Review of currently existing software
   - 2.3 libraries, formats, and resources
   - 2.4 Standard approaches and algorithms
3. [Requirements Specification](#requirements-specification)
4. [Design and implementation](#design-and-implementation)
5. [Artefact evaluation and case studies](#artefact-evaluation-and-case-studies)
6. [Critical appraisal](#critical-appraisal)
7. [Conclusion](#conclusion)
8. [Appendix: Running the program](#appendix-running-the-program)

---

## Introduction

Musical transcription is the process of creating a piece of sheet music which captures the notes played in an already existing performance or recording of a piece of music, similar to how linguistic transcription consists of writing down words corresponding to what a person says out loud. The act of transcribing has for centuries and until quite recently been an activity performed exclusively by human beings, however over the last several decades, advances in computer science have meant that computers can now be used to assist with and sometimes be chiefly responsible for generating transcriptions from musical recordings.

The problem of transcription is a very interesting one, as it is a problem which is not entirely objective (like the answer to the question: What is the loudest moment in this recording of a musical piece?) but also not entirely subjective (like the answer to the question: is this performance of a piece of music good?). Furthermore, humans who are good at performing transcriptions do not gain their skill by memorising a set of techniques and methods to translate what they hear onto the page but instead build an intuition over years of practice, meaning if one were to ask a skilled transcriber "how" they know what to write, it would be extremely difficult for them to explain their process in precise terms.

This project focused on the task of transcribing excerpts of monophonic western tonal music using traditional western music notation, from recordings provided as audio recordings. We investigate how the problem can be broken down into a variety of smaller sub-problems which can then be solved individually and the results of which can be cumulatively combined to form a sensible transcription for a given excerpt. To accomplish this, we draw on and borrow techniques from a variety of domains, including but not limited to, signal processing, statistics, and function optimisation.

---

## Context Survey

This section discusses, introduces, and contextualises key concepts relating to automated music transcription and computer aided music information retrieval (MIR) with respect to the goals of this project.

### Overview of Music Information Retrieval (MIR) Literature

The field of computational MIR is relatively small in the grander scope of computer science disciplines but is the subject of an increasingly broad corpus of literature. There are a variety of widely studied problems in the field of MIR, some of which have been studied at great length and for which there exist robust solutions and standard algorithms for tackling them, while others are still quite challenging for computers to perform with a practical rate of success.

Examples of problems falling into the former would be that of monophonic pitch detection, for which there is a vast corpus of literature dating back to 1970s, and for which there are a plethora of widely implemented solutions which can be found in many everyday places, such as guitar tuner effects pedals, mobile phone apps, and digital audio workstation (DAW) plugins.

Another widely studied problem is the problem of content-based searching, which allows the app to determine from a short, possibly distorted recording, the song being played, and provides the framework used by apps like Shazam. As indicated by the success of the Shazam app, it is a generally reliable and robust system.

The problem of automated monophonic transcription does not fall into the easier category of problems, although there exist standard approaches to some of the constituent sub-problems which automatic transcription can be broken down into (e.g. monophonic pitch detection), the task as a whole represents a relatively challenging one.

### Review of Currently Existing Software

**Digital Audio Workstations**

Digital Audio Workstations (DAWs) are where many professional and hobbyist musicians alike now find themselves spending most of their creative time. Such software suites offer a large variety of tools to aid in the recording, creation and manipulation of audio. Many DAWs contain features for MIR, for example, Ableton Live will attempt to extract from an audio file the onset and pitches of notes played.

**Websites and mobile apps**

There are many websites and mobile phone applications available which offer more niche and specific tools for performing MIR. As already mentioned, there are a plethora of guitar tuner apps and websites available, which effectively perform real-time monophonic pitch detection. It is possible to find a selection of apps and websites offering functionality to accomplish a variety of other basic MIR tasks.

**Automatic sheet music transcription**

Finally we discuss currently existing software which attempts to perform a task similar to what we attempt to do in this paper - full audio-to-sheet transcription. Software such as MelodyScanner and AnthemScore attempt to generate sheet music based on live audio recordings.

### Libraries, Formats, and Resources

**MIDI**

No discussion of music related computing would be complete without some mention of the ubiquitous Musical Instrument Digital Interface (MIDI). Since the 80s MIDI has represented the de facto standard machine readable symbolic representation of music.

**mido**

mido is a Python library which facilitates the straightforward reading and manipulating of MIDI information. mido proved to be a useful tool in extracting information from MIDI files.

**MusicXML**

As the name suggests, MusicXML is an XML-based file format for storing western music notation. It provides a good final format for musical transcriptions to be exported as, since transcriptions exported in this way can then be opened in a variety of music typesetting programs.

**Lilypond**

Lilypond is an open source computer program for typesetting sheet music. It is a long running and well maintained project that provides an easy to use system which produces high quality typesetting.

**MuseScore**

MuseScore is a freely available and open source music engraving suite which provides support for MusicXML files. Similar to Lilypond, MuseScore provides a straightforward way of generating high quality sheet music with minimal effort.

**music21**

music21 is a Python library and self-proclaimed "toolkit for computer-aided musicology" and provides a great many resources for manipulating symbolic representations of music in a standardised way.

**aubio**

aubio is a long running, well featured open source library providing features and functionality for analysing and labelling audio signals. It includes implementations of a variety of standard signal processing algorithms commonly used in MIR.

### Standard Approaches and Algorithms

#### Maximum a-posteriori Estimation

Maximum a-posteriori (MAP) estimation is a standard technique for estimating an unknown quantity within a statistical context. Certain problems in MIR can be modelled as statistical problems which this technique can be applied to.

#### Onset Detection

A common signal process problem in MIR is that of detecting when the beginning, or onset, of a note is. Onset detection is a widely studied problem with various reliable algorithms which exist for detecting onsets.

#### Pitch Detection

Another MIR task overlapping with the field of signal processing is the extraction of musical pitch, which has a broad body of literature associated with it. One of the most common and widely used fundamental frequency algorithms is the YIN algorithm.

---

## Requirements Specification

### Primary Objectives

- Investigate and integrate algorithms and techniques capable of monophonic pitch, and onset detection from audio
- Investigate and integrate algorithms and techniques capable of reasonably estimating the tempo
- Investigate algorithms and techniques capable of reasonably estimating the key signature
- Investigate algorithms and techniques capable of reasonably estimating the time signature
- Investigate and integrate means of systematically typesetting and rendering

### Secondary Objectives

- Investigate extending monophonic pitch and note detection to a larger pool of different instrument timbres
- Investigate extending monophonic pitch and note detection to simple polyphonic recordings
- Investigate and implement means of detecting and analysing dynamical information

---

## Design and Implementation

### Breaking Down the Problem

The main task of generating a transcription from an input audio file was broken down into several smaller sub-tasks:

1. Extract pitch and onset transient information from the input audio file
2. Analyse the transient and pitch information to produce a discrete list of notes
3. From the derived list of notes, deduce an estimation of the tempo
4. Using the estimated tempo, deduce the absolute timestamps of the piece's rhythmic pulse
5. From the downbeat timestamps and note onsets, determine the note durations
6. From the pitches present, estimate the key signature
7. From the notes, estimate the time signature
8. Typeset and export the sheet music

### Pitch and Onset Detection

For pitch detection, the python library aubio was used. The library provides a Python binding for the C Aubio library, which provides a variety of functionality for manipulating and analysing musical signals. From aubio, the YIN fundamental frequency estimation algorithm was used.

For onset detection, the script implemented by Böck et al., which provided python implementations for a variety of transient onset detection algorithms. The algorithm chosen was the spectral flux log filtered algorithm, a modified version of the spectral flux algorithm.

### Determining Discrete Notes

Two approaches were implemented, both with slightly different use cases:

1. **Non-transient method:** Appropriate for instruments which lack distinctive transients
2. **Transient-based method:** Appropriate for instruments with well-defined transients

### Tempo Estimation

Two methods were explored for finding the tempo of the excerpt:

#### Naive Method

The method consists of transforming the problem into an optimisation problem by defining a way of scoring different tempo candidates. Tempo candidates are scored based on how closely they line up with the start of notes in the excerpt.

#### Dixon Method

The method outlined by Dixon does not transform the problem into an optimisation problem as such. Dixon's algorithm outlines a more deliberate and measured approach by considering the gaps between the starts of notes, so-called "inter-onset intervals" (IOIs).

### Tempo Tracking

A key characteristic of music played by human beings is that the exact tempo will vary by small amounts over time. The method involves creating "agents" who traverse through the notes in steps proportional to internal metronomes they possess.

### Quantization

Once the pulse has been determined, it becomes possible to convert the absolute timing information about the notes to the relative timing information required to notate it as sheet music.

#### Maximum a-posteriori (MAP) Approach

The MAP approach scores quantizations based on how well they fit the statistical model of musical notation.

#### Naive Method

The naive method takes the approach of simply "snapping" each note to the closest grid point for a variety of grid resolutions.

### Determining the Key Signature

A relatively simple approach was found to be sufficient for this project. The key signature is determined by counting how many of the notes in the excerpt "belong" to each of the twelve standard musical key signatures.

### Determining the Time Signature

Two methods were investigated:

#### Salience Method

The first method analysed meter through the lens of musical salience. Higher salience notes are typically on strong beats.

#### Rhythmic Similarity Method

The second method investigated compared rhythmic similarity of consecutive measures.

### Typesetting

Once all necessary information has been extracted, the final step is to actually typeset the result as sheet music using the music21 library.

---

## Artefact Evaluation and Case Studies

### Quantisation Evaluation

To compare the two quantization algorithms, both algorithms were run for a number of simulated rhythmic performances. The results showed that the MAP estimation method consistently outperforms the naive method in regards to both accuracy and simplicity.

### Key Signature Evaluation

The algorithm was tested against a selection of well-known pieces. The algorithm achieved approximately 90% accuracy in determining the correct key.

### Time Signature Evaluation

Results for the time signature experiment were somewhat inconclusive. Neither algorithm performs exceedingly well in this task.

### Tempo Detection Evaluation

For less noisy signals, the Dixon algorithm outperforms the naive approach. For noisier signals, the naive approach actually performs slightly better.

### Case Studies

**1. Happy Birthday**

The system manages to produce a good transcription of Happy Birthday. The most obvious mistake is an awkward triplet-based rhythm observed in one measure, which can be traced to the detection of a false positive onset time.

**2. The White Stripes - Seven Nation Army (intro)**

The system produces a satisfactory transcription of the introductory riff to Seven Nation Army. The time signature is not correctly detected.

**3. Yankee Doodle**

The system produces a good transcription of Yankee Doodle. The only mistake made was in the time signature.

**4. Three Blind Mice**

The system struggled in producing a transcription of three blind mice. Several note onsets were not detected by the onset detector.

---

## Critical Appraisal

The main goal undertaken in this project was to create a full audio-to-sheet transcription system which could transcribe simple monophonic musical recordings and export them as typeset sheet music. The initial goals outlined for the project were changed as more understanding about the nature of the problem was learned.

One of the main challenges faced throughout this project was that of time allocation. The goal to build a fully functional audio-to-sheet system was a fairly ambitious one. The task of creating an audio-to-score system is broad enough that it would be possible to approach completing this tasks from a vast number of perspectives.

Another aspect of the project which proved challenging was testing the implemented system. Since the system does so many things, it proved difficult to develop a systematic way of testing the entire system. Instead a hybrid approach was taken, where the most important parts were tested systematically and objectively while a more holistic evaluation was also undertaken.

---

## Conclusion

The chief goal of this project was to develop a full audio-to-score automatic transcription for simple monophonic instruments. This goal was achieved, and a system able to transcribe melodies played by a variety of instruments was developed.

There are many ways the functionality of the system could be developed given more time. The system developed would provide a strong basis for developing a more robust automatic transcription system, or a system with less restrictions on the types of recordings it could process.

For example it was already noted that the system will only transcribe pieces in 4/4, 3/4, or 6/8, and despite these time signatures covering a vast majority of western music, they are not exhaustive. Furthermore, a more fully featured transcription system would probably support polyphonic transcription.

---

## Appendix: Running the Program

The system can be run as a python script on the command line. The modules used throughout the system which may need to be installed are:

- aubio
- colorednoise
- matplotlib
- mido
- music21
- numpy
- Pillow
- PyAudio
- pynput
- scipy

The script can then be run from within the project directory as:

```
python command line tool.py -i [PATH TO INPUT FILE] -o [PATH TO OUTPUT DIRECTORY]
```

To change algorithms used for the different sub-tasks by the system, options can be found at the top of the command line tool.py file.

---

## References

[Abl22] Ableton. Ableton reference manual version 11. 2022.
[BC02] Donald Byrd and Tim Crawford. Problems of music information retrieval in the real world. Information processing & management, 38(2):249–272, 2002.
[BDA+ 05] Juan Pablo Bello, Laurent Daudet, Samer Abdallah, Chris Duxbury, Mike Davies, and Mark B Sandler. A tutorial on onset detection in music signals. IEEE Transactions on speech and audio processing, 13(5):1035–1047, 2005.
[BKS12] Sebastian Böck, Florian Krebs, and Markus Schedl. Evaluating the online capabilities of onset detection methods. In ISMIR, pages 49–54. Citeseer, 2012.
[CDK00] Ali Taylan Cemgil, Peter Desain, and Bert Kappen. Rhythm quantization for transcription. Computer Music Journal, 24(2):60–76, 2000.
[DCK02] Alain De Cheveigné and Hideki Kawahara. Yin, a fundamental frequency estimator for speech and music. The Journal of the Acoustical Society of America, 111(4):1917–1930, 2002.
[Dix01] Simon Dixon. Automatic extraction of tempo and beat from expressive performances. Journal of New Music Research, 30(1):39–58, 2001.
[Dix06] Simon Dixon. Onset detection revisited. In Proceedings of the 9th International Conference on Digital Audio Effects. Citeseer, 2006.

(Additional references continuing as in original document)
