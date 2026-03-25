# zbMATH Web Scraping API

**Author:** Jefferson Orion Portelli  
**University:** University of St Andrews  
**Supervisor:** Olexandr Konovalov  
**Date:** 28th March, 2022

## Abstract

zbMATH and swMATH offer a rich repository of information on mathematics papers and software. Despite this, large-scale exploration of this data is almost entirely unavailable due to the limited functionality offered by their APIs. This project aims to break this barrier by providing users access to this data in an intuitive, easily usable format. To this end, a Python API was developed for web-scraping large quantities of information about different records. Using this system, data analysts with varying levels of technical experience can access mathematical software data in a simple and consistent way. Following this, a use case example of the system for reproducible data analysis is provided to facilitate design and personalisation for future experiments.

## Declaration

I declare that the material submitted for assessment is my own work except where credit is explicitly given to others by citation or acknowledgement. This work was performed during the current academic year except where otherwise stated. The main text of this project report is 11,919 words long, including project specification and plan. In submitting this project report to the University of St Andrews, I give permission for it to be made available for use in accordance with the regulations of the University Library. I also give permission for the title and abstract to be published and for copies of the report to be made and supplied at cost to any bona fide library or research worker, and to be made available on the World Wide Web. I retain the copyright in this work.

---

## Contents

1. [Introduction](#introduction)
   - 1.1 [Impact of UCU Strike Action](#impact-of-ucu-strike-action)
2. [Context Survey](#context-survey)
   - 2.1 zbMATH Systems
   - 2.2 Tools & Concepts
   - 2.3 Existing Systems
3. [Requirements Specification](#requirements-specification)
4. [Software Engineering Process](#software-engineering-process)
5. [Ethics](#ethics)
6. [Design](#design)
7. [Implementation](#implementation)
8. [Experiment](#experiment)
9. [Evaluation & Critical Appraisal](#evaluation--critical-appraisal)
10. [Appendix A - Ethics](#appendix-a---ethics)
11. [Appendix B - Instructions](#appendix-b---instructions)

---

## Introduction

The role of software in modern mathematics is arguably central, however the exact form this takes can vary wildly between two subjects. While statisticians may use packages in R to fit models and run experiments, graph theorists may use entirely different systems to prove fundamental properties about the chromatic numbers of planar graphs. The integration of software itself may also vary dramatically over time subject by subject, with more applied fields finding natural computational uses early on as opposed to pure subjects. Questions such as these may be deeply important to developers of these computational tools in finding their audience usage statistics in academics.

Fortunately, in the era of information this data is all readily stored. zbMATH Open is currently the largest reviewing and abstracting service for mathematics papers. With over four million articles spanning all the way back from the 18th century to the present, it is certainly the most comprehensive database of mathematics research. More importantly though, zbMATH, in conjunction with its sister site swMATH, has the unique trait of including software citation information of many of these papers in a clearly linked way.

Unfortunately, this information is still largely inaccessible in large quantities, and while swMATH offers some small visualisations of citation history, it is hardly sufficient to answer even the questions posed above. To allow greater accessibility to this novel datasource, this project involves the development of a simple Python API which uses web-scraping to find records specific software information at large scales. This API aims to allow easy access to inexperienced users while also facilitating reproducible and consistent experiments. To demonstrate this we also provide an example experiment showing the collection, cleaning, and visualisation of data at a reasonable scale.

This paper follows the execution of these goals in several steps. First, we look into the required concepts, platforms, and other relevant context of the project. Following this, we formalise and prioritise project goals and objectives. Then we analyse the specific approach, implementation steps, and ultimate completion of the stated objectives, before finally returning to analyse the results in relation to our initial requirements and goals.

### Impact of UCU Strike Action

Due to UCU strike action that occurred during the project, a period of no contact with supervisor Olexandr Konovalov occurred from 7th feb to 14th march. This interval coincided with a critical moment in the project wherein the data collection was prepared to be performed. Due to this, I could not access personalised advice setting up the programs to remotely execute for long durations. Further, this was the same time period the issue of rate limits was uncovered, leading to a delay in the development of workaround. These two issues ultimately led to smaller datasets than expected due to time constraints during scraping.

---

## Context Survey

### zbMATH Systems

Naturally, the zbMATH database and platform is the core of this project, hence we spend time exploring it in detail to understand the extent of its capacity. All information referenced here is based on information available on zbMATH.org directly.

#### zbMATH Open

**History**

Originally founded in 1931 as a reviewing service for mathematical literature, Zentralblatt für Mathematik und ihre Grenzgebiete—more simply referred to as Zentralblatt—aimed to provide quick reviews for an international selection of mathematical material. As the number of mathematical publications grew in the 1970s, Zentralblatt began to modernise their database using magnetic tapes to keep up with their aim for promptness. With the advent of the internet, in 1996 Zentralblatt was transformed into an internet accessible service known as zbMATH. Finally, zbMATH transitioned to an openly accessible platform in January 2021.

**Services**

The platform itself is relatively minimalistic, with the initial interface providing only a search line for documents or articles. As the UI suggests, the site is extremely simple, focused solely on exploring the articles contained therein. Each tab at the top of the search bar simply alters the search function to look for the category of content. This selection of options displays zbMATH the comprehensive collection of information zbMATH stores about the articles it indexes and reviews. Detailed searches can be performed allowing filtering by even more categories.

**Record Structure**

The primary resource provided by the platform are records. These records represent mathematical publications and each contain an array of summary information. A typical record contains many details about the document it represents, most notably:

- Title
- Language
- Publication Source
- Publication Date
- Summary
- MSC Codes
- Keywords
- Software
- References

While some variations of these fields can occur, records are generally formatted as described above, only with certain fields (such as software) occasionally omitted.

**MSC**

Mathematics Subject Classification (MSC) is a scheme for categorising mathematical publications by subject. It was last updated in 2020 and is hierarchical in its level of detail. For example, the code 05 represents combinatorics and contains a subset 05CXX, representing graph theory.

#### zbMATH Open API

Along with the transition to being a freely accessible database, zbMATH also created a new API to aid mathematical researchers. This API provides four main endpoints for accessing information from the database:

- **GetRecord:** Retrieves an individual zbMATH entry
- **ListIdentifiers:** Lists the identifiers for all 4,206,870 records currently in the zbMATH database. Several filter parameters can be used to narrow this set
- **ListRecords:** Retrieves all 4,206,870 records in the zbMATH database
- **ListSets:** Lists all 63 high level MSC sets zbMATH can filter by

**Record Structure**

Each record returned by the zbMATH API contains a more segmented and easily machine readable version of the respective website view. These responses are made using XML and contain the following fields:

- Identifier
- Authors
- Title
- Language
- Publication Source
- Publication Date
- Upload Date
- MSC Codes (high level & detailed)
- Keywords
- References

Note the structure is almost identical to the fields provided through the web view, with the most notable exception being the absence of the software field. For the purposes of this project this absence is crucial.

**Multi-page Outputs**

All requests except GetRecord are capable of returning multiple pages of responses (to avoid returning over 4 million items at once). To facilitate this, the zbMATH API utilises resumption tokens, unique strings generated after responses which users must append to subsequent queries to retrieve the next page. These tokens expire within a second and hence require automation to use.

**Usage Rights**

The zbMATH Open API, as well as the zbMATH dataset as a whole is licensed under CC BY-SA 4.0. This allows users to share and adapt any materials therein under the condition that they are appropriately credited and shared under the same licence.

#### swMATH

swMATH is a freely accessible partner site to zbMATH Open that provides information on a variety of mathematical software packages. It includes both a database of these packages as well as a web of links to zbMATH articles which either discuss or utilise them. The site aims to help software authors understand where their software is utilised as well as help users identify the quality of potential packages.

### Tools & Concepts

#### Web Scraping

Typically, obtaining information through websites can be done through either dedicated API's or manual browsing. Using zbMATH as an example, the easiest way to access information about the database is through their dedicated API. Unfortunately, not all websites provide an API, and not all APIs provide all the necessary information. For example, the zbMATH API provides an array of information about records in a clean and easily machine readable XML format. Despite this, it does not contain one crucial piece of information: software. Fortunately, software itself is clearly available, as browsing through the web interface clearly shows us a tab for software. Clearly the information is there, but how do we access it? The answer to this is web scraping.

**Definition & Structure**

Defined as "the process of extracting and combining contents of interest from the Web in a systematic way", web scraping offers an invaluable approach to obtaining web-based content often obscured from automation. Put more simply the process of web scraping operates in three main steps:

1. The page of interest is requested using the standard HTTP GET request. This action is identical to loading a page in the browser and retrieving its HTML. Note, some sites have systems in place to prevent automated bot access.
2. The HTML response is parsed and the fields of interest are extracted. This phase may include simple searching through tags and regex or more complicated text mining techniques.
3. The extracted contents are formatted for output.

Due to the precise nature of this text mining, many web scrapers are extremely weak to changes in site structure.

**Access Prevention**

Naturally, many sites are loath to allow bots access and implement various techniques to identify and prevent these intrusions. Most commonly, these intrusion prevention techniques take the form of rate limits, hard thresholds on the number of requests able to be made in a given interval. These can pose large obstacles to web scrapers, as staying beneath the limits can slow the process down dramatically. zbMATH employs its own limit as a threshold on daily requests. The ramifications of this will be discussed at length during implementation section.

**Legal Concerns**

Web scraping is also subject to legal scrutiny due to its ambiguous interaction with copyright laws and potentially damaging effect on websites. Frequent access attempts may also be virtually indistinguishable from D.O.S attacks. As such, maintaining reasonable access frequencies is important to maintaining an ethical scraper. Fortunately, many sites (such as zbMATH) employ their own rate limits to avoid these concerns. Additionally, zbMATH offers explicit licensing to information in their database, so copyright issues are entirely alleviated.

#### APIs

Application programming interfaces (APIs) provide high level access to potentially complicated applications. These tools expose the end user to reusable and highly abstracted methods to interact with their underlying programs in a unified manner. Most commonly, these take the form of REST web APIs for remote resource access.

**REST APIs**

Representational State Transfer (REST) is a framework for developing Web services that has defined the development of the Web. RESTful API's access resources via URL and generally output JSON or XML. This framework is generally defined by following principles:

- **Resource Addressability:** APIs expose resources from the backend using uniquely addressable Uniform Resource Identifiers (URI's)
- **Representation Separability:** Users do not need to know or be exposed to the underlying storage representation
- **Standardised Accessibility:** Resources are accessed using HTTP protocol methods (Get, Post, Delete, etc.)
- **Statelessness:** Interactions between the user and the API store no state

### Existing Systems

Two other Python based API's for interacting with zbMATH have previously been created, neither of which were referenced in this project due to either their obsoletion or irrelevance to the task at hand.

**ice-MC2 zbMATH API**

This project served as a simple wrapper for the zbMATH search function. It utilised BeautifulSoup and Python to perform a search for matching documents using a provided search term. The provided search term was entered directly into the search line on zbMATH and a list of authors, link, title pairs was returned. Last updated in 2019, this system was made obsolete due to the creation of the zbMATH Open API.

**zbMATH Open Links API**

Developed by members of zbMATH's parent institution—FIZ Karlsruhe – Leibniz Institute for Information Infrastructure—the zbMATH links API aimed to show connections between zbMATH and external platforms which contain links to objects indexed by zbMATH. Last updated 2022/01/05, the project is still being maintained at the time of this report.

---

## Requirements Specification

Several objectives were established during the initial DOER for this project. However, to more accurately encompass the shifting needs that arose during development, these evolved naturally throughout the course of the project.

### Initial Requirements

#### Primary Requirements

1. Develop an API for easily collecting and aggregating data from zbMath & swMath in Python
2. Identify the relationship between individual fields of mathematics and the software used therein
3. Design a clear procedure for collecting, cleaning, and analysing the data according to the following principles:
   - Correctness
   - Repeatability
   - Replicability
   - Reproducibility
   - Reusability

#### Secondary Requirements

1. Employ Python & Jupyter Notebooks to create a strong visual presentation of analytical findings
2. Investigate which facts about the relationship between different mathematical software packages can be inferred from their citations

### Adapted Requirements

While the overarching goals of developing an API and analysing data about software in mathematics remained the same, priorities were greatly shifted.

#### Primary Requirements (Adapted)

1. Develop a web scraper capable of mass collection of software information from zbMATH records
2. Develop an intuitive Python API for integrating the scraping tools into data analytics projects
3. Demonstrate the use of the aforementioned tools through a small scale procedure

#### Secondary Requirements (Adapted)

1. Adapt the API to allow automated, unchecked collection of data beyond rate limits
2. The example procedure should yield a comprehensive dataset capable of reuse
3. The example procedure should create a strong visualisation of its findings through Jupyter Notebooks

---

## Software Engineering Process

### Development Strategy

This project was developed roughly according to Agile procedures, with iterative development being driven by the aim of constantly having a working system. While there was only a single developer, weekly supervisor meetings served as an opportunity to establish goals, discuss obstacles, and reflect on progress. Compared to the traditional waterfall model, this iterative approach was much more favourable for this project.

### Development Structure

The project itself was split into several phases:

1. **Project Scoping:** The structure of the project was established and rough timeframes for each phase were estimated
2. **API Development:** This formed the bulk of the project and consisted of two components:
   - Web Scraper
   - API
3. **Data Collection:** This phase was marked by the shift towards demonstrating the produced tools
4. **Data Analysis:** The final stage involving visual analysis of findings

### Testing

Since these tools work primarily by making requests to zbMATH, unit testing is not entirely feasible. Despite this, a small number of unit tests were written to test certain smaller features. Additionally, the best way to test its features was through large scale deployment, which was achieved during the data collection phase with a selection of over 50,000 records scraped for the final result.

---

## Ethics

As per the DOER, this project raises no special ethical concerns. The tools created grant access to data that is already publicly available via zbMATH. Further, the analysis and sample dataset raise no concerns as they focus solely on generalised statistics about mathematical subjects and software; the names of individual authors will not be stored or utilised in any way.

---

## Design

### Project Architecture

All design considerations surrounding project architecture were made with respect to an abstract view of a hypothetical data analysts requirements. Throughout this section we refer to this hypothetical individual as the end-user.

### Scraper

The aim of the Scraper was aimed towards accessing software information about any specific record. Since software record information is not available through the zbMATH API itself, a web scraper is required.

**Structure**

At an overview, a Scraper represents an object which acts on a single zbMATH record's link. This Scraper then provides access to various attributes of the record by web-scraping the underlying HTML of the associated page.

**Field Choice**

Fields were chosen according to their relevance to a hypothetical data analysis task relating to software usage patterns:

- **ID:** The unique DE number identifying the record
- **Software:** A list of all software cited in the record
- **MSC:** The set of all mathematics subject classifications
- **Date:** The year the paper was published
- **Language:** The language the article was written in

### API

Essentially, this tool is a python accessible interface to the zbMATH website. It makes use of the underlying scraper tool to access the zbMATH records in a way that feels familiar to a web-based API.

**Aims**

- Intuitive Scraper Wrapper
- Access to the zbMATH API
- Out-of-the-box Usability

---

## Implementation

All design considerations surrounding project architecture were made with respect to an abstract view of a hypothetical data analysts requirements. The implementation section details how the design was executed.

### Scraper

The Scraper is implemented as a Python class, with each instance created around a single record link. Each Scraper then uses BeautifulSoup to parse its pages HTML for a variety of fields.

### API

The API acts as the users interface to all their desired information from zbMATH. This functionality was generally split into two types: small functions and large functions.

---

## Experiment

### Collecting Data

Data collection serves as the most important part of this experiment, as it is the main function of these tools.

### Analysis

The data analysis was an experiment demonstrating an example use case of the system.

---

## Evaluation & Critical Appraisal

### Primary Requirements

#### Requirement 1: Web Scraper

Develop a web scraper capable of mass collection of software information from zbMATH records.

The web scraper produced is absolutely capable of collecting mass amounts of data as already displayed through its use collecting over 50,000 records. Additionally, the scraper successfully collects all software used in these packages when present without fail.

#### Requirement 2: Python API

Develop an intuitive Python API for integrating the scraping tools into data analytics projects.

Likely the most important goal, this requirement was achieved to a higher level than any of the other tasks. The systems themselves are immensely intuitive, with complex aggregate functions presented in a simplified manner.

#### Requirement 3: Demonstration

Demonstrate the use of the aforementioned tools through a small scale procedure.

The small scale procedure was explained in detail following the implementation section, demonstrating the exact use case for the tools from experiment inception to analysis.

### Secondary Requirements

#### Requirement 4: Rate Limit Bypass

Adapt the API to allow automated, unchecked collection of data beyond rate limits.

Initially thought to be an unattainable reach goal, this is now the most impressive feature of the system. By overriding internal IP bans the system is able to trick zbMATH and bypass limits entirely, significantly upgrading the usability of the API.

#### Requirement 5: Comprehensive Dataset

The example procedure should yield a comprehensive dataset capable of reuse in other data analysis experiments.

While certainly achieved to a certain extent, the datasets produced for the experiment can by no means be considered comprehensive. Spanning only five years and four subjects, the datasets are likely useful for exploratory purposes only.

#### Requirement 6: Visualisations

Create strong visualisations of its findings through Jupyter Notebooks.

While certainly not to as much of an extent as the first four requirements, this goal was inarguably completed. Visualisation techniques and principles were employed to ensure effective and expressive visualisation of the created datasets.

---

## References

[1] zbmath about.
[2] zbmath open api.
[3] ice-mc2 zbmath api. https://github.com/ice-mc2/zbmath-api, 2019.
[4] zbmath open links api. https://github.com/zbMATHOpen/linksApi, 2022.
[5] Daniel Glez-Peña, Anália Lourenço, Hugo López-Fernández, Miguel Reboiro-Jato, and Florentino Fdez-Riverola. Web scraping technologies in an api world. Briefings in bioinformatics, 15(5):788–797, 2014.
[6] Tamara Munzner. Visualization analysis and design. CRC press, 2014.
[7] Carlos Rodríguez, Marcos Baez, Florian Daniel, Fabio Casati, Juan Carlos Trabucco, Luigi Canali, and Gianraffaele Percannella. Rest apis: a large-scale analysis of compliance with principles and best practices. In International conference on web engineering, pages 21–39. Springer, 2016.
