# The Open-Source Data Science Masters
This is a [fork of this](https://github.com/datasciencemasters/go), experimenting with different curriculum topics and themes.

[License here](license.md).

###### A note on direction
This is an introduction geared toward those with at least **a minimum understanding of programming**, and (perhaps obviously) an interest in the components of Data Science (like statistics and distributed computing). Out of personal preference and need for focus, the curriculum assumes and mainly uses **Python tools and resources**, except where marked as R, Java etc.

## The Open Source Data Science Curriculum
![](http://nirvacana.com/thoughts/wp-content/uploads/2013/07/RoadToDataScientist1.png)

### History
###	Fundamentals
**Intro to Data Science** [UW / Coursera](https://www.coursera.org/course/dat *
*Topics:* Python NLP on Twitter API, Distributed Computing Paradigm, MapReduce/Hadoop & Pig Script, SQL/NoSQL, Relational Algebra, Experiment design, Statistics, Graphs, Amazon EC2, Visualization.asci)
Algebra-Steven-Levandosky/dp/0536667470/ref=sr_1_1?ie=UTF8&qid=1376546498&sr=8-1&keywords=linear+algebra+levandosky#)
 * Forecasting: Principles and Practice [Monash University / Book](http://otexts.com/fpp/) *uses R
 * Problem-Solving Heuristics "How To Solve It" [Polya / Book](http://en.wikipedia.org/wiki/How_to_Solve_It)
 * Think Bayes [Allen Downey / Book](http://www.greenteapress.com/thinkbayes/)
 * Capstone Analysis of Your Own Design; [Quora](http://www.quora.com/Programming-Challenges-1/What-are-some-good-toy-problems-in-data-science)'s Idea Compendium
 * [Toy Data Ideas](http://www.quora.com/Programming-Challenges-1/What-are-some-good-toy-problems-in-data-science)

Skills
	Matrices and Linear Algebra fundamentals
		Linear Algebra / Levandosky [Stanford / Book](http://www.amazon.com/Linear-
		Coding the Matrix: Linear Algebra through Computer Science Applications [Brown / Coursera](https://www.coursera.org/course/matrix)
	Hash Functions, Binary Tree, O(n)
	Relational Algebra
	DB Basics
	Inner, Outer, Cross, Theta join
	CAP Theorem
	abular data
	Entropy
	Data Frames and Series
	Sharding
	OLAP
	Multidimensional Data Model
		ETL
	Reporting vs. BI vs. Analytics
	JSON & XML
	NoSQL
	Regex
	Vendor Landscape
	Env setup
	
###	Maths and Stats
 * Statistics [Stats in a Nutshell / Book](http://shop.oreilly.com/product/9780596510497.do)	   Pick a dataset
 * Linear Programming (Math 407) [University of Washington / Course](http://www.math.washington.edu/~burke/crs/407/lectures/)

Skills
	Descriptive statistics
	Exploratory Data Analysis
	Histograms
	Percentiles and outliers
	Probability theory
	Bayes Theorem
	Random Variables
	Cumulative Distribution Function (CDF)
	Continous Distributions (Normal, Poisson, Gaussian)
	Skewness
	ANOVA
	Probability Density Functions

	Central Limit Theorem
	Monte Carlo Method
	Hypothesis testing
	p-value
	Chi squared test
	Estimation
	Confidence intevals (CI)
	MLE
	Kernel Density Estimate
	Regression
	Covariance
	Correlation
	Pearson Coefficient
	Causation
	Least squares fit
	Euclidean Distance

 * Probabilistic Programming and Bayesian Methods for Hackers [Github / Tutorials](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
 * PGMs / Koller [Stanford / Coursera](https://www.coursera.org/course/pgm)


### Computing
#### Toolbox / Programming Languages / Software stacks
	Unix cli install programs and packages
	Bash basics
		cat, grep, wget etc
		piping
		understand stdio
	Python
	Regex
	MS Excel w/ Analysis ToolPak
	Java
	R, R-studio, Rattle
	IBM SPSS
	Weka, Knime, RapidMiner
	Hadoop ditribution of choice
	Spark, Storm
	Flume, Scibe, Chukwa
	Nutch, Talend, Scraperwiki
	Webscraper, Flume, Sqoop
	tm, RWeka, NLTK
	RHIPE
	D3.js, ggplot2, Shiny
	IBM Languageware
	Cassandra, MongoDB
#### Algorithms, data structures and databases
* **Algorithms**
 * Algorithms Design & Analysis I [Stanford / Coursera](https://www.coursera.org/course/algo)
 * Algorithm Design [Kleinberg & Tardos / Book](http://www.amazon.com/Algorithm-Design-Jon-Kleinberg/dp/0321295358/ref=sr_1_1?ie=UTF8&qid=1376702127&sr=8-1&keywords=kleinberg+algorithms)

* **Databases**
 * SQL Tutorial [W3Schools / Tutorials](http://www.w3schools.com/sql/)
 * Introduction to Databases [Stanford / Online Course](http://class2go.stanford.edu/db/Winter2013/)

#### Programming
* **Python** (Learning)
 * New To Python: [Learn Python the Hard Way](http://learnpythonthehardway.org/), [Google's Python Class](http://code.google.com/edu/languages/google-python-class/)

* **Python** (Libraries)
 * Basic Packages [Python, virtualenv, NumPy, SciPy, matplotlib and IPython ](http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/)
 * [Data Science in iPython Notebooks](http://nborwankar.github.io/LearnDataScience/) (Linear Regression, Logistic Regression, Random Forests, K-Means Clustering)
 * Bayesian Inference | [pymc](https://github.com/pymc-devs/pymc)
 * Labeled data structures objects, statistical functions, etc [pandas](https://github.com/pydata/pandas) (See: Python for Data Analysis)
 * Python wrapper for the Twitter API [twython](https://github.com/ryanmcgrath/twython)
 * Tools for Data Mining & Analysis [scikit-learn](http://scikit-learn.org/stable/)
 * Network Modeling & Viz [networkx](http://networkx.github.io/)
 * Natural Language Toolkit [NLTK](http://nltk.org/)

Skills
	Variables
	Vectors
	Matrices
	Arrays
	Factors
	Lists
	Data Frames
	Reading CSV data
	Reading Raw data
	Manipulate Data Frames
	Functions
	Factor Analysis

### Applied methods	
#### Data Munging and integration
The art of converting or mapping data from one "raw" form into another format that allows for more convenient consumption of the data with the help of semi-automated tools. Expect to spend 80% of your workday doing some sort of data wrangling.
* [What I learned from 2 years of 'data sciencing](http://www.quantisan.com/what-i-learned-from-2-years-of-data-sciencing/) Paul Lam


	Dimensionality & Numerosity Reduction
	Normalization
	Data Scrubbing
	Handling missing values
	Unbiased estimators
	Binning sparse values
	Feature Extraction
	Denoising
	Sampling
	Stratified Sampling
	Principal Component Analysis
	Summary of Data Formats
	Data Discovery
	Data Sources & Acquisition
	Data Integration
	Data Fusion
	Transformation and enrichment
	Data survey
	Google OpenRefine
	How Much Daya
	Using ETL

#### Visualization

Skills
	Data Exploration in R (Hist, boxplot etc)
    Uni, Bi and multivariate Viz
	ggplot2
	Histogram & Pie (Uni)
	Tree and Tree Map
	Scatter Plot
	Line Charts
	Survey Plot
	Timeline
	Decision Tree
	D3.js
	InfoVis
	IBM ManyEyes
	Tableau

### Data mining and analysis
 * Mining Massive Data Sets [Stanford / Book](http://i.stanford.edu/~ullman/mmds.html)
 * Mining The Social Web [O'Reilly / Book](http://shop.oreilly.com/product/0636920010203.do)
 * Introduction to Information Retrieval [Stanford / Book](http://nlp.stanford.edu/IR-book/information-retrieval-book.html)
* **Analysis**
 * Python for Data Analysis [O'Reilly / Book](http://www.kqzyfj.com/click-7040302-11260198?url=http%3A%2F%2Fshop.oreilly.com%2Fproduct%2F0636920023784.do&cjsku=0636920023784)
 * Big Data Analysis with Twitter [UC Berkeley / Lectures](http://blogs.ischool.berkeley.edu/i290-abdt-s12/)
 * Social and Economic Networks: Models and Analysis / [Stanford / Coursera](https://www.coursera.org/course/networksonline)
 * Information Visualization ["Envisioning Information" Tufte / Book](http://www.amazon.com/Envisioning-Information-Edward-R-Tufte/dp/0961392118/ref=sr_1_8?ie=UTF8&qid=1376709039&sr=8-8&keywords=information+design)

###	Machine Learning
 * Machine Learning / Ng [Stanford / Coursera](https://www.coursera.org/course/ml)
 * A Course in Machine Learning / Hal Daumé III UMD [Online Book](http://ciml.info/)
 * Programming Collective Intelligence [O'Reilly / Book](http://shop.oreilly.com/product/9780596529321.do)
 * Statistics [The Elements of Statistical Learning](http://www-stat.stanford.edu/~tibs/ElemStatLearn/)
 * Machine Learning / CaltechX [Caltech / Edx](https://courses.edx.org/courses/CaltechX/CS1156x/Fall2013/)

Skills
	Numerical Var
	Categorical Var
	Supervised Learning
	Unsupervised Learning
	Concepts, Inputs and Attributes
	Training and Test Data
	Classifier
	Prediction
	Lift
	Overfitting
	Bias and variance
	Classification
		Trees and classification
		Classification rate
		Decision trees
		Boosting
		Naive Bayes Classifiers
		K-Nearest neighbour
	Regression
		Logistic regression
		Ranking
		Linear regression
		Perceptron
	Clustering
		Hierarchical clustering
		K-means clustering
	Neural Networks
	Sentiment analysis
	Collaborative Filtering
	Tagging
	   
###	Text Mining / NLP
 * NLP with Python [O'Reilly / Book](http://shop.oreilly.com/product/9780596516499.do)

Skills
	Corpus
	Named Entity Recognition
	Text Analysis
	UIMA
	Term Document Matrix
	Term Frequency and weight
	Support Vector Machines
	Association rules
	Market Based Analysis
	Feature Extraction
	Use Mahout
	Use Weka
	Use NLTK
	Classify Text
	Vocabulaty Mapping

* Healthcare Twitter Analysis [Coursolve & UW Data Science](https://www.coursolve.org/need/54)


### Big Data
	Map reduce fundamentals
	Hadoop
	HDFS
	Data Replication Principles
	Setup Hadoop (IBM / Cloudera / HortonWorks)
	Name & Data nodes
	Job and task tracker
	M/R Programming
	Sqoop: Loading Data in HDFS
	Flube, Scribe: For Unstructured Data
	SQL with Pig
	DWH with Hive
	Scribe, Chukwa For Weblog
	Using Mahout
	Zookeeper Avro
	Storm: Hadoop Realtime
	Rhadoop, RHIPE
	rmr
	Cassandra
	MongoDB, Neo4j

### General Resources:
* [Coursera](http://coursera.org)
* [Khan Academy](https://www.khanacademy.org/math/probability/random-variables-topic/random_variables_prob_dist/v/term-life-insurance-and-death-probability)
* [Wolfram Alpha](http://www.wolframalpha.com/input/?i=torus)
* [Wikipedia](http://en.wikipedia.org/wiki/List_of_cognitive_biases)
* Kindle .mobis
* Great PopSci Read: [The Signal and The Noise](http://www.amazon.com/Signal-Noise-Predictions-Fail-but-ebook/dp/B007V65R54/ref=tmm_kin_swatch_0?_encoding=UTF8&sr=8-1&qid=1376699450) Nate Silver
* Zipfian Academy's [List of Resources](http://blog.zipfianacademy.com/post/46864003608/a-practical-intro-to-data-science)
* [A Software Engineer's Guide to Getting Started w Data Science](http://www.rcasts.com/2012/12/software-engineers-guide-to-getting.html)
* Data Scientist Interviews [Metamarkets](http://metamarkets.com/category/data-science/)

## Contribute
Please Share and Contribute Your Ideas -- **it's Open Source!**
