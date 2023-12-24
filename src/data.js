const FLASHCARD_DATA = [
    {
        id: 1,
        question: 'You have trained an ML model and are deploying it to production. At inference time, the model needs static features in order to make predictions in addition to incoming data from the client. What tool can you use to provide low latency access to static features to your model at inference time?',
        answer: 'SageMaker Feature Store',
        options: ['SageMaker GroundTruth', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker']
    },
    {
        id: 2,
        question: 'What built-in algorithm on SageMaker can be used for translation and summarization tasks?',
        answer: 'Seq2seq is a built-in algorithm in SageMaker where the input is a sequence of tokens and the output is another sequence of tokens. This can be used for translation and summarization tasks.',
        options: ['Seq2seq', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker']
    },
    {
        id: 3,
        question: 'What are some of the LEARNING types of Machine Learning Systems?',
        answer: 'Supervised Learning, Unspuervided Learning, Self-supervided Learning, Semi-supervided Learning, Reinforcement Leaning. There are so many more'
            + ' that you could classify them in broad categories based on the following criteria: - How are they supervised during traning. - Whether they can learn incrementally or on the fly (online vs. batch learning). - Whether they work '
            + ' by comparing new data points with known datapoints, or by detecting patterns in training data and buildng a predictive model (instance based and model based learning)',
        options: [],
        tags: ['Machine Learning']
    },
    {
        id: 4,
        question: 'What is Supervised Learning?',
        answer: 'In supervised learning the training set you feed an algorithm includes the desired solution, called labels',
        options: ['SageMaker GroundTruth', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker']
    },
    {
        id: 5,
        question: 'Provide an overview of phases of machine learning lifecycle?',
        answer: 'Business Stakeholer > Business Problem Identification > ML Problem > Data Collection + Quality Control > Data Prep. > Data Visualization > Feature Engineering > Model Training > Model Eval. > Business Decision > Business workflow integration. See: <a href="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/70" target="_blank">Phases of ML Learning</a>',
        options: [],
        tags: ['Machine Learning', 'Lifecycle']
    },
    {
        id: 6,
        question: 'What is the goal of Machine Learning?',
        answer: 'The goal of machine learning is to predict the value or the class of an unknown quantity using a mathematical model.',
        options: [],
        tags: ['Machine Learning']
    },
    {
        id: 7,
        question: 'Producing the machine leanrning model? What is training data?',
        answer: 'To produce the model you need some data and the model can learn from, which consists of independent varaibles and a dependent variable, known as training data. The model learns to predict the dependent variable from the independent variables from the data provided.'
            + 'Then if the model is well learned/trained should be albe to generalize what it has learned to data it has not seen before. Namely data where dependent variable unknown. See chpater8: <a href="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/99" target="_blank">Model Training</a>',
        options: [],
        tags: ['Machine Learning']
    },
    {
        id: 8,
        question: 'What is a common factor in order for a (business) problem to be a ML problem? If the absence of these common factors can a poblem still be a mchine learning problem?',
        answer: 'In general you need to have data that consists of independent variables and a dpependent variable. AKA labels and target. <br/> Note that you could still have a problem which would lend itself to ML by using ML on the data to discover patterns in absence of dependent variable, AKA unsupervised learning.',
        options: [],
        tags: ['Machine Learning']
    },
    {
        id: 9,
        question: 'When to use covariance correlation coefficient?',
        answer: 'Covariance is used when you have a Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 10,
        question: 'When to use Pearson’s correlation coefficient?',
        answer: 'Pearson’s correlation coefficient is used when you have a Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 11,
        question: 'When to use Spearman’s correlation coefficient?',
        answer: 'Spearman’s correlation coefficient is used when you have a non-Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 12,
        question: 'When to use Polychoric correlation coefficient?',
        answer: 'The polychoric correlation coefficient is used to understand the relationship of variables gathered via surveys such as personality tests and surveys that use rating scales',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 13,
        question: 'What is a Gaussian distribution?',
        answer: 'The normal distribution is also known as a Gaussian distribution or probability bell curve. It is symmetric about the mean and indicates that values near the mean occur more frequently than the values that are farther away from the mean',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 14,
        question: 'What is a Pearson`s correlation?',
        answer: 'The Pearson correlation coefficient can be used to summarize the strength of the linear relationship between two data samples. <br/> A value of 0 means no correlation. The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, and values below those values suggests a less notable correlation.',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis']
    },
    {
        id: 14.1,
        question: 'You are a machine learning specialist working for the social media software development division of your company. The social media features of your web applications allow users to post text messages and pictures about their experiences with your company’s products. You need to be able to block posts that contain inappropriate words quickly. You have defined a vocabulary of words deemed inappropriate for your site. Which of the following algorithms is best suited to your task?',
        answer: 'B. The Bernoulli Naive Bayes algorithm is used in document classification tasks where you wish to know whether a word from your vocabulary appears in your observed text or not. This is exactly what you are trying to accomplish. You need to know whether a word from your vocabulary of inappropriate words appears in the given post text or not.',
        options: ['A. Multinomial Naive Bayes', 'B. Bernouli Naive Bayes', 'C. Gaussian Naive Bayes', 'D. Ploychloric Naive Bayes'],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 15,
        question: 'What is Naive Bayes?',
        answer: 'The Naive Bayes classifier is a supervised machine learning algorithm, which is used for classification tasks, like text classification. They are called naive as it ignores language grammer ruels and common phrases. Dear Friend and Friend Deat is treated the same. See <a href="https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer" target="_blank">Naive Bayes, Clearly Explained!!!</a>',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 16,
        question: 'What is Multinomial Naive Bayes algorithm?',
        answer: 'The Multinomial Naive Bayes algorithm is best suited for document classification tasks where you wish to know the frequency of a given word from your vocabulary in your observed text. You need to know whether a word from your vocabulary appears in the given post text or not.',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 17,
        question: 'What is Bernouli Naive Bayes algorithm?',
        answer: 'The Bernoulli Naive Bayes algorithm is used in document classification tasks where you wish to know whether a word from your vocabulary appears in your observed text or not. This is exactly what you are trying to accomplish. You need to know whether a word from your vocabulary of inappropriate words appears in the given post text or not.',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 18,
        question: 'What is Gaussian Naive Bayes algorithm?',
        answer: 'The Gaussian Naive Bayes algorithm works continuous values in your observations, not discrete values.',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 19,
        question: 'What is Polychoric Naive Bayes algorithm?',
        answer: 'There is no such algorithm',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 20,
        question: 'You are a machine learning specialist working for a government agency that uses a series of web application forms to gather citizen data for census purposes. You have been tasked with finding novel user entries as they are entered by your citizens. A novel user entry is defined as an outlier compared to the established set of citizen entries in your datastore.',
        answer: '<ul><li><strong>Option A is incorrect</strong>. The Multinomial Naive Bayes algorithm is best suited for classification tasks where you wish to know the frequency of a given observation. You are trying to determine whether you have a novel observation.</li><li><strong>Option B is incorrect.</strong> The Bernoulli Naive Bayes algorithm is used in classification tasks where you wish to know whether a known class appears in your observation. You are trying to determine whether you have a novel observation.</li><li><strong>Option C is incorrect</strong>. The Principal Component Analysis algorithm is used to reduce feature dimensionality. You are trying to determine whether you have a novel observation.</li><li><strong>Option D is correct</strong>. The Support Vector Machine algorithm can be used when your training data has no outliers, and you want to detect whether a new observation is a novel entry.</li></ul>',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 21,
        question: 'You are a machine learning specialist working for a translation service company. Your company offers several mobile applications used for translation on smartphones and tablets. As a new feature of one of your translation apps, your company offers a feature to generate handwritten notes from spoken text. Which algorithm is the best choice for your new feature?',
        answer: '<ul><li><strong>Option A is correct</strong>. The Long Short-Term Memory (LSTM) can work with sequences of spoken language and can be used to generate sequenced output such as handwritten text.</li><li><strong>Option B is incorrect</strong>. Convolutional Neural Networks are primarily used to work with image data. You are working with sound data, spoken text.</li><li><strong>Option C is incorrect</strong>. The Multilayer Perceptron algorithm is used primarily for classification predictions and regression predictions. Your problem to solve is to convert spoken text to handwritten text.</li><li><strong>Option D is incorrect</strong>. The Support Vector Machine algorithm is primarily used for classification, regression, and anomaly detection. Your problem to solve is to convert spoken text to handwritten text.</li></ul>',
        options: ['A) Long Short-Term Memory(LSTM)', 'B) Convolutional Neural Network (CNN)', 'C) Multilayer Perception', 'D) Support Vector Machine'],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 22,
        question: 'In data modeling what is mean by dependent variable?',
        answer: 'The variable that researchers are trying to explain or predict is called the response variable. It is also sometimes called the dependent variable because it depends on another variable',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 23,
        question: 'What if you do not have any dependent variable information? Can such problem still be a machine learning problem?',
        answer: 'Yes, using machine learning you can discover patterns within the data in absence of dependent varible. It is called <a href="" target="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/142">unspervised learning</a>',
        options: [],
        tags: ['Machine Learning', 'modeling']
    },
    {
        id: 23.1,
        question: 'Fundamentally how many types data are there? Name them.',
        answer: '3 types <br/> Strucutured Data <br/> Unstructured Data <br/> Semi-structured data.',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data engineering', 'exploratory data analysis']
    },
    {
        id: 24,
        question: 'Provide exmaples of un-structured data.',
        answer: 'unstructured data is type of data that has no schema or well defined structural properties. <br/> Examples include images, videos, audio files, text docs or application log files',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data concept', 'data migration' ]
    },
    {
        id: 25,
        question: 'Provide exmaples of un-structured data.',
        answer: 'unstructured data is type of data that has no schema or well defined structural properties. <br/> Examples include images, videos, audio files, text docs or application log files',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data concept', 'data migration' ]
    },
    {
        id: 26,
        question: 'What is NOT stored in an AMI? mutiple choice A) Boot volume B) Data volumes C) AMI Permissions D) Block Device Mapping E) Instance settings F) Network Settings',
        answer: 'E) Instance Settings F) Network Settings',
        options: [],
        tags: ['Solution Architect', 'AMI' ]
    },
    {
        id: 27,
        question: 'What Permissions options does an AMI have? A) Public Access, Owner only, Specific AWS Accounts B) Public Access, Owner only, Specific IAM users C) Public Access, Owner only, Specific Regions D) Public Access, Specific AWS Accounts, Specific IAM users',
        answer: 'A) Public Access, Owner Only, Specific AWS Accounts',
        options: [],
        tags: ['Solution Architect', 'AMI' ]
    },
    {
        id: 28,
        question: 'Methods to imporove API performance?',
        answer: '<img src="/methods_improve_api_performance.png" />',
        options: [],
        tags: ['API', 'performance' ]
    },
]

export default FLASHCARD_DATA;