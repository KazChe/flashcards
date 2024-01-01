const FLASHCARD_DATA = [
    {
        id: 1,
        question: 'You have trained an ML model and are deploying it to production. At inference time, the model needs static features in order to make predictions in addition to incoming data from the client. What tool can you use to provide low latency access to static features to your model at inference time?',
        answer: 'SageMaker Feature Store',
        options: ['SageMaker GroundTruth', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker'],
        category: 'Machine Learning'
    },
    {
        id: 2,
        question: 'What built-in algorithm on SageMaker can be used for translation and summarization tasks?',
        answer: 'Seq2seq is a built-in algorithm in SageMaker where the input is a sequence of tokens and the output is another sequence of tokens. This can be used for translation and summarization tasks.',
        options: ['Seq2seq', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker'],
        category: 'Machine Learning'
    },
    {
        id: 3,
        question: 'What are some of the LEARNING types of Machine Learning Systems?',
        answer: 'Supervised Learning, Unspuervided Learning, Self-supervided Learning, Semi-supervided Learning, Reinforcement Leaning. There are so many more'
            + ' that you could classify them in broad categories based on the following criteria: - How are they supervised during traning. - Whether they can learn incrementally or on the fly (online vs. batch learning). - Whether they work '
            + ' by comparing new data points with known datapoints, or by detecting patterns in training data and buildng a predictive model (instance based and model based learning)',
        options: [],
        tags: ['Machine Learning'],
        category: 'Machine Learning'
    },
    {
        id: 4,
        question: 'What is Supervised Learning?',
        answer: 'In supervised learning the training set you feed an algorithm includes the desired solution, called labels',
        options: ['SageMaker GroundTruth', 'AWS Comprehend', 'SageMaker Feature Store'],
        tags: ['Machine Learning', 'SageMaker'],
        category: 'Machine Learning'
    },
    {
        id: 5,
        question: 'Provide an overview of phases of machine learning lifecycle?',
        answer: 'Business Stakeholer > Business Problem Identification > ML Problem > Data Collection + Quality Control > Data Prep. > Data Visualization > Feature Engineering > Model Training > Model Eval. > Business Decision > Business workflow integration. See: <a href="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/70" target="_blank">Phases of ML Learning</a>',
        options: [],
        tags: ['Machine Learning', 'Lifecycle'],
        category: 'Machine Learning'
    },
    {
        id: 6,
        question: 'What is the goal of Machine Learning?',
        answer: 'The goal of machine learning is to predict the value or the class of an unknown quantity using a mathematical model.',
        options: [],
        tags: ['Machine Learning'],
        category: 'Machine Learning'
    },
    {
        id: 7,
        question: 'Producing the machine leanrning model? What is training data?',
        answer: 'To produce the model you need some data and the model can learn from, which consists of independent varaibles and a dependent variable, known as training data. The model learns to predict the dependent variable from the independent variables from the data provided.'
            + 'Then if the model is well learned/trained should be albe to generalize what it has learned to data it has not seen before. Namely data where dependent variable unknown. See chpater8: <a href="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/99" target="_blank">Model Training</a>',
        options: [],
        tags: ['Machine Learning'],
        category: 'Machine Learning'
    },
    {
        id: 8,
        question: 'What is a common factor in order for a (business) problem to be a ML problem? If the absence of these common factors can a poblem still be a mchine learning problem?',
        answer: 'In general you need to have data that consists of independent variables and a dpependent variable. AKA labels and target. <br/> Note that you could still have a problem which would lend itself to ML by using ML on the data to discover patterns in absence of dependent variable, AKA unsupervised learning.',
        options: [],
        tags: ['Machine Learning'],
        category: 'Machine Learning'
    },
    {
        id: 9,
        question: 'When to use covariance correlation coefficient?',
        answer: 'Covariance is used when you have a Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 10,
        question: 'When to use Pearson’s correlation coefficient?',
        answer: 'Pearson’s correlation coefficient is used when you have a Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 11,
        question: 'When to use Spearman’s correlation coefficient?',
        answer: 'Spearman’s correlation coefficient is used when you have a non-Gaussian relationship between your variables',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 12,
        question: 'When to use Polychoric correlation coefficient?',
        answer: 'The polychoric correlation coefficient is used to understand the relationship of variables gathered via surveys such as personality tests and surveys that use rating scales',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 13,
        question: 'What is a Gaussian distribution?',
        answer: 'The normal distribution is also known as a Gaussian distribution or probability bell curve. It is symmetric about the mean and indicates that values near the mean occur more frequently than the values that are farther away from the mean',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 14,
        question: 'What is a Pearson`s correlation?',
        answer: 'The Pearson correlation coefficient can be used to summarize the strength of the linear relationship between two data samples. <br/> A value of 0 means no correlation. The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, and values below those values suggests a less notable correlation.',
        options: [],
        tags: ['Machine Learning', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 14.1,
        question: 'You are a machine learning specialist working for the social media software development division of your company. The social media features of your web applications allow users to post text messages and pictures about their experiences with your company’s products. You need to be able to block posts that contain inappropriate words quickly. You have defined a vocabulary of words deemed inappropriate for your site. Which of the following algorithms is best suited to your task?',
        answer: 'B. The Bernoulli Naive Bayes algorithm is used in document classification tasks where you wish to know whether a word from your vocabulary appears in your observed text or not. This is exactly what you are trying to accomplish. You need to know whether a word from your vocabulary of inappropriate words appears in the given post text or not.',
        options: ['A. Multinomial Naive Bayes', 'B. Bernouli Naive Bayes', 'C. Gaussian Naive Bayes', 'D. Ploychloric Naive Bayes'],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 15,
        question: 'What is Naive Bayes?',
        answer: 'The Naive Bayes classifier is a supervised machine learning algorithm, which is used for classification tasks, like text classification. They are called naive as it ignores language grammer ruels and common phrases. Dear Friend and Friend Deat is treated the same. See <a href="https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer" target="_blank">Naive Bayes, Clearly Explained!!!</a>',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 16,
        question: 'What is Multinomial Naive Bayes algorithm?',
        answer: 'The Multinomial Naive Bayes algorithm is best suited for document classification tasks where you wish to know the frequency of a given word from your vocabulary in your observed text. You need to know whether a word from your vocabulary appears in the given post text or not.',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 17,
        question: 'What is Bernouli Naive Bayes algorithm?',
        answer: 'The Bernoulli Naive Bayes algorithm is used in document classification tasks where you wish to know whether a word from your vocabulary appears in your observed text or not. This is exactly what you are trying to accomplish. You need to know whether a word from your vocabulary of inappropriate words appears in the given post text or not.',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 18,
        question: 'What is Gaussian Naive Bayes algorithm?',
        answer: 'The Gaussian Naive Bayes algorithm works continuous values in your observations, not discrete values.',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 19,
        question: 'What is Polychoric Naive Bayes algorithm?',
        answer: 'There is no such algorithm',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 20,
        question: 'You are a machine learning specialist working for a government agency that uses a series of web application forms to gather citizen data for census purposes. You have been tasked with finding novel user entries as they are entered by your citizens. A novel user entry is defined as an outlier compared to the established set of citizen entries in your datastore.',
        answer: '<ul><li><strong>Option A is incorrect</strong>. The Multinomial Naive Bayes algorithm is best suited for classification tasks where you wish to know the frequency of a given observation. You are trying to determine whether you have a novel observation.</li><li><strong>Option B is incorrect.</strong> The Bernoulli Naive Bayes algorithm is used in classification tasks where you wish to know whether a known class appears in your observation. You are trying to determine whether you have a novel observation.</li><li><strong>Option C is incorrect</strong>. The Principal Component Analysis algorithm is used to reduce feature dimensionality. You are trying to determine whether you have a novel observation.</li><li><strong>Option D is correct</strong>. The Support Vector Machine algorithm can be used when your training data has no outliers, and you want to detect whether a new observation is a novel entry.</li></ul>',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 21,
        question: 'You are a machine learning specialist working for a translation service company. Your company offers several mobile applications used for translation on smartphones and tablets. As a new feature of one of your translation apps, your company offers a feature to generate handwritten notes from spoken text. Which algorithm is the best choice for your new feature?',
        answer: '<ul><li><strong>Option A is correct</strong>. The Long Short-Term Memory (LSTM) can work with sequences of spoken language and can be used to generate sequenced output such as handwritten text.</li><li><strong>Option B is incorrect</strong>. Convolutional Neural Networks are primarily used to work with image data. You are working with sound data, spoken text.</li><li><strong>Option C is incorrect</strong>. The Multilayer Perceptron algorithm is used primarily for classification predictions and regression predictions. Your problem to solve is to convert spoken text to handwritten text.</li><li><strong>Option D is incorrect</strong>. The Support Vector Machine algorithm is primarily used for classification, regression, and anomaly detection. Your problem to solve is to convert spoken text to handwritten text.</li></ul>',
        options: ['A) Long Short-Term Memory(LSTM)', 'B) Convolutional Neural Network (CNN)', 'C) Multilayer Perception', 'D) Support Vector Machine'],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 22,
        question: 'In data modeling what is mean by dependent variable?',
        answer: 'The variable that researchers are trying to explain or predict is called the response variable. It is also sometimes called the dependent variable because it depends on another variable',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 23,
        question: 'What if you do not have any dependent variable information? Can such problem still be a machine learning problem?',
        answer: 'Yes, using machine learning you can discover patterns within the data in absence of dependent varible. It is called <a href="" target="https://read.amazon.com/?asin=B09MBGCRQ7&ref_=dbs_t_r_kcr/142">unspervised learning</a>',
        options: [],
        tags: ['Machine Learning', 'modeling'],
        category: 'Machine Learning'
    },
    {
        id: 23.1,
        question: 'Fundamentally how many types data are there? Name them.',
        answer: '3 types <br/> Strucutured Data <br/> Unstructured Data <br/> Semi-structured data.',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data engineering', 'exploratory data analysis'],
        category: 'Machine Learning'
    },
    {
        id: 24,
        question: 'Provide exmaples of un-structured data.',
        answer: 'unstructured data is type of data that has no schema or well defined structural properties. <br/> Examples include images, videos, audio files, text docs or application log files',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data concept', 'data migration' ],
        category: 'Machine Learning'
    },
    {
        id: 25,
        question: 'Provide exmaples of un-structured data.',
        answer: 'unstructured data is type of data that has no schema or well defined structural properties. <br/> Examples include images, videos, audio files, text docs or application log files',
        options: [],
        tags: ['Machine Learning', 'data collection', 'data concept', 'data migration' ],
        category: 'Machine Learning'
    },
    {
        id: 26,
        question: 'What is NOT stored in an AMI? mutiple choice A) Boot volume B) Data volumes C) AMI Permissions D) Block Device Mapping E) Instance settings F) Network Settings',
        answer: 'E) Instance Settings F) Network Settings',
        options: [],
        tags: ['Solution Architect', 'AMI' ],
        category: 'Solution Architect'
    },
    {
        id: 27,
        question: 'What Permissions options does an AMI have? A) Public Access, Owner only, Specific AWS Accounts B) Public Access, Owner only, Specific IAM users C) Public Access, Owner only, Specific Regions D) Public Access, Specific AWS Accounts, Specific IAM users',
        answer: 'A) Public Access, Owner Only, Specific AWS Accounts',
        options: [],
        tags: ['Solution Architect', 'AMI' ],
        category: 'Solution Architect'
    },
    {
        id: 28,
        question: 'Methods to imporove API performance?',
        answer: '<img witdth=200px height=250px src="/methods_improve_api_performance.png" />',
        options: [],
        tags: ['API', 'performance' ],
        category: 'API'
    },
    {
        id: 29,
        question: 'AI/ML Online Resource',
        answer: '<ul class="ZbLneLRe2x_OBOYZMX3M"><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://huggingface.co/" target="_blank" rel="noopener nofollow">Hugging Face</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://syntax.fm/show/679/creator-of-swift-tesla-autopilot-and-tensorflow-new-ai-language-mojo-with-chris-lattner" target="_blank" rel="noopener nofollow"> Creator of Swift, Tesla Autopilot &amp; Tensorflow. New AI language Mojo with Chris Lattner</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/" target="_blank" rel="noopener nofollow"> LLaMA</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://huggingface.co/spaces" target="_blank" rel="noopener nofollow">Spaces - Hugging Face</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://openai.com/" target="_blank" rel="noopener nofollow">OpenAI</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://www.anthropic.com/index/introducing-claude" target="_blank" rel="noopener nofollow">Anthropic \ Introducing Claude</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://replicate.com/" target="_blank" rel="noopener nofollow">Replicate</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://app.fireworks.ai/" target="_blank" rel="noopener nofollow">Fireworks Console</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://gpt-tokenizer.dev/" target="_blank" rel="noopener nofollow">gpt-tokenizer playground</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://github.com/openai/tiktoken" target="_blank" rel="noopener nofollow">openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI’s models.</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://syntax.fm/show/625/supper-club-openai-future-of-programming-llms-and-math-with-andrey-mishchenko" target="_blank" rel="noopener nofollow"> Supper Club × OpenAI, Future of programming, LLMs, and Math with Andrey Mishchenko</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://www.raycast.com/pro" target="_blank" rel="noopener nofollow">Raycast Pro</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://docs.aws.amazon.com/managedservices/latest/userguide/sagemaker.html" target="_blank" rel="noopener nofollow"> Amazon SageMaker (AMS SSPS)</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://github.com/openai/evals" target="_blank" rel="noopener nofollow">openai/evals</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://www.langchain.com/" target="_blank" rel="noopener nofollow">LangChain</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://pytorch.org/" target="_blank" rel="noopener nofollow">PyTorch</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://www.tensorflow.org/" target="_blank" rel="noopener nofollow">TensorFlow</a></li><li class="Type__TypeElement-sc-goli3j-0 cSfdnZ rjdQaIDkSgcGmxkdI2vU" data-encore-id="type"> <a draggable="false" href="https://www.npmjs.com/package/ai" target="_blank" rel="noopener nofollow">ai - npm</a></li></ul>',
        options: [],
        tags: ['Machine Learning', 'online resources' ],
        category: 'Machine Learning'
    },
    {
        id: 30,
        question: 'Think about the document object model of a user interface then assess the following to be true or false.  Document Object Model It is an HTML abstraction of a page. If false please provide more succinct description of the document object model.',
        answer: 'ChatGPT<ol><li>Traditional DOM (Browser DOM):<ol><li>Direct Manipulation:</li><li>In traditional DOM, updates directly manipulate the actual browser DOM. Changes trigger a reflow and repaint, impacting performance.</li><li>Performance Concerns:</li><li>Direct manipulation can be inefficient, especially for complex UIs with frequent updates. Continuous reflows and repaints can lead to performance bottlenecks.</li><li>Complete Rendering:</li><li>When an update occurs, the entire affected subtree is re-rendered in the browser.</li></ol></li><li>React\'s Virtual DOM:<ol><li>Indirect Manipulation:</li><li>React introduces a Virtual DOM, which is an in-memory representation of the actual DOM. Updates are first applied to the Virtual DOM, not directly to the browser DOM.</li><li>Diffing Algorithm:</li><li>React\'s Virtual DOM includes a reconciliation process (diffing algorithm) to identify differences between the current and previous states. Only the necessary changes are then applied to the real DOM.</li><li>Batch Updates:</li><li>React batches multiple updates and applies them in a single operation, reducing the number of reflows and repaints.</li><li>Performance Benefits:</li><li>Minimizes direct manipulation of the DOM, leading to improved performance. Reduces unnecessary re-rendering and updates only the parts of the DOM that changed.</li></ol></li><li>Benefits of Virtual DOM:<ol><li>Performance Optimization:</li><li>Virtual DOM minimizes the number of actual DOM manipulations, leading to better performance.</li><li>Efficient Updates:</li><li>Only the differences between the previous and current states are applied, reducing redundant operations.</li><li>Batching Updates:</li><li>React batches updates, optimizing rendering by grouping multiple changes into a single DOM manipulation.</li><li>Framework-Agnostic:</li><li>The concept of Virtual DOM is not tied to React alone and can be adapted by other frameworks for performance benefits.</li></ol></li><li>Drawbacks of Virtual DOM:<ol><li>Abstraction Overhead:</li><li>The Virtual DOM introduces an abstraction layer that can add some overhead.</li><li>Learning Curve:</li><li>Developers new to React may initially find it challenging to understand the concept of Virtual DOM.</li><li>Not Always Necessary:</li><li>For simple applications with minimal UI updates, the overhead of Virtual DOM might outweigh the benefits.</li><li>Memory Usage:</li><li>Maintaining a separate Virtual DOM in memory consumes additional resources.</li></ol></li></ol><p>In summary, React\'s Virtual DOM provides a performant and efficient way to update the actual DOM by introducing an intermediate step. While it offers significant benefits in terms of performance optimization and efficient updates, there are associated trade-offs, including some abstraction overhead and a learning curve for new developers. The decision to use a Virtual DOM should be based on the specific needs and complexity of the application.</p>',
        options: [],
        tags: ['Front End', 'virtual dom' ],
        category: 'Front End'
    },
    {
        id: 31,
        question: 'What is retrieval-augmented generation RAG? and provide an example of retrievel-augment generation RAG',
        answer: 'Consider a sports league that wants fans and the media to be able to use chat to access its data and answer questions about players, teams, the sport’s history and rules, and current stats and standings. A generalized LLM could answer questions about the history and rules or perhaps describe a particular team’s stadium. <br/> <br/>It wouldn’t be able to discuss last night’s game or provide current information about a particular athlete’s injury because the LLM wouldn’t have that information—and given that an LLM takes significant computing horsepower to retrain, it isn’t feasible to keep the model current. In addition to the large, fairly static LLM, the sports league owns or can access many other information sources, including databases, data warehouses, documents containing player bios, and news feeds that discuss each game in depth. RAG lets the generative AI ingest this information. Now, the chat can provide information that’s more timely, more contextually appropriate, and more accurate. Simply put, RAG helps LLMs give better answers. Retrieval-augmented generation (RAG) typically falls under the category of "hybrid models" in machine learning. It combines elements of both generative models and retrieval models to enhance performance. In the case of RAG, a generative language model (like a large language model, such as GPT) is combined with a retrieval component that can pull in information from external sources. So, to be more specific, RAG can be considered a hybrid approach that combines elements of generative models and retrieval models to improve the quality and relevance of generated responses.',
        options: [],
        tags: ['Machine Learning', 'retrieval-augmented generation RAG' ],
        category: 'Machine Learning'
    },

]

export default FLASHCARD_DATA;