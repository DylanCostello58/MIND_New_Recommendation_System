# MIND_New_Recommendation_System
## Introduction
The MIND is the Microsoft News Dataset which contains data on roughly 1,000,000 users, 160,000 news articles, 15,000,000 impressions, and 24,000,000 clicks. This dataset is massive and can be used in a multitude of ways. We will be using a smaller version of this dataset, the MIND_small dataset with data on roughly 50,000 users, 65,000 news articles, 230,000 impressions, and 347,000 clicks, to develop a news reccomendation system where our AI will attempt to predict what articles a user is likely to click based on their past behavior. 

Both the small and large datasets can be found at this link: [MIND Download](https://msnews.github.io/)

## Exploratory Data Analysis

With a dataset this large, it is a good idea to take some time to understand the data that is being used. So below are six images, each showcasing a feature of the dataset. There are many more explorations to be done on this data, but these six were the ones that I was most interested in. 

![Category Distribution](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/category_distribution.png)

This image displays the number of articles per category, with news and sports being the top two by a large margin.

![Category CTR](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/ctr_by_category.png)

This image displays the click-through-rate of each category, with news and sports, again, being the two highest by a significant margin.

![Hist Len](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/history_length_dist.png)

This image displays the distribution of history lengths across all users. With most users having a history length between 0 and 100.

![Imp Size](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/impression_size_dist.png)

This image displays the distribution of how many news articles a user is shown in a single browsing session. With most users having less than 75 candidates per impression.

![Title Len](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/title_length_dist.png)

This image displays the distribution of title lengths, in number of words, across all given articles. With most titles having between 5 and 15 words. 

![User Click](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/user_click_dist.png)

This image displays the number of users that clicked on a given number of articles. With most users having less than 15 clicks.

Now that we know more about the dataset we are safe to move to the next step.

## Preprocessing

In the notebook 02_preprocessing, we perform several necessary steps to make the data processable by the machine. 

1. Using NLTK, we tokenize the titles of every article, and we build a vocabulary from the training dataset. 
2. We use that vocabulary to extract the corresponding vectors from GloVe to create an embedding matrix (Words not found in GloVe receive random initialization).
3. We map each news title to a fixed-length sequence of word indices, padding or truncating where required to ensure every title has a length 30.
4. We parse each impression into positive and negative samples, corresponding to clicked and not clicked articles respectively.
5. For each positive click, we sample K negative articles from the same impression to form training examples.
6. We use the provided train/dev split from MIND_small.

### Model Architecture and Implementation
The model we used was based on the NRMS (Neural News Recommendation with Multi-Head Self-Attention) architecture. There are two main components to this model, the news encoder which turns articles into a vector representation, and the user encoder which compiles all the user's clicked articles and creates one user representation from that. The final prediction is then the dot product of the user vector and all candidate news vectors.

To run this preprocessing simply open 02_preprocessing, from the "notebooks" folder and run the cells in order.

## Training
Our model is trained using cross-entropy loss over all candidate articles in each impression. The model scores 1 positive and K negative candidates (Where K is a chosen integer) for each training sample. This loss encourages the positive article to receive the highest score.

We ran 3 different trials with 3 different sets of parameters as defined below. 

| Parameter | Run 1 | Run 2 | Run 3 |
|-----------|----------|---------|--------------|
| Learning Rate | 1e-4 | 1e-3 | 1e-4 |
| Batch Size | 64 | 64 | 64 |
| Negative Samples (K) | 4 | 4 | 4 |
| Attention Heads | 16 | 16 | 16 |
| Dropout Rate | 0.2 | 0.2 | 0.4 |
| Max History Length | 50 | 50 | 50 |

We chose the parameters for run 1 by looking at the recommended range provided to us. Then for run 2 we wanted to see what would happen if we raised the learning rate. And for run 3 we wanted to see what would happen if we raised the dropout rate. The results of these runs will be analyzed in the next section. 

## Results
We chose 4 evaluation metrics to track across our 3 runs. 

- AUC (Area Under the ROC Curve)
  - This measures the probability that a randomly chosen clicked article is ranked higher than a randomly chosen non-clicked article. A score of 0.5 represents pure random, and 1.0 represents perfection.
- MRR (Mean Reciprocal Rank)
  - For each impression, find the rank position of the first clicked article. MRR is the average of 1/rank across all impressions. 1.0 represents that the article clicked was the highest ranked. 0.5 represents that the clicked article) is the second highest ranked, and so on.
- nDCG@K (Normalized Discounted Cumulative Gain at K)
  - Evaluates the quality of the top-K ranked articles by giving more credit to relevant articles appearing earlier. We used both K = 5 and K = 10 in this case.

The results and loss curve graphs can be found below. 

| Metric | Run 1 | Run 2 | Run 3 |
|-----------|----------|---------|--------------|
| AUC | n/a | 0.7352 | n/a |
| MRR | n/a | 0.6400 | n/a |
| nDCG@5 | n/a | 0.7303 | n/a |
| nDCG@10 | n/a | 0.7303 | n/a |

![Run2 Loss](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/loss_curve.png)

This is the loss curve for run 1. We can see a decline in the loss as the epochs continue, which is expected and desired. It's minimum loss is 1.2893.

![Run2 Loss](https://raw.githubusercontent.com/DylanCostello58/MIND_New_Recommendation_System/main/results/loss_curve_1.png)

This is the loss curve for run 2. We can see a decline in the loss as the epochs continue, which is expected and desired. It's minimum loss is 1.2528.

Note: This is for the professor of my class so if you somehow found this just call me a bad time manager and move on. This assignment was due at midnight and it's currently 8:32PM and the computer lab is closed. I only had time to get through 2 of the 3 hyperparameter tests and my evaluation code only evaluated the latest run, thus the n/a's that populate my results graph. With my side tangent about not alloting my time well, let's get back to it.

Overall these results show a decent model. For a model built on a smaller version of the original dataset and only trained for 5 epochs, these numbers are actually pretty good. I believe, if there were more epochs and we used the full dataset, that this model would be relatively strong. Likely not as strong as industrial strength models, but for a cheap, undergraduate level model, this seems very good.
