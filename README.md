SemProj Raaj Flores was the initial Notebook, which began with visualizing the dataset to gain some insights; the dataset is available at https://zenodo.org/records/7737983 upon request. It should be noted that users need Python 3.x and some libraries to be installed to run it, namely pandas, matplotlib, seaborn, and ast. All three data sets are named fact_checks.csv. pairs.csv and posts.csv should be in the same directory of work as well. It includes visualization of certain analyses that are: 
1. Number of instances per claim
2. Distribution of Languages in claims
3. Frequency of Claims by Type
4. Distribution of Fact-Checks per Language
5. Posts with and without Retrieved Fact-Checks
6. Analysis of the relationship between social media posts and fact-checks by:
   a. Frequency analysis
   b. Creating a cross-tabulation matrix to see the association between posts and fact-checks
7. Distribution of post-text length
8. Distribution of verdicts in posts
9. Distribution of Posts per Fact Check
10. Average text length per verdict
11. Distribution of platforms in posts [FB, Instagram, Twitter(X)]
12. Distribution of languages in posts and then only the top 20 languages to narrow down the scope that needs to be worked on

Further, for our understanding, BM25 and Sentence BERT had an experimental implementation while calculating top-k indices, precision@10, the rank of correct claim, Mean Reciprocal Rank, Cosine Similarity Score and saving Cosine similarity heatmap as an image. In the case of sentence BERT, we also thought of separately testing OCR, TEXT and Combined TEXT and OCR. These implementations of the two models are not final; they were initial and naïve implementations for our understanding to develop the project later.

Note --> There are some visualizations that presented some errors or did not yield coherent result for example correlation heatmap in post_id, ocr and instances.


 
Upload SemEval contains the code that generated the results submitted to the challenge website. For generating the results a pipeline design was used where first: the text from OCR and text fields is extracted then cleaned of having NAN and empty spaces, where finally they are put together into one string. The function get post text gets the text from either OCR or text. Clear NAN and sep empty clear the Nans and remove the empty spaces in the string. The fact text also has to be extracted with get fact text and clear Nan and sep empty is also used on the string. The post strings and fact strings are used to get an embedding for each post and fact with get embeddings in batches. Then FAISS is used to first build an index based on the fact emebeddings and then search between the fact and post embeddings to generate the similarity scores for the pairs. The resulting dataframe can compared to the true pairs to generate the average success at k for the model. 



BM25 is where the baseline results for the BM25s model was generated along with the category splits that is used in the visualize notebook. 

The functions follow a similar pattern to how SBERT works in SemProj Flores Raaj where in this case callBM is used to call the BM25s model to generate the embeddings and to rank the similarity scores of the facts and posts. The new function in this note book is the get important range function takes a list of character counts and generates x ranges for each bin. afterwards a csv file is created that contains the post id for all of the posts that belong to the new bin. The undefined function is reused in get lang and get source which extracts the language/source for each posts and saves a csv file with the post ids split based on one of the given categories. 

Visualize contains the code that visualized the results of the different categories. Given a list of the post ids seperated by category defined in the testing BM25 notebook. The average success at k is calculated for each category and the plt bar chart is used to visualize the results.



The github for Faiss can be found here: https://github.com/facebookresearch/faiss
The github for the SBERT model used is https://github.com/UKPLab/sentence-transformers
The github for the BM25s model used is https://github.com/xhluca/bm25s  
