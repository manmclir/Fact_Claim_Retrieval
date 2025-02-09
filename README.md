Visualize contains the code that visualized the results of the different categories. 
Upload SemEval contains the code that generated the results submitted to the challenge website.
SemProj Raaj Flores was the notebook where the results with SBERT were generated.
BM25 testing was the notebook where the results for BM25s were generated.

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
