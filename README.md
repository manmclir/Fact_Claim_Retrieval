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


 
UploadSemEval_First_Submission_Raaj_Flores.ipynb is the final implementation of sentence BERT. It should be noted that it is required to install sentence-transformers and faiss-gpu and make sure all the csv files that are fact_checks.csv. pairs.csv and posts.csv are in the same working directory. The libraries pandas, numpy, matplotlib, sentence_transformers, ast, faiss, torch and os should be imported. Also, the notebook must run with GPU. Python version 3.x should be used.

The following Python definitions were included in the notebook for building the SBERT model. 

1. **get_embedding** --> It generates an embedding vector for a given text using a Sentence Transformer model.
2. **get_post_text** --> Retrieves text from a Pandas DataFrame that contains social media posts by selecting the 'ocr' or 'text' elements. Assuming the data is saved as a list or tuple, it iterates through each row, extracting the value of the designated column and parsing it using ast.literal_eval(). An empty string is appended if the value is invalid or empty; if not, the function retrieves the second element from the list's first tuple. This is useful for preparing posts before additional analysis because it guarantees that only relevant text data is retrieved. A list post_text as extracted text.
3. **seq_empty** --> This removes rows from a Pandas DataFrame that are empty in the given column. Each row is iterated through, and non-empty entries are gathered into a new DataFrame (sep_df) and then returned. This is done to separate important information from missing entries.
4. **clean_Nan** --> To ensure there is no issue while text processing, it replaces Nan Values with empty strings and returns the modified and cleaned dataframe.
5. **get_embeddings_in_batches** --> It generates embeddings for a list of text using batch processing with a batch size of 512. Important for embedding a large number of texts efficiently in GPU. It addresses memory constraints and optimizes processing speed through batching. It concatenates the embeddings from all batches into a single NumPy array.
6. **build_faiss_index** --> This creates a FAISS index to enable fast similarity searches within a set of embeddings. It first initializes an index suitable for inner product similarity to get cosine similarity, but for this to happen, it is a must that the method normalizes the input embeddings to unit length. Finally, it adds the normalized embeddings to the FAISS index, which pre-processes them for efficient Exact nearest neighbour lookups since IndexFlatIP is being used. It prepares a data structure that allows one to quickly find embeddings that are most similar to a given query embedding by returning the index.
7. **search_faiss** --> It takes in the index built by **build_faiss_index**, query embeddings and desired top_k value in this implementation top_k=10. After normalizing the query embeddings, it uses the FAISS index to find top_k most similar embeddings in the index to the query embeddings. Finally, it returns the distances, that is, the similarity scores and indices of the found nearest neighbours within the original embedding array.
8. **visualize_metrics** --> It is essentially, visualizing histogram of Success@k for different k values in one plot, and in the other plot, it plots the evolution of success@k at different values of k and MRR score.
9. **pipeline** --> It implements the above discussed definition to get resuslts from them. For result mapping, it maps the search results that is the distances and indices back to the original dataframes, creating a new dataframe containing the post_id, fact_check_id, and similarity score for each retrieved fact-check. For every post, the code generates a record linking the ID of the post to the ID of each of its top_k fact-check matches and their similarity scores after iterating through the distances and indices that FAISS returned. It then calculates the MRR score and Success@k for different k values and then visualizes the results by using def **visualize_metrics**. Finally, it saves the detailed results DataFrame to a CSV file called faiss_results_full.csv containing post_id, fact_check_id and similarity scores corresponding to them.

Finally, the pipeline is run with appropriate parameters to get all the results from the notebook. 


BM25 is where the baseline results for the BM25s model was generated along with the category splits that is used in the visualize notebook. 

The functions follow a similar pattern to how SBERT works in SemProj Flores Raaj where in this case callBM is used to call the BM25s model to generate the embeddings and to rank the similarity scores of the facts and posts. The new function in this note book is the get important range function takes a list of character counts and generates x ranges for each bin. afterwards a csv file is created that contains the post id for all of the posts that belong to the new bin. The undefined function is reused in get lang and get source which extracts the language/source for each posts and saves a csv file with the post ids split based on one of the given categories. 

Visualize contains the code that visualized the results of the different categories. Given a list of the post ids seperated by category defined in the testing BM25 notebook. The average success at k is calculated for each category and the plt bar chart is used to visualize the results.



The github for Faiss can be found here: https://github.com/facebookresearch/faiss
The github for the SBERT model used is https://github.com/UKPLab/sentence-transformers
The github for the BM25s model used is https://github.com/xhluca/bm25s  
