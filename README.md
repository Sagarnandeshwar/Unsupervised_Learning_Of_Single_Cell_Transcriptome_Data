# Unsupervised Learning Of Single Cell Transcriptome Data
![0](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/0_1.png)
![0](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/0_2.png)
![0](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/0_3.png)
![0](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/0_4.png)

## Dataset: Mouse Pancreas single-cell RNA-sequencing data 
The dataset has 1886 cells measured over 4194 genes by scRNA-seq. The dataset were processed and filtered for the purpose of this assignment. Here are the individual files: 
- MP.pickle: The single-cell expression count matrix (matrix of integers) of dimension 1886 cells by 4194 genes plus 2 columns indicating the batch IDs and cell types 
- MP_genes.pickle: gene names for the 4194 genes 
- sample_info.csv: a 1886 by 3 information matrix with the rows as the 1886 cells and columns for the cell IDs, batch IDs, and cell type labels 
- cell_IDs.pkl: cell IDs as a list of strings 

## Evaluating clustering by adjusted Rand Index 
Adjusted Rand Index is a popular metric used to evaluate the unsupervised clustering by comparingthe consistency between the predicted clusters and the ground-truth clusters (i.e., cell type labels in this assignment).  

In the source code file, there is a function called evaluate_ari(cell embed, adata), which takes N x K input cell embedding cell embed with K as the embedding dimensions and the annotated cell label data as adata.  

It will first run UMAP to compute the distance between cells based on their embeddings and then run Louvain clustering using the cells-cells distance matrix from UMAP to cluster cells into groups defined by the resolution parameter (default: 0.15). Finally, it computes the ARI based on the Louvain clusters and the ground-truth cell type using adjusted_rand_score from Scikit-learn.  

We use evaluate_ari to evaluate the cell embedding quality by NMF and scETM below. 

## Implemented non-negative matrix factorization (NMF) to model scRNA-seq counts by minimizing sum of squared reconstruction error 

![1](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/1_1.png)

Besides returning the final matrices W and H, save the mean squared error (MSE) MSE = ||X−WH||2/(N ∗M) and ARI at each iteration into a 3-column ndarray called perf with the first column as the iteration index and return perf as the third output from the function nmf_sse. 

We wan your NMF for 100 iterations and return W, H, and perf. 

## Monitor training progress 
Implement a function called monitor perf that displayed the SSE and ARI at each iteration from the above NMF-SSE training and save the plot as nmf_sse.eps 
![2](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/2_1.png)
![2](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/2_2.png)
We observe that the SSE drops quite rapidly at the first few iterations and the ARI increases in a zigzag way because it is an independent metric from the training objective. 

## Implement NMF to model scRNA-seq counts by maximizing log Poisson likelihood 
Implement the NMF function nmf psn that maximizes the log Poisson likelihood w.r.t. W and H s.t. W,H ≥ 0. Save the average log Poisson likelihood (X logWH −WH)/(N xM) and ARI at each iteration. 

![3](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/3_1.png)

The NMF algorithm requires element-wise division, so to avoid dividing by zeros, we can set the zero values to a small value like so: np.where(A > 0, A, 1e-16). 

![3](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/3_2.png)
![3](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/3_3.png)
![3](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/3_4.png)

Now, comparing the two models, we observe that the NMF-Poisson model led to higher ARI than the NMF-SSE model. This implies that the Poisson likelihood is a better objective function to model the discrete read counts of the scRNA-seq data than the SSE loss. The latter is equivalent to maximizing the log of a univariance Gaussian likelihood. 

## Train Embedding Topic Model (ETM) 
Python file etm.py contains a simpler strip-down version of the scETM. The code is the same as the original ETM implementation that is available from https://github.com/adjidieng/ETM. 

![4](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/4_1.png)

ARI is computed based on the Louvain clustering the cell embedding θ against the groundtruth cell types stored in the mp anndata object. 

Training a neural network (i.e., the VAE in our case) requires many iterations because of the gradient descent updates with small learning rate. We ran the model for 1000 iterations. Record the negative ELBO loss and ARI at each iteration and then use monitor perf to display the training progress. 

## Compare ETM with NMF-Poisson 
We ran NMF-Poisson also for 1000 iterations and compare the ARI scores with scETM over the 1000 iterations.

![5](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/5_1.png)
![5](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/5_2.png)
![5](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/5_3.png)
![5](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/5_4.png)

While the NMF model has converged to a local optimal after only 200 iterations, the scETM continues to improve. We observe some improvement from scETM over the NMF model especially after 200 iterations. This highlights the benefits of having the non-linear encoder function and perhaps the tri-factorization design in the scETM. When training on massive number of single-cell samples, using stochastic gradient training on minibatches, scETM will confer much bigger improvement over the linear model. For this particular dataset, with batch-effect correction, a fine-tuned scETM can reach 0.90 ARI 

## Generate t-SNE to visualize cell embeddings 

We use the model object of ETM saved from the previous training over the 1000 iterations to infer final cell topic embedding θ and generate the two-dimensional t-SNE plot. Also, use the H matrix from the NMF-Poisson model (trained after 1000 iterations) to generate another two- dimensional t-SNE plot.  

![6](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/6_1.png)
![6](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/6_2.png)

We observe a slightly better separation of the alpha cells from other cells from the scETM compared to the NMF-Poisson model. 

## Plot heatmap for the cells under each topic 
An alternative way to present the cell cluster is by heatmap. Heatmap is more effective in identifying which topics correlate well with which cell types.  

We generate a heatmap plot for the same cells-by-topics matrix θ using all of the 1886 cells over the 16 topics. From here, we see that topic 2 correlates well with alpha cell type, topic 12 with endothelial, topic 11 with ductal cell type, and so forth. 

![7](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/7.png)

## Plot heatmap for the top genes under each topic 

To get cell-type-specific gene signature, we can visualize the genes by topics heatmap. Here we will plot the top 5 genes per topic in heatmap 

In plotting the heatmap, I capped the max value at 0.2 instead of letting it set to 1 to make the red intensities more prominent for some of the genes with low absolute value under some of the topics 

Please note: The following observation from heatmap with no cap. We can see the lot of gene in each most topics have high probability. For topic, not many genes have shown high probability, for topics 7, 11 and 13, lot of genes have shown high probability. 

![8](https://github.com/Sagarnandeshwar/Unsupervised_Learning_Of_Single_Cell_Transcriptome_Data/blob/main/images/8.png)

 

 
