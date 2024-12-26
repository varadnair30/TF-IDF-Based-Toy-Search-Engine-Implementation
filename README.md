# TF-IDF-Based-Toy-Search-Engine-Implementation

In this assignment, you will implement a toy "search engine" in Python. You code will read a corpus and produce TF-IDF vectors for documents in the corpus. Then, given a query string, you code will return the query answer--the document with the highest cosine similarity score for the query. 

The instructions on this assignment are written in an .ipynb file. You can use the following commands to install the Jupyter notebook viewer. You can use the following commands to install the Jupyter notebook viewer. "pip" is a command for installing Python packages. You are required to use Python 3.5.1 or more recent versions of Python in this project. 

    pip install jupyter

    pip install notebook (You might have to use "sudo" if you are installing them at system level)

To run the Jupyter notebook viewer, use the following command:

    jupyter notebook P1.ipynb

The above command will start a webservice at http://localhost:8888/ and display the instructions in the '.ipynb' file.

### Dataset

We use a corpus of 30 Inaugural addresses of different US presidents. We processed the corpus and provided you a .zip file, which includes 30 .txt files.

Using the tokens, we would like to compute the TF-IDF vector for each document. Given a query string, we can also calculate the query vector and calcuate similarity.

![weighting_scheme](https://github.com/user-attachments/assets/bb3d98ed-4d09-4433-a609-521e777c1000)


The notation of a weighting scheme is as follows: ddd.qqq, where ddd denotes the combination used for document vector and qqq denotes the combination used for query vector.

A very standard weighting scheme is: ltc.lnc; where the processing for document and query vectors are as follows:
Document: logarithmic tf, logarithmic idf, cosine normalization
Query: logarithmic tf, no idf, cosine normalization
