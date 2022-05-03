# About the data

+ **dataAPI.csv**: Is the all data returned in using the `S2paperAPI()` with the library S2search using the [API](https://api.semanticscholar.org/graph/v1) of Semantic Scholar.

+ **Data(total-citations).json**: Is the all data returned in using the `S2paperWeb()` with the library S2search using the [Semantic Scholar search page engine](https://www.semanticscholar.org/).

+ **SLR_forgetset.csv**: The revisions that were missing from the dataset with the columns `'index', 'abstract', 'title'` and `'decision'`.

+ **SLR-dataset-pico.xlsx**: The main dataset with the reviews.

+ **SLRdata_cleaned_augmentation.csv**: The old cleaned data with data augmentation by colum with name:

    + "text": Concatenation of title and abstract.
    + "text2" Divide the summary into sentences and cut them by the first and last sentence in a sentence number division by 3.
    + "text3": Is the concatenation of title and 'text2'.
    + "text4": Is the middles sentences of 'text2'.
    + "text5": Is the concatenation of title and 'text4'.
    + "text6": It's a summary of the abstract using the T5 model with temperature equal to 50\%.
    + "text7":It is an abstract word insertion using the BERT-base-uncased model.
    + "text8": It is an abstract word replacement using the BERT-base-uncased model.