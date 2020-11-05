This is an automatic Youtube news reader that helps you read a certain Youtube video transcripts, in a format of news article. What's more, it also generate a one-line summarization using a well-trained machine learning a model.


For example:

When we want to watch news like this: https://www.youtube.com/watch?v=3Rzi11Hvyh0

You can type "python3 youtube_reader.py --youtube_id 3Rzi11Hvyh0"

It will generate the whole article, and then give a summarization in one line:

Tonight the sprint to the finish the election now less than two days away president trump's ...

And the summarization will be:

President trump's final rallies in five states today pushing the way to midnight this could be the most important election in our country's history.


To use it:

1. Install Python3.7+;
2. Download all the files in this repo;
3. Install all the libraries in the requirements file;
4. Download a model from here: https://drive.google.com/file/d/1FVzZto4bf5_TCmRy3tNeirhPDdLrvum5/view (Trained based on Google's Pegasus NLP framework)
5. Run the command by typing command like "python3 youtube_reader.py --youtube_id 3Rzi11Hvyh0". Note: it might take minutes to generate the final summarization.

Have fun!
