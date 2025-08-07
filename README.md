# CECS 327 Project

This is my distributed systems and applications project for the summer section of 2025.

This project implements a distributed sentiment analysis system using Ray framework to compare six different machine learning approaches (LLM, API-based, Hugging Face models, VADER, and rule-based) on 1,000 Disneyland reviews. The system provides comprehensive performance analysis including execution time, memory usage, and model agreement rates with automated visualization generation.

## Contents
To run this program on your own machine, please refer to the demo, setup, and requirements below in this *README.md*

For the project **report**, presentation, and requirements, refer to the report folder in the repository.

For the demo video, please use the unlisted YouTube link in the *DEMO* section of this *README.md*.

For the project code and comments, refer to any three python programs; *main.py*, *tasks.py*, and *visualize.py*.

To view the raw data refer to the data folder (DisneylandReviews.csv was used for the report).

Sentiment analysis results can be viewed in the results folder in JSON format.

Generated graphs can be found in the *graphs* folder.

## Demo



## Setup
1. Clone/download the project
2. Install dependencies (see below)
3. Ensure you have the dataset file: `data/DisneylandReviews.csv`
4. Run: `python3 main.py`
5. Select either the Single (4 minutes - faster - recommended) or Average (12 minutes - better accuracy) 

## Requirements 
Python: 
3.8 or higher

Operating System: 
Any

IDE: 
PyCharm (recommend), 
Visual Studios Code

### Python libraries:

```shell
    pip3 install ray
    pip3 install pandas
    pip3 install numpy
    pip3 install matplotlib
    pip3 install psutil
    pip3 install transformers
    pip3 install torch
    pip3 install vaderSentiment
    pip3 install python-dotenv
```


