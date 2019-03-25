#### Python microservice for analyzing user submissions

All survey submissions by logged-in users are fed to this service.
A model is trained for each survey based on the data being fed.
When a random user without a registration submits a response to a survey, the service will make a prediction about their characteristics (age, gender, country of origin) based on the trained model for the specific survey.

##### How to run
Python version = 3.7

1. Create a virtual environment using `venv`
2. Go to the project root and execute `pip install -r requirements.txt`



#### Ideas

1. In the future, models can be trained cross-survey, meaning the submissions
to one survey would impact the ML model for all surveys. For instance, if 
two separate surveys have the same question ("What is your favorite pizza?"),
the answers of both surveys can be summed and used to train a common ML model.

