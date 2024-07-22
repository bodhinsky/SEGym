FROM python:3.12-bookworm

# Set the working directory to /app
WORKDIR /app

RUN pip install pytest numpy requests app certifi charset-normalizer freezegun contourpy cycler fonttools idna kiwisolver matplotlib packaging pandas pillow pyfiglet pyparsing python-dateutil python-dotenv pytz readchar requests-mock seaborn setuptools six tzdata urllib3
#RUN poetry export -f requirements.txt --output requirements.txt

# TODO: Install requirements.txt
#RUN pip install -r requirements.txt
