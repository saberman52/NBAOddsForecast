NBAOddsForecast is a collection of data and code that implements a statistical model for forecasting the odds of NBA games based on historical data.
Given two teams (visitor and home) and their regular season performance as inputs, as measured by each team's averages of every conventional game statistic,
the model outputs the probability that the visiting team beats the home team.
All data used here has been obtained from basketball-reference.com.
Below is a summary of the main components of this project:

pyData - collection of data (hdf) containing every NBA game and outcome from the 2000-01 to 2020-2021 seasons, as output by the downloadGameData() function in BRWebsrapeTools.py

regSeasonData - collection of data (csv) containing each NBA team and their regular season average statistics, both on the offensive and defensive ends.

BRWebscrapeTest.py - code to test the functions in BRWebscrapeTools.py.

BRWebscrapeTools.py - code to web scrape basketball-reference.com and extract and store the outcomes of all NBA games in specified seasons.

loadSeasonData.ipynb - notebook which extracts team season average data from csv files and stores them in a pandas DataFrame.

LogisticRegression1.ipynb - notebook that trains a logistic regression model to reproduce the probabilities of NBA game outcomes given each team's regular season average data.

PCA_Analysis2.ipynb - notebook exploring the dimensional reduction of a team's regular season average data using principal component analysis.
