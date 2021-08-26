import pandas as pd
from bs4 import BeautifulSoup
import requests
import BRWebscrapeTools as br

def testExtractMonthURLs():
    '''
    testExtractMonthURLs tests the BRWebscrapeTools function extractMonthURLs().
    First test is on actual BR webpage with no flags on it.
    Second test is on an HTML string specifically written to trigger the warnings
    of the function.
    '''
    # first, test on actual BR webapge, where output should be normal with no warnings
    url = 'https://www.basketball-reference.com/leagues/NBA_2019_games.html'
    baseURL = 'https://www.basketball-reference.com'
    print('First test: actual url',url)
    monthURLs,monthNames,goodLink = br.extractMonthURLs(url,baseURL)
    print('month Links \n',monthURLs)
    print('good links \n',goodLink)

    # second test: make sure HTML tag processing works properly
    # edits:
    # inserted space after December div tag
    # dropped '.html' from end of January link
    htmlText = """<div class="filter">

<div class=" current">
    <a href="/leagues/NBA_2021_games-december.html">December</a>
</div> <div class="">
    <a href="/leagues/NBA_2021_games-january">January</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-february.html">February</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-march.html">March</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-april.html">April</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-may.html">May</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-june.html">June</a>
</div><div class="">
    <a href="/leagues/NBA_2021_games-july.html">July</a>
</div></div>"""
    buggyTag = BeautifulSoup(htmlText,'html5lib')
    months = buggyTag.find_all(class_ = 'filter')
    monthURLs,monthNames,goodLink = br.processDivTag(months,baseURL)
    print('month Links \n',monthURLs)
    print('good links \n',goodLink)

if __name__ == '__main__':
    testExtractMonthURLs()

