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

def testConvertWL():
    '''
    testConvertWL() tests the convertWL() function.
    '''
    # first test: download known good table from a BR webpage
    url = 'https://www.basketball-reference.com/leagues/NBA_2021_games.html'
    table_list = pd.read_html(url,flavor='bs4')
    processedTable = br.convertWL(table_list[0])
    print(processedTable.head())
    '''
    Expected Output:

        Date                Visitor/Neutral     Home/Neutral    VisitorWin
    0   Tue, Dec 22, 2020   GSW                 BRK             False
    1   Tue, Dec 22, 2020   LAC                 LAL             True
    2   Wed, Dec 23, 2020   CHA                 CLE             False
    3   Wed, Dec 23, 2020   NYK                 IND             False
    4   Wed, Dec 23, 2020   MIA                 ORL             False
    '''

    # second test: give a table with deliberately mispelled entry
    originalTable = table_list[0]
    tableMispell = pd.DataFrame(originalTable[0:1])
    tableMispell.at[1,'Visitor/Neutral'] = 'Los Angeles Clipers' # rename with a mispelled team name
    processedTable = br.convertWL(tableMispell)
    print('processedTable is None: ',processedTable is None)

def testExtractSeasonsGames():
    '''
    testExtractSeasonsGames tests the extractSeasonsGames() function.
    '''
    # test 1: for a webpage known to work, it correctly extracts the data
    print('test 1')
    brURL = 'https://www.basketball-reference.com'
    season = '2021'
    seasonTable,missingMonths = br.extractSeasonsGames(season,brURL)
    if seasonTable.shape == (1171,4):
        print('From webpage extracted table with correct shape')
    else:
        print('Table from webpage has incorrect shape')
    print('missingMonths =',missingMonths) # expected: false

    # test 2: bad season name, leading to broken link
    print('test 2')
    season = '202'
    seasonTable,missingMonths = br.extractSeasonsGames(season,brURL)
    if seasonTable is None and missingMonths is None:
        print('correct output returned')
    else:
        print('incorrect output returned')

    # note: there are several clauses that remain unvisited by this test



# run tests
if __name__ == '__main__':
    print('Test extractMonthURLs()')
    testExtractMonthURLs()
    print('###################################')
    print('Test convertWL()')
    testConvertWL()
    print('###################################')
    print('Test extractSeasonsGames()')
    testExtractSeasonsGames()


