import pandas as pd
from bs4 import BeautifulSoup
import requests

def extractMonthURLs(seasonURL,brURL):
    """
    extractMonthURLs extracts the URLs for BR data of each month of NBA play for a given season,
    whose homepage is seasonURL. Checks that each of the extracted URLs works before returning.
    
    Inputs:
        seasonURL - BR URL containing the links to each month of the season's games
        brURL - https://www.basketball-reference.com
    Outputs:
        monthURLs - links to each month of this season's games
        monthNames - month names
        goodLink - boolean array indicating whether each URL works or not
    """
    webpage = requests.get(seasonURL).text # send http request, store html in 'webpage'
    # create BS object
    soup = BeautifulSoup(webpage,'html5lib')
    # find the tag with class = 'filter'
    months = soup.find_all(class_='filter')

    # extract month URLs and check links
    monthURLs,monthNames,goodLink = processDivTag(months,brURL)

    return monthURLs,monthNames,goodLink

def processDivTag(months,brURL):
    '''
    processDivTag extracts month names and links from months, a BS tag object.

    Inputs:
        months - BS tag object
    Outputs:
        monthURLs - links to each month of this season's games
        monthNames - month names
        goodLink - boolean array indicating whether each URL works or not
    '''
    # check that there is a single tag with class filter
    if len(months) != 1:
        print('Warning: More than one HTML tag with class "filter". Proceeding assuming first tag contains month URLs.')
    
    # iterate over div tags and extract months and links
    monthURLs = []
    monthNames = []
    goodLink = [] # array to store checks (bools) on whether the links are good
    firstDivTag = months[0].div
    sibling = firstDivTag
    while sibling is not None: # sequentially visit all siblings of first div tag until end of this level of tree
        # check that this sibling is a div tag
        if sibling.name != 'div':
            print('Warning: sibling of div tag not a div tag, skipping and continuing.')
            sibling = sibling.next_sibling
            continue
        link = sibling.a
        # extract month names
        monthNames.append(link.text)
        print(monthNames[-1])
        monthURLs.append(brURL+link.get('href'))
        # check that the link works
        r = requests.head(monthURLs[-1])
        goodLink.append(r.status_code == 200)
        if not goodLink[-1]:
            print('Warning: Bad link. Status code',r.status_code)
            
        sibling = sibling.next_sibling
    return monthURLs,monthNames,goodLink

'''
teamNameKey: global variable giving mapping from team names to 3 letter abbreviations
'''
teamNameKey = {'Atlanta Hawks' : 'ATL',
              'Boston Celtics' : 'BOS',
              'Brooklyn Nets' : 'BRK',
              'New Jersey Nets' : 'NJN', # 1977-78 to 2011-12
              'Charlotte Hornets' : 'CHO', 
              'Charlotte Bobcats' : 'CHA', # 2004-05 to 2013-14
              'Chicago Bulls' : 'CHI',
              'Cleveland Cavaliers' : 'CLE',
              'Dallas Mavericks' : 'DAL',
              'Denver Nuggets' : 'DEN',
              'Detroit Pistons' : 'DET',
              'Golden State Warriors' : 'GSW',
              'Houston Rockets' : 'HOU',
              'Indiana Pacers' : 'IND',
              'Los Angeles Clippers' : 'LAC',
              'Los Angeles Lakers' : 'LAL',
              'Memphis Grizzlies' : 'MEM',
              'Vancouver Grizzlies' : 'VAN', # 1995-96 to 2000-01
              'Miami Heat' : 'MIA',
              'Milwaukee Bucks' : 'MIL',
              'Minnesota Timberwolves' : 'MIN',
              'New Orleans Pelicans' : 'NOP',
              'New Orleans/Oklahoma City Hornets' : 'NOK', # 2005-06 to 2006-07
              'New Orleans Hornets' : 'NOH', # 2002-03 to 2012-13
              'New York Knicks' : 'NYK',
              'Oklahoma City Thunder' : 'OKC',
              'Seattle SuperSonics' : 'SEA', # 1967-68 to 2007-08
              'Orlando Magic' : 'ORL',
              'Philadelphia 76ers' : 'PHI',
              'Phoenix Suns' : 'PHO',
              'Portland Trail Blazers' : 'POR',
              'Sacramento Kings' : 'SAC',
              'San Antonio Spurs' : 'SAS',
              'Toronto Raptors' : 'TOR',
              'Utah Jazz' : 'UTA',
              'Washington Wizards' : 'WAS',
              'Playoffs' : 'PLY'} # 'Playoffs' designation appears in every field in table

def convertWL(raw_data):
    '''
    convertWL takes a pandas DataFrame extracted from a BR webpage, and returns a DataFrame with
    a W or L assigned to each game (in the form of True/False if Visitor wins), and maps
    spelled-out team names onto the team abbreviation. If a non-compliant team name is found, 
    returns None.

    Inputs:
    raw_data :  pd DataFrame extracted from a BR webpage

    Outputs:
    my_table :  pd DataFrame containing the date, visitor and home team name abbrevations, and a
                True or False if visitor won.
    '''
    # names for the visitor and home columns
    vis = 'Visitor/Neutral'
    home = 'Home/Neutral'
    my_table = raw_data.reindex(columns=['Date',vis,home,'VisitorWin'])
        
    # assign win or loss for visitor (True or False)
    my_table['VisitorWin'] = raw_data['PTS'] > raw_data['PTS.1'] 
    
    # map team names to abbreviations
    for col in [vis,home]:
        my_table[col] = my_table[col].map(teamNameKey)
        
    # check if any team names didn't map to abbreviations
    for col in [vis,home]:
        if pd.isna(my_table[col]).any():
            print('Error: A',col,'team had a name not matching any keys.')
            return # exit if this is the case

    # drop rows containing PLY
    my_table.where(my_table != 'PLY',inplace=True) # replace 'PLY' with NaN
    my_table.dropna(inplace=True)
    
    return my_table

def extractMonthsGames(url):
    '''
    extractMonthsGames takes a BR url for a season's month's games, and extracts a table which 
    records each team's matchups and the game outcome.

    Inputs: 
    url :               the URL of BR webpage for a specific month of a specific season

    Ouputs:
    processedTable :    Table containing each matchup and outcome (bool - did visitor win?). If
                        one of the team names did not match my keys, then this will be None.
    '''
    # read table from url
    table_list = pd.read_html(url,flavor='bs4')
    if len(table_list) > 1: # check length
        print('Warning: more than one table extracted from',url)
        print('Proceeding with first table.')
        
    processedTable = convertWL(table_list[0])
    return processedTable

def extractSeasonsGames(season,brURL):
    '''
    extractSeasonsGames scrapes BR webpages to extract the outcomes (win or loss) of 
    all NBA games of a particular season. They are returned in a single pd DataFrame.

    Inputs:
        season : string with second year of season (e.g. for 2000-2001 season, '2001')
        brURL :  https://www.basketball-reference.com

    Outputs: (in the event of anassumption not being met, returns None)
        seasonTable : pd DataFrame containing all the games and outcomes for single NBA season
        missingMonths : bool indicating if any month's data was excluded from the table
    '''
    seasonURL = brURL + '/leagues/NBA_' + season + '_games.html'
    # check that the webpage is good
    urlTest = requests.head(seasonURL)
    if urlTest.status_code != 200:
        print('From',seasonURL,'unexpected status code',urlTest.status_code)
        return None,None #expects two outputs
    
    # get URLs for each month of games
    monthURLs,monthNames,goodLink = extractMonthURLs(seasonURL,brURL)
    
    # loop through months to get data for each month
    monthTables = [] # array for storing season's month's tables
    for i,url in enumerate(monthURLs):
        if not goodLink[i]:
            print('Bad link for',monthNames[i],':',url)
            print('Skipping and proceeding.')
            continue
        else:
            processedTable = extractMonthsGames(url)
            if processedTable is not None:
                monthTables.append(processedTable)
            else:
                print('Excluding',monthNames[i])
    
    # check if any months were excluded
    missingMonths = len(monthTables) < len(monthNames)
    
    # combine all months into single DataFrame
    if len(monthTables) > 0:
        seasonTable = pd.concat(monthTables)
    else:
        print('No months loaded')
        seasonTable = None
    
    return seasonTable,missingMonths

def downloadGameData(initialSeason,finalSeason):
    '''
    Inputs:
    initialSeason : int indicating starting season to get data (second year of season e.g. 2000 for 1999-2000 season)
    finalSeason : int indicating final seaon to get data for
    '''
    brURL = 'https://www.basketball-reference.com'
    seasonTables = [] # to store tables from each season
    badSeasons = [] # to store seasons for which downloading data failed to meet all assumptions
    
    # loop over and download each season's data
    for season in range(initialSeason,finalSeason+1):
        print('Dowloading',season,'season data...')
        table,missingMonths = extractSeasonsGames(str(season),brURL)
        # case: no table returned
        if table is None:
            badSeasons.append(season)
        else:
            # store table
            seasonTables.append(table)
            table.to_hdf('pyData/games'+str(season)+'.h5','table',mode='w')
            # case: there was a missing month
            if missingMonths:
                badSeasons.append(season)
        print('##################################')
    
    # if all data was downloaded perfectly, combine into single pd DataFrame and save it
    if len(badSeasons) == 0:
        print('Saving all data to a single table!')
        allGameData = pd.concat(seasonTables)
        allGameData.to_hdf('pyData/allGames'+str(initialSeason)+'_'+str(finalSeason)+'.h5','allGameData',mode='w')

if __name__ == '__main__':
    downloadGameData(2001,2021)


