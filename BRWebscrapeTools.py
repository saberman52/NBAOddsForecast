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
