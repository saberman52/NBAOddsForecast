import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

class HiFOPredict:
    '''
    The HiFOPredict class predicts NBA game win probabilities using a
    logistic regression model trained on 2001-02 to 2020-21 data. The model
    parameters and team statistic means/PCA basis are hard-coded here.
    '''
    
    '''CLASS VARIABLES'''
    # dictionary of team names to abbreviations in BR URLs (based on 2021-22 season)
    teamNameKey = {'Atlanta Hawks' : 'ATL',
              'Boston Celtics' : 'BOS',
              'Brooklyn Nets' : 'NJN',
              'Charlotte Hornets' : 'CHA', 
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
              'Miami Heat' : 'MIA',
              'Milwaukee Bucks' : 'MIL',
              'Minnesota Timberwolves' : 'MIN',
              'New Orleans Pelicans' : 'NOH',
              'New York Knicks' : 'NYK',
              'Oklahoma City Thunder' : 'OKC',
              'Orlando Magic' : 'ORL',
              'Philadelphia 76ers' : 'PHI',
              'Phoenix Suns' : 'PHO',
              'Portland Trail Blazers' : 'POR',
              'Sacramento Kings' : 'SAC',
              'San Antonio Spurs' : 'SAS',
              'Toronto Raptors' : 'TOR',
              'Utah Jazz' : 'UTA',
              'Washington Wizards' : 'WAS'}
    
    # logistic regression coefficients
    w = np.array([-0.3840704 , -0.01434259, -0.03438915,  0.058654  , -0.00311654,
       -0.11415171,  0.02359393, -0.01524577,  0.01094597,  0.02315544,
       -0.06196125,  0.00262423,  0.12908761, -0.02927948,  0.01771595])
    
    # means for each statistical category
    statMean = np.array([ 37.67788945,  82.94857621,   0.45416248,   7.6681742 ,
        21.46046901,   0.35549079,  30.01524288,  61.48693467,
         0.48955276,  18.09296482,  23.83366834,   0.75986767,
        10.97889447,  31.65527638,  42.63031826,  22.12512563,
         7.58308208,   4.88425461,  14.41055276,  20.84706868,
       101.12026801,  37.68492462,  82.94907873,   0.4541474 ,
         7.66834171,  21.45862647,   0.35648409,  30.01490787,
        61.48944724,   0.48927973,  18.09564489,  23.83433836,
         0.75940704,  10.97554439,  31.65661642,  42.63350084,
        22.12579564,   7.58442211,   4.88509213,  14.40871022,
        20.84572864, 101.12713568])
    
    # PCA basis
    PCABasis = np.array([[-1.25891739e-01,  1.17113673e-01,  4.16749492e-02,
        -5.67605167e-02, -2.09278425e-01,  1.55117145e-01,
        -1.67320966e-01],
       [-2.11566440e-01,  2.07785347e-01, -4.96418860e-02,
        -2.08234093e-01,  3.72990567e-03,  3.01820692e-01,
         2.32981406e-01],
       [-3.52156558e-04,  2.73363084e-04,  7.79591334e-04,
         4.71495547e-04, -2.53032510e-03,  2.02950153e-04,
        -3.32325960e-03],
       [-1.66281818e-01, -1.16306862e-01,  1.21650035e-01,
         1.21116268e-03,  3.46076588e-02,  9.42851387e-02,
        -2.15439778e-02],
       [-4.50657361e-01, -3.00713379e-01,  2.60249293e-01,
         1.17175352e-03,  1.85914548e-01,  1.84664250e-01,
         1.13299422e-01],
       [-2.95931858e-04, -4.20821357e-04,  1.47087155e-03,
         6.03016896e-05, -1.21986296e-03,  1.41160637e-03,
        -2.89117494e-03],
       [ 4.05540896e-02,  2.33413554e-01, -8.01139139e-02,
        -5.82855381e-02, -2.44382843e-01,  6.14080767e-02,
        -1.47570189e-01],
       [ 2.39054116e-01,  5.09105344e-01, -3.09219246e-01,
        -2.09304035e-01, -1.81882708e-01,  1.16997325e-01,
         1.19384488e-01],
       [-1.30887051e-03, -2.94475953e-04,  1.20994174e-03,
         7.44615541e-04, -2.53027735e-03,  4.96394611e-05,
        -3.25252381e-03],
       [ 2.33997384e-02,  9.85625842e-02,  1.12174019e-01,
         3.12637844e-01, -1.03219354e-01, -2.61955047e-01,
         1.00275327e-01],
       [ 4.47206296e-02,  1.20908574e-01,  1.43028826e-01,
         4.14008049e-01, -1.05787585e-01, -3.70484065e-01,
         2.27085181e-01],
       [-4.58838329e-04,  2.54290233e-04,  2.02451765e-04,
        -5.42915266e-05, -1.05072402e-03,  8.47429349e-04,
        -3.05386738e-03],
       [ 4.09116527e-02,  7.81955677e-02, -8.00132157e-03,
         6.84515737e-03, -1.41763026e-02,  4.43608559e-03,
         1.95299662e-01],
       [-1.11348130e-01, -1.37992740e-02,  8.52441170e-02,
        -1.19306842e-01, -1.72348067e-01, -3.08300097e-02,
         3.13121272e-01],
       [-7.04350585e-02,  6.40574486e-02,  7.74233359e-02,
        -1.12467911e-01, -1.87227336e-01, -2.64566651e-02,
         5.08498599e-01],
       [-8.00948139e-02,  5.03018453e-02,  4.17156187e-02,
        -7.06667754e-02, -1.91111273e-01,  1.95938327e-01,
        -1.43783458e-01],
       [-4.58802141e-03,  7.73912994e-03, -1.83594656e-02,
         2.04972979e-02, -3.52211704e-02,  7.59772898e-02,
         2.16037247e-02],
       [ 3.27195731e-03,  1.57697245e-02,  3.80944587e-02,
        -1.76731936e-02, -5.49938827e-02,  6.13268541e-03,
         6.74784885e-02],
       [ 1.02173529e-02,  3.63819254e-02, -8.51755496e-03,
         7.43030244e-02,  6.94114924e-02, -6.31883163e-02,
         1.24143425e-01],
       [ 3.23651324e-02,  5.79558116e-02, -3.99043045e-02,
         2.70105246e-01,  6.98442981e-02,  2.21026445e-01,
         1.49154296e-01],
       [-3.94593558e-01,  2.17009108e-01,  3.17313219e-01,
         1.99301424e-01, -4.85549490e-01,  1.44949443e-01,
        -2.54427822e-01],
       [-1.30217488e-01,  1.70333470e-01,  3.84096558e-03,
        -5.61361669e-02,  1.61059348e-01, -1.74350876e-01,
        -1.60686627e-01],
       [-2.22916646e-01,  2.21583466e-01,  1.81306436e-01,
        -2.14740481e-01,  2.89823450e-02, -2.22434847e-01,
         1.73378911e-01],
       [-3.47751389e-04,  8.59453178e-04, -9.38727818e-04,
         4.93962974e-04,  1.77830674e-03, -8.70903156e-04,
        -2.91785132e-03],
       [-1.48420013e-01, -1.67158512e-02, -1.63286409e-01,
         1.12891765e-03, -3.48444858e-02, -9.23720797e-02,
         2.12453505e-02],
       [-4.04832191e-01, -7.53636774e-02, -3.95611155e-01,
        -2.99786781e-02, -1.61373265e-01, -2.13768887e-01,
         1.41210745e-01],
       [-1.71278917e-04,  5.45725801e-04, -9.68038530e-04,
         7.06163188e-04,  1.08450259e-03, -6.84025944e-04,
        -1.41732765e-03],
       [ 1.80829156e-02,  1.86783625e-01,  1.67125308e-01,
        -5.76035071e-02,  1.96432625e-01, -8.16031910e-02,
        -1.81399422e-01],
       [ 1.82009174e-01,  2.97330950e-01,  5.77031711e-01,
        -1.85211291e-01,  1.91238933e-01, -9.04402489e-03,
         3.28997462e-02],
       [-1.20549517e-03,  6.39505509e-04, -1.91364957e-03,
         4.78771743e-04,  1.64680969e-03, -1.29182094e-03,
        -3.17796071e-03],
       [ 2.70926605e-02,  9.19927516e-02, -5.17242768e-02,
         3.12750007e-01,  1.01545788e-01,  3.04314381e-01,
         9.06125963e-02],
       [ 4.78352233e-02,  1.17905241e-01, -5.37684846e-02,
         4.05907864e-01,  1.31429205e-01,  4.03629383e-01,
         1.41105140e-01],
       [-4.04717145e-04,  9.19540916e-05, -4.89990829e-04,
         1.86091865e-04,  4.99712075e-05, -1.23122696e-04,
        -6.68269760e-04],
       [ 3.00669885e-02,  7.83059661e-02,  7.39656546e-02,
         3.14339611e-04,  6.55921009e-02,  1.10495161e-02,
         4.82721432e-03],
       [-1.17015023e-01,  1.75412842e-02, -4.99861619e-02,
        -1.19259711e-01,  2.05375610e-01,  1.01529939e-01,
         1.75582143e-01],
       [-8.68134269e-02,  9.57875204e-02,  2.31558466e-02,
        -1.19425381e-01,  2.71131439e-01,  1.12583510e-01,
         1.81364604e-01],
       [-8.38339393e-02,  1.11545072e-01, -1.26604751e-01,
        -2.83661820e-02,  1.38822135e-01, -1.40356680e-01,
        -3.75809802e-02],
       [-4.80809446e-03,  3.02057893e-02,  1.36566218e-03,
         4.73197035e-03,  4.64760904e-02, -2.58905822e-02,
         6.91374688e-02],
       [ 2.87858835e-03,  3.21373180e-02, -4.17035375e-02,
         8.34704800e-03,  3.96671550e-02, -3.89856802e-02,
         8.28026361e-02],
       [ 7.43176159e-03,  1.33898492e-02, -3.50575705e-02,
         7.49028470e-02, -1.34117479e-02,  1.41030632e-01,
         1.01824699e-02],
       [ 2.95723125e-02,  5.10775154e-02,  7.63156060e-02,
         2.46217149e-01, -5.98751771e-02, -1.39310054e-01,
         1.58319098e-01],
       [-3.81844433e-01,  4.16659663e-01, -2.07678749e-01,
         2.00577342e-01,  3.88600549e-01, -1.34602976e-01,
        -2.08618352e-01]])
    
    '''CONSTRUCTOR'''
    def __init__(self,season,date=pd.to_datetime('today').normalize()):
        '''
        season - second year of current season (e.g. 2022 for 2021-22 season)
        date - pd Timestamp with date to do predictions (defaults to today)
        '''
        # extract team stats and record
        self.season = str(season)
        self.date = date
        self.dataDict, self.teamWL = self._extractStats()
        
        # extract days games
        self.upcoming = self._extractDaysGames()
        
        
    '''CONSTRUCTOR METHODS'''
    def _extractStats(self):
        '''
        Outputs:
        dataDict
        teamWL - pd DataFrame
        '''
        dataDict = {}
        teamList = []
        teamW = []
        teamL = []
        brTeam = 'https://www.basketball-reference.com/teams/'
        offStats = '/stats_per_game_totals.html'
        defStats = '/opp_stats_per_game_totals.html'
        
        for team in self.teamNameKey:
            print(team)
            # extract offensive stats and WLs
            url = brTeam+self.teamNameKey[team]+offStats
            offTableList = pd.read_html(url,flavor='bs4')     #  THIS FUNCTION CALL TAKES THE LONGEST
            offTable = offTableList[0]
            latestSeason = offTable.iloc[0] # most recent season
            offNumbers = (latestSeason['FG':'PTS'].map(float)).to_numpy() # convert to float, store the numbers
            
            # Ws and Ls
            teamList.append(self.teamNameKey[team])
            teamW.append(latestSeason['W'])
            teamL.append(latestSeason['L'])

            
            # extract defensive stats
            url = brTeam+self.teamNameKey[team]+defStats
            defTableList = pd.read_html(url,flavor='bs4')
            defTable = defTableList[0]
            latestSeason = defTable.iloc[0] # most recent season, convert to floart
            defNumbers = (latestSeason['FG':'PTS'].map(float)).to_numpy() # convert to float, store the numbers
            
            # convert to PCA vector
            teamData = np.hstack([offNumbers,defNumbers])
            teamDataZero = teamData - self.statMean
            teamPCA = np.dot(teamDataZero,self.PCABasis)
            dataDict[self.teamNameKey[team]] = teamPCA
        
        teamWL = pd.DataFrame({
            'Tm' : teamList,
            'W' : teamW,
            'L' : teamL
        })
        
        return dataDict,teamWL
    
    def _extractDaysGames(self):
        '''
        Outputs:
        daysGamesComp - pd DataFrame containing columns ('Date','Start (ET)','Visitor','Home') of day's games
            Date - game date
            Start (ET) - start time
            Visitor - abbreviation of visiting team
            Home - abbreviation of home team
        '''
        # ASSUMING THERE WON'T BE ANY TIMEZONE CONFLICT ON TWITTER SERVER
        month = self.date.month_name().lower()
        # form URL for this month's games
        url = 'https://www.basketball-reference.com/leagues/NBA_' + self.season + '_games-' + month + '.html'
        # check that the webpage is good
        urlTest = requests.head(url)
        if urlTest.status_code != 200:
            print('From',url,'unexpected status code',urlTest.status_code)
            return pd.DataFrame() # return empty data frame
        
        # read table
        gameTable = pd.read_html(url,flavor='bs4')[0]
        # find day's games
        gameDates = pd.to_datetime(gameTable['Date'])
        daysGames = gameTable[gameDates == self.date]
        if daysGames.empty:
            print('No games',self.date.strftime('%Y-%m-%d'))
            return pd.DataFrame() # return empty data frame
        
        daysGamesComp = pd.DataFrame({'Date' : daysGames['Date'],
                                     'Start (ET)' : daysGames['Start (ET)'],
                                     'Visitor' : daysGames['Visitor/Neutral'].map(self.teamNameKey),
                                     'Home' : daysGames['Home/Neutral'].map(self.teamNameKey)})
        return daysGamesComp
    
    '''
    PUBLIC METHODS
    '''
    def predict(self):
        '''
        Generates predictions of win probability for the day's games, based on latest team reg. season-average stats.
        '''
        if self.upcoming.empty:
            return pd.DataFrame()
        
        # predictor variables
        xVis = np.stack(self.upcoming['Visitor'].map(self.dataDict))
        xHom = np.stack(self.upcoming['Home'].map(self.dataDict))
        x = np.hstack((np.ones((xVis.shape[0],1)),
                       xVis,
                       xHom)) # prediction variables
        
        # make probability prediction
        sigma = lambda a : 1./(1. + np.exp(-a))
        visWinProbability = sigma(np.dot(x,self.w))
        
        # convert probabilities into money line odds
        convertPtoMLO = lambda p : np.round((-1.)**(p > 0.5) * 100 * ((1-p)/p)**((-1)**(p > 0.5)))
        visMLO = convertPtoMLO(visWinProbability)
        homMLO = -visMLO
        
        # build prediction table
        predictions = pd.DataFrame({
            'Date' : self.upcoming['Date'].to_list(),
            'Visitor' : self.upcoming['Visitor'].to_list(),
            'Home' : self.upcoming['Home'].to_list(),
            'Visitor win probability' : visWinProbability,
            'Visitor Line' : visMLO.astype(int),
            'Home Line' : homMLO.astype(int)
            
        })
        return predictions


if __name__ == '__main__':
  obj1 = HiFOPredict(2022)
  predicTable = obj1.predict()
  print(predicTable)