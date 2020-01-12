from flask import Flask, render_template,request
from sqlalchemy import create_engine
from collections import Counter
import networkx as nx
import pylab as plt

import pymysql
app = Flask(__name__)
class Database:
    def __init__(self):
        host = "35.232.130.107"
        user = "testing123"
        password = "testing123"
        db = "dblp"
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()
    def list_employees(self):
        self.cur.execute("SELECT name FROM dblp.for")
        row = self.cur.fetchall()
        return row
        #print(row)

    def list_journalid(self):
        self.cur.execute("select * from dblp.journal order by id ASC")
        trainingtable = self.cur.fetchall()
        return trainingtable

    def list_years(self):
        self.cur.execute("select distinct year from dblp.trainingtable order by year ASC")
        years = self.cur.fetchall()
        return years

    
@app.route('/test')
def test():
    return "Hello World"
    
    
@app.route('/')
def employees():
    authorsfor=""
    def db_query():
        db = Database()
        emps = db.list_employees()
        journals=db.list_journalid()
        years = db.list_years()
        return (emps,journals,years)
    res = db_query()
    naivejournalid = request.args.get("naivejournal")
    naiveyear = request.args.get("naiveyear")
    naiveclassify=request.args.get("naiveclassify")
    naiveforid=request.args.get("naiveforid")
    naiveforyear=request.args.get("naiveforyear")
    focus = request.args.get("focus")
    print(focus)
    xnumber = request.args.get("xnumber")
    print(xnumber)
    if xnumber:
        print("xnumber accepted")
        List = SearchCoauthorshipAndFor(xnumber, focus)
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, nodes=List[1], edges=List[0], content_type='application/json')
    query = request.args.get("authorname")
    print(query)
    if query !=None:
        authorsfor = get_authors_for(query)
    else:
        authorsfor=""
    journalid=request.args.get("journal")
    year=request.args.get("year")
    if journalid != None and year !=None:
        pred=getPrediction(journalid,year)
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, pred=pred,content_type='application/json')

    elif  naivejournalid !=None :
        naivepred = getnaivePrediction(naivejournalid, naiveyear)
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, naivepred=naivepred,content_type='application/json')

    elif  naiveclassify !=None :
        naiveclassify = getnaiveClassification(naiveclassify)
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, naiveclassify=naiveclassify,content_type='application/json')

    elif  naiveforid !=None :
        naivefor = getnaivefor(naiveforid,naiveforyear)
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, naivefor=naivefor,content_type='application/json')

    else:
        pred = ""
        return render_template('GUI.html', row=res[0], journals=res[1], authorsfor=authorsfor, pred=pred,content_type='application/json')





    return render_template('GUI.html', row=res[0], journals=res[1], years=res[2], authorsfor=authorsfor,content_type='application/json')


def get_authors_for(formvalue):
    if formvalue:
        print(formvalue)
        print("inside get authors for")
        engine = create_engine("mysql+mysqlconnector://testing123:testing123@35.232.130.107/dblp")
        con = engine.connect()

        con.execute(" SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''))")

        query = "SELECT * FROM dblp.conferencedetails WHERE ACML LIKE %s"
        print(query)
        rs = con.execute("select id from authors where name=%s", (formvalue,))
        x = rs.fetchone()
        if x:
            # print(x[0])
            # x = 95
            rs1 = con.execute(
                "SELECT Count(journal.name) as `publications`,journal.name FROM journal,publication,authors_publications WHERE authors_publications.author_id = %s AND publication.id = authors_publications.publ_id AND publication.journal_id=journal.id GROUP BY journal.name",
                (x[0],))
            jcount = 0
            str = ''
            counterJournal = Counter()
            counterconference = Counter()
            count = 0
            for i in rs1:
                jcount += i[0]
                str = i[1].replace('.', '').replace(' ', '%') + '%'
                result1 = con.execute("SELECT * FROM dblp.journalportal WHERE Title like %s", (str,))
                if result1.rowcount > 0:
                    counterJournal[result1.fetchone()[2]] += 1
            # print(jcount)
            confcount = con.execute(
                "SELECT count(*) as `publications`,publication.key FROM dblp.publication,authors_publications WHERE authors_publications.author_id = %s AND authors_publications.publ_id = publication.id AND publication.`key` like 'conf/%' GROUP BY SUBSTRING_INDEX(publication.`key`,'/',2);",
                (x[0],))
            ccount = 0

            for i in confcount:
                ccount += i[0]
                str = i[1].split('/')[1]
                # print(str)
                result2 = con.execute(query, (str,))
                if result2.rowcount > 0:
                    counterconference[result2.fetchone()[5]] += 1

                    print("result2")

            # print(ccount)

            journalMostcommon = counterJournal.most_common(1)
            conferenceMostcommon = counterconference.most_common(1)
            if (journalMostcommon and conferenceMostcommon):
                if (journalMostcommon[0][1] > conferenceMostcommon[0][1]):
                    print(journalMostcommon)
                    finalFoR = journalMostcommon
                else:
                    print(conferenceMostcommon)
                finalFoR = conferenceMostcommon
            elif (journalMostcommon == None and conferenceMostcommon != None):
                print(conferenceMostcommon)
                finalFoR = conferenceMostcommon
            else:
                print(journalMostcommon)
                finalFoR = journalMostcommon

            res3 = con.execute("SELECT * FROM `for` WHERE id= %s", (finalFoR[0][0],))
            finalresult = res3.fetchone()
            print(finalresult)
            return finalresult



def SearchCoauthorshipAndFor(xnumber,focus):

    engine = create_engine("mysql+mysqlconnector://testing123:testing123@35.232.130.107/dblp")
    con = engine.connect()
    con.execute(" SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''))")
    res2=con.execute(" SELECT id FROM dblp.`for` WHERE name=%s",focus)
    forid = res2.fetchone()[0]
    print(forid)
    numberofpapers = xnumber
    res = con.execute(" select * from NODES%s where combinepublication >= %s and authorid1 != authorid2",
                      (forid, numberofpapers))
    for_author = res.fetchall()
    # all authours count greater than 1
    count = len(for_author)

    List=[]
    List2=[]
   # f = open("Create FOR902 Edges.txt", "w")
    # FOR is same and each author has atleast x number of papers now we look whether they written atleast x papers together i.e. co-authorship for X papers together
    for x in for_author:
        res = con.execute(" select authorname from authorsfor where authorid = %s", x[0])
        i = res.fetchone()[0]
        res = con.execute(" select authorname from authorsfor where authorid = %s", x[1])
        j = res.fetchone()[0]
        #print(i, "   ",j, "   ", x[2])
        # authid_names=[x[0],i]
        #nodes going here
        if [x[0],i] not in List2:List2.append([x[0],i])
        if [x[1],j] not in List2: List2.append([x[1], j])

        values=[x[0],x[1],x[2]]
        if values not in List:List.append(values)
        # #f.write(x+";"+y+";"+xx+"\n")
      #  f.write("%s;%s;%s\n" % (i, j, x[2]))
    #f.close()
    print("End Result")
    #Graph()
    print("Graph Displayed")
    print("Waiting for nodes and edge list to complete")
    print(List)
    return (List, List2)

def getPrediction(journalid,year):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = pd.read_csv('training data.csv')
    np.random.seed(25)
    df = data.copy()
    df1 = df[df['Year'] < 2016]
    X1 = df1.drop('Number of Papers published', axis=1)
    Y1 = df1[['Number of Papers published']]
    xtrain, xtest, ytrain, ytest = train_test_split(X1, Y1, test_size=0.3, random_state=25, shuffle=True)
    print(xtrain.shape, ytrain.shape)
    print(xtest.shape, ytest.shape)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    r2_score(ytrain, model.predict(xtrain))
    testing = xtest.iloc[0]
    testing['Year'] = year
    testing['Journal Id'] = journalid
    pred = model.predict([testing])
    print(pred[0][0])
    return pred[0][0]


def getnaivePrediction(journalid,year):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    data = pd.read_csv('training data.csv')
    data['count_group'] = ''
    data.loc[(data['Number of Papers published'] >= 0) & (data['Number of Papers published'] <= 20), 'count_group'] = \
    data.loc[
        (data['Number of Papers published'] >= 0) & (data['Number of Papers published'] <= 20), 'count_group'].replace(
        '', '0-20')
    data.loc[(data['Number of Papers published'] > 20) & (data['Number of Papers published'] <= 40), 'count_group'] = \
    data.loc[
        (data['Number of Papers published'] > 20) & (data['Number of Papers published'] <= 40), 'count_group'].replace(
        '', '21-40')
    data.loc[(data['Number of Papers published'] > 40) & (data['Number of Papers published'] <= 60), 'count_group'] = \
    data.loc[
        (data['Number of Papers published'] > 40) & (data['Number of Papers published'] <= 60), 'count_group'].replace(
        '', '41-60')
    data.loc[(data['Number of Papers published'] > 60) & (data['Number of Papers published'] <= 80), 'count_group'] = \
    data.loc[
        (data['Number of Papers published'] > 60) & (data['Number of Papers published'] <= 80), 'count_group'].replace(
        '', '61-80')
    data.loc[(data['Number of Papers published'] > 80) & (data['Number of Papers published'] <= 100), 'count_group'] = \
    data.loc[
        (data['Number of Papers published'] > 80) & (data['Number of Papers published'] <= 100), 'count_group'].replace(
        '', '81-100')
    data.loc[(data['Number of Papers published'] > 100) & (data['Number of Papers published'] <= 120), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 100) & (
                data['Number of Papers published'] <= 120), 'count_group'].replace('', '101-120')
    data.loc[(data['Number of Papers published'] > 120) & (data['Number of Papers published'] <= 140), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 120) & (
                data['Number of Papers published'] <= 140), 'count_group'].replace('', '121-140')
    data.loc[(data['Number of Papers published'] > 140) & (data['Number of Papers published'] <= 160), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 140) & (
                data['Number of Papers published'] <= 160), 'count_group'].replace('', '141-160')
    data.loc[(data['Number of Papers published'] > 160) & (data['Number of Papers published'] <= 180), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 160) & (
                data['Number of Papers published'] <= 180), 'count_group'].replace('', '161-180')
    data.loc[(data['Number of Papers published'] > 180) & (data['Number of Papers published'] <= 200), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 180) & (
                data['Number of Papers published'] <= 200), 'count_group'].replace('', '181-200')
    data.loc[(data['Number of Papers published'] > 200) & (data['Number of Papers published'] <= 220), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 200) & (
                data['Number of Papers published'] <= 220), 'count_group'].replace('', '201-220')
    data.loc[(data['Number of Papers published'] >= 220) & (data['Number of Papers published'] <= 240), 'count_group'] = \
    data.loc[(data['Number of Papers published'] >= 220) & (
                data['Number of Papers published'] <= 240), 'count_group'].replace('', '221-240')
    data.loc[(data['Number of Papers published'] > 240) & (data['Number of Papers published'] <= 260), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 240) & (
                data['Number of Papers published'] <= 260), 'count_group'].replace('', '241-260')
    data.loc[(data['Number of Papers published'] > 260) & (data['Number of Papers published'] <= 280), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 260) & (
                data['Number of Papers published'] <= 280), 'count_group'].replace('', '261-280')
    data.loc[(data['Number of Papers published'] > 280) & (data['Number of Papers published'] <= 300), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 280) & (
                data['Number of Papers published'] <= 300), 'count_group'].replace('', '281-300')
    data.loc[(data['Number of Papers published'] > 300) & (data['Number of Papers published'] <= 320), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 300) & (
                data['Number of Papers published'] <= 320), 'count_group'].replace('', '301-320')
    data.loc[(data['Number of Papers published'] > 320) & (data['Number of Papers published'] <= 340), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 320) & (
                data['Number of Papers published'] <= 340), 'count_group'].replace('', '321-340')
    data.loc[(data['Number of Papers published'] > 340) & (data['Number of Papers published'] <= 360), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 340) & (
                data['Number of Papers published'] <= 360), 'count_group'].replace('', '341-360')
    data.loc[(data['Number of Papers published'] > 360) & (data['Number of Papers published'] <= 380), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 360) & (
                data['Number of Papers published'] <= 380), 'count_group'].replace('', '361-380')
    data.loc[(data['Number of Papers published'] > 380) & (data['Number of Papers published'] <= 400), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 380) & (
                data['Number of Papers published'] <= 400), 'count_group'].replace('', '381-400')
    data.loc[(data['Number of Papers published'] > 400) & (data['Number of Papers published'] <= 420), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 400) & (
                data['Number of Papers published'] <= 420), 'count_group'].replace('', '401-420')
    data.loc[(data['Number of Papers published'] > 420) & (data['Number of Papers published'] <= 440), 'count_group'] = \
    data.loc[(data['Number of Papers published'] > 420) & (
                data['Number of Papers published'] <= 440), 'count_group'].replace('', '421-440')
    data.loc[data['Number of Papers published'] > 440, 'count_group'] = data.loc[
        data['Number of Papers published'] > 440, 'count_group'].replace('', 'greater than 440')
    training = data.copy()
    X1 = training.drop('Number of Papers published', axis=1).astype(str)
    Y1 = training[['count_group']]
    X1 = X1.drop('count_group', axis=1).astype(str)
    xtrain, xtest, ytrain, ytest = train_test_split(X1, Y1, test_size=0.3, random_state=25, shuffle=True)
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    testing = xtrain.iloc[0]
    testing['Journal Id'] = journalid
    testing['Year'] = year
    pred = model.predict([testing])
    return pred[0]


def getnaiveClassification(stringin):
    import pandas as pd
    sms = pd.read_csv('publications.csv')
    train = sms
    X = train.title
    y = train.forid
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    string = stringin
    test0 = [string]
    test1 = vect.transform(test0)
    y_pred_class = nb.predict(test1)
    z = y_pred_class[0].astype(str)
    z1 = str(z)
    from sqlalchemy import create_engine
    engine = create_engine("mysql+mysqlconnector://testing123:testing123@35.232.130.107/dblp")
    con = engine.connect()
    res2 = con.execute("select name from `for` where id=%s", z1)
    result = res2.fetchall()
    return result[0][0]



def getnaivefor(naiveforname,naiveforyear):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from sklearn.naive_bayes import GaussianNB
    data = pd.read_csv('for_mining.csv')
    train = data
    X1 = train.drop('count', axis=1).astype(str)
    Y1 = train[['count']]
    xtrain, xtest, ytrain, ytest = train_test_split(X1, Y1, test_size=0.3, random_state=25, shuffle=True)
    model = GaussianNB()
    model.fit(xtrain, ytrain)
    testing = xtrain.iloc[0]
    from sqlalchemy import create_engine
    engine = create_engine("mysql+mysqlconnector://testing123:testing123@35.232.130.107/dblp")
    con = engine.connect()
    naiveforname = naiveforname
    res2 = con.execute("select id from `for` where name=%s", naiveforname)
    result = res2.fetchall()
    naiveforid = result[0][0]
    year = naiveforyear
    forid = naiveforid
    testing['year'] = year
    testing['for_id'] = forid
    pred = model.predict([testing])
    return pred[0]



if __name__ == '__main__':
    app.run()



#Ipke Wachsmuth
