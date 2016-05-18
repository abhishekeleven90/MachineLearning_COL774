from math import log
import pandas as pd

def checkequal(paramdict1, paramdict2):
    
    #paramdict1 = {'B_H': {'B0_H0': 0.945514584479912, 'B0_H1': 0.5802752293577982, 'B1_H1': 0.4197247706422018, 'B1_H0': 0.05448541552008806}, 'H': {'H0': 0.8083428899082569, 'H1': 0.19165711009174313}, 'L_H': {'L1_H1': 0.296551724137931, 'L1_H0': 0.005012531328320802, 'L0_H0': 0.9949874686716792, 'L0_H1': 0.7034482758620689}, 'X_L': {'X1_L1': 0.610909090909091, 'X1_L0': 0.02420321111909897, 'X0_L0': 0.975796788880901, 'X0_L1': 0.3890909090909091}, 'F_BL': {'F0_B1L1': 0.2923076923076923, 'F0_B1L0': 0.9112903225806451, 'F1_B0L1': 0.6105263157894737, 'F1_B0L0': 0.05537150970184572, 'F1_B1L0': 0.08870967741935484, 'F1_B1L1': 0.7076923076923077, 'F0_B0L0': 0.9446284902981543, 'F0_B0L1': 0.3894736842105263}}
    #paramdict2 = {'B_H': {'B0_H0': 0.945414584479912, 'B0_H1': 0.5802752293577982, 'B1_H1': 0.4197247706422018, 'B1_H0': 0.05448541552008806}, 'H': {'H0': 0.8083428899082569, 'H1': 0.19165711009174313}, 'L_H': {'L1_H1': 0.296551724137931, 'L1_H0': 0.005012531328320802, 'L0_H0': 0.9949874686716792, 'L0_H1': 0.7034482758620689}, 'X_L': {'X1_L1': 0.610909090909091, 'X1_L0': 0.02420321111909897, 'X0_L0': 0.975796788880901, 'X0_L1': 0.3890909090909091}, 'F_BL': {'F0_B1L1': 0.2923076923076923, 'F0_B1L0': 0.9112903225806451, 'F1_B0L1': 0.6105263157894737, 'F1_B0L0': 0.05537150970184572, 'F1_B1L0': 0.08870967741935484, 'F1_B1L1': 0.7076923076923077, 'F0_B0L0': 0.9446284902981543, 'F0_B0L1': 0.3894736842105263}}
    
    alltrue = True
    epsilon = 1e-8
    
    allkeys =  paramdict1.keys()
    
    for key in allkeys:
        dict1  = paramdict1[key]
        dict2 = paramdict2[key]
        innerkeys = dict1.keys()
        for innerkey in innerkeys:
            if (dict1[innerkey]-dict2[innerkey])>epsilon:
                return False
    
    return True

def printTabular(paramdict):
    
    print '##############################################\n'
    
    #paramdict = {'B_H': {'B0_H0': 0.945514584479912, 'B0_H1': 0.5802752293577982, 'B1_H1': 0.4197247706422018, 'B1_H0': 0.05448541552008806}, 'H': {'H0': 0.8083428899082569, 'H1': 0.19165711009174313}, 'L_H': {'L1_H1': 0.296551724137931, 'L1_H0': 0.005012531328320802, 'L0_H0': 0.9949874686716792, 'L0_H1': 0.7034482758620689}, 'X_L': {'X1_L1': 0.610909090909091, 'X1_L0': 0.02420321111909897, 'X0_L0': 0.975796788880901, 'X0_L1': 0.3890909090909091}, 'F_BL': {'F0_B1L1': 0.2923076923076923, 'F0_B1L0': 0.9112903225806451, 'F1_B0L1': 0.6105263157894737, 'F1_B0L0': 0.05537150970184572, 'F1_B1L0': 0.08870967741935484, 'F1_B1L1': 0.7076923076923077, 'F0_B0L0': 0.9446284902981543, 'F0_B0L1': 0.3894736842105263}}
    
    ##print H
    print 'H'
    print str(paramdict['H']['H0']) + ' ' + str(paramdict['H']['H1'])
    print ' '
    
    ##print B given H
    print 'B_H'
    print str(paramdict['B_H']['B0_H0']) + ' ' + str(paramdict['B_H']['B1_H0'])
    print str(paramdict['B_H']['B0_H1']) + ' ' + str(paramdict['B_H']['B1_H1'])
    print ' '
    
    ##print L given H
    print 'L_H'
    print str(paramdict['L_H']['L0_H0']) + ' ' + str(paramdict['L_H']['L1_H0'])
    print str(paramdict['L_H']['L0_H1']) + ' ' + str(paramdict['L_H']['L1_H1'])
    print ' '
    
    ##print X given L
    print 'X_L'
    print str(paramdict['X_L']['X0_L0']) + ' ' + str(paramdict['X_L']['X1_L0'])
    print str(paramdict['X_L']['X0_L1']) + ' ' + str(paramdict['X_L']['X1_L1'])
    print ' '

    ##print F given BL
    print 'F_BL'
    print str(paramdict['F_BL']['F0_B0L0']) + ' ' + str(paramdict['F_BL']['F1_B0L0'])
    print str(paramdict['F_BL']['F0_B0L1']) + ' ' + str(paramdict['F_BL']['F1_B0L1'])
    print str(paramdict['F_BL']['F0_B1L0']) + ' ' + str(paramdict['F_BL']['F1_B1L0'])
    print str(paramdict['F_BL']['F0_B1L1']) + ' ' + str(paramdict['F_BL']['F1_B1L1'])
    print ' '    
    
    print '##############################################'

##printTabular({})

def calcLengthNew(somedf,colNameOrder,colValueOrder):
    #import copy
    #copy.deepcopy()
    tempdf = somedf
    for i in range(len(colNameOrder)):
        tempdf = tempdf[tempdf[colNameOrder[i]] == colValueOrder[i]]
    #print '##' 
    #print tempdf['prob_wt']
    return sum(tempdf['prob_wt'])
    #return sum(tempdf['prob_wt'].sum())
    
def returnDictNew(df, dependent, given):
    X_L = {}
    lenX0_L0 =  calcLengthNew(df,[dependent,given],[0,0])
    lenX1_L0 =  calcLengthNew(df,[dependent,given],[1,0])
    lenX0_L1 =  calcLengthNew(df,[dependent,given],[0,1])
    lenX1_L1 =  calcLengthNew(df,[dependent,given],[1,1])
    lenL0 = lenX0_L0 + lenX1_L0
    lenL1 = lenX0_L1 + lenX1_L1 
    X_L[dependent+'0_'+given+'0'] = lenX0_L0/float(lenL0)
    X_L[dependent+'1_'+given+'0'] = lenX1_L0/float(lenL0)
    X_L[dependent+'0_'+given+'1'] = lenX0_L1/float(lenL1)
    X_L[dependent+'1_'+given+'1'] = lenX1_L1/float(lenL1)
    return X_L

def returnDict2New(df, dependent, given1, given2):
    X_L = {}
    
    lenX0_L00 =  calcLengthNew(df,[dependent,given1,given2],[0,0,0])
    lenX0_L01 =  calcLengthNew(df,[dependent,given1,given2],[0,0,1])
    lenX0_L10 =  calcLengthNew(df,[dependent,given1,given2],[0,1,0])
    lenX0_L11 =  calcLengthNew(df,[dependent,given1,given2],[0,1,1])
    
    lenX1_L00 =  calcLengthNew(df,[dependent,given1,given2],[1,0,0])
    lenX1_L01 =  calcLengthNew(df,[dependent,given1,given2],[1,0,1])
    lenX1_L10 =  calcLengthNew(df,[dependent,given1,given2],[1,1,0])
    lenX1_L11 =  calcLengthNew(df,[dependent,given1,given2],[1,1,1])
    
    lenL00 = lenX0_L00 + lenX1_L00
    lenL01 = lenX0_L01 + lenX1_L01 
    lenL10 = lenX0_L10 + lenX1_L10
    lenL11 = lenX0_L11 + lenX1_L11
    
    X_L[dependent+'0_'+given1+'0'+given2+'0'] = lenX0_L00/float(lenL00)
    X_L[dependent+'0_'+given1+'0'+given2+'1'] = lenX0_L01/float(lenL01)
    X_L[dependent+'0_'+given1+'1'+given2+'0'] = lenX0_L10/float(lenL10)
    X_L[dependent+'0_'+given1+'1'+given2+'1'] = lenX0_L11/float(lenL11)
    
    X_L[dependent+'1_'+given1+'0'+given2+'0'] = lenX1_L00/float(lenL00)
    X_L[dependent+'1_'+given1+'0'+given2+'1'] = lenX1_L01/float(lenL01)
    X_L[dependent+'1_'+given1+'1'+given2+'0'] = lenX1_L10/float(lenL10)
    X_L[dependent+'1_'+given1+'1'+given2+'1'] = lenX1_L11/float(lenL11)
    
    return X_L


def getAllParamsNew(df):
    ##parameter 1
    H={}

    lenH0 = calcLengthNew(df,['H'],[0]) ##len(df_H0)
    #print lenH0
    lenH1 = calcLengthNew(df,['H'],[1]) ##len(df_H1)
    H['H0']=lenH0/float(lenH0+lenH1)
    H['H1']=lenH1/float(lenH0+lenH1)
    
    
    L_H= returnDictNew(df,'L','H') ##L given H _ is given 


#     ##parameter 3
    B_H = returnDictNew(df,'B','H')


   

#     ## parameter 4
    X_L = returnDictNew(df,'X','L')

    
#     ## parameter 5
    F_BL = returnDict2New(df, 'F', 'B', 'L')

    
    masterDict = {'H':H,'L_H':L_H,'B_H':B_H,'X_L':X_L,'F_BL':F_BL}
    
#     return masterDict

#    print H
#    print L_H

    return masterDict



def calcLength(somedf,colNameOrder,colValueOrder):
    #import copy
    #copy.deepcopy()
    tempdf = somedf
    for i in range(len(colNameOrder)):
        tempdf = tempdf[tempdf[colNameOrder[i]] == colValueOrder[i]]
    return len(tempdf)

def returnDict(df, dependent, given):
    X_L = {}
    lenX0_L0 =  calcLength(df,[dependent,given],['0','0'])
    lenX1_L0 =  calcLength(df,[dependent,given],['1','0'])
    lenX0_L1 =  calcLength(df,[dependent,given],['0','1'])
    lenX1_L1 =  calcLength(df,[dependent,given],['1','1'])
    lenL0 = lenX0_L0 + lenX1_L0
    lenL1 = lenX0_L1 + lenX1_L1 
    X_L[dependent+'0_'+given+'0'] = lenX0_L0/float(lenL0)
    X_L[dependent+'1_'+given+'0'] = lenX1_L0/float(lenL0)
    X_L[dependent+'0_'+given+'1'] = lenX0_L1/float(lenL1)
    X_L[dependent+'1_'+given+'1'] = lenX1_L1/float(lenL1)
    return X_L

# def returnDictCopy(df, dependent, given):
#     X_L = {}
#     lenL0 = calcLength(df,[given],['0'])
#     lenL1 = calcLength(df,[given],['1'])
#     lenX0_L0 =  calcLength(df,[dependent,given],['0','0'])
#     lenX1_L0 =  calcLength(df,[dependent,given],['1','0'])
#     lenX0_L1 =  calcLength(df,[dependent,given],['0','1'])
#     lenX1_L1 =  calcLength(df,[dependent,given],['1','1'])
#     X_L[dependent+'0_'+given+'0'] = lenX0_L0/float(lenL0)
#     X_L[dependent+'1_'+given+'0'] = lenX1_L0/float(lenL0)
#     X_L[dependent+'0_'+given+'1'] = lenX0_L1/float(lenL1)
#     X_L[dependent+'1_'+given+'1'] = lenX1_L1/float(lenL1)
#     return X_L

def returnDict2(df, dependent, given1, given2):
    X_L = {}
    
    lenX0_L00 =  calcLength(df,[dependent,given1,given2],['0','0','0'])
    lenX0_L01 =  calcLength(df,[dependent,given1,given2],['0','0','1'])
    lenX0_L10 =  calcLength(df,[dependent,given1,given2],['0','1','0'])
    lenX0_L11 =  calcLength(df,[dependent,given1,given2],['0','1','1'])
    
    lenX1_L00 =  calcLength(df,[dependent,given1,given2],['1','0','0'])
    lenX1_L01 =  calcLength(df,[dependent,given1,given2],['1','0','1'])
    lenX1_L10 =  calcLength(df,[dependent,given1,given2],['1','1','0'])
    lenX1_L11 =  calcLength(df,[dependent,given1,given2],['1','1','1'])
    
    lenL00 = lenX0_L00 + lenX1_L00
    lenL01 = lenX0_L01 + lenX1_L01 
    lenL10 = lenX0_L10 + lenX1_L10
    lenL11 = lenX0_L11 + lenX1_L11
    
    X_L[dependent+'0_'+given1+'0'+given2+'0'] = lenX0_L00/float(lenL00)
    X_L[dependent+'0_'+given1+'0'+given2+'1'] = lenX0_L01/float(lenL01)
    X_L[dependent+'0_'+given1+'1'+given2+'0'] = lenX0_L10/float(lenL10)
    X_L[dependent+'0_'+given1+'1'+given2+'1'] = lenX0_L11/float(lenL11)
    
    X_L[dependent+'1_'+given1+'0'+given2+'0'] = lenX1_L00/float(lenL00)
    X_L[dependent+'1_'+given1+'0'+given2+'1'] = lenX1_L01/float(lenL01)
    X_L[dependent+'1_'+given1+'1'+given2+'0'] = lenX1_L10/float(lenL10)
    X_L[dependent+'1_'+given1+'1'+given2+'1'] = lenX1_L11/float(lenL11)
    
    return X_L

# def returnDict2Copy(df, dependent, given1, given2):
#     X_L = {}
    
#     lenL00 = calcLength(df,[given1, given2],['0','0'])
#     lenL01 = calcLength(df,[given1, given2],['0','1'])
#     lenL10 = calcLength(df,[given1, given2],['1','0'])
#     lenL11 = calcLength(df,[given1, given2],['1','1'])
    
#     lenX0_L00 =  calcLength(df,[dependent,given1,given2],['0','0','0'])
#     lenX0_L01 =  calcLength(df,[dependent,given1,given2],['0','0','1'])
#     lenX0_L10 =  calcLength(df,[dependent,given1,given2],['0','1','0'])
#     lenX0_L11 =  calcLength(df,[dependent,given1,given2],['0','1','1'])
    
#     lenX1_L00 =  calcLength(df,[dependent,given1,given2],['1','0','0'])
#     lenX1_L01 =  calcLength(df,[dependent,given1,given2],['1','0','1'])
#     lenX1_L10 =  calcLength(df,[dependent,given1,given2],['1','1','0'])
#     lenX1_L11 =  calcLength(df,[dependent,given1,given2],['1','1','1'])
    
#     X_L[dependent+'0_'+given1+'0'+given2+'0'] = lenX0_L00/float(lenL00)
#     X_L[dependent+'0_'+given1+'0'+given2+'1'] = lenX0_L01/float(lenL01)
#     X_L[dependent+'0_'+given1+'1'+given2+'0'] = lenX0_L10/float(lenL10)
#     X_L[dependent+'0_'+given1+'1'+given2+'1'] = lenX0_L11/float(lenL11)
    
#     X_L[dependent+'1_'+given1+'0'+given2+'0'] = lenX1_L00/float(lenL00)
#     X_L[dependent+'1_'+given1+'0'+given2+'1'] = lenX1_L01/float(lenL01)
#     X_L[dependent+'1_'+given1+'1'+given2+'0'] = lenX1_L10/float(lenL10)
#     X_L[dependent+'1_'+given1+'1'+given2+'1'] = lenX1_L11/float(lenL11)
    
#     return X_L

def getAllParams(df):
    ##parameter 1
    H={}

    lenH0 = calcLength(df,['H'],['0']) ##len(df_H0)
    lenH1 = calcLength(df,['H'],['1']) ##len(df_H1)
    H['H0']=lenH0/float(lenH0+lenH1)
    H['H1']=lenH1/float(lenH0+lenH1)
    
    L_H= returnDict(df,'L','H') ##L given H _ is given 




    ##parameter 3
    B_H = returnDict(df,'B','H')


   

    ## parameter 4
    X_L = returnDict(df,'X','L')

    
    ## parameter 5
    F_BL = returnDict2(df, 'F', 'B', 'L')

    
    masterDict = {'H':H,'L_H':L_H,'B_H':B_H,'X_L':X_L,'F_BL':F_BL}
    #print masterDict
    
    #print H
    #print L_H
    #print B_H
    #print X_L
    #print F_BL


    


    
    return masterDict

##gives likelihood of one example, if log True gives log likelihood too!
def likelihood(masterDict, **kwargs):
    kwargs['H'] = str(kwargs['H'])
    kwargs['B'] = str(kwargs['B'])
    kwargs['L'] = str(kwargs['L'])
    kwargs['X'] = str(kwargs['X'])
    kwargs['F'] = str(kwargs['F'])
    
#     print kwargs['H']
#     print kwargs['B']
#     print kwargs['L']
#     print kwargs['X']
#     print kwargs['F']
    
    probH = masterDict['H']['H'+kwargs['H']]
   
    
    probL_H = masterDict['L_H']['L'+kwargs['L']+'_'+'H'+kwargs['H']]
    
    probB_H = masterDict['B_H']['B'+kwargs['B']+'_'+'H'+kwargs['H']]
    
    probX_L = masterDict['X_L']['X'+kwargs['X']+'_'+'L'+kwargs['L']]
    
    probF_BL = masterDict['F_BL']['F'+kwargs['F']+'_'+'B'+kwargs['B']+'L'+kwargs['L']]
    #print 'F'+kwargs['F']+'_'+'B'+kwargs['B']+'L'+kwargs['L']
    
    #print probH, probL_H, probB_H, probX_L, probF_BL
    
    if kwargs['log']==True:
        return log(probH) + log(probL_H) + log(probB_H) + log(probX_L) + log(probF_BL)
    else:
        return probH * probL_H * probB_H * probX_L * probF_BL
    
    
##Usage:  
##print loglikelihood(H=1,B=0,X=1,L=1,F=0)

def get_mle(somedf2, masterDict3):
    mle = 0.0
    for i in range(len(somedf2)):
        H = somedf2['H'][i]
        X = somedf2['X'][i]
        B = somedf2['B'][i]
        L = somedf2['L'][i]
        F = somedf2['F'][i]
        mle = mle + likelihood(masterDict3,log=True, H=H,X=X,B=B,L=L,F=F)
    #print mle
    return mle

def part1():
    df = pd.read_csv('data_EM/train.data', delimiter=' ', dtype='str')
    paramDict = getAllParams(df)
    printTabular(paramDict)
    #print masterDict
    df2 = pd.read_csv('data_EM/test.data', delimiter=' ', dtype='str')
    print get_mle(df2,paramDict)
        
##Usage ##part1()

def getMissing(row): ##pandas row
    allList = ['H','L','X','B','F']
    missing = []
    nonmissing = []
    for rv in allList: ##rv is short for random variable here
        if row[rv]=='?':
            missing.append(rv)
        else:
            nonmissing.append(rv)
    return missing, nonmissing

def part3():
    
    df = pd.read_csv('data_EM/train-m1.data', delimiter=' ', dtype='str')
    dftest = pd.read_csv('data_EM/test.data', delimiter=' ', dtype='str')
    
    #print df
    masterDict = getAllParams(df)
    
    print 'Initial Params'
    printTabular(masterDict)
    #print masterDict
    
    df2 = df.copy(deep=True)
    
    prev_mle = 0
    
    k = 0
    while True:
        
        k = k + 1 ##iteration variable
        
        ##init the dict
        ##for each iteration? TODO!
        tempdict = {}
        for column in df.columns:
            tempdict[column]=[]
            #print column
            
        ##add the prob_wt column from now on itself
        tempdict['prob_wt'] = []
        
        
        #print 'masterDict'
        #printTabular(masterDict)
        ##print masterDict
        
    
        for index, row in df.iterrows():
            missing, nonmissing = getMissing(row)
            
            if len(missing)==1:
                
                #print 'one'

                ##when missing rv is 1
                temp1 = {}
                for rv in missing: ##rv is short for random variable here
                    temp1[rv]='1'
                for rv in nonmissing:
                    temp1[rv]=row[rv]

                p1 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])

                ##when missing rv is 0
                for rv in missing:
                    temp1[rv] = '0'

                p0 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])

                prob1 = p1/float(p0+p1)
                prob0 = p0/float(p0+p1)

                #print temp1,missing,prob1,prob0

                ##now you write new code that will be used
                ###########################
                ##assuming len missing == 1
                for col in nonmissing:
                    tempdict[col].append(int(row[col]))
                    tempdict[col].append(int(row[col]))
                for col in missing:
                    tempdict[col].append(0) ##0 first
                    tempdict[col].append(1) ##1 afterwards
                tempdict['prob_wt'].append(prob0) ##0 first
                tempdict['prob_wt'].append(prob1) ##1 afterwards
                ############################
                
            elif(len(missing)==2):
                
                #print 'two'
                
                ##first calc the probs
                temp1 = {}
                a,b = missing[0], missing[1]
                
                ##handle non-missing first!
                for rv in nonmissing:
                    temp1[rv]=row[rv]
                
                ##set 00
                temp1[a] = '0'
                temp1[b] = '0'
                p00 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])
                
                ##set 01
                temp1[a] = '0'
                temp1[b] = '1'
                p01 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])
                
                ##set 10
                temp1[a] = '1'
                temp1[b] = '0'
                p10 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])
                
                ##set 11
                temp1[a] = '1'
                temp1[b] = '1'
                p11 = likelihood(masterDict,log=False, H=temp1['H'],L=temp1['L'],X=temp1['X'],B=temp1['B'],F=temp1['F'])
                
                
                prob00 = p00/float(p00+p01+p10+p11)
                prob01 = p01/float(p00+p01+p10+p11)
                prob10 = p10/float(p00+p01+p10+p11)
                prob11 = p11/float(p00+p01+p10+p11)
                
                ##constructing the dict/df!! with multiplicated rows!
                ##assuming len missing == 2
                for col in nonmissing:
                    tempdict[col].append(int(row[col]))
                    tempdict[col].append(int(row[col]))
                    tempdict[col].append(int(row[col]))
                    tempdict[col].append(int(row[col]))
                    
                tempdict[a].append(0) 
                tempdict[a].append(0)
                tempdict[a].append(1)
                tempdict[a].append(1)
                
                tempdict[b].append(0) 
                tempdict[b].append(1)
                tempdict[b].append(0)
                tempdict[b].append(1)
                
                
                tempdict['prob_wt'].append(prob00) 
                tempdict['prob_wt'].append(prob01)
                tempdict['prob_wt'].append(prob10)
                tempdict['prob_wt'].append(prob11)
                
                ############################
            
    
#         for key in tempdict:
#             print len(tempdict[key])
            
        df2 = pd.DataFrame(tempdict)
        print len(df2)
        newParams = getAllParamsNew(df2)
        #print newParams

        boolval = checkequal(masterDict, newParams)
        
        #print get_mle(df2,newParams), get_mle(dftest, newParams)
        new_mle = get_mle(dftest, newParams)
        print str(k)+', '+str(new_mle)
        #boolval  = abs(prev_mle-new_mle)<1e-6
        prev_mle = new_mle
        
        
        
        
        import copy
        masterDict = copy.deepcopy(newParams) ##update here!
        #print masterDict
        print '-----'
        
        if boolval: 
            break
        else:
            print k
        
    print 'Final Params'
    printTabular(masterDict)
        
        #print nonmissing, missing

##usage:part1()
##part3()
part3()
