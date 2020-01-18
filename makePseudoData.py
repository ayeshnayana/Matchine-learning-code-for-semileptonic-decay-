import numpy as np
import time #used to seed RGNs
import csv

# creates inputs from flat (uniform) prior
def makeInputs(tMin, tMax):
    # uses time to create a seed
    t = int(time.time() * 1000.0)
    np.random.seed( ((t & 0xff000000) >> 24) +
                    ((t & 0x00ff0000) >>  8) +
                    ((t & 0x0000ff00) <<  8) +
                    ((t & 0x000000ff) << 24)   )

    # returns uniform random number
    return np.random.uniform(tMin, tMax)

# creates target data using compilcated algorithm
    # RGNs are zero mean Gaussian random numbers
def makeTargets(mean, statU, sysUtot, sysCorr, statCorr, col):
    # uses time to create a seed
    t = int(time.time() * 1000.0)
    np.random.seed( ((t & 0xff000000) >> 24) +
                    ((t & 0x00ff0000) >>  8) +
                    ((t & 0x0000ff00) <<  8) +
                    ((t & 0x000000ff) << 24)   )

    totU = (statU**2 + sysUtot**2)**(0.5)

    # finds uncorrelated uncertainty for the experimental point
    uncorrUncer = np.random.normal(0, 1)*totU

    # finds correlated uncertainty for the experimental point
    sysCorrUnc = np.sum(np.random.normal(0, 1)*sysCorr[:,col])
    statCorrUnc = np.sum(np.random.normal(0, 1)*statCorr[:,col])
    corrUncer = sysCorrUnc + statCorrUnc

    # returns the pseudo data point
    return mean + corrUncer + uncorrUncer

# converts correlation matrix to sigma_{x,y}
    # which is the input needed for the target data point algorithm
def corrTOsigmaXY(corr, fileData, k):
    sigmaXY = np.zeros((len(corr),len(corr[0])))

    for i in range(len(sigmaXY)):
        for j in range(len(sigmaXY[0])):
            sigmaXY[i,j] = (abs(corr[i,j]*fileData[i,k]*fileData[j,k]))**(0.5)

    return sigmaXY

# creation of two TXT files: one for q^2 and one for dGamma pseudo-data
def txtPDGen(q2File, dGFile, pseudoDataAmount, fileData, sysUtot, sysCorr, statCorr):
    File = open(q2File, 'w')
    for l in range(pseudoDataAmount):
        for k in range(len(fileData)):
            File.write(str(makeInputs(fileData[k,0], fileData[k,1])) + " ")
        if(l<(pseudoDataAmount-1)):
            File.write("\n")
    File.close();

    print("q2 pseudo-data gen is complete. Waiting on dGamma...")

    File = open(dGFile, 'w')
    for l in range(pseudoDataAmount):
        for k in range(len(fileData)):
            t = makeTargets(fileData[k,2],fileData[k,2]*fileData[k,3]/100,
                    sysUtot, sysCorr, statCorr, k)
            File.write(str(t) + " ")
        if ((l+1) % (pseudoDataAmount/100) == 0):
            print("Pseudo-data generation is ", 100*(l+1)/pseudoDataAmount, "% finished")
        if(l<(pseudoDataAmount-1)):
            File.write("\n")
    File.close();

# creation of two CSV files: one for q^2 and one for dGamma pseudo-data
def csvPDGen(q2File, dGFile, pseudoDataAmount, fileData, sysUtot, sysCorr, statCorr):
    with open(q2File, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for l in range(pseudoDataAmount):
            for k in range(len(fileData)):
                i = makeInputs(fileData[k,0], fileData[k,1])
                writer.writerow(i)
    csvFile.close();

    print("q2 pseudo-data gen is complete. Waiting on dGamma...")

    with open(dGFile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for l in range(pseudoDataAmount):
            for k in range(len(fileData)):
                t = makeTargets(fileData[k,2],fileData[k,2]*fileData[k,3]/100,
                        sysUtot, sysCorr, statCorr, k)
                writer.writerow(t)
            if ((l+1) % (pseudoDataAmount/100) == 0):
                print("dGamma pseudo-data gen is ",100*(l+1)/pseudoDataAmount,"% finished")
    csvFile.close();




# load in data and two correlation matrices
fileData = np.loadtxt("data\\DtoPi_PRD92_072012_VF.txt")
sysCorr = np.loadtxt("data\\DtoPi_PRD92_072012_sysCorr.txt")
statCorr = np.loadtxt("data\\DtoPi_PRD92_072012_statCorr.txt")

# make matrices symmetrical because half of txt file was filled with zeroes
for i in range(len(sysCorr)):
    for j in range(len(sysCorr[0])):
        if (i<j):
            sysCorr[i,j] = sysCorr[j,i]
            statCorr[i,j] = statCorr[j,i]

# find the correlation matrices which will be used to create dGamma/VF values
sysCorr = corrTOsigmaXY(sysCorr, fileData, 4)
statCorr = corrTOsigmaXY(statCorr, fileData, 3)

# find total uncorrelated uncertainty, will be used to create dGamma values
sysUtot = (np.sum((fileData[:,2]*fileData[:,4]/100)**2))**(0.5);

# file names pseudo data will be saved to
q2File = 'data\\DtoPi_PD_q2_b_SD1.txt' #.txt or .csv methods below
dGFile = 'data\\DtoPi_PD_VF_b_SD1.txt' #.txt or .csv methods below
# not a dG file if we are calculating VF

# number of pseudo data points generated for each exp data point
pseudoDataAmount = 2000000 #per each q^2 bin

# both methods below will create the inputs (q^2) while writing to the q^2 file
    # and the target data while writing to the target (dG or VF) file
    # just need to uncomment which file type you'd like to create!

#csvPDGen(q2File, dGFile, pseudoDataAmount, fileData, sysUtot, sysCorr, statCorr)
txtPDGen(q2File, dGFile, pseudoDataAmount, fileData, sysUtot, sysCorr, statCorr)