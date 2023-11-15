# Read and plot the data...

def plot_VIMS_data(infile,figfile_rawdata):
    with open(infile,newline='') as csvfile:
        reader = csv.reader(csvfile)
        counts = np.array(list(reader),dtype='float')[:,0]

# three-panel plot of raw data, saved as figfile_rawdata
# log scale
# linear scale
# zoomed linear cale

    fig1=plt.figure(1,figsize=(12,14)) # open up figure 
    plt.rcParams.update({'font.size': 18})

    plt.subplot(3,1,1)
    plt.plot(counts)
    #plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.ylim(0.1,max(counts)*1.5)
    #plt.ylim(max(min([counts,.1]),max(counts)*1.5))
    plt.yscale('log')
    plt.title(VIMS['event']+' raw observations')

    plt.subplot(3,1,2)

    plt.plot(counts)
    #plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.ylim(0.0,200)
    plt.yscale('linear')

    plt.subplot(3,1,3)

    plt.plot(counts)
    plt.xlabel('Frame number')
    plt.ylabel('DN')
    plt.xlim(800,1450)
    plt.ylim(-10,200)
    plt.yscale('linear')

    plt.show()
    plt.savefig(figfile_rawdata)
    print("Saved",figfile_rawdata,"\n")
    
    return counts # for later use

