class Cdata():
    def __init__(self):
        self.rawline=[]
        self.data=[]
        self.label=[]
        self.count=1000
        #self.windowsize=windowsize
        self.load_file()
        self.trans_data()

    def load_file(self):
        f=open("/Users/fdhcg/Desktop/clshen/data/1998.txt","r")
        self.rawline=f.readlines()
        f.close
    def trans_data(self):
        f=open("/Users/fdhcg/Desktop/clshen/data/test.txt","w")
        length=len(self.rawline)
        for i in range(length):           

            rawline=self.rawline[i].split(" ")
            line=[]
            for x in rawline:
                if x:
                    if x.endswith("/n"):
                        line.append(x[:-2])
                    else:
                        line.append(x)
            
            label=int(line[-2])
            if label!=0:
                f.write(self.rawline[i])
                self.count=1000
            if label==0 and self.count>0:
                f.write(self.rawline[i])
                self.count-=1
a=Cdata()
a.trans_data()