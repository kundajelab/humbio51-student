#creates mini dataset for students to test alignment algorithm.
data=open("reference.fasta").read().strip().split('\n')
data_len=len(data)

#create 4 uniquely aligned reads
import random
unique_reads=[]
used_indices=[] 
for i in range(4):
    cur_index=random.randint(1,data_len)
    while cur_index in used_indices:
        cur_index=random.randint(1,data_len)
    used_indices.append(cur_index)
    unique_reads.append(data[cur_index])
    
#create 3 multimapped reads to several locations
multimappers=[] 
for i in range(3):
    cur_index=random.randint(1,data_len)
    while cur_index in used_indices:
        cur_index=random.randint(1,data_len)
    used_indices.append(cur_index)
    multimappers.append(data[cur_index])
    
#create 3 reads that map in many locations 
nonspecific=[]
for i in range(3):
    cur_index=random.randint(1,data_len)
    while cur_index in used_indices:
        cur_index=random.randint(1,data_len)
    used_indices.append(cur_index)
    nonspecific.append(data[cur_index])

#insert the multimappers
for read in multimappers:
    for i in range(3):
        insert_index=random.randint(1,data_len)
        data.insert(insert_index,read)
        
#insert the nonspecific reads 
for read in nonspecific:
    for i in range(100):
        insert_index=random.randint(1,data_len)
        data.insert(insert_index,read)
        

#Write out the spiked fasta file & the reads to align 
outf=open("prepped.fasta",'w')
outf.write("\n".join(data))
outf=open("sample.fasta",'w')
samples=unique_reads+multimappers+nonspecific
sample_index=0
for read in samples:
    outf.write(">read"+str(sample_index)+'\n')
    outf.write(read+'\n')
    sample_index+=1
    
