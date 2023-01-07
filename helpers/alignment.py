import subprocess
def align(sample,reference,outputf):
    command_args=["bowtie2","-x",reference,"-f","-U",sample,"-S",outputf+".sam"]
    print(str(command_args))
    proc = subprocess.Popen (command_args, shell=False, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    proc.wait() 
    #out = proc.communicate()
    #print the output of the child process to stdout
    #convert the aligned file from sam to tagAlign format
    convert_sam_to_tagalign(outputf)
    
def convert_sam_to_tagalign(sam_filename):
    data=open(sam_filename+".sam",'r').read().strip().split('\n') 
    outf=open(sam_filename+".bed",'w')
    for line in data:
        if line.startswith("@"):
            continue
        tokens=line.split()
        if (len(tokens)<10):
            continue
        #print(tokens)
        chrom=tokens[2]
        startpos=tokens[3]
        seq=tokens[9]
        if tokens[1]!="4": 
            endpos=str(int(startpos)+len(seq))
        else:
            endpos="0"
        q=tokens[4]
        outf.write(chrom+'\t'+startpos+'\t'+endpos+'\t'+seq+'\t'+q+'\t'+'+'+'\n')

def select_column(outputf,column):
    data=open(outputf,'r').read().strip().split('\n')
    tokens=[line.split('\t')[column] for line in data[3::]]
    print('\n'.join(tokens))

def bowtie_index(reference,outf_prefix):
    create_spiked_dataset(reference)
    command_args=["bowtie2-build","-f",reference,outf_prefix]
    proc=subprocess.Popen(command_args,shell=False,stdout=subprocess.PIPE)
    out=proc.communicate()[0]

def create_spiked_dataset(reference):
    #creates mini dataset for students to test alignment algorithm.
    data=open(reference,'r').read().strip().split('\n')
    data_len=len(data)
    first_insert_loc=201 
    #create 4 uniquely aligned reads
    import random
    unique_reads=[]
    used_indices=[] 
    for i in range(4):
        cur_index=random.randint(first_insert_loc,data_len)
        while cur_index in used_indices:
            cur_index=random.randint(first_insert_loc,data_len)
        used_indices.append(cur_index)
        unique_reads.append(data[cur_index])

    #create 3 multimapped reads to several locations
    multimappers=[] 
    for i in range(3):
        cur_index=random.randint(first_insert_loc,data_len)
        while cur_index in used_indices:
            cur_index=random.randint(first_insert_loc,data_len)
        used_indices.append(cur_index)
        multimappers.append(data[cur_index])

    #create 3 reads that map in many locations 
    nonspecific=[]
    for i in range(3):
        cur_index=random.randint(first_insert_loc,data_len)
        while cur_index in used_indices:
            cur_index=random.randint(first_insert_loc,data_len)
        used_indices.append(cur_index)
        nonspecific.append(data[cur_index])
    
    #insert the multimappers
    for read in multimappers:
        for i in range(3):
            insert_index=random.randint(first_insert_loc,data_len)
            data.insert(insert_index,read)

    #insert the nonspecific reads 
    for read in nonspecific:
        for i in range(100):
            insert_index=random.randint(first_insert_loc,data_len)
            data.insert(insert_index,read)

            
    #Write out the spiked fasta file & the reads to align 
    outf=open(reference,'w')
    outf.write("\n".join(data))
    outf.close() 
    outf=open("samples.fasta",'w')
    samples=unique_reads+multimappers+nonspecific
    sample_index=0
    for read in samples:
        outf.write(">read"+str(sample_index)+'\n')
        outf.write(read+'\n')
        sample_index+=1
    #create a single random read that does not map to the reference
    outf.write(">read"+str(sample_index)+'\n')
    outf.write("ATCGATCGATATCGATCGATATCGATCGATATCGATCGATATCGATCGAT\n")


