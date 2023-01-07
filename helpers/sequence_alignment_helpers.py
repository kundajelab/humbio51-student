
def insert_newlines(string,every=50):
    lines=[]
    for i in range(0,len(string),every):
        lines.append(string[i:i+every])
    return lines

def format_alignment_linebreak(align1_linebreaks,align2_linebreaks,score,begin,end,seq1_id,seq2_id):
    s=[]
    for line in range(0,len(align1_linebreaks)):
        s.append(seq1_id + ":" + "%s%s\n" %(" "*((max(len(seq1_id),len(seq2_id))+1)-len(seq1_id)),align1_linebreaks[line]))
        s.append("%s%s\n" % (" "*(max(len(seq1_id),len(seq2_id))+2), "|" * (len(align1_linebreaks[line]))))
        s.append(seq2_id + ":" + "%s%s\n" %(" "*((max(len(seq1_id),len(seq2_id))+1)-len(seq2_id)),align2_linebreaks[line]))
        s.append('\n')
    s.append("  Score=  %g\n" % score)
    s.append("  Begin=  %g\n" % begin)
    s.append("  End  =  %g\n" % end)
    s.append("  Length= %g\n" % (end-begin))
    return ''.join(s)

def format_alignment_linebreak_withnumbering(align1_linebreaks,align2_linebreaks,score,begin,end,seq1_id,seq2_id):
    s=[]
    bpstart1=0
    bpend1=0
    bpstart2=0
    bpend2=0
    for line in range(0,len(align1_linebreaks)):
        bpstart1 = bpend1 + 1
        bpend1 = bpend1 + 50 - align1_linebreaks[line].count('-')
        bpstart2 = bpend2 + 1
        bpend2 = bpend2 + 50 - align2_linebreaks[line].count('-')
        s.append(seq1_id + ' Range:'+str(bpstart1)+'-'+str(bpend1))
        s.append('\n')
        s.append(seq1_id + ":" + "%s%s\n" %(" "*((max(len(seq1_id),len(seq2_id))+1)-len(seq1_id)),align1_linebreaks[line]))
        s.append("%s%s\n" % (" "*(max(len(seq1_id),len(seq2_id))+2), "|" * (len(align1_linebreaks[line]))))
        s.append(seq2_id + ":" + "%s%s\n" %(" "*((max(len(seq1_id),len(seq2_id))+1)-len(seq2_id)),align2_linebreaks[line]))
        s.append(seq2_id + ' Range:'+str(bpstart2)+'-'+str(bpend2))
        s.append('\n')
        s.append('\n')
    s.append("  Score=  %g\n" % score)
    s.append("  Begin=  %g\n" % begin)
    s.append("  End  =  %g\n" % end)
    s.append("  Length= %g\n" % (end-begin))
    return ''.join(s)