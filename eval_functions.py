def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def parse_krn_content(krn, ler_parsing=False, cer_parsing=False):
    if cer_parsing:
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("\t", " <t> ")
        tokens = krn.split(" ")
        characters = []
        for token in tokens:
            if token not in ['<b>', '<t>']:
                characters.append(token)
            else:
                for char in token:
                    characters.append(char)
        return characters
    elif ler_parsing:
        krn_lines = krn.split("\n")
        for i, line in enumerate(krn_lines):
            line = line.replace("\n", " <b> ")
            line = line.replace("\t", " <t> ")
            krn_lines[i] = line
        return krn_lines
    else:
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("\t", " <t> ")
        return krn.split(" ")

import re
def parse_abc_content(abc, ler_parsing=False, cer_parsing=False):
    assert not (ler_parsing and cer_parsing)
    if cer_parsing:
        characters = []
        for token in re.split(r'(<\|text\|>)', ''.join(abc)):
            if token not in ['<|text|>']:
                for char in token:
                    characters.append(char)
            else:
                characters.append(token)
        return characters
    if ler_parsing:
        abc_lines = ''.join(abc).split("\n")
        return abc_lines
    else:
        return abc
        

def compute_metric(a1, a2):
    acc_ed_dist = 0
    acc_len = 0
    
    for (h, g) in zip(a1, a2):
        acc_ed_dist += levenshtein(h, g)
        acc_len += len(g)
    
    return 100.*acc_ed_dist / acc_len
    
def compute_poliphony_metrics_abc(hyp_array, gt_array):
    hyp_cer = []
    gt_cer = []

    hyp_ser = []
    gt_ser = []

    hyp_ler = []
    gt_ler = []
    
    for h_tokens, gt_tokens in zip(hyp_array, gt_array):
        hyp_ler.append(parse_abc_content(h_tokens, ler_parsing=True, cer_parsing=False))
        gt_ler.append(parse_abc_content(gt_tokens, ler_parsing=True, cer_parsing=False))

        hyp_ser.append(parse_abc_content(h_tokens, ler_parsing=False, cer_parsing=False))
        gt_ser.append(parse_abc_content(gt_tokens, ler_parsing=False, cer_parsing=False))

        hyp_cer.append(parse_abc_content(h_tokens, ler_parsing=False, cer_parsing=True))
        gt_cer.append(parse_abc_content(gt_tokens, ler_parsing=False, cer_parsing=True))
    
    acc_ed_dist = 0
    acc_len = 0

    cer = 0
    ser = 0
    ler = 0

    for (h, g) in zip(hyp_cer, gt_cer):
        acc_ed_dist += levenshtein(h, g)
        acc_len += len(g)
    
    cer = compute_metric(hyp_cer, gt_cer)
    ser = compute_metric(hyp_ser, gt_ser)
    ler = compute_metric(hyp_ler, gt_ler)

    return cer, ser, ler


def compute_poliphony_metrics(hyp_array, gt_array):
    hyp_cer = []
    gt_cer = []

    hyp_ser = []
    gt_ser = []

    hyp_ler = []
    gt_ler = []
    
    for h_string, gt_string in zip(hyp_array, gt_array):
        hyp_ler.append(parse_krn_content(h_string, ler_parsing=True, cer_parsing=False))
        gt_ler.append(parse_krn_content(gt_string, ler_parsing=True, cer_parsing=False))

        hyp_ser.append(parse_krn_content(h_string, ler_parsing=False, cer_parsing=False))
        gt_ser.append(parse_krn_content(gt_string, ler_parsing=False, cer_parsing=False))

        hyp_cer.append(parse_krn_content(h_string, ler_parsing=False, cer_parsing=True))
        gt_cer.append(parse_krn_content(gt_string, ler_parsing=False, cer_parsing=True))
    
    acc_ed_dist = 0
    acc_len = 0

    cer = 0
    ser = 0
    ler = 0

    for (h, g) in zip(hyp_cer, gt_cer):
        acc_ed_dist += levenshtein(h, g)
        acc_len += len(g)
    
    cer = compute_metric(hyp_cer, gt_cer)
    ser = compute_metric(hyp_ser, gt_ser)
    ler = compute_metric(hyp_ler, gt_ler)

    return cer, ser, ler

def extract_music_text(array):
    lines = array.split("\n")
    lyrics = []
    symbols = []
    for idx, l in enumerate(lines):
        if '.\t.\n' in l:
            continue
        if idx > 0 and len(l.rstrip().split('\t')) > 1:
            symbols.append(l.rstrip().split('\t')[0])
            lyrics.append(l.rstrip().split('\t')[1])
 
    return lyrics, symbols, " ".join(lyrics)

def extract_music_textllevel(array):
    lines = []
    lcontent = []
    completecontent = []
    krn = array.split("\n")
    for line in krn:
        line = line.replace("\n", "<b>")
        line = line.split("\t")
        if len(line)>1:
            lcontent.append(line[0])
            completecontent.append(line[0])
            lcontent.append("<t>")
            completecontent.append("<t>")
            for token in line[1]:
                if token != '<':
                    lcontent.append(token)
                    completecontent.append(token)
                else:
                    lcontent.append("<b>")
                    break
        
        lines.append(lcontent)
        lcontent = []
                
    return lines, completecontent
    