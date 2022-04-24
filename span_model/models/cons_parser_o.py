import re
import benepar

def constituency_parse(sentence_str, threshold):
    parser = benepar.Parser("benepar_en3")
    input_sentence = benepar.InputSentence(
    words=sentence_str,
)

    results = []
    for i in range(len(list(parser.parse(input_sentence)))):
        tree = str(parser.parse(input_sentence)[i])
        sent = tree.replace('\n','',999)
        sent = re.sub(' +',' ',sent)
        sent = sent.replace('(. ', '(', 999)
        sent = sent.replace('(: ', '(', 999)
        sent = sent.replace('($ ', '(', 999)
        sent = sent.replace('(! ', '(', 999)
        sent = sent.replace('(, ', '(', 999)
        sent = sent.replace('(-LRB- ', '(', 999)
        sent = sent.replace('(-RRB- ', '(', 999)
        sent = sent.replace('(\'\' \'\')', '(\'\')', 999)
        sent = sent.replace('(\'\' \')', '(\')', 999)
        sent = sent.replace('(\'\' *', '(*', 999)
        sent = sent.replace('(\'\' !)', '(!)', 999)
        sent = sent.replace('(`` ', '(', 999)
        ## 使用正则去除词性标签，但是保留s根节点
        rule = '[\(][A-Z]+[\s\$]+'
        obj = re.findall(rule,sent)
        # print(obj)
        # print(len(obj))
        for item in obj:
            #if item != '(S ':
            sent = sent.replace(item,"(")
        b=[]
        c =sent.split(")"+')'*threshold)
        b = b + c
        result = []
        for item in b:
            # print(item)
            item = item.replace("(", "")
            # print(item)
            item = item.replace(")", "")
            # print(item)
            result.append(item)

        result = list(filter(None, result))
        for i in range(len(result)):
            result[i] = result[i].strip().split(' ')
        results.extend(result)
    return results

def span_process(sentence_str, span_len, start):
    span = []
    for i in range(len(sentence_str)):
        if i + span_len <= len(sentence_str):
            for j in range(span_len):
                span.append([i + start, i + j + start])
        else:
            for j in range(len(sentence_str) - i):
                span.append([i + start, i + j + start])
    return span


def constituency_span(sentence_str, span_len, threshold):
    results = constituency_parse(sentence_str, threshold)
    #print(results)
    span_results = []

    for i in range(len(results)):
        start = 0
        for j in range(i):
            start += len(results[j])
        span_results.extend(span_process(results[i], span_len, start))
    # print(len(span_results))
    return span_results


sentence_str = ['–', 'I', 'do', "n't", 'understand', 'how', 'I', 'was', 'a', 'stranger', 'to', 'this', 'place', 'for', 'so', 'long', '...', 'the', 'fajita', 'salad', ',', 'the', 'colorado', ',', 'the', 'fajitas', '-', 'EVERYTHING', 'is', 'delicious', '.']
span_results = constituency_span(sentence_str, 8, 3)
print(span_results)