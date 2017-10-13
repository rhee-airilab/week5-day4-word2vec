# coding: utf-8
import re

def remove_non_words(line):
    re_flags = re.IGNORECASE # re.UNICODE | re.IGNORECASE
    # 영문자,숫자,일부특수문자 제외 모든 문자
    # 숫자사이에 있는 `.' 를 제외한 `.'
    re_nonwords = r'([^A-Za-z0-9가-힣%. ]|[.,](?!\d))+'
    line = re.sub(re_nonwords, ' ', line, flags=re_flags)
    line = re.sub(r'\s+', ' ', line, flags=re_flags) # collapse repetitive spaces
    line = line.strip() # remove leading/training spaces
    return line
