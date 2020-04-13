from flask import Flask, render_template, redirect, url_for, request, Markup
import pickle
from os.path import join
import numpy as np

import requests
from nltk import sent_tokenize, regexp_tokenize
import pymorphy2

from natasha import NamesExtractor
import re

path = 'D:\\hse\\3 course\\course_work\\CourseWork2020\\06. App creation\\app\\Pickles'

with open(join(path, 'corr_matrix.pickle'), 'rb') as data:
    corr_matrix = pickle.load(data)

with open(join(path, 'groups_in_matrix.pickle'), 'rb') as data:
    groups_in_matrix = pickle.load(data)

with open(join(path, 'vectorizer.pickle'), 'rb') as data:
    bof = pickle.load(data)

with open(join(path, 'tfidf.pickle'), 'rb') as data:
    tfidf = pickle.load(data)

with open(join(path, 'svc_classifiers.pickle'), 'rb') as data:
    svc_classifiers = pickle.load(data)

with open(join(path, 'rfc_classifiers.pickle'), 'rb') as data:
    rfc_classifiers = pickle.load(data)

with open(join(path, 'svc_named_entity.pickle'), 'rb') as data:
    svc_named_entity = pickle.load(data)

url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"


def get_text(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    else:
        raise Exception('parameter [url] can be either URL or a filename')


def normalize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok
              for tok in tokens
              if tok not in stopwords and len(tok) >= min_length]
    return tokens


def tokenize_n_lemmatize(text, stopwords=None, normalize=True, regexp=r'(?u)\b\w{4,}\b'):
    words = [w for sent in sent_tokenize(text)
             for w in regexp_tokenize(sent, regexp)]
    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)
    return ' '.join(words)

stopwords_ru = get_text(url_stopwords_ru).splitlines()


# парсинг текста, приведение в нормальную форму
def parse_text(text):
    return tokenize_n_lemmatize(text)


# создание вектора tfidf
def make_tfidf_from_text(text_parsed):
    vect = tfidf.transform([text_parsed]).toarray()
    return vect


# создание bag of words из направления сочинения
def make_bof_from_direction(direction):
    direction_parsed = tokenize_n_lemmatize(direction)
    vect = bof.transform([direction_parsed]).toarray()
    return vect


# предсказание вероятности, что текст принадлежит определенной теме, с помощью svc
def make_text_prediction_svc(text):
    prediction = []
    text_parsed = parse_text(text)
    text_tfidf = make_tfidf_from_text(text_parsed)
    for svc in svc_classifiers:
        prediction.append(svc.predict_proba(text_tfidf)[0][1])
    return prediction


# предсказание вероятности, что текст принадлежит определенной теме, с помощью rfc
def make_text_prediction_rfc(text):
    prediction = []
    text_parsed = parse_text(text)
    text_tfidf = make_tfidf_from_text(text_parsed)
    for rfc in rfc_classifiers:
        prediction.append(rfc.predict_proba(text_tfidf)[0][1])
    return prediction


# создание направления сочинения
def make_direction_vector(prediction=None, direction=None, topic=None, threshold=0.12):
    k = len(prediction)
    direction_vect = np.array([0] * k)
    if prediction is not None:
        indices = np.argsort(prediction)[-5:]
        for i in indices:
            if prediction[i] >= threshold:
                direction_vect[i] = 1
    if direction is not None:
        direction_parsed = parse_text(direction)
        for word in direction_parsed.split():
            if word in bof.get_feature_names():
                for i in range(len(prediction)):
                    if bof.get_feature_names()[i] == word:
                        prediction[i] = 0.8
                        direction_vect[i] = 1
                        break
    if topic is not None:
        topic_parsed = parse_text(topic)
        for word in topic_parsed.split():
            if word in bof.get_feature_names():
                for i in range(len(prediction)):
                    if bof.get_feature_names()[i] == word:
                        prediction[i] = 0.8
                        direction_vect[i] = 1
                        break
    return direction_vect


# создание вектора именованных сущностей
def make_named_entity_vector(enteties):
    ne_vect = np.array([0] * len(groups_in_matrix))
    for _, _, g in enteties:
        ne_vect[g] = 1
    return ne_vect


def make_named_entity_fit(direction_vect):
    return direction_vect.dot(corr_matrix.T)


# проверка того, хорошее ли сочинение
def make_conclusion_for_writing(ne_vect, ne_fit):
    X = np.multiply(ne_vect, ne_fit)
    return svc_named_entity.predict(np.array([X]))[0]


# получение индексы альтернативных именованных сущностей, подходящих тексту
def get_alternative_named_enteties(ne_fit, threshold=0.5):
    m = len(ne_fit)
    result = []
    indices = np.argsort(ne_fit)[-5:]
    for i in indices:
        if ne_fit[i] >= threshold:
            result.append(i)
    return result


def get_longest_ne(group):
    max_l = 0
    l_el = ''
    for el in group:
        m = re.search('[\"](.+)[\"]', el)
        if m is not None:
            return '\'{}\''.format(m.group(1))
        if len(el) > max_l:
            l_el = el
            max_l = len(el)

    return l_el.title()


# Функция расчета схожести строк по коэффициенту Жаккара
def dist_jaccard(string1, string2):
    str1 = string1
    str2 = string2
    a = len(str1)
    b = len(str2)
    c = 0

    while str1 != '':
        s = str1[0]
        r = re.compile(s)
        c += min(len(r.findall(str1)), len(r.findall(str2)))
        str1 = r.sub('', str1)
        str2 = r.sub('', str2)

    return c / (a + b - c)


# Сравнение двух именованных сущностей
def compare_str(str1, str2):
    s1 = str1.split()
    s2 = str2.split()
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    n = len(s1)
    m = len(s2)
    result = False
    b1 = re.search('[\"](.+)[\"]', str1)
    b2 = re.search('[\"](.+)[\"]', str2)
    if b1 is not None and b2 is not None:
        if b1.group(1) == b2.group(1):
            return True
        else:
            return False
    elif b1 is not None or b2 is not None:
        return False
    if n == 3:
        if m == 3:
            if s1[0] == s2[0] and s1[1] == s2[1] and s1[2] == s2[2]:
                result = True
            elif s1[0][0] == s2[0][0] and s1[1][0] == s2[1][0] and s1[2] == s2[2] and (
                    len(s1[0]) == 1 or len(s2[0]) == 1):
                result = True
        elif m == 2:
            if s1[0] == s2[0] and s1[2] == s2[1]:
                result = True
            elif s1[0][0] == s2[0][0] and s1[2] == s2[1] and (len(s1[0]) == 1 or len(s2[0]) == 1):
                result = True
        else:
            if s1[2] == s2[0]:
                result = True
    elif n == 2:
        if m == 2:
            if s1[0] == s2[0] and s1[1] == s2[1]:
                result = True
            elif s1[0][0] == s2[0][0] and s1[1] == s2[1] and (len(s1[0]) == 1 or len(s2[0]) == 1):
                result = True
        else:
            if s1[1] == s2[0]:
                result = True
    elif n == 1:
        if s1[0] == s2[0]:
            result = True
    elif dist_jaccard(str1, str2) >= 0.9:
        result = True
    else:
        result = False

    return result


# Функция сравнения двух групп именовааных сущностей
def compare_with_group(name, group):
    for el in group:
        if compare_str(name, el):
            return True
    return False


def get_group(name):
    for i in range(len(groups_in_matrix)):
        if compare_with_group(name, groups_in_matrix[i]):
            return i
    return -1


def extract_names(text):
    extractor = NamesExtractor()
    matches = extractor(text)
    result = []
    pattern = re.compile('[\"\«\“](.+)[\"\»\”]')

    for match in matches:

        name = []
        start, stop = match.span

        if match.fact.first != None:
            name.append(match.fact.first)
        if match.fact.middle != None:
            name.append(match.fact.middle)
        if match.fact.last != None:
            name.append(match.fact.last)

        name = ' '.join(name).lower()
        group = get_group(name)

        result.append((start, stop, group))

        res_regexp = pattern.search(text, max(start - 75, 0), min(stop + 75, len(text)))
        if res_regexp is not None:
            book = name + ' \"' + res_regexp.group(1) + '\"'
            book_start, book_stop = res_regexp.span()
            book_group = get_group(book)
            if book_group != -1:
                result.append((book_start, book_stop, book_group))

    return result


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def number_to_rgb(i, up=1, down=-1):
    i = i - down
    up = up - down
    if i >= up/2:
        green = 255
        red = int(round(255 * (1-(i-up/2)/(up/2))))
    else:
        red = 255
        green = int(round(255 * (i/(up/2))))
    return red, green, 0


app = Flask(__name__)

message = {}


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html', message=message)

@app.route('/check_writing', methods=['POST'])
def check_writing():
    topic = request.form['topic']
    direction = request.form['direction']
    text = request.form['text']

    # Получаем именованные сущности
    named_enteties = extract_names(text)

    # Превращаем именованные сущности в вектор
    ne_vect = make_named_entity_vector(named_enteties)

    # Прогнозируем вероятности принадлежности к каждой теме
    prediction = make_text_prediction_rfc(text)

    # Из прогноза делаем вектор тем
    direction_vect = make_direction_vector(prediction=prediction, topic=topic, direction=direction)

    # Считаем, насколько бы подходили к данному тексту всевозможные именованные сущности
    ne_fit = make_named_entity_fit(direction_vect)

    # Делаем вывод, хорошее ли сочинение
    conclusion = make_conclusion_for_writing(ne_vect, ne_fit)

    directions_out = ''
    for i in range(len(direction_vect)):
        if direction_vect[i] == 1:
            directions_out += '<li>{} ({:.3f})</li>'.format(bof.get_feature_names()[i], prediction[i])
    message['directions'] = Markup('<ul>{}</ul>'.format(directions_out))

    message['text'] = ''
    for i in range(len(text)):
        f = False
        for s, e, g in named_enteties:
            if s == i:
                f = True
                if g == -1:
                    message['text'] += '<mark style="background-color:#b3b3b3;">' + text[i]
                else:
                    color = rgb_to_hex(number_to_rgb(ne_fit[g]))
                    message['text'] += '<mark style="background-color:#{};">'.format(color) + text[i]
                break
            elif e == i:
                f = True
                message['text'] += text[i] + '</mark>'
                break
        if not f:
            message['text'] += text[i]

    message['text'] = Markup(message['text'])

    alt_out = ''
    alt_ne = get_alternative_named_enteties(ne_fit)
    for ne_i in alt_ne:
        group = groups_in_matrix[ne_i]
        alt_out += '<li>{}</li>'.format(get_longest_ne(group))
    message['alt'] = Markup('<ul>{}</ul>'.format(alt_out))


    if conclusion == 1:
        is_good_out = '<h2 style="background-color: #4CAF50;" align="center">Хорошее сочинение!</h2>'
    else:
        is_good_out = '<h2 style="background-color: #F2502D;" align="center">Аргументы не соответствуют теме!</h2>'
    message['is_good'] = Markup(is_good_out)

    message['show'] = True
    if text == '' and topic == '' and direction == '':
        message['show'] = False

    return redirect(url_for('main',  _anchor='result'))
