import vk
import random
import requests
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

st.set_page_config(layout="wide")
st.title('Граф друзей')
st.write(' ')
st.subheader('Данный проект создает граф дружественных связей для случайного участника группы НИУ ВШЭ во вконтакте')
st.write('### Подключаемся к VK Api')
st.write('Считываем токен для подключения')
with st.echo():
    token = open('/Users/kseniayakunina/PycharmProjects/Ksenia/token.txt', "r")
    g = str(token.readline())
st.write('Запускаем сессию')
with st.echo():
    session = vk.Session(access_token = g)
    api_vk = vk.API(session, v='5.131')
st.write("### Получаем список подписчиков группы, максимальное количество 1000 человек для одного запроса")
st.write(' Сообщество - официальная группа НИУ ВШЭ')
with st.echo():
    url = "https://api.vk.com/method/groups.getMembers?group_id=25205856&v=5.21&access_token={}".format(g)
    response = requests.get(url)
    d = response.json()
    followers = d.get('response', {})['items'] #Список подписчиков

st.write("### Bыбираем одного пользователя случайным образом используя метод random.choice()")
st.write('''
Проверяем открыта ли у него страница путем отправки запроса. 
Eсли открыта, то просто получаем всех его друзей, если нет, то выбираем другого пользователя.''')
with st.echo():
    open = 0
    while open != 1:
        try:
            a = int(random.choice(followers))
            st.write("Cheking {}".format(a))
            f = api_vk.friends.get(user_id= a)
            open = 1
        except :
            open = 0

st.write("### Создаем словарь для всех друзей нашего подписчика и сохраняем информацию о всех их друзьях")
with st.echo():
    df = {}
    for friend in f['items']:
        try:
            df[friend] = api_vk.friends.get(user_id= int(friend))
        except:
            continue

st.write("### Создаем граф дружественных связей")
st.write("Ребро образуется если два друга выбранного изначально нами подписика дружат между собой, также оно образуется для всех пар (подписчик,друг)")
with st.echo():
    G = nx.Graph()
    G.add_node(a)
    for i in df:
        G.add_node(i)
        G.add_edge(i, a)
        for j in df[i]['items']:
            if i in f['items'] and j in f['items']:
                G.add_node(j)
                G.add_edge(i, j)

st.write("### Рисуем граф")
st.write("Размер вершины будет зависить от ее степени")
degree = []
with st.echo(code_location='below'):
    for i in G.nodes:
        degree.append(G.degree[i]+20) # прибавим 20 чтобы совсем маленькие вершины не пропадали
    fig = plt.figure(figsize=(30, 30))
    nx.draw(G, with_labels=True, node_color='#6f9de3', edge_color='#dfaa5a', node_size = degree)
    st.pyplot(fig)
