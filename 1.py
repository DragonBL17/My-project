import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import plotly.graph_objs as go

st.set_page_config(layout="wide")
color_pallete = ['#6f9de3', '#dfaa5a', '#e3986f', '#afc9ef', '#2f71d7']
sns.set_palette(sns.color_palette(color_pallete))
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv', delimiter=",")
st.title('Прогнозирование оттока сотрудников')
st.write(' ')
st.subheader("Данный проект направлен на создание программы, которая прогнозирует увольнение сотрудников на основе открытого набора данных (https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)")
st.write(' ')
st.markdown(
    ''' 
    Содержание :
    
    1. Исследовательский анализ данных
    
    2. Предобработка данных
    
    3. Построение моделей
    
    4. Заключение

    Информация о датасете:
    
    Данные содержат 1470 объектов и 35 признаков.
    
    Признаки : 
    Age, Daily Rate, Distance FromHome, Education, Employee Count, Employee Number, 
    Environment Satisfaction, Hourly Rate, Job Involvement, Job Level,Job Satisfaction,
    Monthly Income, Monthly Rate, NumCompanies Worked,	Percent Salary Hike, Performance Rating,
    Relationship Satisfaction, Standard Hours, Stock Option Level, Total Working Years,
    Training Times Last Year, Work Life Balance, Years At Company, Years In Current Role,
    Years Since Last Promotion, Years With Curr Manager.

    ''')
st.write(' ')
if st.checkbox('Показать данные'):
    st.write('### Данные')
    st.write(data)
st.write('### Исследовательский анализ данных')
st.write(' ')
st.write(data.head())
st.write(' ')
st.write('#### 1. Проверим данные на пропущенные значения')
with st.echo():
    data.columns[data.isnull().sum() != 0]
st.write("В наших данных нет пропущенных начений")
st.write(' ')
st.write("#### 2. посмотрим на статестические показатели признаков")
with st.echo():
    data.describe()
st.write(data.describe())
st.write(' ')
st.write(" Установим уникальный номер сотрудника как индекс и удалим три контсантных признака, они не имеют веса для модели")
with st.echo():
    data = data.set_index('EmployeeNumber')
    data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis = 1, inplace=True)
st.write(' ')
st.write('#### 3.Визуализация данных')
st.write(' ')
st.write("#### Кореляция всех числовых признаков")
st.write(' ')
with st.echo(code_location='below'):
    n = data.columns[data.dtypes == 'float64'].union(data.columns[data.dtypes == 'int64'])
    fig, ax = plt.subplots(figsize=(23, 20))
    fig = go.Figure(data=go.Heatmap(z=data[n].corr(),
                                      x = data[n].columns,
                                      y = data[n].columns,
                                      colorscale = "bluyl",))
    fig.update_layout(autosize=False, width=900, height=800)
    fig.update_xaxes(side="top")
    st.plotly_chart(fig)
st.write(''' 
    Из графика видно, что большиство столбцов плохо коррелируют друг с другом.
    Cильная кореляция: TotalWorkingYears и JobLevel, MonthlyIncome и JobLevel, MonthlyIncome и TotalWorkingYears, PercentSalaryHike и PerformanceRating, YearsInCurrentRole и YearsAtCompany, YearsWithCurrManager и YearsAtCompany.''')
st.write(' ')
st.write("#### Целевая переменная : Attrition")
st.write(' ')
with st.echo(code_location='below'):
    f, axes = plt.subplots(1, 2, figsize=(15, 10))
    data.Attrition.value_counts(normalize=True).plot.pie(explode=[0, 0.20], autopct='%1.1f%%', ax=axes[0], colors = ['#6f9de3', '#dfaa5a'])
    sns.countplot('Attrition', data=data, palette = ['#6f9de3', '#dfaa5a'],ax = axes[1])
    st.pyplot(f)
st.write('Заметно, что данные несбалансированны, так как 1 класс значительно больше другого по количеству')
st.write(' ')
st.write("#### Выгорание взависимости от пола")
with st.echo(code_location='below'):
    f = plt.figure(figsize=(25, 10))
    sns.countplot(x = 'Gender', data = data, hue = 'Attrition')
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    data[data['Gender'] == "Female"].Attrition.value_counts(normalize=True).plot.pie(autopct='%1.1f%%', ax = axes[0])
    axes[0].set_title("Female")
    data[data['Gender'] == "Male"].Attrition.value_counts(normalize=True).plot.pie(autopct='%1.1f%%', ax = axes[1])
    axes[1].set_title("Male")
    st.pyplot(f)
    st.pyplot(fig)
st.write("Мужчины склонны выгорать чаще, чем женщины.")
st.write(' ')
st.write("#### Переработка относительно выгорания сотрудников")
with st.echo(code_location='below'):
    f, axes = plt.subplots(1, 2, figsize=(25, 10))
    data[data.Attrition == 'No'].OverTime.value_counts(normalize=True).plot.pie(explode=[0,0.07], autopct='%1.1f%%',  ax=axes[0])
    axes[0].set_title("Attrition = No")
    axes[0].yaxis.tick_left()

    data[data.Attrition == 'Yes'].OverTime.value_counts(normalize=True).plot.pie(explode=[0,0.07], autopct='%1.1f%%',  ax=axes[1],colors = ['#dfaa5a','#6f9de3'])
    axes[1].yaxis.tick_left()
    axes[1].set_title("Attrition = Yes")
    st.pyplot(f)
st.write("Среди уволившихся более половины сотрудников перерабатывали, а среди оставшихся только четверть перерабатывают.")
st.write(' ')
st.write("#### Pейтинг производительности относительно процента повышения зарплаты")
with st.echo(code_location='below'):
    f = plt.figure(figsize=(25,10))
    sns.boxplot(x = data['PerformanceRating'], y = data['PercentSalaryHike'], hue = data['Attrition'])
    st.pyplot(f)
st.write('''
        Средний процент увеличения заработной платы больше у сотрудников с более высокой степенью продуктивности, как видно на графике справа.
        Также, сотрудники, которые увольняются с отличной производительностью, получают более низкий средний процент повышения заработной платы, чем остальные сотрудники категории PerformanceRating.''')
st.write(' ')
st.write("#### Частота командирвок относительно ухода сотрудников")
with st.echo():
    f, axes = plt.subplots(1, 3, figsize=(15, 10))
    data[data['BusinessTravel'] == "Travel_Rarely"].Attrition.value_counts(normalize=True).plot.pie(autopct='%1.2f%%', ax = axes[0])
    axes[0].set_title("Travel Rarely")
    data[data['BusinessTravel'] == "Travel_Frequently"].Attrition.value_counts(normalize=True).plot.pie(autopct='%1.2f%%', ax = axes[1])
    axes[1].set_title("Travel Frequently")
    data[data['BusinessTravel'] == 'Non-Travel'].Attrition.value_counts(normalize=True).plot.pie(autopct='%1.2f%%', ax = axes[2])
    axes[2].set_title("Non Travel")
    st.pyplot(f)
st.write('Из графика следует вывод, кто путешествует чаще - выгорает. Большинство сотрудников редко путешествовали, поэтому большинство тех, кто ушел, относятся к этой категории. Интересным явлением является то, что самая большая разница в процентном соотношении  наблюдается в графике людей, которые не путешествовали.')
st.write(' ')
st.write('### Предобработка данных')
with st.echo():
    #Кодируем категориальные признаки с помощью Label Encoder
    category = data.columns[data.dtypes == 'object']
    df = data.copy()
    encoder = LabelEncoder()
    for i in category:
        df[i] = encoder.fit_transform(df[i])
    #Задаем переменные x и y
    x = df.drop('Attrition', axis=1)
    y = df.Attrition
    #Делим на train и test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    #Используем scaller
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    # Установим парметры разбиения для кросс валидации
    kfold = KFold(n_splits = 10, shuffle=True, random_state = 42)
st.write(' ')
st.write('### Обучение моделей')
space1, row_1, space2, row_2, space3, row_3, space4 = st.beta_columns((.1, 2, .1, 2, .1, 2, .1))
with row_1:
    st.markdown('Logistic regression')
    with st.echo(code_location='below'):
        LogReg = LogisticRegression(random_state=42, class_weight = {0:0.4,1:0.6},C=1, max_iter=20, penalty = 'l2',) # установим веса, так как у нас несбалансированные данные.
        #Обучение модели на кросс валидациию Метрики качества : Accuracy and F1-score
        st.write("Cross Validation on Logistic Regression")
        accuracy = (cross_val_score(LogReg, X_train, y_train, cv= kfold, scoring ='accuracy').mean())
        f1 = (cross_val_score(LogReg, X_train, y_train, cv=kfold, n_jobs=1, scoring='f1').mean())
        st.write("Accuracy: {}".format(accuracy))
        st.write("F1 score: {}".format(f1))
        st.write(' ')
        st.write("Test score on Logistic Regression")
        LogReg.fit(X_train, y_train)
        pred_log = LogReg.predict(X_test)
        st.write("Accuracy: {}".format(metrics.accuracy_score(y_test, pred_log)))
        st.write("F1 score: {}".format(metrics.f1_score(y_test, pred_log)))
        st.write(" ")
with row_2:
    st.write("Extra Tree Classifier")
    with st.echo(code_location='below'):
        extree = ExtraTreesClassifier(criterion = "entropy")
        #Обучение модели на кросс валидациию Метрики качества : Accuracy and F1-score
        st.write("Cross Validation on Extra Tree Classifier")
        accuracy = (cross_val_score(extree, X_train, y_train, cv= kfold, scoring ='accuracy').mean())
        f1 = (cross_val_score(extree, X_train, y_train, cv=kfold, scoring='f1').mean())
        st.write("Accuracy: {}".format(accuracy))
        st.write("F1 score: {}".format(f1))
        st.write(' ')
        st.write("Test score on Extra Tree Classifier")
        extree.fit(X_train, y_train)
        pred_extree = extree.predict(X_test)
        st.write("Accuracy: {}".format(metrics.accuracy_score(y_test, pred_extree)))
        st.write("F1 score: {}".format(metrics.f1_score(y_test, pred_extree)))
        st.write(' ')
with row_3:
    st.write("Cat Boost Classifier")
    st.write ('Произведем подбор параметров и обучим модель с учетом их')
    with st.echo(code_location='below'):
        params ={
        'iterations': [40,60,80,100],
        'learning_rate' : [0.2,0.4,0.5,0.6,0.7],
        'random_state' : [42],
        'verbose': [0],
        'depth': [4,6,8,10]}
        catb = CatBoostClassifier()
        Grid = GridSearchCV(estimator=catb, param_grid = params, cv = 3, n_jobs= 1)
        Grid.fit(X_train, y_train)
        parameters = Grid.best_params_
        st.write(Grid.best_params_)
        catb = CatBoostClassifier(**parameters)
        kfold = KFold(n_splits = 10, shuffle=True, random_state = 42)
        #Обучение модели на кросс валидациию Метрики качества : Accuracy and F1-score
        st.write("Cross Validation on Cat Boost Classifier")
        accuracy = (cross_val_score(catb, X_train, y_train, cv= kfold, scoring ='accuracy').mean())
        f1 = (cross_val_score(catb, X_train, y_train, cv=kfold, scoring='f1').mean())
        st.write("Accuracy: {}".format(accuracy))
        st.write("F1 score: {}".format(f1))
        st.write(' ')
        st.write("Test score on Cat Boost Classifier")
        catb.fit(X_train, y_train)
        pred_catb = catb.predict(X_test)
        st.write("Accuracy: {}".format(metrics.accuracy_score(y_test, pred_catb)))
        st.write("F1 score: {}".format(metrics.f1_score(y_test, pred_catb)))
        st.write(" ")

st.write('### Заключение')
st.write(''' Исходя из полученных результатов всех трех моделей можно сказать, что самой лучшей моделью для предсказания сотрудников является Логистическая Регрессия.
Boost Classifier показал самый высокий результат accuracy на тестовой выборке, однако так как мы имеем несбалансированные данные, то лучше всего ориентироваться на метрику F1 score.
''')
st.write(' ')
st.write('Посмотрим на коэфиценты регресии')
with st.echo(code_location="below"):
    st.write(np.std(X_train, 0) * LogReg.coef_)
st.write(' ')
st.write('Feature importnatce для Логистическй регресии')
with st.echo(code_location="below"):
    fig,ax = plt.subplots(figsize=(15, 10))
    plt.bar([i for i in range(len(LogReg.coef_[0]))], LogReg.coef_[0])
    ax.xaxis.set_ticks(np.arange(0, 30))
    ax.set_xticklabels(x.columns, rotation = "vertical")
    st.pyplot(fig)
st.write('Исходя из этого можно сказать, что наибольшую роль для прогнозирования оттока сотрудников имеют следующие факторы : Переработка, Количество лет в компании, Сколько времени прошло с прошлого повышения')











