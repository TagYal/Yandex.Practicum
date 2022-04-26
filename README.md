# Яндекс.Практикум
Здесь представлены завершенные проекты, которые были выполнены в рамках обучения на программе Data Scientist (Practicum by Yandex) https://practicum.yandex.com <br>
Проекты разделены на блоки: условно DA (первичная работа с данными) и DS (работа с данными + применение алгоритмов DS, работа с моделями)

Python, Pandas, matplotlib, LinearRegression, RandomForestRegressor, CatBoostRegressor, временные ряды (statsmodels.tsa.seasonal)



import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


## DA

Номер| Название проекта| Содержание файла | Навыки, библиотеки и инструменты
----------------|----------------|----------------------|----------------------
1 | [Исследование надёжности заёмщиков - финтех (DA)](https://github.com/TagYal/Yandex.Practicum/tree/main/01.%20%D0%98%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%D0%B4%D1%91%D0%B6%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%B7%D0%B0%D1%91%D0%BC%D1%89%D0%B8%D0%BA%D0%BE%D0%B2%20-%20%D1%84%D0%B8%D0%BD%D1%82%D0%B5%D1%85%20(DA))       | На основе статистики о платёжеспособности клиентов исследовать влияет ли семейное положение и количество детей клиента на факт возврата кредита в срок  | EDA, предобработка данных, Python, Pandas, pivot-tables
2 | [Недвижимость (DA)](https://github.com/TagYal/Yandex.Practicum/tree/main/02.%20%D0%9D%D0%B5%D0%B4%D0%B2%D0%B8%D0%B6%D0%B8%D0%BC%D0%BE%D1%81%D1%82%D1%8C%20(DA))     | Продажа квартир в Санкт-Петербурге — анализ рынка недвижимости. На основе данных сервиса Яндекс.Недвижимость определена рыночная стоимость объектов недвижимости разного типа, типичные параметры квартир, в зависимости от удаленности от центра. | EDA, предобработка данных, Python, Pandas, pivot-tables, matplotlib
3 | [Определение оптимального тарифа - телеком (SDA)](https://github.com/TagYal/Yandex.Practicum/tree/main/03.%20%D0%9E%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BE%D0%BF%D1%82%D0%B8%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B3%D0%BE%20%D1%82%D0%B0%D1%80%D0%B8%D1%84%D0%B0%20-%20%D1%82%D0%B5%D0%BB%D0%B5%D0%BA%D0%BE%D0%BC%20(SDA))   | Определение оптимального тарифного плана. Коммерческий отдел хочет знать, какой из планов более выгоден, чтобы скорректировать рекламный бюджет.  | EDA, предобработка данных, Python, Pandas, pivot-tables, matplotlib, numpy, scipy, stats, критерии значимости, проверка null-гипотез (сравнение выборок)
4 | [Поиск наиболее успешной платформы для игр (DA, осн)](https://github.com/TagYal/Yandex.Practicum/tree/main/04.%20%D0%9F%D0%BE%D0%B8%D1%81%D0%BA%20%D0%BD%D0%B0%D0%B8%D0%B1%D0%BE%D0%BB%D0%B5%D0%B5%20%D1%83%D1%81%D0%BF%D0%B5%D1%88%D0%BD%D0%BE%D0%B9%20%D0%BF%D0%BB%D0%B0%D1%82%D1%84%D0%BE%D1%80%D0%BC%D1%8B%20%D0%B4%D0%BB%D1%8F%20%D0%B8%D0%B3%D1%80%20(DA%2C%20%D0%BE%D1%81%D0%BD))       | Используя исторические данные о продажах компьютерных игр, оценки пользователей и экспертов, жанры и платформы, выявить закономерности, определяющие успешность игры | EDA, предобработка данных, Python, Pandas, pivot-tables, matplotlib, numpy, scipy, stats, критерии значимости, проверка null-гипотез (сравнение выборок)
5 | [Спрос на авиабилеты (SQL) <br>](https://github.com/TagYal/Yandex.Practicum/tree/main/05.%20%D1%81%D0%BF%D1%80%D0%BE%D1%81%20%D0%BD%D0%B0%20%D0%B0%D0%B2%D0%B8%D0%B0%D0%B1%D0%B8%D0%BB%D0%B5%D1%82%D1%8B%20(SQL))       | Исследование данных авиакомпании — проверить гипотезу о повышении спроса во время фестивалей. Важно понять предпочтения пользователей, покупающих билеты на разные направления. Извлечены данные запросами на языке SQL и методами библиотеки PySpark. Изучена база данных и проанализирован спрос пассажиров на рейсы в города, где проходят крупнейшие культурные фестивали.  | SQL, EDA, предобработка данных, Python, Pandas, seaborn, scipy, проверка null-гипотез


## DS

Номер| Название проекта| Содержание файла | Стек
----------------|----------------|----------------------|----------------------
6 | [Тариф - телеком (DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/06.%20%D0%A2%D0%B0%D1%80%D0%B8%D1%84%20-%20%D1%82%D0%B5%D0%BB%D0%B5%D0%BA%D0%BE%D0%BC%20(DS))       | Разработать модель DS, которая бы предложила новый оптимальный тарифный план для каждого клиента оператора телеком. многие из его абонентов пользуются устаревшими тарифными планами. Они хотят разработать модель, которая будет анализировать поведение абонентов и рекомендовать один из новых тарифных планов Megaline: Smart или Ultra.  | EDA, предобработка данных, Python, Pandas, sklearn, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, accuracy
7 | [Отток клиентов банка (DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/07.%20%D0%9E%D1%82%D1%82%D0%BE%D0%BA%20%D0%BA%D0%BB%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2%20%D0%B1%D0%B0%D0%BD%D0%BA%D0%B0%20(DS))       | Из банка стали уходить клиенты каждый месяц. Спрогнозирована вероятность ухода клиента из банка в ближайшее время.  | EDA, предобработка данных, Python, Pandas, numpy, seaborn, matplotlib, sklearn, OHE, get_dummies, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, best parameters, accuracy, precision, recall, f1, roc_auc
8 | [Нефтяное месторождение (DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/08.%20%D0%9D%D0%B5%D1%84%D1%82%D1%8F%D0%BD%D0%BE%D0%B5%20%D0%BC%D0%B5%D1%81%D1%82%D0%BE%D1%80%D0%BE%D0%B6%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20(DS))       | Выбрать один из трёх регионов для разработки нового нефтяного месторождения. Построена модель для предсказания объёма запасов в новых скважинах. Выбраны скважины с самыми высокими оценками значений. Определены регионы с максимальной суммарной прибылью отобранных скважин. Построена модель для определения региона, где добыча принесёт наибольшую прибыль. Проанализирована возможная прибыль и риски.   | EDA, предобработка данных, Python, Pandas, numpy, matplotlib, scipy.stats, LinearRegression, RandomForestRegressor, mean_squared_error, оценка рисков через доверительный интервал 
9 | [Оптимизация добычи золота (DS, осн 2)](https://github.com/TagYal/Yandex.Practicum/tree/main/09.%20%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B4%D0%BE%D0%B1%D1%8B%D1%87%D0%B8%20%D0%B7%D0%BE%D0%BB%D0%BE%D1%82%D0%B0%20(DS%2C%20%D0%BE%D1%81%D0%BD%202))       | Компания разрабатывает решения для эффективной работы золотодобывающей отрасли. Построена модель, предсказывающая коэффициент восстановления золота из золотосодержащей руды. Проанализированы данные с параметрами добычи и очистки. Построена и обучена модель, помогающая оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.   | EDA, предобработка данных, Python, Pandas, numpy, matplotlib, LinearRegression, RandomForestRegressor, DummyRegressor, cross_val_score, mean_absolute_error, mean_squared_error
10 | [Cтрахование - поиск похожих клиентов](https://github.com/TagYal/Yandex.Practicum/tree/main/10.%20%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20-%20%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5)       | Для защиты данных клиентов страховой компании разработаны методы преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Была проведена предобработка данных. Произведена проверка работы алгоритма модели линейной регрессии при перемножении на обратимую матрицу.   | Python, Pandas, numpy, matplotlib, seaborn, NearestNeighbors, KNeighborsClassifier
11 | [Цена авто (DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/11.%20%D1%86%D0%B5%D0%BD%D0%B0%20%D0%B0%D0%B2%D1%82%D0%BE%20(DS))       | Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. Проанализированы данные: технические характеристики, комплектации и цены автомобилей. Построена модель для определения стоимости автомобиля с пробегом. Использованы численные методы, приближённые вычисления, оценка сложности алгоритма, градиентный спуск.   | Python, Pandas, matplotlib, LinearRegression, RandomForestRegressor, CatBoostRegressor, lgbm
12 | [Прогноз заказов такси (врем ряды, DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/12.%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%20%D0%B7%D0%B0%D0%BA%D0%B0%D0%B7%D0%BE%D0%B2%20%D1%82%D0%B0%D0%BA%D1%81%D0%B8%20(%D0%B2%D1%80%D0%B5%D0%BC%20%D1%80%D1%8F%D0%B4%D1%8B%2C%20DS))       | Проанализированы исторические данные о заказах такси в аэропортах. Спрогнозировано количество заказов такси на следующий час, чтобы привлекать больше водителей в период пиковой нагрузки. Построена модель для такого предсказания.   | Python, Pandas, matplotlib, LinearRegression, RandomForestRegressor, CatBoostRegressor, временные ряды (statsmodels.tsa.seasonal)
13 | [Классификация комментариев (текст, DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/13.%20%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BA%D0%BE%D0%BC%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%80%D0%B8%D0%B5%D0%B2%20(%D1%82%D0%B5%D0%BA%D1%81%D1%82%2C%20DS))       | Для запуска нового сервиса интернет-магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. Пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Обучена модель классифицировать комментарии на позитивные и негативные. Проанализирован набор данных с разметкой о токсичности правок. Построена модель со значением метрики качества F1 не меньше 0.75.   | numpy, pandas, matplotlib, seaborn, tqdm, BERT, tokenization, vectorizer
14 | [Финальный проект - телеком (DS)](https://github.com/TagYal/Yandex.Practicum/tree/main/14.%20%D0%A4%D0%B8%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%20-%20%D1%82%D0%B5%D0%BB%D0%B5%D0%BA%D0%BE%D0%BC%20(DS))       | Оператор связи Interconnect хотел бы иметь возможность прогнозировать отток клиентов.Если обнаружится, что пользователь собирается уйти, ему будут предложены промокоды и специальные тарифные планы. Маркетинговая команда Interconnect собрала некоторые личные данные своих клиентов, включая информацию об их планах и контрактах.   | Python, Pandas, Matplotlib, numpy, SciPy, описательная статистика, проверка статистических гипотез, math, Seaborn, sklearn, машинное обучение (CatBoost, LightGBM, LogisticRegression, DecisionTree, RandomForest), оптимальные параметры (RandomizedSearchCV)




Стек технологий/инструменты/подходы:
- Bootstrap
- CatBoost
- LightGBM
- math
- Matplotlib
- NLTK
- numpy
- Pandas
- Python
- SciKitLearn
- SciPy
- Seaborn
- sklearn
- SQL
- StatsModels
- визуализация данных
- исследовательский анализ данных
- исследовательский анализ данных
- лемматизация
- машинное обучение
- описательная статистика
- предобработка данных
- проверка статистических гипотез

   
