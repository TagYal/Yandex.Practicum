Предсказание цены автомобиля

# Цель
Предсказание цены автомобиля

# Задачи
Обучить модель для определения рыночной стоимости автомобиля. 

# Описание
Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. 
Проанализированы данные: технические характеристики, комплектации и цены автомобилей. Построена модель для определения стоимости автомобиля с пробегом.
Использованы численные методы, приближённые вычисления, оценка сложности алгоритма, градиентный спуск.


# Что использовано
Pandas
sklearn
numpy
LightGBM
машинное обучение
CatBoost

# Краткий вывод (без графиков и сводных таблиц)
Эта часть является результатом и анализом предыдущей части.
Подведем итоги и просмотрим их.

Итак... Подведем итоги:
RMSE_linear_model = 2763.0 (проверка работоспособности) очень быстро
RMSE_random_forest = 2444,9 быстро
RMSE_random_forest_tuned = 1996,8 (диапазоны: n_estimators, max_depth) медленно 3 часа
RMSE_lgbm_ver.1 = 1711,6 (по умолчанию) быстро
RMSE_lgbm_ver.2 = 1661,4 (n_estimators=200) быстро
RMSE_cat_ver.1 = 1714,7 (по умолчанию) быстро
RMSE_cat_tuned = 1773,2 (диапазоны: n_estimators, max_depth) средние 8 минут

Все эти модели кажутся не слишком образованными. Наилучший результат дает библиотека lgbm. Я думаю, мы можем настроить параметры, чтобы улучшить качество модели, чтобы получить тот же результат для cat_boost. Круто, что мы можем получить такой хороший результат прямо из коробки.

Чтобы улучшить модель, мы можем включить даты, которые я пропустил.
Мы можем попробовать настроить гиперпараметры модели повышения градиента. Похоже, мы не нашли ключевые параметры, которые могли бы резко повысить качество.
Можно попробовать использовать XGBoost (аналог CatBoost), но вряд ли это может повысить качество.
Может нам стоит лучше обработать данные. Было очень странно, что некоторые автомобили имеют очень низкую цену (<100 евро).