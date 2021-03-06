Прогнозирование заказов такси

# Цель
Прогнозирование заказов такси

# Задачи
Обучить модель для предсказания количества заказов такси на следующий час. 

# Описание
Проанализированы исторические данные о заказах такси в аэропортах.  
Спрогнозировано количество заказов такси на следующий час, чтобы привлекать больше водителей в период пиковой нагрузки. 
Построена модель для такого предсказания.
Значение метрики RMSE на тестовой выборке должно меньше 48.


# Что использовано
Pandas
sklearn
numpy
LightGBM
Matplotlib
StatsModels
CatBoost
машинное обучение

# Краткий вывод (без графиков и сводных таблиц)
**Вывод**
Лучшей моделью является буст-модель кота с параметрами по умолчанию. <br>
Это дает RMSE (тест): 40,9, что лучше, чем необходимое значение 48,0. <br>

*Проверка работоспособности* <br>
RMSE для модели медианного прогнозирования: 87,2 <br>
RMSE для предыдущей модели стоимости: 58,9 <br><br>
*Линейный* number_lag=77 и window_size=26 <br>
RMSE для линейной модели (тест): 42,9 <br><br>
*Случайный лес* number_lag=81, window_size=21, max_depth=10, n_estimators=200 <br>
RMSE для модели случайного леса (тест): 41,6 <br><br>
*Cat boost* max_lag = 3 roll_mean_size = 12<br>
RMSE для модели повышения кота (тест): 40,9 <br>