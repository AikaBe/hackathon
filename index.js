// Получение данных о автобусных остановках
fetch('/api/bus_stops')
    .then(response => response.json())
    .then(data => {
        data.forEach(stop => {
            // Добавление маркера для каждой остановки
            new mapgl.Marker(map, {
                coordinates: [stop.longitude, stop.latitude], // Используйте ваши координаты
            });
        });
    })
    .catch(error => console.error('Ошибка:', error));
