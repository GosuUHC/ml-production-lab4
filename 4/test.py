# test.py

import requests
import time
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm

# --- Настройки ---
API_URL = "http://localhost:8000/predict"
NUM_REQUESTS = 200  # Общее количество запросов
CONCURRENCY = 10     # Количество одновременных запросов

# --- Глобальные переменные для сбора статистики ---
latencies = []
errors = 0


async def send_request(session, pbar):
    """Асинхронно отправляет один запрос и записывает результат."""
    global errors

    # Генерируем случайный ID пациента для каждого запроса
    random_patient_id = np.random.randint(1, 30000)
    payload = {"patient_id": random_patient_id}

    start_time = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload, timeout=10) as response:
            if response.status == 200:
                # Успешный запрос, добавляем задержку в список
                latency = (time.perf_counter() - start_time) * \
                    1000  # в миллисекундах
                latencies.append(latency)

                # Можно также логировать ответ для анализа
                data = await response.json()
                if np.random.random() < 0.01:  # Логируем 1% ответов для примера
                    print(f"Sample response - Patient {data['patient_id']}: "
                          f"prediction={data['prediction']}, "
                          f"model_version={data['model_version']}")

            else:
                # Ошибка на стороне сервера (500, 404 и т.д.)
                errors += 1
                print(
                    f"HTTP Error {response.status} for patient {random_patient_id}")

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        # Ошибка сети или таймаут
        errors += 1
        print(f"Request failed for patient {random_patient_id}: {e}")

    finally:
        # Обновляем прогресс-бар после каждого запроса
        pbar.update(1)


async def main():
    """Основная функция для запуска асинхронного нагрузочного теста."""
    print(
        f"Starting hospital readmission load test with {NUM_REQUESTS} requests and concurrency of {CONCURRENCY}...")

    # Сначала проверяем health endpoint
    try:
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health check: {health_response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Используем tqdm для визуализации прогресса
    with tqdm(total=NUM_REQUESTS) as pbar:
        async with aiohttp.ClientSession() as session:
            # Создаем задачи для всех запросов
            tasks = [send_request(session, pbar) for _ in range(NUM_REQUESTS)]
            # Запускаем задачи с определенным уровнем параллелизма
            semaphore = asyncio.Semaphore(CONCURRENCY)

            async def run_with_semaphore(task):
                async with semaphore:
                    await task

            await asyncio.gather(*[run_with_semaphore(task) for task in tasks])


def print_results(total_time):
    """Выводит результаты тестирования."""
    print("\n--- Hospital Readmission Load Test Results ---")
    if not latencies:
        print("No successful requests were made.")
        return

    total_requests_made = len(latencies) + errors
    error_rate_percent = (errors / total_requests_made) * \
        100 if total_requests_made > 0 else 0
    throughput = len(latencies) / total_time if total_time > 0 else 0

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total requests: {total_requests_made}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Failed requests: {errors}")
    print(f"Error Rate: {error_rate_percent:.2f}%")
    print(f"Throughput: {throughput:.2f} req/s")

    print("\n--- Latency Percentiles (ms) ---")
    print(f"p50 (Median): {np.percentile(latencies, 50):.2f} ms")
    print(f"p75: {np.percentile(latencies, 75):.2f} ms")
    print(f"p90: {np.percentile(latencies, 90):.2f} ms")
    print(f"p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"Min Latency: {np.min(latencies):.2f} ms")
    print(f"Max Latency: {np.max(latencies):.2f} ms")
    print(f"Average Latency: {np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    # Установка библиотек, если они не установлены
    try:
        import aiohttp
        from tqdm import tqdm
    except ImportError:
        print("Required libraries not found. Installing aiohttp and tqdm...")
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "aiohttp", "tqdm"])
        print("Installation complete. Please run the script again.")
        sys.exit(1)

    # Запускаем тест
    start_test_time = time.time()
    asyncio.run(main())
    end_test_time = time.time()

    # Выводим результаты
    print_results(total_time=end_test_time - start_test_time)
