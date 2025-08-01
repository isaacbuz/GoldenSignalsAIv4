from celery import Celery

# TODO: Configure Celery with a proper broker URL from settings
app = Celery("tasks", broker="redis://localhost:6379/0")


@app.task
def example_task(x, y):
    return x + y
