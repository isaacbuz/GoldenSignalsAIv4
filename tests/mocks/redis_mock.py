"""Mock Redis client for testing."""

class MockRedisClient:
    def __init__(self):
        self.data = {}
        self.pubsub_messages = []
        self.pubsub_subscribers = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    async def hget(self, name, key):
        hash_data = self.data.get(name, {})
        return hash_data.get(key)

    async def hset(self, name, key, value):
        if name not in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return 1

    async def publish(self, channel, message):
        self.pubsub_messages.append((channel, message))
        return len(self.pubsub_subscribers.get(channel, []))

    def pubsub(self):
        return MockPubSub(self)

    async def close(self):
        pass

class MockPubSub:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.channels = set()

    async def subscribe(self, *channels):
        self.channels.update(channels)

    async def unsubscribe(self, *channels):
        for channel in channels:
            self.channels.discard(channel)

    async def get_message(self, timeout=None):
        return None
