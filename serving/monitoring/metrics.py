from common.logger import get_logger

logger = get_logger("metrics")

class RequestMetrics:
    def __init__(self):
        self.count = 0
        self.total_latency = 0.0

    def log_request(self, latency):
        self.count += 1
        self.total_latency += latency
        avg_latency = self.total_latency / self.count
        logger.info("Request #%d, latency=%.3f sec, avg_latency=%.3f sec", self.count, latency, avg_latency)

# Usage in API:
# from metrics import RequestMetrics
# metrics = RequestMetrics()
# start = time()
# ... process request ...
# metrics.log_request(time()-start)
