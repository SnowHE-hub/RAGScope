"""Verify ownership each batch and observe other GPU users at a fixed cadence."""
import time
from ragscope.experiments.runtime import validate_gpu_lease


class EmbeddingServiceGuard:
    def __init__(self,service):self.service,self.last_gpu_check=service,0.0
    def __getattr__(self,name):return getattr(self.service,name)
    def verify(self):
        validate_gpu_lease(self.service.gpu.lease_path)
        self.service.owned_process()
        if self.service.record['status']!='READY':raise RuntimeError('Owned embedding service not ready')
        if time.monotonic()-self.last_gpu_check>=5:
            self.service.gpu.check();self.last_gpu_check=time.monotonic()
        return self.service.record
