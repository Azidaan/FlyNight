"""Keep causal checkpoints but reproduce native sequential decoding exactly."""
from sparse_views import SparseFrames,SparseViews

class SequentialFrames(SparseFrames):
    def _reset(self,frame):
        super()._reset(frame)
        start=self.cursor
        self.close()
        self.cap=self.cv2.VideoCapture(str(self.path));assert self.cap.isOpened()
        # The saved retina has already consumed0..start-1. Decode and discard
        # those raw frames, without advancing it again. cap.set was not pixel
        # equivalent to the native decode path in the observed MOV source.
        for _ in range(start):
            ok,_image=self.cap.read();assert ok
        self.cursor=start

class SequentialViews(SparseViews):
    def __init__(self,video,*,force_initial=False):
        super().__init__(video,force_initial=force_initial)
        self.frames.close()
        self.frames=SequentialFrames(video,force_initial=force_initial)
