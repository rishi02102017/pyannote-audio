import traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from importlib.metadata import version
print("pyannote.audio", version("pyannote.audio"), "core", version("pyannote.core"), "torch", torch.__version__)
from pyannote.core import SlidingWindow, SlidingWindowFeature, Annotation, Segment, Timeline

def run(name, f):
    print(f"\n### {name}")
    try:
        r = f()
        print("OK ->", r)
    except Exception as e:
        print("EXC ->", type(e).__name__, e)
        traceback.print_exc(limit=3)

from pyannote.audio.utils.signal import Binarize, Peak, binarize
sw = SlidingWindow(start=0.0, duration=0.02, step=0.02)
run("Binarize 1 frame active", lambda: Binarize()(SlidingWindowFeature(np.array([[0.9]]), sw)))
run("Binarize 1 frame inactive", lambda: Binarize()(SlidingWindowFeature(np.array([[0.1]]), sw)))
run("Binarize 2 frames active", lambda: Binarize()(SlidingWindowFeature(np.array([[0.9],[0.9]]), sw)))
run("Binarize 0 frames", lambda: Binarize()(SlidingWindowFeature(np.zeros((0,1)), sw)))
run("Binarize offset=0.0 (onset .5) on [.9,.3,.3]", lambda: Binarize(onset=0.5, offset=0.0)(SlidingWindowFeature(np.array([[0.9],[0.3],[0.3],[0.3]]), sw)))
run("binarize fn offset=0.0", lambda: binarize(np.array([[0.9,0.3,0.3]]), onset=0.5, offset=0.0))
run("Peak", lambda: Peak(alpha=0.1, min_duration=0.04)(SlidingWindowFeature(np.array([[0.],[0.5],[0.],[0.],[0.6],[0.]]), sw)))

from pyannote.audio.utils.preprocessors import DeriveMetaLabels
def dml_unions():
    ann = Annotation(uri="f")
    ann[Segment(0, 5), 0] = "male"
    ann[Segment(6, 8), 1] = "music"
    d = DeriveMetaLabels(classes=["male","music"], unions={"speech": ["male"], "anything": ["male","music"]})
    out = d({"annotation": ann})
    return out.labels(), list(out.itertracks(yield_label=True))
run("DeriveMetaLabels two unions sharing a segment", dml_unions)
def dml_track_collision():
    ann = Annotation(uri="f")
    ann[Segment(0, 5), "A"] = "male"
    ann[Segment(0, 5), "B"] = "female"
    d = DeriveMetaLabels(classes=["male","female"], unions={"speech": ["male"]})
    out = d({"annotation": ann})
    return out.labels(), list(out.itertracks(yield_label=True))
run("DeriveMetaLabels union track 'A' collides with existing track", dml_track_collision)
def dml_inter():
    ann = Annotation(uri="f")
    ann[Segment(0, 5)] = "male"     # track "_"
    ann[Segment(0, 5), "x"] = "music"
    d = DeriveMetaLabels(classes=["male","music"], intersections={"male_music": ["male","music"]})
    out = d({"annotation": ann})
    return out.labels(), list(out.itertracks(yield_label=True))
run("DeriveMetaLabels intersection '_' track collides", dml_inter)
def dml_two_inter():
    ann = Annotation(uri="f")
    ann[Segment(0, 5), 0] = "a"; ann[Segment(0, 5), 1] = "b"; ann[Segment(0, 5), 2] = "c"
    d = DeriveMetaLabels(classes=["a","b","c"], intersections={"ab": ["a","b"], "abc": ["a","b","c"]})
    out = d({"annotation": ann})
    return out.labels(), list(out.itertracks(yield_label=True))
run("DeriveMetaLabels two intersections same segment", dml_two_inter)

from pyannote.audio import Audio
run("Audio stereo 1 sample", lambda: Audio()({"waveform": torch.zeros(2,1), "sample_rate": 16000})[0].shape)
run("Audio channel=3 on stereo", lambda: Audio(mono="downmix")({"waveform": torch.zeros(2,1600), "sample_rate": 16000, "channel": 3})[0].shape)

from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio import Inference
from pyannote.audio.pipelines import VoiceActivityDetection
torch.manual_seed(0)
for dur in [0.005, 0.01, 0.02, 0.03, 0.05, 0.3]:
    def vad(dur=dur):
        m = PyanNet(sample_rate=16000)
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        m.specifications = Specifications(problem=Problem.MONO_LABEL_CLASSIFICATION if False else Problem.BINARY_CLASSIFICATION, resolution=Resolution.FRAME, duration=2.0, classes=["speech"])
        m.build()
        class Always(type(m)): pass
        p = VoiceActivityDetection(segmentation=m)
        p.instantiate({"onset": -1.0, "offset": -1.0, "min_duration_on": 0.0, "min_duration_off": 0.0})
        n = int(dur*16000)
        return p({"waveform": torch.randn(1, n), "sample_rate": 16000, "uri": "x"})
    run(f"VAD always-on dur={dur}", vad)
