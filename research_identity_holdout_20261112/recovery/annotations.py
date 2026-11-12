"""Evaluation-only exact clocks; never used by runtime selection or policy."""
from common import ROOT, read

def overlap(left, right):
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(0., min(left[3], right[3]) - max(left[1], right[1]))
    union = max(0., left[2] - left[0]) * max(0., left[3] - left[1]) + max(0., right[2] - right[0]) * max(0., right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.

class ExactAnnotations:
    def __init__(self, plan):
        self.plan = plan
        self.videos = {}
        self.cache = {}

    def at(self, video, frame):
        if video not in self.videos:
            metadata = self.plan['videos'][video]
            if not (ROOT / metadata['annotation_clock']).exists() or not (ROOT / metadata['annotation_boxes']).exists():
                self.videos[video] = None
                return dict(exact=False, reason='missing_exact_annotation_source', objects=[])
            clock = read(ROOT / metadata['annotation_clock'])['anchors']
            raw = read(ROOT / metadata['annotation_boxes'])
            labels = {int(row.get('frameIndex', i)): row for i, row in enumerate(raw)}
            self.videos[video] = clock, labels
        if self.videos[video] is None:
            return dict(exact=False, reason='missing_exact_annotation_source', objects=[])
        clock, labels = self.videos[video]
        key = video, int(frame)
        if key not in self.cache:
            if str(frame) not in clock:
                self.cache[key] = dict(exact=False, reason='missing_exact_raw_frame_clock_anchor', objects=[])
            else:
                anchor = clock[str(frame)]
                row = labels[int(anchor['frameIndex'])]
                assert row['name'] == anchor['name']
                objects = [dict(identity=str(obj['id']), box=[float(obj['box2d'][k]) for k in ['x1', 'y1', 'x2', 'y2']]) for obj in row.get('labels') or [] if obj['category'] == 'car' and not (obj.get('attributes') or {}).get('crowd', False)]
                self.cache[key] = dict(exact=True, reason='exact_clock_name_frameIndex_verified', anchor=anchor, objects=objects)
        return self.cache[key]

    def hits(self, video, frame, box):
        annotation = self.at(video, frame)
        values = [dict(identity=obj['identity'], box=obj['box'], iou=overlap(box, obj['box'])) for obj in annotation['objects'] if overlap(box, obj['box']) >= 0.5]
        return dict(exact=annotation['exact'], reason=annotation['reason'], hits=values)

    def historical_expected(self, video, observations, descriptor_source_frame):
        unique = {}
        for observation in observations:
            key = int(observation['frame']), tuple(observation['box'])
            unique[key] = observation
        anchors = []
        for _, observation in sorted(unique.items()):
            value = self.hits(video, observation['frame'], observation['box'])
            if value['exact']:
                anchors.append(dict(frame=int(observation['frame']), box=observation['box'], evaluation=value))
        identities = [item['evaluation']['hits'][0]['identity'] for item in anchors if len(item['evaluation']['hits']) == 1]
        problems = []
        if not anchors:
            problems.append('no_exact_past_historical_anchor')
        if any(len(item['evaluation']['hits']) != 1 for item in anchors):
            problems.append('annotated_history_unmatched_or_ambiguous')
        if len(set(identities)) > 1:
            problems.append('conflicting_annotated_history_identities')
        valid = bool(anchors) and not problems
        return dict(valid=valid, problems=problems, expected_identity=identities[0] if valid else None,
                    actual_source_anchor_frame=max(item['frame'] for item in anchors) if valid else None, all_annotated_history=anchors,
                    descriptor_last_source_frame=descriptor_source_frame, source_pixel_identity_assumed=False,
                    task='Secondary expected identity continuity from exact historical observation(s) to exact current GT; never labels unannotated last descriptor source.')

def classify(reference, current):
    left = [hit['identity'] for hit in reference['hits']]
    right = [hit['identity'] for hit in current['hits']]
    if not reference['exact'] or not current['exact'] or len(left) != 1 or len(right) != 1:
        return 'unknown'
    return 'own' if left == right else 'foreign'

def classify_historical(expected, current):
    if not expected['valid'] or not current['exact'] or len(current['hits']) != 1:
        return 'unknown'
    return 'own' if expected['expected_identity'] == current['hits'][0]['identity'] else 'foreign'
